import os
import tyro
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pixel3dmm.utils.utils_3d import rotation_6d_to_matrix
from pixel3dmm.env_paths import TRACKING_OUTPUT, PREPROCESSED_DATA
import trimesh
from skimage.draw import polygon
import torchvision.transforms.functional as TF

def barycentric_coords(p, a, b, c):
    v0, v1, v2 = b - a, c - a, p - a
    d00, d01, d11 = np.dot(v0, v0), np.dot(v0, v1), np.dot(v1, v1)
    d20, d21 = np.dot(v2, v0), np.dot(v2, v1)
    denom = d00 * d11 - d01 * d01 + 1e-8
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    return u, v, w

def visible_faces_mask(verts_proj, verts_cam, faces, image_size):
    H, W = image_size
    verts_z = -verts_cam[:, 2]

    z_buffer = np.full((H, W), np.inf, dtype=np.float32)
    tri_buffer = np.full((H, W), -1, dtype=np.int32)
    face_mask = np.zeros(len(faces), dtype=bool)

    for tri_id, (i0, i1, i2) in enumerate(faces):
        pts = verts_proj[[i0, i1, i2]]
        z_vals = verts_z[[i0, i1, i2]]

        if not np.all(np.isfinite(pts)) or not np.all(np.isfinite(z_vals)):
            continue

        # Bounding box
        min_x = max(int(np.floor(np.min(pts[:, 0]))), 0)
        max_x = min(int(np.ceil(np.max(pts[:, 0]))), W - 1)
        min_y = max(int(np.floor(np.min(pts[:, 1]))), 0)
        max_y = min(int(np.ceil(np.max(pts[:, 1]))), H - 1)

        if max_x < min_x or max_y < min_y:
            continue

        # Generate all pixels in bounding box
        xx, yy = np.meshgrid(np.arange(min_x, max_x + 1),
                             np.arange(min_y, max_y + 1))
        pix = np.stack([xx.ravel(), yy.ravel()], axis=-1).astype(np.float32)

        # Vectorized barycentric coordinates
        u, v, w = barycentric_coords(pix, pts[0], pts[1], pts[2])
        inside = (u >= 0) & (v >= 0) & (w >= 0)
        if not np.any(inside):
            continue

        pix_inside = pix[inside]
        depth_inside = (u[inside] * z_vals[0] +
                        v[inside] * z_vals[1] +
                        w[inside] * z_vals[2])

        # Depth test
        for (x, y), depth in zip(pix_inside.astype(int), depth_inside):
            if depth < z_buffer[y, x]:
                z_buffer[y, x] = depth
                tri_buffer[y, x] = tri_id

    visible_tri_indices = np.unique(tri_buffer[tri_buffer >= 0])
    face_mask[visible_tri_indices] = True
    return face_mask


def project_points(points_3d, intrinsics, extrinsics):
    points_3d_h = np.hstack([points_3d, np.ones((points_3d.shape[0], 1))])
    camera_coords = (extrinsics @ points_3d_h.T).T[:, :3]
    pixels_h = (intrinsics @ camera_coords.T).T
    pixels = pixels_h[:, :2] / pixels_h[:, 2:3]
    return pixels, camera_coords

def calculate_visible_faces(ckpt_path, mesh_path, IMAGE_WIDTH, IMAGE_HEIGHT):
    meshes = sorted(f for f in os.listdir(mesh_path) if f.endswith('.ply') and 'canonical' not in f)
    ckpts = sorted(f for f in os.listdir(ckpt_path) if f.endswith('.frame'))
    N = min(len(meshes), len(ckpts))
    assert len(meshes) == len(ckpts)

    vis_data = {}
    for idx in range(N):
        ckpt = torch.load(os.path.join(ckpt_path, ckpts[idx]), weights_only=False)
        mesh = trimesh.load(os.path.join(mesh_path, meshes[idx]), process=False)
        verts = mesh.vertices.copy()

        head_rot = rotation_6d_to_matrix(torch.from_numpy(ckpt['flame']['R'])).numpy()[0]
        verts_h = np.hstack([verts, np.ones((verts.shape[0], 1))])
        verts_h = verts_h @ np.linalg.inv(ckpt['joint_transforms'][0, 1])
        mesh.vertices = verts_h[:, :3]

        extr = np.eye(4)
        extr[:3, :3] = ckpt['camera']['R_base_0'][0]
        extr[:3, 3] = ckpt['camera']['t_base_0'][0]

        flame2world = np.eye(4)
        flame2world[:3, :3] = head_rot
        flame2world[:3, 3] = ckpt['flame']['t'].squeeze()
        extr = extr @ flame2world

        intr = np.eye(3)
        fl = ckpt['camera']['fl'][0, 0] * IMAGE_WIDTH
        intr[0, 0], intr[1, 1] = fl, fl
        intr[0, 2] = ckpt['camera']['pp'][0, 0] * (IMAGE_WIDTH / 2 + 0.5) + IMAGE_WIDTH / 2 + 0.5
        intr[1, 2] = ckpt['camera']['pp'][0, 1] * (IMAGE_HEIGHT / 2 + 0.5) + IMAGE_HEIGHT / 2 + 0.5

        verts_proj, verts_cam = project_points(mesh.vertices, intr, extr)

        valid_mask = (
            (verts_proj[:, 0] >= 0) & (verts_proj[:, 0] < IMAGE_WIDTH) &
            (verts_proj[:, 1] >= 0) & (verts_proj[:, 1] < IMAGE_HEIGHT)
        )

        face_normals_cam = mesh.face_normals
        face_centers_cam = verts_cam[mesh.faces].mean(axis=1)
        view_dir = -face_centers_cam / (np.linalg.norm(face_centers_cam, axis=1, keepdims=True) + 1e-8)
        normal_dot_view = np.einsum('ij,ij->i', face_normals_cam, view_dir)
        face_front_mask = normal_dot_view > 0
        visible_faces = mesh.faces[face_front_mask]

        valid_faces_mask = valid_mask[visible_faces].any(axis=1)
        visible_faces = visible_faces[valid_faces_mask]

        face_vis_mask = visible_faces_mask(verts_proj, verts_cam, mesh.faces, (IMAGE_HEIGHT, IMAGE_WIDTH))
        visible_faces = mesh.faces[face_vis_mask]

        vis_data[idx] = {
            'face_mask': face_vis_mask,     
            'face_mesh': mesh.copy(),
            'verts_proj': verts_proj.copy()
        }

    return vis_data


def load_reference_image(vid_name, ref_frame, w, h):
    """Load and horizontally flip the reference frame image."""
    ref_img_path = f'{PREPROCESSED_DATA}/{vid_name}/cropped/{ref_frame:05d}.jpg'
    ref_img = Image.open(ref_img_path).resize((w, h))
    return ref_img


def save_face_mask_image(out_dir, mesh, faces_idx, verts_proj, img_size, fname):
    """Save a binary mask of given faces for visualization."""
    mask_img = np.zeros(img_size, dtype=np.uint8)
    for tri in mesh.faces[faces_idx]:
        rr, cc = polygon(verts_proj[tri, 1], verts_proj[tri, 0], img_size)
        mask_img[rr, cc] = 255
    plt.imsave(os.path.join(out_dir, fname), mask_img, cmap='gray')


def compute_dense_flow(ref_faces, src_faces, ref_proj, src_proj, w, h):
    """Compute dense barycentric-based flow between ref and src faces."""
    dense_flow = np.zeros((h, w, 2), dtype=np.float32)

    ref_proj_flipped = ref_proj.copy()
    src_proj_flipped = src_proj.copy()

    ref_proj_flipped[:, 0] = w - 1 - ref_proj_flipped[:, 0]
    src_proj_flipped[:, 0] = w - 1 - src_proj_flipped[:, 0]

    for face_ref, face_src in zip(ref_faces, src_faces):
        ref_tri = ref_proj_flipped[face_ref]
        src_tri = src_proj_flipped[face_src]
        rr, cc = polygon(ref_tri[:, 1], ref_tri[:, 0], (h, w))
        if len(rr) == 0:
            continue
        P = np.stack([cc, rr], axis=-1).astype(np.float32)
        u, v, w_ = barycentric_coords(P, ref_tri[0], ref_tri[1], ref_tri[2])
        src_pix = u[:, None] * src_tri[0] + v[:, None] * src_tri[1] + w_[:, None] * src_tri[2]
        dense_flow[rr, cc, 0] = (src_pix[:, 0] / (w - 1)) * 2 - 1
        dense_flow[rr, cc, 1] = (src_pix[:, 1] / (h - 1)) * 2 - 1
    return dense_flow


def warp_image(src_img_path, dense_flow, w, h):
    """Warp source image using given dense flow."""
    img_target = Image.open(src_img_path).resize((w, h))
    img_tensor = TF.to_tensor(img_target).unsqueeze(0)
    flow_torch = torch.from_numpy(dense_flow).unsqueeze(0)
    warped_tensor = F.grid_sample(img_tensor, flow_torch, mode='bilinear', align_corners=True)
    return TF.to_pil_image(warped_tensor.squeeze(0).clamp(0, 1)), img_target


def main(vid_name: str):
    IMAGE_WIDTH, IMAGE_HEIGHT = 256, 256
    ref_frame = 12 # front pose

    tracking_dir = f'{TRACKING_OUTPUT}/{vid_name}_nV1_noPho_uv2000.0_n1000.0'
    mesh_path = os.path.join(tracking_dir, 'mesh')
    ckpt_path = os.path.join(tracking_dir, 'checkpoint')
    out_dir = f'/home/gustav/pixel3dmm/tmp_out/{vid_name}'
    os.makedirs(out_dir, exist_ok=True)

    print("Calculating visible faces...")
    vis_data = calculate_visible_faces(ckpt_path, mesh_path, IMAGE_WIDTH, IMAGE_HEIGHT)

    ref_data = vis_data[ref_frame]
    ref_img = load_reference_image(vid_name, ref_frame, IMAGE_WIDTH, IMAGE_HEIGHT)

    correspondences = {}
    for idx, data in vis_data.items():
        if idx == ref_frame:
            continue
        common_faces = np.where(ref_data['face_mask'] & data['face_mask'])[0]
        if len(common_faces) == 0:
            print("No common_faces!!!!")
            continue
        correspondences[idx] = {
            **data,
            'common_faces': common_faces,
            'ref_mesh': ref_data['face_mesh'],
            'ref_verts_proj': ref_data['verts_proj']
        }

    for src_idx, corrs in correspondences.items():
        save_face_mask_image(
            out_dir,
            corrs['ref_mesh'],
            corrs['common_faces'],
            corrs['ref_verts_proj'],
            (IMAGE_HEIGHT, IMAGE_WIDTH),
            f"visible_faces_mask_{src_idx:03d}.png"
        )

        ref_faces = corrs['ref_mesh'].faces[corrs['common_faces']]
        src_faces = corrs['face_mesh'].faces[corrs['common_faces']]
        dense_flow = compute_dense_flow(ref_faces, src_faces,
                                        corrs['ref_verts_proj'], corrs['verts_proj'],
                                        IMAGE_WIDTH, IMAGE_HEIGHT)

        src_img_path = f'{PREPROCESSED_DATA}/{vid_name}/cropped/{src_idx:05d}.jpg'
        warped_img, img_target = warp_image(src_img_path, dense_flow, IMAGE_WIDTH, IMAGE_HEIGHT)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for ax, img, title in zip(
            axes, [ref_img, warped_img, img_target],
            ["Frame 12 (ref)", f"Frame {src_idx} warped", f"Frame {src_idx}"]
        ):
            ax.imshow(img)
            ax.set_title(title)
            ax.axis('off')
        plt.tight_layout()
        plt.savefig(f"{out_dir}/warped_img_12_from_{src_idx:03d}.png", dpi=200)
        plt.close()

if __name__ == '__main__':
    tyro.cli(main)
