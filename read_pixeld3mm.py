import os
import torch
import numpy as np
from tqdm import tqdm

def extract_shape(frame):
    """Extracts the FLAME shape parameter if present."""
    shape = frame.get('flame', {}).get('shape')
    return np.array(shape) if shape is not None else None

def collect_all_shapes(root_folder):
    """
    Walks through all subfolders in root_folder and collects all shape vectors.
    
    Returns:
        identity_ids: np.array of shape (N,) with identity strings
        frame_filenames: np.array of shape (N,) with frame file names
        shape_array: np.array of shape (N, D) with shape vectors
    """
    identity_ids = []
    frame_filenames = []
    shape_array = []

    # Traverse subfolders
    for subdir in sorted(os.listdir(root_folder)):
        subpath = os.path.join(root_folder, subdir, "checkpoint")
        if not os.path.isdir(subpath):
            continue
        
        frame_files = sorted([
            f for f in os.listdir(subpath) if f.endswith(".frame")
        ])

        for fname in frame_files:
            full_path = os.path.join(subpath, fname)
            try:
                frame_data = torch.load(full_path, map_location='cpu', weights_only=False)
                shape = extract_shape(frame_data)
                if shape is not None:
                    identity_ids.append(subdir.split('_')[0])
                    frame_filenames.append(subdir)
                    shape_array.append(shape)
                    break
            except Exception as e:
                print(f"Failed to load {full_path}: {e}")

    # Convert to numpy arrays
    identity_ids = np.array(identity_ids)
    frame_filenames = np.array(frame_filenames)
    shape_array = np.stack(shape_array) if shape_array else np.array([])
    shape_array = np.squeeze(shape_array)  # Remove any size-1 dimensions

    return identity_ids, frame_filenames, shape_array


# Example usage
if __name__ == "__main__":
    root_folder = "/home/gustav/pixel3dmm/tracking_results/"
    ids, file_names, shapes = collect_all_shapes(root_folder)

    print(f"Total frames collected: {len(shapes)}")
    print(f"Shape array shape: {shapes.shape}")
    print(f"First identity/file_names/shape: {ids[0]}, {file_names[0]}, {shapes[0][:5]}")
    for i in range(0,5): 
        print(f"First identity/frame/shape: {ids[i]}, {file_names[i]}, {shapes[i][:5]}")
    
    # Optional: save
    np.save("shape_array.npy", shapes)
    np.save("identity_ids.npy", ids)
    np.save("file_names.npy", file_names)

