import os
import torch
import numpy as np
import pprint

def summarize_array(arr, max_elements=5):
    """Returns a compact preview of an array for printing."""
    arr = np.array(arr)
    shape = arr.shape
    preview = arr.flatten()[:max_elements]
    return f"shape={shape}, preview={preview.tolist()}"

def print_frame_data(frame):
    """
    Nicely print the content of a loaded .frame dictionary.
    """
    print(f"\n--- Frame ID: {frame.get('frame_id')} ---")
    print(f"Global step: {frame.get('global_step')}")
    print(f"Image size: {frame.get('img_size')}")

    # Print flame parameters
    print("\n[FLAME Params]")
    flame = frame.get('flame', {})
    for key, value in flame.items():
        print(f"  {key}: {summarize_array(value)}")

    # Print joint transforms
    print("\n[Joint Transforms]")
    jt = frame.get('joint_transforms', None)
    if jt is not None:
        print(f"  joint_transforms: {summarize_array(jt)}")

    # Print camera info
    print("\n[Camera Params]")
    cam = frame.get('camera', {})
    for key, value in cam.items():
        print(f"  {key}: {summarize_array(value)}")

def load_all_frames(folder_path):
    """
    Loads all .frame files from the given folder and returns a list of frame dictionaries.
    
    Args:
        folder_path (str): Path to the folder containing .frame files.
    
    Returns:
        List[dict]: List of loaded frame data.
    """
    frames = []

    # List and sort the files to ensure chronological order (e.g., 00001.frame, 00002.frame, ...)
    frame_files = sorted([
        f for f in os.listdir(folder_path) if f.endswith('.frame')
    ])

    if not frame_files:
        print("No .frame files found in the folder.")
        return []

    for fname in frame_files:
        full_path = os.path.join(folder_path, fname)
        try:
            frame_data = torch.load(full_path, map_location='cpu', weights_only=False)
            frames.append(frame_data)
        except Exception as e:
            print(f"Failed to load {fname}: {e}")

    print(f"Loaded {len(frames)} frame(s).")
    return frames

# Example usage:
if __name__ == "__main__":
    folder = "/home/duck/pixel3dmm/tracking_results/vio_nV1_noPho_uv2000.0_n1000.0/checkpoint"
    all_frames = load_all_frames(folder)
    
    # Example: Print the first frame's ID
    for i in all_frames:
        print_frame_data(i)
