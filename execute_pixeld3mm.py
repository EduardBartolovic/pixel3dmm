import os
import sys
import time
from omegaconf import OmegaConf
from pixel3dmm import env_paths
import torch
import gc
import multiprocessing as mp

script_dir = os.path.join(os.path.dirname(__file__), 'scripts')
sys.path.append(script_dir)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Import scripts
import run_preprocessing
import network_inference
import track
import viz_head_centric_cameras

def is_already_processed(vid_name):
    """
    Check if the video has already been processed.
    """
    output_dir = os.path.join(env_paths.CODE_BASE, "tracking_results", vid_name+"_nV1_noPho_uv2000.0_n1000.0")
    return os.path.exists(output_dir) and os.path.isdir(output_dir)

def process_video(video_path):
    start_time = time.time()

    # Extract base name
    base_name = os.path.basename(video_path)
    vid_name = os.path.splitext(base_name)[0]

    if is_already_processed(vid_name):
        print(f"⏩ Skipping '{vid_name}' – already processed.")
        return

    print(f"Processing video: {vid_name}")

    # Step 1: Preprocessing
    run_preprocessing.main(video_or_images_path=video_path)
    print("✅ Preprocessing done!")

    # Step 2: Load base config
    base_cfg_path = os.path.join(env_paths.CODE_BASE, 'configs', 'base.yaml')
    base_conf = OmegaConf.load(base_cfg_path)

    # Step 3: Run network_inference for both prediction types
    for prediction_type in ['normals', 'uv_map']:
        cli_conf = OmegaConf.from_dotlist([
            f'model.prediction_type={prediction_type}',
            f'video_name={vid_name}'
        ])
        cfg = OmegaConf.merge(base_conf, cli_conf)
        network_inference.main(cfg)
        print(f"✅ Network inference for '{prediction_type}' done!")

    # Step 4: Tracking
    track_cfg_path = os.path.join(env_paths.CODE_BASE, 'configs', 'tracking.yaml')
    track_conf = OmegaConf.load(track_cfg_path)
    track_cli_conf = OmegaConf.from_dotlist([
        f'video_name={vid_name}',
        'iters=100',
        'iters=1500',
        'include_neck=False',
        'w_exp=0.1',
        'use_mouth_lmk=False',
        'w_shape=0.01',
        'w_shape_general=0.001',
        'normal_super=2000.0',
        'sil_super=1000.0',
        'use_flame2023=True',
        'ignore_mica=True',
        'is_discontinuous=True'
    ])
    track_cfg = OmegaConf.merge(track_conf, track_cli_conf)
    # print(track_cfg)
    track.main(track_cfg)
    print("✅ Tracking done!")

    # Step 5: Visualization
    viz_head_centric_cameras.main(vid_name=vid_name)
    print("✅ Visualization done!")

    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    # Timing
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"🕒 Total processing time: {elapsed:.2f} seconds")

# Example usage
if __name__ == '__main__':
    #PATH_TO_VIDEO = "/home/duck/pixel3dmm/example_videos/test1.mp4"
    #process_video(PATH_TO_VIDEO)
    mp.set_start_method('spawn', force=True)
    video_folder = "/home/gustav/pixel3dmm/vox2test_crop_vids"
    for video in sorted(os.listdir(video_folder)):
        video_path = os.path.join(video_folder, video)
        print(f"Processing: {video_path}")
        p = mp.Process(target=process_video, args=(video_path,))
        p.start()
        p.join()

        # exit()

        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()

