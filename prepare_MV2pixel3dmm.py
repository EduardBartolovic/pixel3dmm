import os
import subprocess
import tempfile
from collections import defaultdict
from tqdm import tqdm

ROOT_DIR = "/home/gustav/pixel3dmm/rgb_bff_crop"
OUTPUT_DIR = "/home/gustav/pixel3dmm/rgb_bff_crop_vids/"
FPS = 2  # Frames per second of output video

os.makedirs(OUTPUT_DIR, exist_ok=True)

person_folders = sorted([
    f for f in os.listdir(ROOT_DIR)
    if os.path.isdir(os.path.join(ROOT_DIR, f))
])

for person_folder in tqdm(person_folders, desc="Iterating over IDs"):
    person_path = os.path.join(ROOT_DIR, person_folder)
    if not os.path.isdir(person_path):
        continue

    # Organize images by their 40-char hash prefix
    hash_groups = defaultdict(list)
    for filename in sorted(os.listdir(person_path)):
        if not filename.endswith(".jpg"):
            continue
        hash_prefix = filename[:40]
        hash_groups[hash_prefix].append(os.path.join(person_path, filename))

    for hash_prefix, image_list in hash_groups.items():
        # Output file path
        if '@' in person_folder:
            id_folder = person_folder.split("@")[0].replace(".","")
        else:
            id_folder = person_folder
        output_filename = f"{id_folder}_{hash_prefix}.mp4"
        output_path = os.path.join(OUTPUT_DIR, output_filename)

        # Write image list to temp file
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as list_file:
            for image_path in image_list:
                list_file.write(f"file '{os.path.abspath(image_path)}'\n")
            list_file_path = list_file.name

        # ffmpeg command to encode images as video
        cmd = [
            "ffmpeg",
            "-f", "concat",
            "-safe", "0",
            "-i", list_file_path,
            "-framerate", str(FPS),
            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2,pad=iw+200:ih+200:100:100", # With extra Padding
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-preset", "fast",
            "-y",
            output_path
        ]


        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            #print(f"✅ Video gespeichert: {output_path}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Fehler beim Erstellen des Videos für {hash_prefix}: {e}")

        os.remove(list_file_path)
