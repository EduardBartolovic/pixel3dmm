import os
import subprocess
import tempfile
from tqdm import tqdm

ROOT_DIR = "/home/gustav/vox/vox2test/"
OUTPUT_DIR = "/home/gustav/vox/vox2test_preped_vid_cap250/"

os.makedirs(OUTPUT_DIR, exist_ok=True)

person_folders = sorted([
    f for f in os.listdir(ROOT_DIR)
    if os.path.isdir(os.path.join(ROOT_DIR, f))
])

person_count = 0

for person_folder in tqdm(person_folders, desc="Iterrating over ids"):
    person_path = os.path.join(ROOT_DIR, person_folder)
    if not os.path.isdir(person_path):
        continue

    person_count += 1
    video_count = 0

    for video_folder in sorted(os.listdir(person_path)):
        video_path = os.path.join(person_path, video_folder)
        if not os.path.isdir(video_path):
            continue

        video_files = [
            os.path.join(video_path, f)
            for f in sorted(os.listdir(video_path))
            if f.endswith(".mp4")
        ]

        if not video_files:
            continue

        # Erstelle temporäre Datei mit Liste der Videos
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as list_file:
            for video in video_files:
                list_file.write(f"file '{os.path.abspath(video)}'\n")
            list_file_path = list_file.name

        output_filename = f"{person_folder}_{video_folder}.mp4"
        output_path = os.path.join(OUTPUT_DIR, output_filename)

        # ffmpeg-Kommando zum Zusammenfügen und Kürzen auf 500 Frames
        cmd = [
            "ffmpeg",
            "-f", "concat",
            "-safe", "0",
            "-i", list_file_path,
            "-c:v", "libx264",         # Re-encode to allow frame trimming
            "-preset", "fast",
            "-frames:v", "250",        # Max N frames
            "-y",                      # Overwrite output if exists
            output_path
        ]

        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"✅ Gespeichert (max N Frames): {output_path}")
        except subprocess.CalledProcessError as e:
            raise Exception(f"❌ Fehler beim Zusammenfügen: {e}")

        video_count += 1

        # Aufräumen
        os.remove(list_file_path)
