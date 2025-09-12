import os
import shutil

# paths
src_root = "/home/duck/pixel3dmm/corr_out/"   # folder with idXXXXX_hash folders
dst_root = "/home/duck/pixel3dmm/vox2test_corr/"    # output

os.makedirs(dst_root, exist_ok=True)

# mapping from file index to string
mapping = {
    0: "-10_-10",
    1: "-10_-25",
    2: "-10_0",
    3: "-10_10",
    4: "-10_25",

    5: "-25_-10",
    6: "-25_-25",
    7: "-25_0",
    8: "-25_10",
    9: "-25_25",

    10: "0_-10",
    11: "0_-25",
    12: "0_0",
    13: "0_10",
    14: "0_25",

    15: "10_-10",
    16: "10_-25",
    17: "10_0",
    18: "10_10",
    19: "10_25",

    20: "25_-10",
    21: "25_-25",
    22: "25_0",
    23: "25_10",
    24: "25_25",
}

allowed = {'0_0', '25_-25', '25_25', '10_-10', '10_10', '0_-25', '0_25', '25_0'}

for folder in os.listdir(src_root):
    folder_path = os.path.join(src_root, folder)
    if not os.path.isdir(folder_path):
        continue
    
    # split folder name into id + sample
    class_id, sample = folder.split("_", 1)

    # make class folder inside output
    dst_class_path = os.path.join(dst_root, class_id)
    os.makedirs(dst_class_path, exist_ok=True)
    
    # copy and rename files
    for file in os.listdir(folder_path):
        if file.endswith("_corr.npz"):
            # extract file index, e.g. "00012_corr.npz" -> 12
            idx = int(file.split("_")[0])
            if idx not in mapping:
                print(f"⚠️ Skipping {file}, no mapping found")
                continue
            
            mapped_name = mapping[idx]
            if mapped_name not in allowed:
                # skip if not in the whitelist
                continue
            
            src_file = os.path.join(folder_path, file)
            new_name = f"{sample}{mapped_name}_corr.npz"
            dst_file = os.path.join(dst_class_path, new_name)

            shutil.copy2(src_file, dst_file)

print("✅ Done copying + renaming filtered files!")
