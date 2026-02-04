"""
Precompute image paths for training data.
Run once: python precompute_paths.py
Then use nusscenes_reasons_with_paths.jsonl for training.
"""
import os
import json
from tqdm import tqdm
from nuscenes.nuscenes import NuScenes

NUSCENES_DATAROOT = "/scratch/group/p.cis250376.000/dataset/nuscenes"
NUSCENES_VERSION = "v1.0-trainval"
CAMERAS = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_RIGHT", "CAM_BACK", "CAM_BACK_LEFT"]

INPUT_FILE = "nusscenes_reasons.jsonl"
OUTPUT_FILE = "nusscenes_reasons_with_paths.jsonl"

print("Loading NuScenes...")
nusc = NuScenes(version=NUSCENES_VERSION, dataroot=NUSCENES_DATAROOT, verbose=False)

def get_image_paths_from_token(token):
    sample = nusc.get('sample', token)
    image_paths = []
    for cam in CAMERAS:
        cam_data = nusc.get('sample_data', sample['data'][cam])
        image_path = os.path.join(NUSCENES_DATAROOT, cam_data['filename'])
        image_paths.append(image_path)
    return image_paths

print(f"Processing {INPUT_FILE}...")
with open(INPUT_FILE, 'r') as f_in, open(OUTPUT_FILE, 'w') as f_out:
    lines = f_in.readlines()
    for line in tqdm(lines, desc="Precomputing paths"):
        data = json.loads(line)
        token = data['token']
        if isinstance(token, list):
            token = token[0]
        
        image_paths = get_image_paths_from_token(token)
        data['image_paths'] = image_paths
        
        f_out.write(json.dumps(data) + '\n')

print(f"Done! Saved to {OUTPUT_FILE}")
