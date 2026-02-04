import os
import json
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import transform_matrix
from utils.results_utils import project_wp_to_image

# 配置参数
JSONL_FILE = "driver_predicted_waypoints_3.jsonl"
DATA_ROOT = "/home/ximeng/Dataset/nuscenes_full_v1_0/"
VERSION = 'v1.0-mini'
OUTPUT_DIR = "figures"  # 改为输出文件夹名称

def main():
    print("Loading NuScenes...")
    nusc = NuScenes(version=VERSION, dataroot=DATA_ROOT, verbose=False)
    
    with open(JSONL_FILE, 'r') as f:
        lines = f.readlines()
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: {OUTPUT_DIR}")
    else:
        print(f"Output directory exists: {OUTPUT_DIR}")

    print(f"Generating images to {OUTPUT_DIR} ({len(lines)} frames)...")
    
    for i, line in enumerate(tqdm(lines)):
        data = json.loads(line)
        token = data['token'][0] if isinstance(data['token'], list) else data['token']        
        
        # Get the image
        sample_rec = nusc.get('sample', token)
        cam_front_data = nusc.get('sample_data', sample_rec['data']['CAM_FRONT'])
        img_path = os.path.join(nusc.dataroot, cam_front_data['filename'])
        raw_img = cv2.imread(img_path)

        pred_pts = np.array(data['predicted_output'], dtype=np.float32)
        gt_pts = np.array(data['gt_waypoints'], dtype=np.float32)
        
        # Draw predicted waypoints 
        vis_img = project_wp_to_image(
            nusc, token, pred_pts, raw_img,
            color_waypoints=(255, 0, 0),
            color_polygon=(200, 0, 0),
            plot_polygon=True
        )

        # Draw ground truth waypoints
        vis_img = project_wp_to_image(
            nusc, token, gt_pts, vis_img,
            color_waypoints=(0, 255, 0),
            plot_polygon=False
        )

        file_name = f"{i:05d}_{token}.jpg"
        save_path = os.path.join(OUTPUT_DIR, file_name)
        
        cv2.imwrite(save_path, vis_img)

    print(f"\nAll images saved successfully to directory: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()