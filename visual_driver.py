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

JSONL_FILE = "driver_predicted_waypoints.jsonl"
DATA_ROOT = "/home/ximeng/Dataset/nuscenes_full_v1_0/"
VERSION = 'v1.0-mini'
VIDEO_OUTPUT = "driving_prediction.mp4"
FPS = 2

def main():
    print("Loading NuScenes...")
    nusc = NuScenes(version=VERSION, dataroot=DATA_ROOT, verbose=False)
    with open(JSONL_FILE, 'r') as f:
        lines = f.readlines() 
        
    first_data = json.loads(lines[0])
    token = first_data['token'][0] if isinstance(first_data['token'], list) else first_data['token']
    sample_rec = nusc.get('sample', token)
    cam_data = nusc.get('sample_data', sample_rec['data']['CAM_FRONT'])
    width, height = cam_data['width'], cam_data['height']

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(VIDEO_OUTPUT, fourcc, FPS, (width, height))
    print(f"Generating video: {VIDEO_OUTPUT} ({len(lines)} frames)...")
    
    for line in tqdm(lines):
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

        video_writer.write(vis_img)

    video_writer.release()
    print(f"\nVideo saved successfully to {VIDEO_OUTPUT}")

if __name__ == "__main__":
    main()