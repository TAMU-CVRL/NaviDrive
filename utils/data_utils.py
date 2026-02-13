import os
import json
from tqdm import tqdm
from PIL import Image
import numpy as np
from pathlib import Path
from nuscenes.utils.geometry_utils import view_points
import yaml

def save_triplet_dataset_jsonl(dataset, save_jsonl_path, split, rel_image_dir, rel_lidar_dir, image_format='png', lidar_format='npy'):
    # 1. Anchor the absolute path relative to where the JSONL is stored
    jsonl_abs_path = Path(save_jsonl_path).resolve()
    base_root = jsonl_abs_path.parent

    # 2. Define Absolute Paths for Disk I/O
    abs_image_dir = base_root / rel_image_dir
    abs_lidar_dir = base_root / rel_lidar_dir

    # 3. Create directories physically
    abs_image_dir.mkdir(parents=True, exist_ok=True)
    abs_lidar_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Saving images to: {abs_image_dir}")
    print(f"[INFO] Saving LiDAR to: {abs_lidar_dir}")

    # Open JSONL with line buffering
    with open(save_jsonl_path, "w", buffering=1) as f_out:
        for idx in tqdm(range(len(dataset)), desc=f"Processing {split} triplets"):
            triplets = dataset[idx].get('til_triplet', [])
            if not triplets:
                continue

            for i, triplet in enumerate(triplets):
                label, img_pil, lidar, bbox = triplet

                # Common Filename Logic
                filename_base = f"{split}_{idx}_{i}_{label}"
                img_filename = f"{filename_base}.{image_format}"
                lidar_filename = f"{filename_base}.{lidar_format}"

                # --- 1. Disk Saving (Use Absolute Paths) ---
                abs_img_path = abs_image_dir / img_filename
                abs_lidar_path = abs_lidar_dir / lidar_filename
                
                # Save Image
                img_pil.save(str(abs_img_path))

                # Save LiDAR
                if lidar_format == 'npy':
                    np.save(str(abs_lidar_path), lidar)
                elif lidar_format == 'txt':
                    np.savetxt(str(abs_lidar_path), lidar, fmt="%.4f")
                else:
                    raise ValueError(f"Unsupported lidar_format: {lidar_format}")

                # --- 2. JSON Record (Use Relative Paths) ---
                # We use os.path.join or Path / to combine the relative dir prefix with filename
                json_obj = {
                    "label": label,
                    "image_path": str(Path(rel_image_dir) / img_filename),
                    "lidar_path": str(Path(rel_lidar_dir) / lidar_filename),
                    "bbox": bbox
                }
                
                f_out.write(json.dumps(json_obj) + "\n")
                # f_out.flush() # Not strictly necessary if buffering=1 is set

    print(f"[DONE] Triplets saved to {save_jsonl_path}")
    
# https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/nuscenes.py, render_annotation
def crop_annotation_nusc(nusc, ann_token, sample_record, margin=5, min_ratio=0.8):
    assert 'LIDAR_TOP' in sample_record['data'].keys(), 'Error: No LIDAR_TOP in data, unable to render.'

    # Figure out which camera the object is fully visible in (this may return nothing).
    boxes, cam = [], []
    cams = [key for key in sample_record['data'].keys() if 'CAM' in key]
    for cam in cams:
        _, boxes, _ = nusc.get_sample_data(sample_record['data'][cam], box_vis_level=1, selected_anntokens=[ann_token])
        if len(boxes) > 0:
            break  # We found an image that matches. Let's abort.
    if len(boxes) == 0:
        return None  # skip if not visible

    cam_token = sample_record['data'][cam]

    # Plot CAMERA view.
    data_path, boxes, camera_intrinsic = nusc.get_sample_data(cam_token, selected_anntokens=[ann_token])
    im = Image.open(data_path)

    # Crop the box from the image
    box = boxes[0]
    corners = view_points(box.corners(), camera_intrinsic, normalize=True)[:2, :]
    x_min, y_min = corners.min(axis=1)
    x_max, y_max = corners.max(axis=1)
    
    # Calculate the area inside the image, if too small, skip this box
    x_min_clip = max(int(x_min), 0)
    y_min_clip = max(int(y_min), 0)
    x_max_clip = min(int(x_max), im.width)
    y_max_clip = min(int(y_max), im.height)

    area_box = max(int(x_max - x_min), 1) * max(int(y_max - y_min), 1)
    area_clipped = max(x_max_clip - x_min_clip, 0) * max(y_max_clip - y_min_clip, 0)
    ratio_inside = area_clipped / area_box

    if ratio_inside < min_ratio:
        return None

    # Add margin and ensure within image bounds
    x_min_final = max(int(x_min) - margin, 0)
    y_min_final = max(int(y_min) - margin, 0)
    x_max_final = min(int(x_max) + margin, im.width)
    y_max_final = min(int(y_max) + margin, im.height)

    cropped_im = im.crop((x_min_final, y_min_final, x_max_final, y_max_final)) 
    
    return cropped_im

def crop_annotation_kitti(image: Image.Image, bbox_2d, margin: int = 5, min_ratio: float = 0.8):
    width, height = image.size
    x_min, y_min, x_max, y_max = bbox_2d

    # calculate box area
    area_box = max(int(x_max - x_min), 1) * max(int(y_max - y_min), 1)

    # clip to image boundaries
    x_min_clip = max(int(x_min), 0)
    y_min_clip = max(int(y_min), 0)
    x_max_clip = min(int(x_max), width)
    y_max_clip = min(int(y_max), height)

    # calculate clipped area
    area_clipped = max(x_max_clip - x_min_clip, 0) * max(y_max_clip - y_min_clip, 0)
    ratio_inside = area_clipped / area_box

    # skip if too small
    if ratio_inside < min_ratio or area_clipped <= 0:
        return None

    # Add margin and ensure within image bounds
    x_min_final = max(int(x_min_clip) - margin, 0)
    y_min_final = max(int(y_min_clip) - margin, 0)
    x_max_final = min(int(x_max_clip) + margin, width)
    y_max_final = min(int(y_max_clip) + margin, height)

    # crop
    cropped_im = image.crop((x_min_final, y_min_final, x_max_final, y_max_final))

    return cropped_im

def compute_box_corners(x, y, z, h, w, l, ry):
    """
    Compute 8 corners of 3D bounding box (KITTI camera coordinates)
    Args:
        x, y, z: center of box
        h, w, l: box size
        ry: yaw rotation around Y-axis
    Returns:
        corners_3d: (8, 3) array of box corners
    """
    # Define 8 corners in local coordinates
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

    corners = np.vstack([x_corners, y_corners, z_corners])  # (3, 8)

    # Rotation around Y-axis
    R = np.array([
        [np.cos(ry), 0, np.sin(ry)],
        [0, 1, 0],
        [-np.sin(ry), 0, np.cos(ry)]
    ])

    corners_3d = (R @ corners).T
    corners_3d += np.array([x, y, z])
    return corners_3d

def points_in_3d_box(box_corners, points):
    """
    Accurately determine if points are inside a 3D bounding box
    Args:
        box_corners: (8,3) box corner coordinates
        points: (N,3)
    Returns:
        mask: (N,) bool
    """
    # Assume the corner order is:
    # 0–3: bottom, 4–7: top
    p1 = box_corners[0]
    p2 = box_corners[1]
    p4 = box_corners[3]
    p5 = box_corners[4]

    # Three edge vectors
    i = p2 - p1  # x-direction
    j = p4 - p1  # y-direction
    k = p5 - p1  # z-direction

    v = points - p1.reshape(1, 3)

    iv = np.dot(v, i)
    jv = np.dot(v, j)
    kv = np.dot(v, k)

    mask_x = np.logical_and(0 <= iv, iv <= np.dot(i, i))
    mask_y = np.logical_and(0 <= jv, jv <= np.dot(j, j))
    mask_z = np.logical_and(0 <= kv, kv <= np.dot(k, k))

    mask = np.logical_and.reduce((mask_x, mask_y, mask_z))
    return mask

def load_calib(calib_path):
    """Reads KITTI calib.txt and returns transformation matrices."""
    calib = {}
    with open(calib_path, "r") as f:
        for line in f.readlines():
            if ":" not in line: continue
            key, value = line.split(":", 1)
            calib[key] = np.array([float(x) for x in value.split()])
    
    # Construct matrices
    # Tr_velo_to_cam: 3x4 matrix transforming Velodyne to Camera coordinates
    Tr_velo_to_cam = calib.get("Tr_velo_to_cam", np.eye(12)[:12]).reshape(3, 4)
    # R0_rect: 3x3 rectifying rotation matrix
    R0_rect = calib.get("R0_rect", np.eye(9)).reshape(3, 3)
    return Tr_velo_to_cam, R0_rect

def camera_box_to_lidar(x, y, z, h, w, l, ry, Tr_velo_to_cam, R0_rect):
    """
    Converts KITTI 3D bounding box from Camera coordinates to LiDAR coordinates.
    """
    # Box center in Camera homogeneous coordinates
    cam_center = np.array([x, y, z, 1.0])
    
    # Construct 4x4 transformation matrices
    Tr = np.eye(4)
    Tr[:3, :4] = Tr_velo_to_cam
    R0 = np.eye(4)
    R0[:3, :3] = R0_rect
    
    # Compute Camera-to-LiDAR transformation (Inverse of LiDAR-to-Camera)
    T_cam2lidar = np.linalg.inv(R0 @ Tr)
    
    # Transform center coordinates
    lidar_center = (T_cam2lidar @ cam_center)[:3]
    
    # Flip yaw direction (KITTI camera yaw is around Y-axis, LiDAR yaw is around Z-axis)
    yaw_lidar = -ry - np.pi / 2
    
    return lidar_center[0], lidar_center[1], lidar_center[2], h, w, l, yaw_lidar

def lsm_tikhonov(output, step, lambda_reg=0.1):
    # Tikhonov regularization
    n = len(step)
    A = np.eye(n)*step
    b = output
    
    Gamma = np.eye(n) # Identity regularization matrix
    
    u = np.linalg.inv(A.T @ A + lambda_reg * Gamma) @ A.T @ b
    
    return u

# https://huggingface.co/docs/trl/en/dataset_formats
def preprocess_data(examples):
    all_prompts = []
    all_completions = []
    system_prompt = (
        "You are an expert autonomous driving planning module (Driver). Your goal is to output a safe, smooth, and kinematically feasible future trajectory.\n"
        "Rules:\n"
        "1. Coordinate System: Current ego position is (0,0). X-axis positive is forward, Y-axis positive is left.\n"
        "2. Trajectory Timing: Output exactly 12 waypoints (except origin (0,0)) representing the next 6 seconds (sampled at 2Hz, 0.5s intervals).\n"
        "3. Kinematic Constraints: Ensure the gaps between waypoints are consistent with the current velocity and acceleration. Avoid sudden jumps or unrealistic lateral shifts.\n"
        "4. Safety Alignment: The trajectory must strictly follow the Navigator's safety analysis.\n"
        "5. Output Format: Only output the coordinates: (x1, y1), (x2, y2), ..., (x12, y12)."
    )
    
    for i in range(len(examples['token'])):
        ego_status_prompt = (
            f"Current Dynamics:\n"
            f"- Velocity: {examples['vel_val'][i]:.2f} m/s\n"
            f"- Yaw Rate: {examples['yr_val'][i]:.2f} rad/s\n"
            f"- Acceleration (Longitudinal x, Lateral y): {examples['acc_val'][i]}\n"
            f"- Past Trajectory (2Hz): {examples['wp_past'][i]}\n"
            # f"- High-level Command: {examples['command'][i]}\n\n"
        )
        driver_user_prompt = (
            "Predict the next 12 waypoints. "
        )        
        reasons_list = examples['reasons'][i]
        future_wp = examples['wp_future'][i]
        
        for reason_text in reasons_list:
            full_driver_prompt = (
                f"Navigator's Analysis and Instructions:\n{reason_text}\n\n"
                f"{ego_status_prompt}\n"
                f"{driver_user_prompt}"
            )
            all_prompts.append([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": full_driver_prompt}
            ])
            all_completions.append([
                {"role": "assistant", "content": f"Future Waypoints: {future_wp}."}
            ])
            
    return {
        "prompt": all_prompts,
        "completion": all_completions
    }
    
def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)