import os
import json
from tqdm import tqdm
from PIL import Image
import numpy as np
from pathlib import Path
from scipy.sparse import diags
from scipy.linalg import solve
from nuscenes.utils.geometry_utils import view_points
import yaml
import re
from qwen_vl_utils import process_vision_info

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

def lsm_tikhonov(x, y, theta, v0, a0, dt, lambda_reg=0.1):
    n = len(x)
    num_vars = n - 1  # acceleration steps
    
    # v_{i+1} = v_i + a_i * dt
    # v_target = sqrt((dx/dt)^2 + (dy/dt)^2)
    v_gt = np.sqrt((np.diff(x)/dt)**2 + (np.diff(y)/dt)**2) # length n-1
    
    # v_k = v0 + dt * sum_{j=0}^{k-1} a_j
    # A * a = v_gt - v0
    A = np.tril(np.ones((num_vars, num_vars))) * dt
    target = v_gt - v0
    
    # A_sub * a_unknown = target - A[:, 0]*a0
    A_sub = A[:, 1:]
    target_sub = target - A[:, 0] * a0
    
    # Tikhonov Matrix D (penalizes change in acceleration: jerk)
    num_unknowns = num_vars - 1
    D = np.zeros((num_unknowns, num_unknowns))
    for i in range(num_unknowns - 1):
        D[i, i] = -1
        D[i, i+1] = 1
        
    # (A.T @ A + lambda * D.T @ D) a = A.T @ target
    # a[0] = a0
    lhs = A_sub.T @ A_sub + lambda_reg * D.T @ D
    rhs = A_sub.T @ target_sub
    a_unknown = solve(lhs, rhs)
    accel = np.concatenate(([a0], a_unknown))
    
    # Reconstruct smooth velocity
    v_smooth = np.zeros(n)
    v_smooth[0] = v0 # initial velocity
    for i in range(num_vars):
        v_smooth[i+1] = v_smooth[i] + accel[i] * dt

    # curvature (kappa)
    # theta_{i+1} = theta_i + dt*k*v + (dt^2/2)*k*a
    # theta_{i+1} - theta_i = kappa_i * (dt * v_i + 0.5 * dt^2 * a_i)
    d_theta = np.diff(theta)
    denom = (v_smooth[:-1] * dt + 0.5 * accel * dt**2)
    
    # Tikhonov for kappa (penalizes steering rate)
    # (diag(coeff)^2 + lambda * D.T @ D) kappa = diag(coeff) @ d_theta
    W = np.diag(denom)
    D_k = np.zeros((num_vars - 1, num_vars))
    for i in range(num_vars - 1):
        D_k[i, i] = -1
        D_k[i, i+1] = 1
    lhs_k = W.T @ W + lambda_reg * D_k.T @ D_k
    rhs_k = W.T @ d_theta
    kappa = solve(lhs_k, rhs_k)

    return accel, kappa, v_smooth

def compute_action(waypoints, dt, v0, a0) -> str:
    waypoints = waypoints.cpu().numpy()
    x_seq = waypoints[:, 0]
    y_seq = waypoints[:, 1]
    theta_seq = waypoints[:, 2]
    accel_seq, kappa_seq, v_smooth = lsm_tikhonov(x_seq, y_seq, theta_seq, v0, a0, dt)
    action_list = [f"({a:.4f}, {k:.4f})" for a, k in zip(accel_seq, kappa_seq)]
    action = ", ".join(action_list)
    return action

def compute_trajectory(accel, kappa, x0, y0, theta0, v0, dt):
    n = len(accel) + 1
    
    x = np.zeros(n)
    y = np.zeros(n)
    theta = np.zeros(n)
    v = np.zeros(n)
    
    x[0], y[0], theta[0], v[0] = x0, y0, theta0, v0
    
    for i in range(len(accel)):
        # update velocity v
        v[i+1] = v[i] + accel[i] * dt
        
        # theta^{i+1} = theta^i + dt * k^i * v^i + (dt^2 / 2) * k^i * a^i
        theta[i+1] = theta[i] + dt * kappa[i] * v[i] + (0.5 * dt**2) * kappa[i] * accel[i]
        
        # x^{i+1} = x^i + (dt/2) * (v^i*cos(theta^i) + v^{i+1}*cos(theta^{i+1}))
        x[i+1] = x[i] + (dt / 2.0) * (v[i] * np.cos(theta[i]) + v[i+1] * np.cos(theta[i+1]))
        y[i+1] = y[i] + (dt / 2.0) * (v[i] * np.sin(theta[i]) + v[i+1] * np.sin(theta[i+1]))
        
    return x, y, theta, v

def compute_trajectory_2(pred_actions, x0, y0, theta0, v0, dt) -> np.ndarray:
    actions = np.array(pred_actions)

    # (N, 2) -> [[a, k], [a, k], ...]
    accel = actions[:, 0]
    kappa = actions[:, 1]
    
    n = len(accel) + 1
    
    x = np.zeros(n)
    y = np.zeros(n)
    theta = np.zeros(n)
    v = np.zeros(n)
    
    x[0], y[0], theta[0], v[0] = x0, y0, theta0, v0
    
    for i in range(len(accel)):
        # update velocity v
        v[i+1] = v[i] + accel[i] * dt
        
        # theta^{i+1} = theta^i + dt * k^i * v^i + (dt^2 / 2) * k^i * a^i
        theta[i+1] = theta[i] + dt * kappa[i] * v[i] + (0.5 * dt**2) * kappa[i] * accel[i]
        
        # x^{i+1} = x^i + (dt/2) * (v^i*cos(theta^i) + v^{i+1}*cos(theta^{i+1}))
        x[i+1] = x[i] + (dt / 2.0) * (v[i] * np.cos(theta[i]) + v[i+1] * np.cos(theta[i+1]))
        y[i+1] = y[i] + (dt / 2.0) * (v[i] * np.sin(theta[i]) + v[i+1] * np.sin(theta[i+1]))
    
    trajectory = np.column_stack((x, y))
    
    return trajectory

# https://huggingface.co/docs/trl/en/dataset_formats
def preprocess_data(examples, driver_user_prompt, system_prompt, enable_action=False):
    all_prompts = []
    all_completions = []
    
    for i in range(len(examples['token'])):
        # only keep x,y for the prompt, remove theta if exists
        wp_past = filter_to_xy_str(examples['wp_past'][i])
        if not enable_action:
            wp_future = filter_to_xy_str(examples['wp_future'][i])
        else:
            wp_future = filter_to_xy_str(examples['action_future'][i])
        
        ego_status_prompt = (
            f"Current Dynamics:\n"
            f"- Velocity: {examples['vel_val'][i]:.2f} m/s\n"
            f"- Yaw Rate: {examples['yr_val'][i]:.2f} rad/s\n"
            f"- Acceleration (Longitudinal x, Lateral y): {examples['acc_val'][i]}\n"
            f"- Past Trajectory (2Hz): {wp_past}\n"
            # f"- High-level Command: {examples['command'][i]}\n\n"
        )
            
        reasons_list = examples['reasons'][i]
        
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
                {"role": "assistant", "content": f"{wp_future}."}
            ])
            
    return {
        "prompt": all_prompts,
        "completion": all_completions
    }

def preprocess_data_img(examples, driver_user_prompt, enable_action=False, enable_reason=True):
    all_prompts = []
    all_completions = []
    all_image_paths = []
    
    for i in range(len(examples['token'])):
        # only keep x,y for the prompt, remove theta if exists
        wp_past = filter_to_xy_str(examples['wp_past'][i])
        if not enable_action:
            wp_future = filter_to_xy_str(examples['wp_future'][i])
        else:
            wp_future = filter_to_xy_str(examples['action_future'][i])
        
        ego_status_prompt = (
            f"Current Dynamics:\n"
            f"- Velocity: {examples['vel_val'][i]:.2f} m/s\n"
            f"- Yaw Rate: {examples['yr_val'][i]:.2f} rad/s\n"
            f"- Acceleration (Longitudinal x, Lateral y): {examples['acc_val'][i]}\n"
            f"- Past Trajectory (2Hz): {wp_past}\n"
            # f"- High-level Command: {examples['command'][i]}\n\n"
        )
            
        reasons_list = examples['reasons'][i]
        relative_paths = examples['image_paths'][i]
        
        if enable_reason:
            for reason_text in reasons_list:           
                full_driver_prompt = (
                    f"Navigator's Analysis and Instructions:\n{reason_text}\n\n"
                    f"{ego_status_prompt}\n"
                    f"{driver_user_prompt}"
                )
                all_image_paths.append(relative_paths)
                all_prompts.append(full_driver_prompt)
                all_completions.append(wp_future)
        else:
            full_driver_prompt = (
                f"{ego_status_prompt}\n"
                f"{driver_user_prompt}"
            )
            all_image_paths.append(relative_paths)
            all_prompts.append(full_driver_prompt)
            all_completions.append(wp_future)
        
    return {
        "prompt": all_prompts,
        "completion": all_completions,
        "image_paths": all_image_paths
    }
    
def collate_fn(batch, processor, system_prompt, nuscenes_dataroot, enable_image=False):
    messages_batch = []
    
    for item in batch:
        text_prompt = item['prompt']
        completion = item['completion']
        image_paths = item['image_paths']

        user_content = []
        if enable_image:
            for p in image_paths:
                full_path = os.path.join(nuscenes_dataroot, p)
                user_content.append({"type": "image", "image": full_path})
                
        user_content.append({"type": "text", "text": text_prompt})
                        
        prompt_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]
        completion_message = [{"role": "assistant", "content": [{"type": "text", "text": f"{completion}."}]}]
        messages_batch.append(prompt_messages + completion_message)     

    texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=False) for m in messages_batch]
    image_inputs, video_inputs = process_vision_info(messages_batch) # If there is no image, image_inputs is None

    inputs = processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )        

    labels = inputs["input_ids"].clone()
    
    for i in range(len(batch)):
        prompt_m = messages_batch[i][:-1] # [System, User, Assistant]
        p_text = processor.apply_chat_template(prompt_m, tokenize=False, add_generation_prompt=True)
        
        current_img = None
        if image_inputs is not None:
            current_img = image_inputs[i]
        
        p_inputs = processor(text=[p_text], 
                            images=[current_img] if current_img is not None else None,
                            return_tensors="pt") # text + image
        prompt_len = p_inputs["input_ids"].shape[1]
        
        # Mask prompt
        labels[i, :prompt_len] = -100

    labels[inputs["attention_mask"] == 0] = -100
    inputs["labels"] = labels   

    return inputs

def preprocess_data_action(examples):
    all_prompts = []
    all_completions = []
    system_prompt = (
        "You are an expert autonomous driving planning module (Driver). Your goal is to output a safe, smooth, and kinematically feasible future trajectory.\n"
        "Rules:\n"
        "1. Coordinate System: Current ego position is (0,0). X-axis positive is forward, Y-axis positive is left.\n"
        "2. Trajectory Timing: Output exactly 12 waypoints (except origin (0,0)) representing the next 6 seconds (sampled at 2Hz, 0.5s intervals).\n"
        "3. Kinematic Constraints: Ensure the gaps between waypoints are consistent with the current velocity and acceleration. Avoid sudden jumps or unrealistic lateral shifts.\n"
        "4. Safety Alignment: The trajectory must strictly follow the Navigator's safety analysis.\n"
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
            "Predict the next 11 control actions (acceleration, curvature): (a1, k1), (a2, k2), ..., (a11, k11). "
        )        
        reasons_list = examples['reasons'][i]
        action_future = examples['action_future'][i]
        
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
                {"role": "assistant", "content": f"Future Actions: {action_future}."}
            ])
            
    return {
        "prompt": all_prompts,
        "completion": all_completions
    }
    
def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
    
def filter_to_xy_str(original_str):
    pattern = r"\(([^)]+)\)"
    matches = re.findall(pattern, original_str)

    new_points = []
    for m in matches:
        parts = [p.strip() for p in m.split(',')]
        
        if len(parts) > 2:
            new_points.append(f"({parts[0]}, {parts[1]})")
        else:
            new_points.append(f"({m})")
            
    return ", ".join(new_points)