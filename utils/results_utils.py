import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pyquaternion import Quaternion
import cv2
import torch
from datetime import datetime
import json
import textwrap
import unicodedata
import re

def save_predicated_waypoints(pre_waypoints, gt_waypoints, index):
    # pre_waypoints: [pre_frames, 3]  nparray
    # gt_waypoints: [gtframes, 3]  nparray

    x_pre = pre_waypoints[:, 0]  # x coordinates for previous waypoints
    y_pre = pre_waypoints[:, 1]  # y coordinates for previous waypoints

    x_gt = gt_waypoints[:, 0]  # x coordinates for gt waypoints
    y_gt = gt_waypoints[:, 1]  # y coordinates for gt waypoints

    # Create the figure
    plt.figure(figsize=(8, 6))

    plt.scatter(x_gt, y_gt, color='red', label='Groundtruth Waypoints', zorder=5)
    plt.scatter(x_pre, y_pre, color='blue', label='Predicated Waypoints', zorder=5)
    plt.scatter(0, 0, color='green', label='Current position', s=50, zorder=10)

    # Set up the legend and other properties
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')  # Ensure equal scaling on both axes
    plt.title("Waypoints")
    # plt.show()

    # Save the plot to the specified directory with a timestamp
    folder = "results"
    if not os.path.exists(folder):
        os.makedirs(folder)
    save_path = os.path.join(folder, f"waypoints_plot_{index}.png")
    plt.savefig(save_path)
    plt.close()

def show_rv(range_view):
    range_depth = range_view[:, :, -1].numpy()
    plt.figure(figsize=(12*4, 0.375*4), dpi=200)
    
    # plt.imshow(color_map_rgb, aspect='equal')
    plt.imshow(range_depth, cmap="jet", aspect="equal")
    plt.show()

def show_cam(cam_image):
    image = (cam_image + 1)/2 # [-1, 1] -> [0, 1]
    plt.imshow(image)
    plt.show()

def show_panoramic(images):
    # images: [num_images, H, W, C]
    num_images, H, W, C = images.shape
    concat_image = np.zeros((H, W * num_images, C))
    for i in range(num_images):
        concat_image[:, i * W:(i + 1) * W, :] = (images[i] + 1)/2
    shift_image = np.zeros((H, W * num_images, C))
    shift_image[:, :W * num_images - W//2, :] = concat_image[:, W//2:, :]
    shift_image[:, W * num_images - W//2:, :] = concat_image[:, :W//2, :]

    fig, ax = plt.subplots(figsize=(W * num_images / 100, H / 100))
    ax.imshow(shift_image)
    ax.axis("off")

    plt.show()

def show_lidar(lidar):
    x = lidar[:, 0].numpy()
    y = lidar[:, 1].numpy()
    z = lidar[:, 2].numpy()

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x, y, z, c=z, cmap='viridis', s=1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Lidar Point Cloud')

    # ax.set_xlim([-30, 30])
    # ax.set_ylim([-30, 30])
    # ax.set_zlim([0, 20])
    plt.show()

def project_wp_to_image(nusc, sample_token, waypoints, image, color_waypoints=(0, 255, 0), color_polygon=(90, 0, 0), plot_polygon=True, plot=False):
    """
    nusc : NuScenes
        The NuScenes dataset.
    sample_token : str
        The token of the sample.
    waypoints : torch.Tensor
        The predicted waypoints. Shape: [N, 3].
    image : torch.Tensor or np.ndarray
        The image tensor. Shape: [C, H, W].
        The numpy array. Shape: [H, W, C].
    """
    car_width = 1.73 # vehicle width, meters
    # Preprocess the waypoints
    if isinstance(waypoints, torch.Tensor):
        waypoints = waypoints.cpu().numpy()
    elif isinstance(waypoints, np.ndarray):
        pass
    else:
        raise TypeError("Unsupported waypoints type: must be torch.Tensor or np.ndarray")

    # Add a zero row to the waypoints
    zero_row = np.zeros((1, waypoints.shape[1]), dtype=waypoints.dtype)
    waypoints = np.vstack((zero_row, waypoints))

    if isinstance(image, torch.Tensor):
        # Tensor: assume shape [C, H, W]
        front_image = image.permute(1, 2, 0).cpu().numpy()  # [H, W, C]
    elif isinstance(image, np.ndarray):
        # NumPy array: assume already in [H, W, C]
        front_image = image
    else:
        raise TypeError("Unsupported image type: must be torch.Tensor or np.ndarray")
    
    front_image = np.ascontiguousarray(front_image)

    # Get the camera calibration matrix
    sample = nusc.get('sample', sample_token)
    cam_token = sample['data']['CAM_FRONT']
    cam_data = nusc.get('sample_data', cam_token)
    cam_calib = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])

    ego2cam = transform_matrix(cam_calib['translation'], Quaternion(cam_calib['rotation']), inverse=True)

    # Calculate contour waypoints
    x = waypoints[:, 0]
    y = waypoints[:, 1]
    if waypoints.shape[1] > 2:
        theta = waypoints[:, 2]
    else:
        theta = theta_from_xy(waypoints)
    heading = np.array([np.sin(theta), np.cos(theta)]) 
    dxdy = heading * car_width/2
    x_left = x - dxdy[0]
    y_left = y + dxdy[1]
    x_right = x + dxdy[0]
    y_right = y - dxdy[1]

    points = np.vstack((x, y, np.zeros(len(waypoints)), np.ones(len(waypoints))))  # [4, N]
    points_left = np.vstack((x_left, y_left, np.zeros(len(waypoints)), np.ones(len(waypoints))))  # [4, N]
    points_right = np.vstack((x_right, y_right, np.zeros(len(waypoints)), np.ones(len(waypoints))))  # [4, N]

    # Transform points to camera coordinates
    points_cam = ego2cam @ points  # [4, N]
    points_cam_left = ego2cam @ points_left  # [4, N]
    points_cam_right = ego2cam @ points_right  # [4, N]

    points_cam3 = points_cam[:3, :]
    points_cam_left3 = points_cam_left[:3, :]
    points_cam_right3 = points_cam_right[:3, :]

    # Move the inital point to the front of the camera
    points_cam3[-1, 0] = 1
    points_cam_left3[-1, 0] = 1
    points_cam_right3[-1, 0] = 1

    # Project points to image
    view = np.array(cam_calib['camera_intrinsic'])
    uv_left = project_points2image(points_cam_left3, view, front_image)
    uv_right = project_points2image(points_cam_right3, view, front_image)

    # Create the polygon
    polygon = np.vstack([uv_left, uv_right[::-1]]).astype(np.int32)
    if plot_polygon:
        front_image = fill_poly(front_image, polygon, color_polygon)
    # Draw the waypoints on the image
    project_points2image(points_cam3, view, front_image, color_waypoints, True)

    if plot:
        plt.imshow(front_image)
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', label='Ground Truth',
                    markerfacecolor=np.array([0, 255, 0])/255, markersize=6),
            plt.Line2D([0], [0], marker='o', color='w', label='Predicted',
                    markerfacecolor=np.array([0, 0, 255])/255, markersize=6),
        ]
        # plt.legend(handles=legend_elements, loc='upper right')
        plt.axis('off')
        plt.show()

    return front_image




# =================================================================================================
def project_points2image(points, view, front_image, color=(255, 255, 255),plot=False):
    viewpad = np.eye(4)
    viewpad[:view.shape[0], :view.shape[1]] = view
    # Filter out points behind the camera
    mask = points[2, :] > 0.5
    points = points[:, mask]
    nbr_points = points.shape[1]

    points = np.concatenate((points, np.ones((1, nbr_points))))
    points = np.dot(viewpad, points)
    points = points[:3, :]

    points = points / points[2:3, :].repeat(3, 0).reshape(3, nbr_points)
    img_points_uv = points[:2, :].T

    img_points_uv = np.round(img_points_uv).astype(np.int32)

    if plot:
        for (u, v) in img_points_uv:
            cv2.circle(front_image, (u, v), radius=6, color=color, thickness=-1)

    return img_points_uv

def fill_poly(front_image, polygon, color=(90, 90, 90)):
    r_, g_, b_ = color
    blend_ratio = 0.5
    h, w = front_image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], color=255)

    gradient_layer = np.zeros_like(front_image)
    ys = polygon[:, 1]
    y_min, y_max = max(0, ys.min()), min(h - 1, ys.max())

    for y in range(y_min, y_max + 1):
        t = 1.0 - (y - y_min) / max(1, y_max - y_min)
        
        t = t ** 1  # speed, 1: far -> 0: near

        # r = r_ + (255 - r_) * (t) # Red to Maroon
        r = int(r_ * (t))
        g = int(g_ * (t))
        b = int(b_ * (t))

        gradient_layer[y, mask[y] > 0] = (r, g, b)

    result = front_image.copy()
    result[mask > 0] = (
        (1 - blend_ratio) * front_image[mask > 0] + 
        blend_ratio * gradient_layer[mask > 0]
    ).astype(np.uint8)

    return result

def theta_from_xy(waypoints):
    x = waypoints[:, 0]
    y = waypoints[:, 1]
    
    # Compute differences between consecutive points
    dx = np.diff(x)
    dy = np.diff(y)
    
    # Calculate theta for each segment
    theta = np.arctan2(dy, dx)
    
    # Pad last theta (e.g. repeat the last one) to match original length
    theta = np.append(theta, theta[-1])
    
    return theta

def transform_matrix(translation: np.ndarray = np.array([0, 0, 0]),
                     rotation: Quaternion = Quaternion([1, 0, 0, 0]),
                     inverse: bool = False) -> np.ndarray:
    """
    Convert pose to transformation matrix.
    :param translation: <np.float32: 3>. Translation in x, y, z.
    :param rotation: Rotation in quaternions (w ri rj rk).
    :param inverse: Whether to compute inverse transform matrix.
    :return: <np.float32: 4, 4>. Transformation matrix.
    """
    # from nuscenes.utils.geometry_utils import transform_matrix
    tm = np.eye(4)

    if inverse:
        rot_inv = rotation.rotation_matrix.T
        trans = np.transpose(-np.array(translation))
        tm[:3, :3] = rot_inv
        tm[:3, 3] = rot_inv.dot(trans)
    else:
        tm[:3, :3] = rotation.rotation_matrix
        tm[:3, 3] = np.transpose(np.array(translation))

    return tm

def calculate_metrics(gt, pred, threshold=2.0):
    """
    gt: np.array (N, 2)
    pred: np.array (N, 2)
    """
    gt = np.array(gt)
    pred = np.array(pred)
    
    min_len = min(len(gt), len(pred))
    gt, pred = gt[:min_len], pred[:min_len]
    errors = np.linalg.norm(gt - pred, axis=1)
    
    # idx 0=0.5s, 1=1s, 2=1.5s, 3=2s, 4=2.5s, 5=3s...
    metrics = {
        "l2_1s": errors[1] if min_len > 1 else np.nan,
        "l2_2s": errors[3] if min_len > 3 else np.nan,
        "l2_3s": errors[5] if min_len > 5 else np.nan,
        "l2_6s": errors[11] if min_len > 11 else errors[-1],
        
        "ade_3s": np.mean(errors[:6]) if min_len >= 6 else np.mean(errors),
        
        "ade": np.mean(errors)
    }
    
    # Failure rate
    metrics['is_failure'] = 1 if metrics['l2_6s'] > threshold else 0
    
    return metrics

def format_results(avg_metrics, input_file, total_samples, threshold):
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    width = 95
    double_line = "=" * width
    single_line = "-" * width

    header_section = (
        f"{double_line}\n"
        f"   TRAJECTORY EVALUATION REPORT  |  {date_str}\n"
        f"{double_line}\n"
        f"   [Input File]    : {input_file}\n"
        f"   [Sample Count]  : {total_samples}\n"
        f"   [Failure Thresh]: {threshold} m\n"
        f"{single_line}\n"
    )
    
    table_header = (
        f"   {'Metric':<15} | {'1.0s':<8} | {'2.0s':<8} | {'3.0s':<8} | {'6.0s(FDE)':<10} | {'minADE@3s':<10} | {'minADE(All)':<10}\n"
        f"   {'-'*15}-|-{'-'*8}-|-{'-'*8}-|-{'-'*8}-|-{'-'*10}-|-{'-'*10}-|-{'-'*10}\n"
    )
    
    table_row = (
        f"   {'L2 Error (m)':<15} | "
        f"{avg_metrics['L2_1s']:<8.2f} | "
        f"{avg_metrics['L2_2s']:<8.2f} | "
        f"{avg_metrics['L2_3s']:<8.2f} | "
        f"{avg_metrics['L2_6s']:<10.2f} | "
        f"{avg_metrics['ADE_3s']:<10.2f} | "
        f"{avg_metrics['ADE_avg']:<10.2f}\n"
    )
    
    summary_section = (
        f"{single_line}\n"
        f"   {'OVERALL PERFORMANCE':<15}\n"
        f"   > Failure Rate : {avg_metrics['Failure_Rate']:>6.2f} %\n"
        f"   > Reliability  : {100 - avg_metrics['Failure_Rate']:>6.2f} % (within {threshold}m)\n"
        f"{double_line}\n"
    )
    
    return header_section + table_header + table_row + summary_section

def render_frame(nusc, line_data, best_pred=None):
    data = json.loads(line_data) if isinstance(line_data, str) else line_data
    
    token = data['token'][0] if isinstance(data['token'], list) else data['token']
    
    sample_rec = nusc.get('sample', token)
    cam_data = nusc.get('sample_data', sample_rec['data']['CAM_FRONT'])
    img_path = os.path.join(nusc.dataroot, cam_data['filename'])
    raw_img = cv2.imread(img_path)
    img_h, img_w = raw_img.shape[:2]
    
    if best_pred is not None:
        pred_pts = np.array(best_pred, dtype=np.float32)
    else:
        pred_key = 'pred_waypoints' if 'pred_waypoints' in data else 'predicted_output'
        raw_preds = data[pred_key]
        
        if isinstance(raw_preds, list) and len(raw_preds) > 0 and isinstance(raw_preds[0][0], list):
            pred_pts = np.array(raw_preds[0], dtype=np.float32)
        else:
            pred_pts = np.array(raw_preds, dtype=np.float32)

    gt_pts = np.array(data['gt_waypoints'], dtype=np.float32)

    vis_img = project_wp_to_image(
        nusc, token, pred_pts, raw_img,
        color_waypoints=(255, 0, 0),
        color_polygon=(200, 0, 0),
        plot_polygon=True
    )

    vis_img = project_wp_to_image(
        nusc, token, gt_pts, vis_img,
        color_waypoints=(0, 255, 0),
        plot_polygon=False
    )
    
    reason = data.get('reasons', "")
    if isinstance(reason, list): 
        reason = reason[0] if len(reason) > 0 else ""
    
    if reason:
        reason = format_reasoning_for_cv2(reason)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        thickness = 1
        line_height = 25

        colors = {
            "Perception:": (0, 255, 255),
            "Action:": (255, 100, 0),
            "Reasoning:": (0, 255, 0)
        }
        
        raw_lines = reason.split('\n')
        final_lines = []
        for line in raw_lines:
            line = line.strip()
            if not line: continue
            wrapped = textwrap.wrap(line, width=170)
            final_lines.extend(wrapped)

        if final_lines:
            bg_height = len(final_lines) * line_height + 20

            overlay = vis_img.copy()
            cv2.rectangle(overlay, (0, 0), (img_w, bg_height), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, vis_img, 0.4, 0, vis_img)
            
            for i, line_text in enumerate(final_lines):
                y_text = 30 + i * line_height
                x_pos = 20
                
                found_keyword = False
                for kw, color in colors.items():
                    if line_text.startswith(kw):
                        cv2.putText(vis_img, kw, (x_pos, y_text), font, font_scale, color, thickness, cv2.LINE_AA)
                        
                        (tw, _), _ = cv2.getTextSize(kw, font, font_scale, thickness)
                        content = line_text[len(kw):]
                        cv2.putText(vis_img, content, (x_pos + tw + 5, y_text), font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
                        
                        found_keyword = True
                        break
                
                if not found_keyword:
                    cv2.putText(vis_img, line_text, (x_pos, y_text), font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
    
    return vis_img, token, cam_data['width'], cam_data['height']

def format_reasoning_for_cv2(text):
    if not text:
        return ""
    
    # 1. Handle escaped newlines if the JSON loaded them as literal strings
    text = text.replace('\\n', '\n')
    
    # 2. Pull the text up to the same line as the keywords
    # \1 refers to whichever keyword it found, adding one clean space after it
    text = re.sub(r'(Perception:|Action:|Reasoning:)\s*\n\s*', r'\1 ', text)
    
    # 3. Swap out OpenCV-breaking Unicode characters
    replacements = {
        '\u2019': "'",   # Smart apostrophe
        '\u2018': "'",   # Left single quote
        '\u201c': '"',   # Left double quote
        '\u201d': '"',   # Right double quote
        '\u2014': '--',  # Em dash
        '\u2013': '-',   # En dash
        '\u00b2': '^2',  # Superscript 2 (fixes your m/s^2)
        '\u2212': '-',   # Math minus sign
        '\u00b0': ' deg' # Degree symbol
    }
    
    for og_char, ascii_char in replacements.items():
        text = text.replace(og_char, ascii_char)
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')    
    
    return text
