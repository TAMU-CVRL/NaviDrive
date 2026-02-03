import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pyquaternion import Quaternion
import cv2
import torch

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
