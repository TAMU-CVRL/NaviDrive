import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import open3d as o3d
from mpl_toolkits.mplot3d import Axes3D
import wandb
from matplotlib.patches import Polygon
from pyquaternion import Quaternion
from PIL import Image

# CLIP normalization parameters for image de-normalization
CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073])
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711])

def get_bbox_corners(bbox):
    """
    Calculate the 8 corners of a 3D bounding box.
    Args:
        bbox: list/array of 7 elements [x, y, z, w, l, h, yaw]
    Returns:
        corners: numpy array of shape (8, 3)
    """
    x, y, z, w, l, h, yaw = bbox
    
    # Rotation matrix around the Z-axis
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1]
    ])
    
    # Define 8 corners relative to the center (local coordinates)
    # Order: Top 4 corners, then Bottom 4 corners
    x_corners = l / 2 * np.array([1, 1, 1, 1, -1, -1, -1, -1])
    y_corners = w / 2 * np.array([1, -1, -1, 1, 1, -1, -1, 1])
    z_corners = h / 2 * np.array([1, 1, -1, -1, 1, 1, -1, -1])
    
    corners = np.vstack([x_corners, y_corners, z_corners])
    # Rotate and translate to global coordinates
    corners = R @ corners + np.array([[x], [y], [z]])
    
    return corners.T

def draw_bbox_plt(ax, bbox, color='red'):
    """Draw a 3D bounding box on a Matplotlib 3D axis."""
    corners = get_bbox_corners(bbox)
    
    # Define edges by connecting corner indices
    lines = [
        (0, 1), (1, 2), (2, 3), (3, 0), # Top face
        (4, 5), (5, 6), (6, 7), (7, 4), # Bottom face
        (0, 4), (1, 5), (2, 6), (3, 7)  # Vertical pillars
    ]
    
    for start, end in lines:
        ax.plot(corners[[start, end], 0], 
                corners[[start, end], 1], 
                corners[[start, end], 2], c=color, linewidth=1)
    
    # Mark the center point
    ax.scatter([bbox[0]], [bbox[1]], [bbox[2]], color='green', s=10)

def draw_bbox_o3d(bbox, color=[1, 0, 0]):
    """
    Create an Open3D LineSet for a 3D bounding box.
    Args:
        bbox: list/array of 7 elements [x, y, z, w, l, h, yaw]
        color: list of 3 elements for RGB color
    """
    corners = get_bbox_corners(bbox)
    lines = [
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7]
    ]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(corners),
        lines=o3d.utility.Vector2iVector(lines)
    )
    line_set.colors = o3d.utility.Vector3dVector([color for _ in range(len(lines))])
    return line_set

def visualize_pc_plt(point_cloud, title="Point Cloud (Matplotlib)"):
    """
    Visualize point cloud using Matplotlib with fixed aspect ratio.
    Args:
        point_cloud: numpy array or torch tensor of shape (N, 3)
        title: Title of the plot
    """
    if isinstance(point_cloud, torch.Tensor):
        point_cloud = point_cloud.cpu().numpy()
    
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Color by Z-height for better depth perception
    sc = ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2],
                    c=point_cloud[:, 2], cmap='jet', s=0.2, alpha=0.8)
    
    # Set equal axis scaling
    mid_x, mid_y, mid_z = point_cloud[:, :3].mean(axis=0)
    max_range = (point_cloud[:, :3].max(axis=0) - point_cloud[:, :3].min(axis=0)).max() / 2
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    ax.set_title(title)
    plt.colorbar(sc, ax=ax, shrink=0.5, label='Z elevation')
    plt.show()

def visualize_sample_o3d(sample, point_size=1.0):
    """
    Visualize a complete SparseCLIP sample (LiDAR + BBoxes) in Open3D.
    Args:
        sample: dict containing 'raw_lidar' and 'all_bboxes' keys
    """
    # Extract LiDAR points (handle sequence dimensions if present)
    lidar = sample['raw_lidar'][-1] if sample['raw_lidar'].ndim == 3 else sample['raw_lidar']
    if isinstance(lidar, torch.Tensor):
        lidar = lidar.cpu().numpy()
    
    # Initialize PointCloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(lidar[:, :3])
    pcd.paint_uniform_color([0.5, 0.5, 0.5]) # Gray points
    
    # Collect all geometries
    geometries = [pcd]
    if 'all_bboxes' in sample:
        for b in sample['all_bboxes']:
            geometries.append(draw_bbox_o3d(b['bbox']))
            
    # Set up visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="SparseCLIP 3D Inspector")
    for g in geometries:
        vis.add_geometry(g)
    
    vis.get_render_option().point_size = point_size
    vis.get_render_option().background_color = np.asarray([0, 0, 0]) # Black background
    vis.run()
    vis.destroy_window()

def show_clip_image(img_tensor, index=0):
    """
    De-normalize and display an image processed for CLIP.
    Args:
        img_tensor: torch tensor of shape (B, C, H, W)
    """
    img = img_tensor[index].cpu().clone()
    
    # Reverse normalization: img * std + mean
    # Reshape mean/std to (C, 1, 1) for broadcasting
    img = img * CLIP_STD[:, None, None] + CLIP_MEAN[:, None, None]
    img = torch.clamp(img, 0, 1)
    
    plt.figure(figsize=(8, 8))
    # Change format from (C, H, W) to (H, W, C) for plt.imshow
    plt.imshow(img.permute(1, 2, 0))
    plt.axis("off")
    plt.title(f"De-normalized CLIP Image (Index {index})")
    plt.show()

def scene_visualization(data, dataroot):
    """
    Visualize a bev view and 6 surround-view images for a scene.
    """
    token = data['token'][0]
    instances = data['instance'][0]
    lidar_t = torch.tensor(data['lidar_t'], dtype=torch.float32)
    lidar_q = Quaternion(data['lidar_q'])
    
    fig_bev = plt.figure(figsize=(10, 10))
    ax_bev = fig_bev.add_subplot(111)
    
    lidar_path = os.path.join(dataroot, data['lidar_path'][0])
    pc_raw = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)
    points = torch.tensor(pc_raw[:, :3], dtype=torch.float32)
    R = torch.tensor(lidar_q.rotation_matrix, dtype=torch.float32)
    pc_ego = (R @ points.T + lidar_t.view(3, 1)).T.numpy()
    
    mask = (np.abs(pc_ego[:, 0]) < 50) & (np.abs(pc_ego[:, 1]) < 50)
    pc_plot = pc_ego[mask]
    ax_bev.scatter(pc_plot[:, 1], pc_plot[:, 0], s=0.1, c='gray', alpha=0.5)
    
    for inst in instances:
        x, y, z, w, l, h, yaw = inst['bbox']
        rot_matrix = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        corners = np.array([[l/2, w/2], [l/2, -w/2], [-l/2, -w/2], [-l/2, w/2]])
        rotated_corners = (rot_matrix @ corners.T).T + np.array([x, y])
        poly = Polygon(rotated_corners[:, [1, 0]], closed=True, fill=False, edgecolor='red', linewidth=2)
        ax_bev.add_patch(poly)
        ax_bev.text(y, x, inst['label'], color='blue', fontsize=8)

    ax_bev.plot(0, 0, 'gx', markersize=10)
    ax_bev.invert_xaxis()
    ax_bev.set_title(f"BEV View - {token[:8]}")
    ax_bev.grid(True)
    
    bev_wandb = wandb.Image(fig_bev)
    plt.close(fig_bev)

    fig_cam = plt.figure(figsize=(18, 8))
    cam_names = ["BACK", "BACK_LEFT", "FRONT_LEFT", "FRONT", "FRONT_RIGHT", "BACK_RIGHT"]
    for i, img_path in enumerate(data['image_paths'][0]):
        img = Image.open(os.path.join(dataroot, img_path))
        ax = fig_cam.add_subplot(2, 3, i + 1)
        ax.imshow(img)
        ax.set_title(cam_names[i])
        ax.axis('off')
    
    plt.tight_layout()
    
    cam_wandb = wandb.Image(fig_cam)
    plt.close(fig_cam)

    return {"bev": bev_wandb, "camera": cam_wandb}
