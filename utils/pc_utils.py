import torch
import numpy as np
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion
from open3d import open3d as o3d
from sklearn.neighbors import NearestNeighbors
import math

def segment_ground_o3d(points, dist_thresh=0.15, ransac_n=3, num_iterations=1000):
    # Convert to numpy
    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()
    if points.shape[1] > 3:
        points = points[:, :3]

    # Convert to open3d point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # RANSAC
    plane_model, inliers = pcd.segment_plane(distance_threshold=dist_thresh,
                                             ransac_n=ransac_n,
                                             num_iterations=num_iterations)
    [a, b, c, d] = plane_model
    # print(f"[INFO] Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

    ground_points = points[inliers]
    non_ground_points = np.delete(points, inliers, axis=0)

    # print(f"[INFO] Ground points: {len(ground_points)}, Non-ground points: {len(non_ground_points)}")
    return ground_points, non_ground_points

def lidar2camera_fov(nusc, points_lidar, token, camera_name):
    # Get sample and sensors
    points_lidar = points_lidar[:, :3] # [N, 3]
    sample_record = nusc.get('sample', token)
    lidar_token = nusc.get('sample_data', sample_record['data']['LIDAR_TOP'])
    cam_token = nusc.get('sample_data', sample_record['data'][camera_name])

    lidar_calib = nusc.get('calibrated_sensor', lidar_token['calibrated_sensor_token'])
    cam_calib = nusc.get('calibrated_sensor', cam_token['calibrated_sensor_token'])

    # LiDAR -> Ego
    q_l2e = Quaternion(lidar_calib['rotation'])
    R_l2e = torch.tensor(q_l2e.rotation_matrix, dtype=torch.float32)
    t_l2e = torch.tensor(lidar_calib['translation'], dtype=torch.float32).view(3, 1)
    points_ego = R_l2e @ points_lidar.T + t_l2e  # [3, N]

    # Ego -> Camera
    q_e2c = Quaternion(cam_calib['rotation'])
    R_e2c = torch.tensor(q_e2c.rotation_matrix, dtype=torch.float32)
    t_e2c = torch.tensor(cam_calib['translation'], dtype=torch.float32).view(3, 1)
    cam_intrinsic = np.array(cam_calib['camera_intrinsic'])

    pc_cam = R_e2c.T @ (points_ego - t_e2c.view(3, 1))  # [3, N]
    pc_cam_np = pc_cam.detach().cpu().numpy()  # to numpy

    points_img = view_points(pc_cam_np, cam_intrinsic, normalize=True)

    W, H = (1600, 900) # image size for nuScenes camera
    mask = (points_img[0, :] > 0) & (points_img[0, :] < W) & \
           (points_img[1, :] > 0) & (points_img[1, :] < H) & \
           (pc_cam_np[2, :] > 0)

    visible_points = points_ego[:, mask].T  # [3,N] → [M,3]
    return visible_points, mask

def load_lidar_bin(bin_path):
    points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 5)[:, :4]  # or :4 if 5 columns
    return torch.tensor(points[:, :3], dtype=torch.float32)  # Keep only xyz

def load_sparse_method(method):
    if method == "knn_jitter":
        sparse_to_dense_fn = knn_jitter
    elif method == "zero_pad":
        sparse_to_dense_fn = zero_pad
    elif method == "repeat_pad":
        sparse_to_dense_fn = repeat_pad
    else:
        raise ValueError(f"Unknown dense method: {method}")
    return sparse_to_dense_fn

def zero_pad(points, target_num = 1024):
    if points.shape[0] >= target_num:
        target_points = farthest_point_sampling(points, npoint=target_num)  # [target_num, 3]
    else:
        target_points = torch.cat([points, torch.zeros((target_num - points.shape[0], 3), device=points.device)], dim=0)
    return target_points

def repeat_pad(points, target_num=1024):
    n = points.shape[0]
    if n >= target_num:
        target_points = farthest_point_sampling(points, npoint=target_num)
    else:
        num_to_add = target_num - n
        idx = torch.randint(0, n, (num_to_add,), device=points.device)
        extra_points = points[idx] 
        target_points = torch.cat([points, extra_points], dim=0)
    return target_points

def knn_jitter(points, target_num = 1024):
    if points.shape[0] >= target_num:
        target_points = farthest_point_sampling(points, npoint=target_num)  # [target_num, 3]
    else:
        scalr = math.ceil(target_num / points.shape[0])
        factor1 = 2
        pts = knn_upsample(points, upsample_factor=factor1, k=3)  # [N', 3]
        factor2 = math.ceil(scalr / factor1)
        pts_2 = jitter_upsample(pts, upsample_factor=factor2, jitter_std=0.01)  # [N'', 3]
        target_points = farthest_point_sampling(pts_2, npoint=1024)  # [1024, 3]
    return target_points  # [target_num_points, 3]

# Different upsampling methods
def knn_upsample(points, upsample_factor=2, k=3):
    is_tensor = isinstance(points, torch.Tensor)
    if is_tensor:
        device = points.device
        points = points.cpu().numpy()

    N, D = points.shape
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(points)
    _, indices = nbrs.kneighbors(points)

    new_points = []
    for i in range(N):
        for _ in range(upsample_factor - 1):
            j = np.random.choice(indices[i][1:])  # skip itself
            alpha = np.random.rand()
            interp = alpha * points[i] + (1 - alpha) * points[j]
            new_points.append(interp)

    new_points = np.array(new_points)  # shape: [(upsample_factor - 1) * N, D]
    all_points = np.concatenate([points, new_points], axis=0)

    if is_tensor:
        return torch.tensor(all_points, dtype=torch.float32, device=device)
    else:
        return all_points

def jitter_upsample(points, upsample_factor=2, jitter_std=0.01):
    N, D = points.shape
    num_new = N * (upsample_factor - 1)
    
    idx = torch.randint(0, N, (num_new,), device=points.device)
    new_points = points[idx] + torch.randn((num_new, D), device=points.device) * jitter_std
    all_points = torch.cat([points, new_points], dim=0)

    return all_points  # [N * upsample_factor, D]

# Farthest Point Sampling
def farthest_point_sampling(points, npoint):
    if isinstance(points, np.ndarray):
        points = torch.from_numpy(points).float()

    device = points.device
    N, C = points.shape

    centroids = torch.zeros(npoint, dtype=torch.long, device=device)
    distance = torch.ones(N, device=device) * 1e10
    farthest = torch.randint(0, N, (1,), dtype=torch.long, device=device)[0]

    for i in range(npoint):
        centroids[i] = farthest
        centroid = points[farthest].view(1, C)  # [1, C]
        dist = torch.sum((points - centroid) ** 2, dim=1)  # [N]
        distance = torch.min(distance, dist)
        farthest = torch.max(distance, dim=0)[1]

    sampled_points = points[centroids]  # [npoint, C]
    return sampled_points
