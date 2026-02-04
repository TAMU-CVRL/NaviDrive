# Reference: https://github.com/E2E-AD/AD-MLP/blob/main/deps/stp3/stp3/datas/NuscenesData.py
import os
import numpy as np
from PIL import Image

from nuscenes.utils.splits import create_splits_scenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import transform_matrix
from nuscenes.eval.common.utils import quaternion_yaw
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

from utils.pc_utils import farthest_point_sampling

class NuscenesData(Dataset):
    def __init__(self, nusc, is_train, pre_frames, future_frames):
        self.nusc = nusc
        self.is_train = is_train # 0: train, 1: val, 2: test

        self.pre_frames = pre_frames + 1 # previous frames + current frame
        self.future_frames = future_frames
        self.sequence_length = self.pre_frames + self.future_frames # previous frames + future framess
        
        self.max_lidar_points = 35000
        self.front_fov = 77
        self.cameras = ["CAM_BACK", "CAM_BACK_LEFT", "CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_RIGHT"] # (H, W, 3): (900, 1600, 3) <class 'numpy.ndarray'>

        self.can_bus = NuScenesCanBus(dataroot=self.nusc.dataroot)

        self.scenes = self.get_scenes()
        self.samples = self.get_samples()
        self.indices = self.get_indices()
        
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        data = {}
        keys = ['token', 'raw_images', 'raw_lidar', 'future_waypoints', 
                'cur_waypoint', 'pre_waypoints', 'instance', 'velocity', 'accel', 
                'yaw_rate', 'command', 'image_paths', 'lidar_path', 'lidar_cs_record']
        for key in keys:
            data[key] = []

        cur_frame = self.indices[index][self.pre_frames - 1] # current frame
        cur_sample = self.samples[cur_frame] # current sample
        T_ego2w_cur = self.get_transform_w2ego(cur_sample, True) # current transform matrix, ego -> world
        data['token'].append(cur_sample['token'])

        for idx, frame in enumerate(self.indices[index]):
            sample = self.samples[frame]
            # Get trainning data
            if idx < self.pre_frames: # previous frames + current frame
                # Camera images
                images, image_paths = self.get_images(sample, self.cameras) # [num_cameras, 3, H, W]
                images = images.unsqueeze(0) # [1, num_cameras, 3, H, W]
                data['raw_images'].append(images)
                data['image_paths'].append(image_paths)

                # Lidar points
                raw_lidar, lidar_path, lidar_cs_record = self.get_lidar(sample)
                raw_lidar = raw_lidar.unsqueeze(0) # [1, N, 4], in ego coord system
                data['raw_lidar'].append(raw_lidar)
                data['lidar_path'].append(lidar_path)
                data['lidar_cs_record'].append(lidar_cs_record)
                
                # previous waypoints
                pre_waypoints = self.get_waypoints(sample, T_ego2w_cur).float() # [1, waypoint] 
                data['pre_waypoints'].append(pre_waypoints)

                # Record ego states
                instance = self.get_instance(sample)
                data['instance'].append(instance) # list[T][B]
                velocity, accel, yaw_rate = self.get_ego_state(sample)
                data['velocity'].append(velocity)
                data['accel'].append(accel)
                data['yaw_rate'].append(yaw_rate)
            else:
                # future waypoints
                future_waypoints = self.get_waypoints(sample, T_ego2w_cur).float() # [1, waypoint]
                data['future_waypoints'].append(future_waypoints)
        
        data['raw_images'] = torch.cat(data['raw_images'], dim=0) # [batch_size, pre_frames, num_cameras, 3, H, W]
        data['raw_lidar'] = torch.cat(data['raw_lidar'], dim=0) # [batch_size, pre_frames, max_N, 4], in lidar coord system
        data['pre_waypoints'] = torch.cat(data['pre_waypoints'], dim=0) # [batch_size, pre_frames, waypoint]
        data['cur_waypoint'] = data['pre_waypoints'][-1].unsqueeze(0) # the last previous waypoint is the current waypoint
        data['pre_waypoints'] = data['pre_waypoints'][:-1] # remove the current waypoint
        data['velocity'] = torch.stack(data['velocity'], dim=0)  # [pre_frames, 2]
        data['accel'] = torch.stack(data['accel'], dim=0)  # [pre_frames, 2]
        data['yaw_rate'] = torch.stack(data['yaw_rate'], dim=0)  # [pre_frames]
        if self.future_frames != 0:
            data['future_waypoints'] = torch.cat(data['future_waypoints'], dim=0) # [batch_size, future_frames, waypoint]
        data['command'] = self.classify_command(data['future_waypoints'], data['cur_waypoint']) # [batch_size, command]   
        return data
    
    # Get splits scenes
    def get_scenes(self):
        # filter by scene split
        split = {'v1.0-trainval': {0: 'train', 1: 'val', 2: 'test'},
                 'v1.0-mini': {0: 'mini_train', 1: 'mini_val'},}[self.nusc.version][self.is_train]

        blacklist = [419] + self.can_bus.can_blacklist  # scene-0419 does not have vehicle monitor data
        blacklist = ['scene-' + str(scene_no).zfill(4) for scene_no in blacklist]

        scenes = create_splits_scenes()[split][:]
        for scene_no in blacklist:
            if scene_no in scenes:
                scenes.remove(scene_no)

        return scenes

    # Get samples from splits scenes and sort them
    def get_samples(self):
        samples = [samp for samp in self.nusc.sample]

        # remove samples that aren't in this split
        samples = [samp for samp in samples if self.nusc.get('scene', samp['scene_token'])['name'] in self.scenes]

        # sort by scene, timestamp (only to make chronological viz easier)
        samples.sort(key=lambda x: (x['scene_token'], x['timestamp']))

        return samples

    # Get frames index in same scene
    def get_indices(self):
        indices = []
        for index in range(len(self.samples)):
            is_valid_data = True
            previous_sample = None
            current_indices = []
            for t in range(self.sequence_length):
                index_t = index + t
                # Going over the dataset size limit.
                if index_t >= len(self.samples):
                    is_valid_data = False
                    break
                sample = self.samples[index_t]
                # Check if scene is the same
                if (previous_sample is not None) and (sample['scene_token'] != previous_sample['scene_token']):
                    is_valid_data = False
                    break

                current_indices.append(index_t)
                previous_sample = sample

            if is_valid_data:
                indices.append(current_indices)

        return np.asarray(indices)
    
    def get_images(self, sample, cameras):
        images = []
        image_paths = []

        for cam in cameras:
            cam_token = self.nusc.get('sample_data', sample['data'][cam])
            image_path = self.nusc.get_sample_data_path(cam_token['token'])
            relative_path = os.path.relpath(image_path, self.nusc.dataroot)
            image_paths.append(relative_path)

            img = np.array(Image.open(image_path))
            img_tensor = torch.from_numpy(img).unsqueeze(0).permute(0, 3, 1, 2) # [1, H, W, 3] -> [1, 3, H, W]
            images.append(img_tensor)

        images = torch.cat(images, dim=0) # [cam_num, 3, H, W]
        return images, image_paths
    
    def get_lidar(self, sample):
        target_num_points = self.max_lidar_points
        lidar_token = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        cs_record = self.nusc.get('calibrated_sensor', lidar_token['calibrated_sensor_token'])
        lidar_path = self.nusc.get_sample_data_path(lidar_token['token'])
        relative_path = os.path.relpath(lidar_path, self.nusc.dataroot)
        lidar = LidarPointCloud.from_file(lidar_path).points.T
        lidar = torch.tensor(lidar, dtype=torch.float32) # [N, 4]

        # lidar -> ego
        points = lidar[:, :3]  # [N, 3]
        # translation vector
        t = torch.tensor(cs_record['translation'], dtype=torch.float32)  # [3]
        # rotation quaternion
        q = Quaternion(cs_record['rotation'])
        points_ego = torch.tensor(q.rotation_matrix, dtype=torch.float32) @ points.T + t.view(3, 1)  # [3, N]
        points_ego = points_ego.T  # [N, 3]

        lidar_ego = torch.cat([points_ego, lidar[:, 3:4]], dim=1)  # [N, 4]

        total_num_points = lidar_ego.size(0)
        # Pad lidar to ensure consistent tensor dimensions
        if total_num_points < target_num_points:
            padding = (0, 0, 0, target_num_points - total_num_points)
            padded_lidar = F.pad(lidar_ego, padding, value=0)
        else:
            padded_lidar = farthest_point_sampling(lidar_ego, target_num_points)
        return padded_lidar, relative_path, cs_record # [max_N, 4]
    
    def get_transform_w2ego(self, sample, inverse = False):
        if 'data' in sample:
            lidar_token = sample['data']['LIDAR_TOP']
        else:
            token = sample['token'][-1] if isinstance(sample['token'], list) else sample['token']
            full_sample = self.nusc.get('sample', token)
            lidar_token = full_sample['data']['LIDAR_TOP']
        sample_data = self.nusc.get('sample_data', lidar_token)
        ego_pose = self.nusc.get('ego_pose', sample_data['ego_pose_token'])
        T_w2ego = transform_matrix(ego_pose['translation'], Quaternion(ego_pose["rotation"]), inverse = inverse)
        # inverse=False： world -> ego
        # inverse=True ego -> world
        return T_w2ego
    
    def get_waypoints(self, sample, T_ego2w_cur):
        T_w2ego = self.get_transform_w2ego(sample)
        T_cur2ego = T_ego2w_cur @ T_w2ego # currnet frame as origin
        theta = quaternion_yaw(Quaternion(matrix = T_cur2ego)) # yaw angle in radians
        
        origin = np.array(T_cur2ego[:3, 3])
        waypoint = torch.tensor([origin[0], origin[1], theta]).unsqueeze(0) # [1, waypoint]
        return waypoint

    def get_ego_state(self, sample):
        timestamp = self.nusc.get('ego_pose', sample['data']['LIDAR_TOP'])['timestamp']

        scene_token = sample['scene_token']
        scene_name = [s['name'] for s in self.nusc.scene if s['token'] == scene_token][0]

        pose_msgs = self.can_bus.get_messages(scene_name, 'pose')
        closest = min(pose_msgs, key=lambda x: abs(x['utime'] - timestamp))
        velocity_vector = torch.tensor(closest['vel'][:2], dtype=torch.float32)
        accel = torch.tensor(closest['accel'][:2], dtype=torch.float32)
        yaw_rate = torch.tensor(closest['rotation_rate'][2], dtype=torch.float32)

        velocity = torch.norm(velocity_vector)

        return velocity, accel, yaw_rate

    def get_instance(self, sample):
        token = sample['token'][-1] if isinstance(sample['token'], list) else sample['token']
        sample_record = self.nusc.get('sample', token)
        # lidar_token = sample['data']['LIDAR_TOP']
        lidar_token = self.nusc.get('sample_data', sample_record['data']['LIDAR_TOP'])
        ego_pose = self.nusc.get('ego_pose', lidar_token['ego_pose_token'])
        all_instances = []

        for ann_token in sample['anns']:
            ann_record = self.nusc.get('sample_annotation', ann_token)
            visible = int(ann_record['visibility_token'])
            if visible < 2: # 0: unknown, 1: not_visible, 2: partly, 3: fully
                continue

            box = Box(ann_record['translation'], ann_record['size'], Quaternion(ann_record['rotation']))

            # Step 1: global -> ego
            box.translate(-np.array(ego_pose['translation']))
            box.rotate(Quaternion(ego_pose['rotation']).inverse)

            # Filter out instances that are too far away
            distance = np.linalg.norm(box.center[:2])
            if distance > 30:
                continue

            raw_cat = ann_record['category_name']
            parts = raw_cat.split('.')
            # Use parts[1] if available, otherwise fallback to the first part
            category = parts[1] if len(parts) > 1 else parts[0]

            yaw = box.orientation.yaw_pitch_roll[0]
            # bbox format: [x, y, z, w, l, h, yaw]
            bbox_inf = [
                round(float(box.center[0]), 2), round(float(box.center[1]), 2), round(float(box.center[2]), 2),
                round(float(box.wlh[0]), 2), round(float(box.wlh[1]), 2), round(float(box.wlh[2]), 2),
                round(float(yaw), 2)
            ]
            all_instances.append({
                # 'sample_token': token,
                # 'instance_token': ann_record['instance_token'],
                'label': category,
                'bbox': bbox_inf,
                'distance': distance
            })

        # Sort instances by distance for better readability in the JSONL
        all_instances = sorted(all_instances, key=lambda x: x['distance'])

        return all_instances

    #TODO: commands are not accurate enough, need to improve
    def classify_command(self, future_waypoints, cur_waypoint, stop_thresh=1.0, lane_width=3.5, uturn_thresh=np.deg2rad(150), turn_thresh=np.deg2rad(45)):
        # 1. Data Extraction
        data = future_waypoints.numpy() if isinstance(future_waypoints, torch.Tensor) else future_waypoints
        pts = data[:, :2]
        headings = data[:, 2]
        
        curr_pose_np = cur_waypoint.numpy().flatten() if isinstance(cur_waypoint, torch.Tensor) else cur_waypoint.flatten()
        curr_heading = curr_pose_np[2]

        start_pos = pts[0]
        end_pos = pts[-1]
        total_dist = np.linalg.norm(end_pos - start_pos)

        # 2. Stop Identification
        if total_dist < stop_thresh:
            return "Stop"

        # 3. Reverse Identification
        start_move_vec = pts[min(5, len(pts)-1)] - start_pos
        current_heading_vec = np.array([np.cos(curr_heading), np.sin(curr_heading)])
        if np.dot(start_move_vec, current_heading_vec) < 0:
            return "Reverse"

        # 4. Cumulative Yaw Calculation
        yaw_diffs = np.diff(headings)
        yaw_diffs = (yaw_diffs + np.pi) % (2 * np.pi) - np.pi
        total_yaw_change = np.sum(yaw_diffs)

        # 5. Lateral Displacement Calculation
        abs_yaw_change = abs(total_yaw_change)
        
        if abs_yaw_change > uturn_thresh:
                return "U-turn"
        
        if abs_yaw_change > turn_thresh:
                return "Turn Left" if total_yaw_change > 0 else "Turn Right"

        # norm_vec = np.array([-np.sin(curr_heading), np.cos(curr_heading)])
        # lat_offsets = [np.dot(p - start_pos, norm_vec) for p in pts]
        # final_lat_shift = lat_offsets[-1]        
        # abs_lat_shift = abs(final_lat_shift)
        # lc_yaw_tolerance = np.deg2rad(20)
        
        # if abs_lat_shift > lane_width * 0.5:
        #     if abs_yaw_change < lc_yaw_tolerance:
        #         return "Lane Change Left" if final_lat_shift > 0 else "Lane Change Right"
        #     else:
        #         return "Go Straight"
        
        return "Go Straight"
