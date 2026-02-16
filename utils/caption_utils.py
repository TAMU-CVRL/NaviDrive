from xml.parsers.expat import model
import torch
import numpy as np
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion
from collections import Counter
import re

def describe_camera_annotations(nusc, sample_record, camera_name, box_vis_level=1):
    cam_token = sample_record["data"][camera_name]
    _, boxes, _ = nusc.get_sample_data(cam_token, box_vis_level=box_vis_level)

    if len(boxes) == 0:
        description = "No annotated objects are visible in this camera view."
        return description

    # find categories and counts
    categories = [
        nusc.get("sample_annotation", box.token)["category_name"].split(".")[-1]
        for box in boxes
    ]
    counter = Counter(categories)

    # build description
    parts = [f"{count} {cat}{'s' if count > 1 else ''}" for cat, count in counter.items()]
    if len(parts) == 1:
        description = f"This scene includes {parts[0]}."
    elif len(parts) == 2:
        description = f"This scene includes {parts[0]} and {parts[1]}."
    else:
        description = f"This scene includes {', '.join(parts[:-1])}, and {parts[-1]}."

    return description

def lidar2camera_fov(nusc, points_ego, token, camera_name):
    sample_record = nusc.get('sample', token)
    cam_token = nusc.get('sample_data', sample_record['data'][camera_name])
    cam_calib = nusc.get('calibrated_sensor', cam_token['calibrated_sensor_token'])

    # Camera extrinsics (ego -> camera)
    q = Quaternion(cam_calib['rotation'])
    t = torch.tensor(cam_calib['translation'], dtype=torch.float32)  # [3]
    R = torch.tensor(q.rotation_matrix, dtype=torch.float32)         # [3,3]
    cam_intrinsic = np.array(cam_calib['camera_intrinsic'])

    pc_cam = R.T @ (points_ego[:, :3].T - t.view(3, 1))  # [3, N]
    pc_cam_np = pc_cam.detach().cpu().numpy()  # to numpy

    points_img = view_points(pc_cam_np, cam_intrinsic, normalize=True)

    W, H = (1600, 900) # image size for nuScenes camera
    mask = (points_img[0, :] > 0) & (points_img[0, :] < W) & \
           (points_img[1, :] > 0) & (points_img[1, :] < H) & \
           (pc_cam_np[2, :] > 0)

    visible_points = points_ego[mask]
    return visible_points, mask

def caption_generate(describe, prompt, system, image, processor, model):
    text = (
        f"{describe}\n"
        f"{prompt}"
    )

    messages = [
        {
            "role": "system",
            "content": system,
        },
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": text},
            ],
        },
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
    with torch.no_grad():
        generate_ids = model.generate(**inputs, max_new_tokens=77)

    caption = processor.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return caption

def scene_generate(instance_data, prompt, system, images, processor, model):
    objs = instance_data[0] if isinstance(instance_data[0], list) else instance_data
    instance_context = "Nearby objects relative to the ego vehicle (label, distance, bbox [x, y, z, w, l, h, yaw]):\n"
    for obj in objs[:20]:
        label = obj['label']
        dist = obj['distance']
        x, y = obj['bbox'][0], obj['bbox'][1]
        direction = get_cardinal_direction(x, y)
        bbox_str = "[" + ", ".join([f"{x:.1f}" for x in obj['bbox']]) + "]"
        instance_context += f"- {label} is {direction} at {dist:.1f}m, bbox: {bbox_str}\n"

    content = [{"type": "image"} for _ in range(len(images))]
    
    view_order = "Image sequence: [1:BACK, 2:BACK_LEFT, 3:FRONT_LEFT, 4:FRONT, 5:FRONT_RIGHT, 6:BACK_RIGHT]."
    
    text_content = f"{view_order}\n\n{instance_context}\nTask: {prompt}"
    content.append({"type": "text", "text": text_content})

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": content},
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=images, return_tensors="pt", padding=True)
    device = next(model.parameters()).device
    inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}

    with torch.no_grad():
        generate_ids = model.generate(**inputs, max_new_tokens=2048)

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generate_ids)
    ]
    caption = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    
    return caption

def scene_generate_v2(full_prompt, system, processor, model, images=None, do_sample=False):
    if images is None:
        images = []

    if isinstance(full_prompt, list):
        content = full_prompt
    else:
        content = [{"type": "image"} for _ in range(len(images))]
        content.append({"type": "text", "text": full_prompt})

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": content},
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if len(images) > 0:
        inputs = processor(text=[text], images=images, return_tensors="pt", padding=True)
    else:
        inputs = processor(text=[text], return_tensors="pt", padding=True)
    device = next(model.parameters()).device
    inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}

    with torch.no_grad():
        generate_ids = model.generate(**inputs, max_new_tokens=2048, do_sample=do_sample)

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generate_ids)
    ]
    caption = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    
    return messages, caption

def reason_generate(user, system, processor, model, images=None, do_sample=False, max_new_tokens=2048, **kwargs):
    if images is None:
        images = []

    if isinstance(user, list):
        content = user
    else:
        content = [{"type": "image"} for _ in range(len(images))]
        content.append({"type": "text", "text": user})

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": content},
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if len(images) > 0:
        inputs = processor(text=[text], images=images, return_tensors="pt", padding=True)
    else:
        inputs = processor(text=[text], return_tensors="pt", padding=True)
    device = next(model.parameters()).device
    inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}

    with torch.no_grad():
        generate_ids = model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens, # 512
            do_sample=do_sample,
            **kwargs 
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generate_ids)
    ]
    caption = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    
    return messages, caption

def get_cardinal_direction(x, y):
    """Converts ego coordinates to spatial text."""
    angle = np.arctan2(y, x) * 180 / np.pi
    # Standard nuScenes: x-front, y-left
    if -22.5 <= angle <= 22.5: return "directly ahead"
    if 22.5 < angle <= 67.5: return "front-left"
    if 67.5 < angle <= 112.5: return "to the left"
    if 112.5 < angle <= 157.5: return "rear-left"
    if angle > 157.5 or angle <= -157.5: return "directly behind"
    if -157.5 < angle <= -112.5: return "rear-right"
    if -112.5 < angle <= -67.5: return "to the right"
    if -67.5 < angle <= -22.5: return "front-right"
    return "nearby"

def parse_waypoints(vlm_str):
    pattern = r"\((-?\d+\.?\d*),\s*(-?\d+\.?\d*)\)"
    matches = re.findall(pattern, vlm_str)
    
    waypoints = np.array(matches, dtype=float)
    return waypoints
    
def parse_string(vlm_str):
    pattern = r"\(([^)]+)\)"
    matches = re.findall(pattern, vlm_str)
    
    results = []
    for m in matches:
        point = [float(x.strip()) for x in m.split(',')]
        results.append(point)   
    
    results = np.array(results)
    return results
