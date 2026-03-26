from xml.parsers.expat import model
import torch
import numpy as np
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion
from collections import Counter
import re

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
    pattern = r"\(([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\)" # match (x, y) pairs with optional signs and decimal points
    matches = re.findall(pattern, vlm_str)

    if not matches:
        pattern_2d = r"\(([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\)"
        matches = re.findall(pattern_2d, vlm_str)    
        
    results = [[float(x) for x in m] for m in matches]
    results = np.atleast_2d(np.array(results))
    
    return results
