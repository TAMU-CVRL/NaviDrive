import os
import json
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForImageTextToText

from utils.img_utils import resize_long_edge
from utils.caption_utils import reason_generate
from nuscenes.nuscenes import NuScenes
from data.nuscenes_data import NuscenesData

def reasonGen(model_id, data_path, output_file, version, system_prompt, is_train=0, pre_frame=4, future_frame=12, num_reasons=3, device="auto"):
    print(f"Loading model: {model_id}...")
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, 
        dtype=torch.bfloat16, 
        device_map=device,
        attn_implementation="flash_attention_2"
    )

    print(f"Loading NuScenes dataset from {data_path}...")
    nusc = NuScenes(version=version, dataroot=data_path)
    dataset = NuscenesData(nusc, is_train, pre_frame, future_frame)

    print(f"Starting inference on {len(dataset)} samples...")
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'a', encoding='utf-8') as f:
        for i in tqdm(range(len(dataset))):
            try:
                sample = dataset[i]
                token = sample['token']
                
                raw_images = sample['raw_images']
                pre_waypoints = sample['pre_waypoints']
                velocity = sample['velocity']
                acceleration = sample['accel']
                yaw_rate = sample['yaw_rate']
                future_waypoints = sample['future_waypoints']
                command = sample['command']
                image_paths = sample['image_paths'][-1] # Get the latest frame image paths
                
                new_order = [2, 3, 4, 5, 0, 1]
                current_images = raw_images[-1]
                pil_images = []
                for idx in new_order:
                    img_np = current_images[idx].permute(1, 2, 0).cpu().numpy()
                    img_pil = Image.fromarray(img_np.astype('uint8'))
                    img_pil = resize_long_edge(img_pil, 512)
                    pil_images.append(img_pil)

                pts = pre_waypoints.cpu().numpy().tolist()
                wp_past = ", ".join([f"({pt[0]:.2f}, {pt[1]:.2f})" for pt in pts])
                vel_val = round(velocity[-1].item(), 2)
                acc_val = [round(float(a), 2) for a in acceleration[-1]]
                yr_val = round(yaw_rate[-1].item(), 2)
                
                ego_status_prompt = (
                    "Current Dynamics:\n"
                    f"- Velocity: {vel_val:.2f} m/s.\n"
                    f"- Yaw Rate: {yr_val:.2f} rad/s.\n"
                    f"- Acceleration (Longitudinal x, Lateral y): ({acc_val[0]:.2f}, {acc_val[1]:.2f}) m/s^2.\n"
                    f"- Past Trajectory (2Hz): {wp_past} m.\n\n"
#                    f"- High-level Command: {command}\n"
                )
                
                user_prompt = (
                    "Inputs: 6 images (Full Surround View) and Ego-Vehicle Status.\n"
                    "1:FRONT_LEFT, 2:FRONT, 3:FRONT_RIGHT, 4:BACK_RIGHT, 5:BACK, 6:BACK_LEFT.\n"
                    f"{ego_status_prompt}"
                    "Task: Analyze the current situation and provide the safest next action with reasons."
                )
                
                reasons = []
                for _ in range(num_reasons):
                    _, res = reason_generate(
                        user=user_prompt,
                        system=system_prompt,
                        images=pil_images,
                        processor=processor,
                        model=model,
                        do_sample=True,
                        temperature=1.0,
                        top_p=1.0,
                    )
                    reasons.append(res.strip())
                
                wp_future = ", ".join([f"({pt[0]:.2f}, {pt[1]:.2f})" for pt in future_waypoints])
                
                data_record = {
                    "token": token,
                    "wp_past": wp_past,
                    "wp_future": wp_future,
                    "vel_val": vel_val,
                    "acc_val": acc_val,
                    "yr_val": yr_val,
                    "command": command,
                    "image_paths": image_paths,
                    "reasons": reasons
                }

                f.write(json.dumps(data_record, ensure_ascii=False) + "\n")
                f.flush()

            except Exception as e:
                print(f"\nError processing sample {i}: {e}")
                continue

if __name__ == "__main__":
#    model_id = "Qwen/Qwen3-VL-30B-A3B-Instruct" # MOE model, existing some issues
    model_id = "Qwen/Qwen3-VL-32B-Instruct"
    data_path = Path("/mnt/nuscenes")
    if not data_path.exists() or not any(data_path.iterdir()):
        raise FileNotFoundError(f"Data path {data_path} does not exist or is empty: {list(data_path.glob('*'))}")
    output_file = f"/scratch/user/ximeng/dllm/nusscenes_reasons.jsonl"
    
    system_prompt = (
        "You are an expert autonomous driving navigator. Your task is to analyze a 360-degree surround-view driving environment and provide concise, safety-oriented driving guidance.\n"
        "Guidelines:\n"
        "1. Coordinate System: The x-axis positive is forward, the y-axis positive is left.\n"
        "2. Attention Priority: Focus on 'Dynamic Hazards' (pedestrians, moving vehicles) and 'Traffic Regulators' (lights, signs, lane markings).\n"
        "3. Camera Emphasis: Focus primarily on information from the front cameras.\n"
        "4. Output Format: Start with a concise 'Perception' summary, followed by 'Action', and a brief 'Reasoning'."
    )
    
    reasonGen(
        model_id=model_id, 
        data_path=data_path, 
        output_file=output_file, 
        version='v1.0-trainval',
        system_prompt=system_prompt, 
        num_reasons=1, 
        is_train=0 # 0 train, 1 val
    )
    