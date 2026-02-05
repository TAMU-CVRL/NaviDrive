import cmd
import os
import sys
import json
from matplotlib.pylab import sample
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import re
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
from utils.caption_utils import reason_generate

MODEL_ID = "checkpoints/qwen3-1.7b-dllm-sft-0203"  # Replace with your local checkpoint path
DATA_ROOT = "/home/ximeng/Dataset/nuscenes_full_v1_0/"
INPUT_JSONL = "nusscenes_reasons_mini_0201.jsonl"
OUTPUT_JSONL = "driver_predicted_waypoints_3.jsonl"
SYSTEM_PROMPT = (
        "You are an expert autonomous driving planning module (Driver). Your goal is to output a safe, smooth, and kinematically feasible future trajectory.\n"
        "Rules:\n"
        "1. Coordinate System: Current ego position is (0,0). X-axis positive is forward, Y-axis positive is left.\n"
        "2. Trajectory Timing: Output exactly 12 waypoints representing the next 6 seconds (sampled at 2Hz, 0.5s intervals).\n"
        "3. Kinematic Constraints: Ensure the gaps between waypoints are consistent with the current velocity and acceleration. Avoid sudden jumps or unrealistic lateral shifts.\n"
        "4. Safety Alignment: The trajectory must strictly follow the Navigator's safety analysis.\n"
        "5. Output Format: Only output the coordinates: (x1, y1), (x2, y2), ..., (x12, y12)."
)

def load_model(model_id):
    print(f"Loading model: {model_id}")
    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct", trust_remote_code=True) # 1.7B and 8B share the same processor, but 1.7B cannot handle system and user prompt separately
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        dtype=torch.bfloat16, 
        device_map="auto",
        trust_remote_code=True
    )
    return processor, model

def parse_waypoints(vlm_str):
    pattern = r"\((-?\d+\.?\d*),\s*(-?\d+\.?\d*)\)"
    matches = re.findall(pattern, vlm_str)
    
    if matches:
        waypoints = np.array(matches, dtype=float)
        return waypoints
    else:
        print("Warning: No waypoints found in VLM output!")
        return np.array([])
    
def main():
    driver_processor, driver_model = load_model(MODEL_ID)
    driver_user_prompt_base = "Predict the next 12 waypoints."
    
    with open(INPUT_JSONL, 'r', encoding='utf-8') as f_in, \
         open(OUTPUT_JSONL, 'w', encoding='utf-8') as f_out:

        lines = f_in.readlines()
        for line in tqdm(lines, desc="Inference"):
            data = json.loads(line)
            
            ego_status_prompt = (
                "Current Dynamics:\n"
                f"- Velocity: {data['vel_val']} m/s.\n"
                f"- Yaw Rate: {data['yr_val']} rad/s.\n"
                f"- Acceleration (Longitudinal x, Lateral y): {data['acc_val']} m/s^2.\n"
                f"- Past Trajectory (2Hz): {data['wp_past']} m.\n\n"
                # f"- High-level Command: {data['command']}\n"
            )
            
            reason = data['reasons'][0] if isinstance(data['reasons'], list) else data['reasons']
            
            full_driver_prompt = (
                f"Navigator's Analysis and Instructions:\n{reason}\n\n"
                f"{ego_status_prompt}\n"
                f"{driver_user_prompt_base}"
            )

            # Model Inference
            _, output = reason_generate(
                user=full_driver_prompt,
                system=SYSTEM_PROMPT,
                # images=pil_images,
                processor=driver_processor,
                model=driver_model,
                do_sample=True,
                max_new_tokens=128
            )
            
            pred_pts = parse_waypoints(output)
            gt_pts = parse_waypoints(data['wp_future'])
            
            # Save Record
            record = {
                "token": data['token'],
                "gt_waypoints": gt_pts.tolist(),
                "pred_waypoints": pred_pts.tolist(),
                "reasons": data['reasons'],
            }
            f_out.write(json.dumps(record) + "\n")
            f_out.flush()

    print(f"Success! Predictions saved to {OUTPUT_JSONL}")

if __name__ == "__main__":
    main()
