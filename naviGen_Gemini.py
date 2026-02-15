import json
import argparse
import threading
import time
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed

from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type

# Google GenAI SDK
from google import genai
from google.genai import types
from google.api_core.exceptions import ResourceExhausted, ServiceUnavailable, InternalServerError

from nuscenes.nuscenes import NuScenes
from data.nuscenes_data import NuscenesData
from utils.data_utils import compute_action

PROJECT_ID = "norse-limiter-487520-c0"
LOCATION = "us-south1" # Dallas
client = genai.Client(
    vertexai=True,
    project=PROJECT_ID,
    location=LOCATION
)
file_lock = threading.Lock()

@retry(
    wait=wait_random_exponential(min=2, max=60),
    stop=stop_after_attempt(10),
    retry=retry_if_exception_type((ResourceExhausted, ServiceUnavailable, InternalServerError))
)

def call_gemini_api(model_id, contents, config):
    return client.models.generate_content(
        model=model_id,
        contents=contents,
        config=config
    )
    
def process_single_sample(sample, system_prompt, model_id, num_reasons):
    try:
        token = sample['token']
        pre_waypoints = sample['pre_waypoints']
        velocity = sample['velocity']
        acceleration = sample['accel']
        yaw_rate = sample['yaw_rate']
        future_waypoints = sample['future_waypoints']
        
        image_paths = sample['image_paths'][-1] # 
        
        new_order = [2, 3, 4, 5, 0, 1] # front_left, front, front_right, back_right, back, back_left
        
        pil_images = []
        for idx in new_order:
            img_path = image_paths[idx]
            with Image.open(img_path) as img:
                img.thumbnail((1024, 1024))
                pil_images.append(img.copy())

        wp_past = ", ".join([f"({pt[0]:.2f}, {pt[1]:.2f}, {pt[2]:.2f})" for pt in pre_waypoints])
        vel_val = round(velocity[-1].item(), 2)
        acc_val = [round(float(a), 2) for a in acceleration[-1]]
        yr_val = round(yaw_rate[-1].item(), 2)
        
        ego_status_prompt = (
            "Current Dynamics:\n"
            f"- Velocity: {vel_val:.2f} m/s.\n"
            f"- Yaw Rate: {yr_val:.2f} rad/s.\n"
            f"- Acceleration (Longitudinal x, Lateral y): ({acc_val[0]:.2f}, {acc_val[1]:.2f}) m/s^2.\n"
            f"- Past Trajectory (2Hz): {wp_past} m.\n\n"
        )
        
        user_prompt_text = (
            "Inputs: 6 images (Full Surround View) and Ego-Vehicle Status.\n"
            "1:FRONT_LEFT, 2:FRONT, 3:FRONT_RIGHT, 4:BACK_RIGHT, 5:BACK, 6:BACK_LEFT.\n"
            f"{ego_status_prompt}"
            "Task: Analyze the current situation and provide the safest next action with reasons."
        )

        contents = [
            types.Content(role="system", parts=[types.Part.from_text(system_prompt)]),
            types.Content(role="user", parts=[
                # Images Prompt
                *[types.Part.from_image(img) for img in pil_images],
                # Text Prompt
                types.Part.from_text(user_prompt_text)
            ])
        ]
        
        config = types.GenerateContentConfig(
            temperature=1.0,
            max_output_tokens=1024,
        )

        reasons = []
        for _ in range(num_reasons):
            response = call_gemini_api(model_id, contents, config)
            if response.text:
                reasons.append(response.text.strip())
            else:
                reasons.append("Error: Empty response")

        # calculate actions
        dt = 0.5 
        v0 = velocity[-1].item()
        a0 = acceleration[-1][0].item() 
        action_past = compute_action(pre_waypoints, dt, v0, a0)
        action_future = compute_action(future_waypoints, dt, v0, a0)
        wp_future = ", ".join([f"({pt[0]:.2f}, {pt[1]:.2f}, {pt[2]:.2f})" for pt in future_waypoints])
        
        return {
            "token": token,
            "wp_past": wp_past,
            "wp_future": wp_future,
            "vel_val": vel_val,
            "acc_val": acc_val,
            "yr_val": yr_val,
            "action_past": action_past,
            "action_future": action_future,
            "image_paths": image_paths, 
            "reasons": reasons
        }

    except Exception as e:
        print(f"\n[Error] Sample Token {sample.get('token', 'Unknown')}: {e}")
        return None    
    
def reasonGen_Gemini(model_id, data_path, output_file, version, system_prompt, is_train = 0, pre_frame = 4, 
                     future_frame = 12, num_reasons = 3, device = "auto", total_shards = 1, shard_id = 0,
                     max_workers = 4):

    print(f"Loading NuScenes dataset ({version})...")
    nusc = NuScenes(version=version, dataroot=data_path)
    dataset = NuscenesData(nusc, is_train, pre_frame, future_frame)
    
    total_samples = len(dataset)
    shard_size = total_samples // total_shards
    start_idx = shard_id * shard_size    
    end_idx = total_samples if shard_id == total_shards - 1 else (shard_id + 1) * shard_size
    indices = range(start_idx, end_idx)
    print(f"Shard {shard_id}/{total_shards}: Processing samples from {start_idx} to {end_idx}...")
    print(f"Target Model: {model_id} | Location: {LOCATION}")
    print(f"Concurrency: {max_workers} threads")
    
    with open(output_file, 'a', encoding='utf-8') as f:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(process_single_sample, dataset[i], system_prompt, model_id, num_reasons): i 
                for i in indices
            }
        for future in tqdm(as_completed(future_to_idx), total=len(indices), desc="Inference"):
            idx = future_to_idx[future]
            result = future.result()
            
            if result:
                with file_lock:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    f.flush()
                    
    print(f"\nDone! Results saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a samll-size driver LLM")
    parser.add_argument("--model_id", type=str, default="gemini-2.5-flash", help="Path to the configuration YAML file")
    parser.add_argument("--output_file", type=str, default="data/nuscenes_reasons_Gemini.jsonl", help="Path to the configuration YAML file")
    parser.add_argument("--data_path", type=str, default="/home/ximeng/Dataset/nuscenes_full_v1_0/", help="Path to the NuScenes dataset")
    parser.add_argument("--version", type=str, default="v1.0-trainval", choices=['v1.0-mini', 'v1.0-trainval'], help="Version of NuScenes dataset")
    parser.add_argument("--is_train", type=int, default=0, help="Whether to generate training data")
    parser.add_argument("--num_reasons", type=int, default=1, help="Number of reasons to generate")
    parser.add_argument("--total_shards", type=int, default=1, help="Total number of shards")
    parser.add_argument("--shard_id", type=int, default=0, help="Current shard index (0 to total_shards-1)")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel threads")
    
    args = parser.parse_args()

    system_prompt = (
        "You are an expert autonomous driving navigator. Your task is to analyze a 360-degree surround-view driving environment and provide concise, safety-oriented driving guidance.\n"
        "Guidelines:\n"
        "1. Coordinate System: The x-axis positive is forward, the y-axis positive is left.\n"
        "2. Attention Priority: Focus on 'Dynamic Hazards' (pedestrians, moving vehicles) and 'Traffic Regulators' (lights, signs, lane markings) on the front cameras.\n"
        "3. Output Format: Start with a concise 'Perception' summary, followed by 'Action', and a brief 'Reasoning'. Do not use '*' in the response."
    )
    reasonGen_Gemini(
        model_id=args.model_id,
        data_path=Path(args.data_path),
        output_file=args.output_file,
        version=args.version,
        system_prompt=system_prompt,
        is_train=args.is_train, # 0 train, 1 val
        num_reasons=args.num_reasons,
        total_shards=args.total_shards,
        shard_id=args.shard_id,
        max_workers=args.workers
    )
