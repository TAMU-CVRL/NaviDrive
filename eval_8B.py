import os

# Set cache directories to scratch space
#SCRATCH_DIR = "/scratch/group/p.cis250376.000"
#os.environ["HF_HOME"] = f"{SCRATCH_DIR}/hf_cache"
#os.environ["TRANSFORMERS_CACHE"] = f"{SCRATCH_DIR}/hf_cache/transformers"
#os.environ["HF_DATASETS_CACHE"] = f"{SCRATCH_DIR}/hf_cache/datasets"
#os.environ["TORCH_HOME"] = f"{SCRATCH_DIR}/torch_cache"

import json
import re
from datetime import datetime
import cv2
import torch
import numpy as np
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from peft import PeftModel
from nuscenes.nuscenes import NuScenes
from utils.results_utils import project_wp_to_image

# Hardcoded paths
CHECKPOINT_PATH = "checkpoints/qwen25-7b-dllm-sft-0203/checkpoint-1036"
#CHECKPOINT_PATH = "/home/avalocal/Desktop/iros26/NaviDrive/checkpoints/qwen25-7b-dllm-sft-vislora-waypoints-w-vision-tower-lora/checkpoint-777"
BASE_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
NUSCENES_ROOT = "/media/avalocal/T7/NuScenes"
NUSCENES_VERSION = "v1.0-trainval"
INPUT_JSONL = "data/nuscenes_reasons_val_Qwen_32B.jsonl"
OUTPUT_JSONL = "eval_results_8B.jsonl"
VIDEO_OUTPUT = "test.mp4"
FPS = 2

# NuScenes official val split scene names (150 scenes)
VAL_SCENES = [
    'scene-0003', 'scene-0012', 'scene-0013', 'scene-0014', 'scene-0015',
    'scene-0016', 'scene-0017', 'scene-0018', 'scene-0035', 'scene-0036',
    'scene-0038', 'scene-0039', 'scene-0092', 'scene-0093', 'scene-0094',
    'scene-0095', 'scene-0096', 'scene-0097', 'scene-0098', 'scene-0099',
    'scene-0100', 'scene-0101', 'scene-0102', 'scene-0103', 'scene-0104',
    'scene-0105', 'scene-0106', 'scene-0107', 'scene-0108', 'scene-0109',
    'scene-0110', 'scene-0221', 'scene-0268', 'scene-0269', 'scene-0270',
    'scene-0271', 'scene-0272', 'scene-0273', 'scene-0274', 'scene-0275',
    'scene-0276', 'scene-0277', 'scene-0278', 'scene-0329', 'scene-0330',
    'scene-0331', 'scene-0332', 'scene-0344', 'scene-0345', 'scene-0346',
    'scene-0347', 'scene-0348', 'scene-0349', 'scene-0350', 'scene-0351',
    'scene-0352', 'scene-0353', 'scene-0354', 'scene-0355', 'scene-0356',
    'scene-0357', 'scene-0358', 'scene-0359', 'scene-0360', 'scene-0361',
    'scene-0362', 'scene-0363', 'scene-0364', 'scene-0365', 'scene-0366',
    'scene-0367', 'scene-0368', 'scene-0369', 'scene-0370', 'scene-0371',
    'scene-0372', 'scene-0373', 'scene-0374', 'scene-0375', 'scene-0376',
    'scene-0377', 'scene-0378', 'scene-0379', 'scene-0380', 'scene-0381',
    'scene-0382', 'scene-0383', 'scene-0384', 'scene-0385', 'scene-0386',
    'scene-0388', 'scene-0389', 'scene-0390', 'scene-0391', 'scene-0392',
    'scene-0393', 'scene-0394', 'scene-0395', 'scene-0396', 'scene-0397',
    'scene-0398', 'scene-0399', 'scene-0400', 'scene-0401', 'scene-0402',
    'scene-0403', 'scene-0405', 'scene-0406', 'scene-0407', 'scene-0408',
    'scene-0410', 'scene-0411', 'scene-0412', 'scene-0413', 'scene-0414',
    'scene-0415', 'scene-0416', 'scene-0417', 'scene-0418', 'scene-0419',
    'scene-0420', 'scene-0421', 'scene-0422', 'scene-0423', 'scene-0424',
    'scene-0425', 'scene-0426', 'scene-0427', 'scene-0428', 'scene-0429',
    'scene-0430', 'scene-0431', 'scene-0432', 'scene-0433', 'scene-0434',
    'scene-0435', 'scene-0436', 'scene-0437', 'scene-0438', 'scene-0439',
    'scene-0440', 'scene-0441', 'scene-0442', 'scene-0443', 'scene-0444',
    'scene-0445', 'scene-0446', 'scene-0447', 'scene-0448', 'scene-0449',
]

SYSTEM_PROMPT = (
    "You are an expert autonomous driving planning module (Driver). Your goal is to output a safe, smooth, and kinematically feasible future trajectory.\n"
    "Rules:\n"
    "1. Coordinate System: Current ego position is (0,0). X-axis positive is forward, Y-axis positive is left.\n"
    "2. Trajectory Timing: Output exactly 12 waypoints (except origin (0,0)) representing the next 6 seconds (sampled at 2Hz, 0.5s intervals).\n"
    "3. Kinematic Constraints: Ensure the gaps between waypoints are consistent with the current velocity and acceleration. Avoid sudden jumps or unrealistic lateral shifts.\n"
    "4. Safety Alignment: The trajectory must strictly follow the Navigator's safety analysis.\n"
    "5. Output Format: Only output the coordinates: (x1, y1), (x2, y2), ..., (x12, y12)."
)

def load_model():
    print(f"Loading base model: {BASE_MODEL_ID}")
    processor = AutoProcessor.from_pretrained(
        BASE_MODEL_ID,
        trust_remote_code=True,
        min_pixels=384*216,      # ~83k pixels (16:9 aspect ratio)
        max_pixels=512*288,      # ~147k pixels (16:9 aspect ratio)
    )
    
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
        #attn_implementation="flash_attention_2",
    )
    
    print(f"Loading LoRA adapter: {CHECKPOINT_PATH}")
    model = PeftModel.from_pretrained(base_model, CHECKPOINT_PATH)
    model.eval()
    
    return processor, model

def parse_waypoints(vlm_str):
    """Extract waypoints (x, y) from model output. Handles both (x,y) and (x,y,heading) formats."""
    pattern = r"\((-?\d+\.?\d*),\s*(-?\d+\.?\d*)(?:,\s*-?\d+\.?\d*)?\)"
    matches = re.findall(pattern, vlm_str)
    
    if matches:
        return np.array(matches, dtype=float)
    else:
        return np.array([])

# def compute_metrics(pred_pts, gt_pts):
#     """Compute ADE and FDE"""
#     if len(pred_pts) == 0 or len(gt_pts) == 0:
#         return float('inf'), float('inf')
#     
#     # Truncate to same length
#     min_len = min(len(pred_pts), len(gt_pts))
#     pred_pts = pred_pts[:min_len]
#     gt_pts = gt_pts[:min_len]
#     
#     # ADE: Average Displacement Error
#     distances = np.linalg.norm(pred_pts - gt_pts, axis=1)
#     ade = np.mean(distances)
#     
#     # FDE: Final Displacement Error
#     fde = distances[-1]
#     
#     return ade, fde

def main():
    processor, model = load_model()
    
    # Load NuScenes for visualization
    print("Loading NuScenes for visualization...")
    nusc = NuScenes(version=NUSCENES_VERSION, dataroot=NUSCENES_ROOT, verbose=False)
    
    # Camera order for 6-view input
    CAMERAS = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_RIGHT", "CAM_BACK", "CAM_BACK_LEFT"]
    
    def get_image_paths_from_token(token):
        """Get 6 camera image paths from a sample token"""
        sample = nusc.get('sample', token)
        paths = []
        for cam in CAMERAS:
            cam_data = nusc.get('sample_data', sample['data'][cam])
            paths.append(os.path.join(NUSCENES_ROOT, cam_data['filename']))
        return paths
    
    # Load JSONL. Set MAX_SAMPLES=None for all, or a number to limit.
    MAX_SAMPLES = None
    with open(INPUT_JSONL, 'r') as f_in:
        lines = f_in.readlines()
    if MAX_SAMPLES is not None:
        lines = lines[:MAX_SAMPLES]
    
    print(f"Loaded {len(lines)} samples from {INPUT_JSONL}")
    
    # L2 at specific time horizons (2Hz: 1s=idx1, 2s=idx3, 3s=idx5, 6s=idx11)
    all_l2_1s = []
    all_l2_2s = []
    all_l2_3s = []
    all_l2_6s = []
    all_ade = []
    all_fde = []
    # Collision: trajectory feasibility (consecutive waypoint jump > threshold)
    coll_1s = []
    coll_2s = []
    coll_3s = []
    coll_6s = []
    frames = []
    
    def check_collision(waypoints, max_wp, speed_thresh=10.0):
        """Check if trajectory has unrealistic jumps (proxy for collision/infeasibility)
        At 2Hz, max reasonable distance between waypoints = speed * 0.5s
        speed_thresh: max reasonable speed (m/s), default 10 m/s ~ 36 km/h
        """
        max_step_dist = speed_thresh * 0.5  # max distance per 0.5s step
        pts = waypoints[:max_wp]
        if len(pts) < 2:
            return False
        for i in range(1, len(pts)):
            dist = np.linalg.norm(pts[i] - pts[i-1])
            if dist > max_step_dist:
                return True
        return False
    
    print(f"Evaluating {len(lines)} samples...")
    
    for line in tqdm(lines, desc="Evaluating"):
        data = json.loads(line)
        
        # Get token and compute image paths from NuScenes
        token = data['token'][0] if isinstance(data['token'], list) else data['token']
        image_paths = get_image_paths_from_token(token)
        
        # Build ego status prompt
        ego_status_prompt = (
            f"Current Dynamics:\n"
            f"- Velocity: {data['vel_val']:.2f} m/s\n"
            f"- Yaw Rate: {data['yr_val']:.2f} rad/s\n"
            f"- Acceleration (Longitudinal x, Lateral y): {data['acc_val']}\n"
            f"- Past Trajectory (2Hz): {data['wp_past']}\n"
        )
        
        driver_user_prompt = (
            "Inputs: 6 images (Full Surround View) and Ego-Vehicle Status.\n"
            "1:FRONT_LEFT, 2:FRONT, 3:FRONT_RIGHT, 4:BACK_RIGHT, 5:BACK, 6:BACK_LEFT.\n"
            f"{ego_status_prompt}"
            "Predict the next 12 waypoints."
        )
        
        # Build prompt messages
        prompt_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": (
                [{"type": "image", "image": p} for p in image_paths] +
                [{"type": "text", "text": driver_user_prompt}]
            )}
        ]
        
        # Process inputs
        prompt_text = processor.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info([prompt_messages])
        
        inputs = processor(
            text=[prompt_text],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
        ).to(model.device)
        
        # Generate
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=256)
        
        # Decode output
        generated_text = processor.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        
        # Parse waypoints
        pred_pts = parse_waypoints(generated_text)
        gt_pts = parse_waypoints(data['wp_future'])
        
        # Compute metrics: L2 at specific time points (standard NuScenes convention)
        # 2Hz sampling: 1s=idx1, 2s=idx3, 3s=idx5, 6s=idx11
        if len(pred_pts) > 0 and len(gt_pts) > 0:
            min_len = min(len(pred_pts), len(gt_pts))
            pred_pts_eval = pred_pts[:min_len]
            gt_pts_eval = gt_pts[:min_len]
            distances = np.linalg.norm(pred_pts_eval - gt_pts_eval, axis=1)
            
            if min_len > 1:
                all_l2_1s.append(distances[1])
                coll_1s.append(check_collision(pred_pts_eval, 2))
            if min_len > 3:
                all_l2_2s.append(distances[3])
                coll_2s.append(check_collision(pred_pts_eval, 4))
            if min_len > 5:
                all_l2_3s.append(distances[5])
                coll_3s.append(check_collision(pred_pts_eval, 6))
            if min_len > 11:
                all_l2_6s.append(distances[11])
            else:
                all_l2_6s.append(distances[-1])
            coll_6s.append(check_collision(pred_pts_eval, min_len))
            all_ade.append(np.mean(distances))
            all_fde.append(distances[-1])
        
        # Visualize and store frame
        try:
            sample_rec = nusc.get('sample', token)
            cam_front_data = nusc.get('sample_data', sample_rec['data']['CAM_FRONT'])
            img_path = os.path.join(NUSCENES_ROOT, cam_front_data['filename'])
            raw_img = cv2.imread(img_path)
            
            if raw_img is not None and len(pred_pts) > 0 and len(gt_pts) > 0:
                # Draw predicted waypoints (red)
                vis_img = project_wp_to_image(
                    nusc, token, pred_pts.astype(np.float32), raw_img,
                    color_waypoints=(0, 0, 255),
                    color_polygon=(0, 0, 200),
                    plot_polygon=True
                )
                # Draw ground truth waypoints (green)
                vis_img = project_wp_to_image(
                    nusc, token, gt_pts.astype(np.float32), vis_img,
                    color_waypoints=(0, 255, 0),
                    plot_polygon=False
                )
                frames.append(vis_img)
        except Exception as e:
            print(f"Warning: Could not visualize frame {token}: {e}")
    
    # Save video
    if frames:
        print(f"\nSaving video to {VIDEO_OUTPUT}...")
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(VIDEO_OUTPUT, fourcc, FPS, (width, height))
        for frame in frames:
            video_writer.write(frame)
        video_writer.release()
        print(f"Video saved: {VIDEO_OUTPUT} ({len(frames)} frames)")
    
    # Load adapter config for hyperparameter reporting
    adapter_cfg_path = os.path.join(CHECKPOINT_PATH, "adapter_config.json")
    lora_info = {}
    if os.path.exists(adapter_cfg_path):
        with open(adapter_cfg_path) as f:
            lora_info = json.load(f)
    
    training_args_path = os.path.join(CHECKPOINT_PATH, "training_args.bin")
    train_args = {}
    if os.path.exists(training_args_path):
        train_args = torch.load(training_args_path, map_location="cpu", weights_only=False)
    
    # Compute metrics
    l2_1s = np.mean(all_l2_1s) if all_l2_1s else float('nan')
    l2_2s = np.mean(all_l2_2s) if all_l2_2s else float('nan')
    l2_3s = np.mean(all_l2_3s) if all_l2_3s else float('nan')
    l2_6s = np.mean(all_l2_6s) if all_l2_6s else float('nan')
    ade = np.mean(all_ade) if all_ade else float('nan')
    fde = np.mean(all_fde) if all_fde else float('nan')
    
    failure_thresh = 2.0
    failure_rate = np.mean([1 if d > failure_thresh else 0 for d in all_l2_6s]) * 100 if all_l2_6s else 0
    reliability = 100.0 - failure_rate
    
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    w = 85
    dl = "=" * w
    sl = "-" * w
    
    # Build report
    report = (
        f"\n{dl}\n"
        f"  Training Hyperparameters:\n"
        f"{dl}\n"
        f"  • Model:                  {BASE_MODEL_ID}\n"
        f"  • Checkpoint:             {CHECKPOINT_PATH}\n"
        f"  • LoRA r:                 {lora_info.get('r', 'N/A')}\n"
        f"  • LoRA alpha:             {lora_info.get('lora_alpha', 'N/A')}\n"
        f"  • LoRA dropout:           {lora_info.get('lora_dropout', 'N/A')}\n"
        f"  • Target Modules:         {lora_info.get('target_modules', 'N/A')}\n"
        f"  • Epochs:                 {getattr(train_args, 'num_train_epochs', 'N/A')}\n"
        f"  • Batch Size:             {getattr(train_args, 'per_device_train_batch_size', 'N/A')}\n"
        f"  • Gradient Accumulation:  {getattr(train_args, 'gradient_accumulation_steps', 'N/A')}\n"
        f"  • Learning Rate:          {getattr(train_args, 'learning_rate', 'N/A')}\n"
        f"  • LR Scheduler:           {getattr(train_args, 'lr_scheduler_type', 'N/A')}\n"
        f"  • Optimizer:              {getattr(train_args, 'optim', 'N/A')}\n"
        f"{dl}\n"
        f"{dl}\n"
        f"  TRAJECTORY EVALUATION REPORT  |  {date_str}\n"
        f"{dl}\n"
        f"  [Input File]  : {INPUT_JSONL}\n"
        f"  [Sample Count]: {len(all_ade)}\n"
        f"  [Failure Thresh] : {failure_thresh} m\n"
        f"{sl}\n"
        f"  {'Metric':<15} | {'1.0s':<10} | {'2.0s':<10} | {'3.0s':<10} | {'6.0s (FDE)':<12} | {'Avg (ADE)':<10}\n"
        f"  {'-'*15}-|-{'-'*10}-|-{'-'*10}-|-{'-'*10}-|-{'-'*12}-|-{'-'*10}\n"
        f"  {'L2 Error (m)':<15} | {l2_1s:<10.3f} | {l2_2s:<10.3f} | {l2_3s:<10.3f} | {l2_6s:<12.3f} | {ade:<10.3f}\n"
        f"{sl}\n"
        f"  OVERALL PERFORMANCE\n"
        f"  > Failure Rate :  {failure_rate:.2f} %\n"
        f"  > Reliability  :  {reliability:.2f} % (within {failure_thresh}m)\n"
        f"{dl}\n"
    )
    
    print(report)
    
    # Save to file
    os.makedirs("results", exist_ok=True)
    result_file = f"results/eval_8B_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(result_file, 'w') as f:
        f.write(report)
    print(f"Results saved to: {result_file}")

if __name__ == "__main__":
    main()
