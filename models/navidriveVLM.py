import re
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoProcessor, AutoModelForImageTextToText

class NaviDriveVLM(nn.Module):
    def __init__(self, driver_model_id, navi_model = "Qwen3", navi_model_id = "Qwen/Qwen3-VL-8B-Instruct", device="cuda"):
        super().__init__()
        self.navi = Navigator(navi_model, navi_model_id, device)
        self.driver = Driver(driver_model_id)
        
    def forward(self, navi_user_prompt, driver_user_prompt, images):
        reason = self.navi.generate_reason(navi_user_prompt, images)
        all_pred_waypoints = self.driver.generate_waypoints(reason, driver_user_prompt, images)
        return reason, all_pred_waypoints
    
class Navigator(nn.Module):
    def __init__(self, navi_model, model_id, device="auto"):
        super().__init__()
        self.model, self.processor = self.get_reason_model(navi_model, model_id, device)
        # Freeze the navigator model
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()
        
        self.system_prompt = (
            "You are an expert autonomous driving navigator. Your task is to analyze a 360-degree surround-view "
            "driving environment and provide concise, safety-oriented driving guidance.\n"
            "Guidelines:\n"
            "1. Coordinate System: The x-axis positive is forward, the y-axis positive is left.\n"
            "2. Attention Priority: Focus on 'Dynamic Hazards' (pedestrians, moving vehicles) and 'Traffic Regulators' "
            "(lights, signs, lane markings) on the front cameras.\n"
            "3. Output Format: Start with a concise 'Perception' summary, followed by 'Action', and a brief 'Reasoning'. "
            "Do not use '*' in the response."
        )
        self.gen_config = {"temperature": 1.0, "top_p": 1.0, "max_new_tokens": 2048}

    def get_reason_model(self, navi_model, model_id, device):
        if navi_model == "Qwen3":
            model = AutoModelForImageTextToText.from_pretrained(
                model_id,
                dtype=torch.bfloat16,
                device_map=device,
                trust_remote_code=True
                )
            processor = AutoProcessor.from_pretrained(
                model_id,
                min_pixels=128*28*28,
                max_pixels=224*28*28, 
                trust_remote_code=True
                )
            return model, processor
        else:
            print(f"Unsupported reason model: {navi_model}")
            raise NotImplementedError
    
    def generate_reason(self, user_prompt, images):
        inputs = get_vlm_inputs(self.processor, self.system_prompt, user_prompt, images, next(self.model.parameters()).device)
        
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                do_sample=True,
                **self.gen_config
            )
        
        input_len = inputs["input_ids"].shape[1]
        reason = self.processor.decode(output_ids[0][input_len:], skip_special_tokens=True)
        return reason

class Driver(nn.Module):
    def __init__(self, model_id, device="auto", is_training=False):
        super().__init__()
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            dtype=torch.bfloat16, 
            device_map=device,
            trust_remote_code=True
        )
        # Individual training driver model
        if is_training:
            for param in self.model.parameters():
                param.requires_grad = True
            self.model.train()
            if device and device != "auto": 
                self.model.to(device)
        else:
            self.model.eval()
            
        self.processor = AutoProcessor.from_pretrained(
            model_id,
            min_pixels=128*28*28,
            max_pixels=224*28*28, 
            trust_remote_code=True
        )
        self.system_prompt = (
            "You are an expert autonomous driving planning module (Driver). Your goal is to output a safe, "
            "smooth, and kinematically feasible future trajectory. "
            "Rules: "
            "1. Coordinate System: Current ego position is (0,0). X-axis positive is forward, Y-axis positive is left. "
            "2. Trajectory Timing: Output exactly 12 waypoints representing the next 6 seconds (2Hz, 0.5s intervals). "
            "3. Kinematic Constraints: Ensure consistent velocity and acceleration. Avoid unrealistic lateral shifts. "
            "4. Safety Alignment: The trajectory must strictly follow the Navigator's safety analysis."
        )
        self.gen_config = {"temperature": 0.7, "top_p": 0.8, "max_new_tokens": 1024}
        self.num_trajectories = 6
    
    def forward(self, input_ids, attention_mask, pixel_values, image_grid_thw=None, labels=None):
        kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
        if pixel_values is not None:
            kwargs["pixel_values"] = pixel_values
        if image_grid_thw is not None:
            kwargs["image_grid_thw"] = image_grid_thw

        outputs = self.model(**kwargs)
        return outputs.loss
    
    def generate_waypoints(self, reason, user_prompt, images):
        full_prompt = (
            f"{reason}\n\n" 
            f"{user_prompt}"
            )
        if isinstance(images, list) and len(images) > 1:
            front_image = images[1] # front_left, front, front_right, back_right, back, back_left
        else:
            front_image = images if not isinstance(images, list) else images[0]
            
        inputs = get_vlm_inputs(self.processor, self.system_prompt, full_prompt, front_image, next(self.model.parameters()).device)
        
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                do_sample=True,
                num_return_sequences=self.num_trajectories,
                **self.gen_config
            )
        
        input_len = inputs["input_ids"].shape[1]
        all_pred_waypoints = []
        
        for i in range(self.num_trajectories):
            raw_output = self.processor.decode(output_ids[i][input_len:], skip_special_tokens=True)
            try:
                pred_pts = parse_string(raw_output)
                all_pred_waypoints.append(pred_pts[:, :2].tolist())
            except Exception as e:
                print(f"Parsing error for sequence {i}: {e}")
                all_pred_waypoints.append([])
        return all_pred_waypoints    

def get_vlm_inputs(processor, system_prompt, user_prompt, images, target_device):
    if not isinstance(images, list):
        images = [images]
        
    content = [{"type": "image", "image": img} for img in images]
    content.append({"type": "text", "text": user_prompt})
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=images, return_tensors="pt", padding=True).to(target_device)
    
    return inputs

def parse_string(vlm_str):
    pattern = r"\(([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\)" # match (x, y) pairs with optional signs and decimal points
    matches = re.findall(pattern, vlm_str)

    if not matches:
        pattern_2d = r"\(([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\)"
        matches = re.findall(pattern_2d, vlm_str)    

    if not matches:
        return np.array([[]])
            
    results = [[float(x) for x in m] for m in matches]
    results = np.atleast_2d(np.array(results))
    
    return results
