import argparse
import json
import os
import numpy as np
import cv2
import torch
import wandb
import time

from tqdm import tqdm
from datetime import datetime
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
from transformers import (
    Trainer,
    TrainingArguments,
    AutoConfig, 
    AutoModelForCausalLM, 
    AutoModelForImageTextToText, 
    AutoProcessor, 
    BitsAndBytesConfig
)
from datasets import load_dataset
from nuscenes.nuscenes import NuScenes
from qwen_vl_utils import process_vision_info

from utils.data_utils import preprocess_data_img, load_config, compute_trajectory_2, filter_to_xy_str
from utils.caption_utils import parse_string
from utils.results_utils import calculate_metrics, format_results, render_frame

class driverEngine():
    def __init__(self, cfg):
        self.cfg = cfg
        self.name = cfg["Name"]
        self.model_id = cfg["Model"]["model_id"]
        self.attention_type = cfg["Model"]["attention"]
        self.date_str = datetime.now().strftime("%d%H%M")
        
        # Quantization & LoRA
        self.enable_quant = cfg["Model"]["Quantization"]["enable_quantization"]
        
        # Data Paths
        self.train_data_path = cfg["Dataset"]["train_data_path"]
        self.mini_data_path = cfg["Dataset"]["mini_data_path"]
        self.nuscenes_dataroot = cfg["Dataset"]["nuscenes_dataroot"]
        # Prompts
        self.system_prompt = cfg["Dataset"]["system_prompt"]
        self.driver_user_prompt = cfg["Dataset"]["driver_user_prompt"]
        
        # Training Hyperparameters
        self.train_cfg = cfg["Train"]
        self.epochs = self.train_cfg["epochs"]
        self.batch_size = self.train_cfg["batch_size"]
        self.gradient_accumulation_steps = self.train_cfg["gradient_accumulation_steps"]
        self.learning_rate = self.train_cfg["learning_rate"]
        self.lr_scheduler_type = self.train_cfg["lr_scheduler_type"]
        self.optimizer = self.train_cfg["optimizer"]
        self.weight_decay = self.train_cfg["weight_decay"]
        self.log_to = self.train_cfg["log_to"]
        self.max_length = self.train_cfg["max_length"]
        
        # Flags
        self.enable_action = self.train_cfg.get("enable_action", False)
        self.enable_image = self.train_cfg.get("enable_image", False)
        self.image_indices = self.train_cfg.get("image_indices", None)
        self.enable_reason = self.train_cfg.get("enable_reason", True)
        self.enable_command = self.cfg["Dataset"].get("enable_command", False)
        
    def init_wandb(self):
        wandb.init(
            project="dllm",
            # name=self.name + "_" + self.date_str,
            name=self.name,
            config={
                "model_id": self.model_id,
                "attention_type": self.attention_type,
                "enable_quantization": self.enable_quant,
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "gradient_accumulation_steps": self.gradient_accumulation_steps,
                "learning_rate": self.learning_rate,
                "lr_scheduler_type": self.lr_scheduler_type,
                "optimizer": self.optimizer,
                "weight_decay": self.weight_decay,
                "max_length": self.max_length
            }
        )
        
    def load_model(self):
        bnb_config = None
        if self.enable_quant:
            print(f"Model loaded with quantization...")
            quant_config = self.cfg["Model"]["Quantization"]
            if quant_config.get("load_in_4bit", False):
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
            elif quant_config.get("load_in_8bit", False):
                bnb_config = BitsAndBytesConfig(load_in_8bit=True)           

        # Determine model type based on architecture
        model_config = AutoConfig.from_pretrained(
            self.model_id, 
            trust_remote_code=True
        )   
        arch_list = model_config.architectures if model_config.architectures else []
        is_visual_model = any("VL" in arch for arch in arch_list)
        
        if is_visual_model:
            print(f"Loading visual model: {self.model_id} with attention: {self.attention_type}")
            model_class = AutoModelForImageTextToText
        else:
            print(f"Loading text-only model: {self.model_id} with attention: {self.attention_type}")
            model_class = AutoModelForCausalLM
            
        # Load the model and processor
        self.model = model_class.from_pretrained(
            self.model_id,
            dtype=torch.bfloat16,
            device_map="auto",
            quantization_config=bnb_config,
            trust_remote_code=True,
            attn_implementation=self.attention_type
        )
        
        if self.enable_quant:
            self.model = prepare_model_for_kbit_training(self.model)
            
        # Text-only processor has issue with completion_only_loss=True, use VL processor for both visual and text-only models
        processor_model = self.model_id if is_visual_model else "Qwen/Qwen3-VL-8B-Instruct"
        self.processor = AutoProcessor.from_pretrained(
            processor_model,
            min_pixels=128*28*28,
            max_pixels=224*28*28, # limit image resolution
            trust_remote_code=True
        )

        if self.processor.tokenizer.pad_token is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
        self.processor.tokenizer.padding_side = "right"
    
    def _load_dataset(self):
        print("Loading dataset from:", self.train_data_path)
        raw_dataset = load_dataset("json", data_files=self.train_data_path, split="train")
        self.train_dataset = raw_dataset.map(
            preprocess_data_img,
            batched=True,
            remove_columns=raw_dataset.column_names,
            fn_kwargs={
                "driver_user_prompt": self.driver_user_prompt,
                "enable_action": self.enable_action,
                "enable_reason": self.enable_reason,
                "enable_command": self.enable_command
            },
        )
        print(f"Dataset expanded: {len(raw_dataset)} -> {len(self.train_dataset)} samples.")
    
    def get_lora_config(self):
        lora_cfg = self.cfg["Train"]["LoRA"]
        return LoraConfig(
            r=lora_cfg["lora_rank"], 
            lora_alpha=lora_cfg["lora_alpha"],
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj", 
                "gate_proj", "up_proj", "down_proj"
            ],
            lora_dropout=lora_cfg["lora_dropout"],
            bias="none",
            task_type="CAUSAL_LM" # or "IMAGE_TEXT_TO_TEXT" depending on model
        )
    
    def train(self, ckpt_path=None):
        if ckpt_path:
            self.load_model_from_checkpoint(ckpt_path)
        else:
            self.load_model()
        self.init_wandb()
        self._load_dataset()
        print(f"Hyperparameters:\n {self.hyper_info}")
        output_dir = os.path.join("checkpoints", f"{self.name}")
        
        # Trainer configuration
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=self.batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            learning_rate=self.learning_rate,
            num_train_epochs=self.epochs,
            lr_scheduler_type=self.lr_scheduler_type,
            optim=self.optimizer,
            weight_decay=self.weight_decay,
            report_to=self.log_to,
            logging_strategy="steps",
            logging_steps=10,
            save_strategy="epoch",
            save_total_limit=1,
            remove_unused_columns=False, 
        )
        
        peft_config = self.get_lora_config() if self.enable_quant else None # When using quantization, we enable LoRA by default to allow fine-tuning
        if peft_config is not None:
            self.model = prepare_model_for_kbit_training(self.model)
            self.model = get_peft_model(self.model, peft_config)
            self.model.print_trainable_parameters()
            
        collator = dataCollator(
            processor=self.processor,
            system_prompt=self.system_prompt,
            nuscenes_dataroot=self.nuscenes_dataroot,
            enable_image=self.enable_image,
            image_indices=self.image_indices
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            data_collator=collator, 
        )
        
        print("Starting training...")
        trainer.train()
        
        trainer.save_model(output_dir)
        print(f"Model saved to {output_dir}")

    def load_model_from_checkpoint(self, checkpoint_path=None):
        if checkpoint_path is None:
            checkpoint_path = os.path.join("checkpoints", f"{self.name}")

        print(f"Loading model from checkpoint: {checkpoint_path}")
        
        bnb_config = None
        if self.enable_quant:
            model_id = self.model_id # Load base model
            quant_config = self.cfg["Model"]["Quantization"]
            # Load quantization config
            if quant_config.get("load_in_4bit", False):
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
            elif quant_config.get("load_in_8bit", False):
                bnb_config = BitsAndBytesConfig(load_in_8bit=True)      
        else:
            model_id = checkpoint_path
        
        # Determine model type based on architecture
        model_config = AutoConfig.from_pretrained(
            model_id,
            trust_remote_code=True
        )
        arch_list = model_config.architectures if model_config.architectures else []
        is_visual_model = any("VL" in arch for arch in arch_list)
        
        if is_visual_model:
            print(f"Loading visual model: {model_id} with attention: {self.attention_type}")
            model_class = AutoModelForImageTextToText
        else:
            print(f"Loading text-only model: {model_id} with attention: {self.attention_type}")
            model_class = AutoModelForCausalLM
            
        # Load the model and processor
        self.model = model_class.from_pretrained(
            model_id,
            quantization_config=bnb_config if self.enable_quant else None,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation=self.attention_type
        )
        
        if self.enable_quant:
            print(f"Loading QLoRA adapter from: {checkpoint_path}")
            self.model = PeftModel.from_pretrained(self.model, checkpoint_path, is_trainable=False)
            
        print("Model loaded successfully from checkpoint.")
        processor_model = self.model_id if is_visual_model else "Qwen/Qwen3-VL-8B-Instruct"
        self.processor = AutoProcessor.from_pretrained(
            processor_model,
            min_pixels=128*28*28,
            max_pixels=512*28*28, # limit image resolution
            trust_remote_code=True
        )
                
    def inference(self, inference_path=None, temperature=0.7, top_p=0.8, num_trajectories=6):
        output_dir = os.path.join("results/inference", f"{self.name}.jsonl")
        os.makedirs(os.path.dirname(output_dir), exist_ok=True)
        
        overall_start = time.perf_counter()
        model_latencies = []
        with open(inference_path or self.mini_data_path, 'r', encoding='utf-8') as f_in, \
             open(output_dir, 'w', encoding='utf-8') as f_out:
            lines = f_in.readlines()
            num_samples = len(lines)
            for line in tqdm(lines, desc="Inference"):
                data = json.loads(line)
                wp_past = filter_to_xy_str(data['wp_past'])

                command_str = f"High-level Command: {data['command']}\n" if self.enable_command else ""
                
                ego_status_prompt = (
                    "Current Dynamics:\n"
                    f"- Velocity: {data['vel_val']:.2f} m/s\n"
                    f"- Yaw Rate: {data['yr_val']:.2f} rad/s\n"
                    f"- Acceleration (Longitudinal x, Lateral y): {data['acc_val']}\n"
                    f"Past Trajectory (2Hz): {wp_past}\n"
                    f"{command_str}\n"
                )
            
                reason = data['reasons'][0] if isinstance(data['reasons'], list) else data['reasons']
                full_driver_prompt = (
                    # f"Navigator's Analysis and Instructions:\n{reason}\n\n"
                    f"{reason}\n\n"
                    f"{ego_status_prompt}"
                    f"{self.driver_user_prompt}"
                )

                # Process the prompt, either with or without images
                if self.enable_image:
                    image_paths = data['image_paths']
                    if self.image_indices is not None:
                        selected_paths = [image_paths[i] for i in self.image_indices if i < len(image_paths)]
                    else:
                        selected_paths = image_paths
                        
                    prompt_messages = [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": (
                            [{"type": "image", "image": os.path.join(self.nuscenes_dataroot, p)} for p in selected_paths] +
                            [{"type": "text", "text": full_driver_prompt}]
                        )}
                    ]
                else:
                    prompt_messages = [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": full_driver_prompt}
                    ]
                    
                prompt_text = self.processor.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
                image_inputs, video_inputs = process_vision_info([prompt_messages])

                inputs = self.processor(
                    text=[prompt_text],
                    images=image_inputs if image_inputs else None,
                    videos=video_inputs if video_inputs else None,
                    padding=True,
                    return_tensors="pt",
                ).to(self.model.device)
                             
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                m_start = time.perf_counter()  
                
                with torch.no_grad():
                    output_ids = self.model.generate(**inputs, 
                                                     max_new_tokens=1024, 
                                                     do_sample=True, 
                                                     temperature=temperature, 
                                                     top_p=top_p,
                                                     num_return_sequences=num_trajectories)
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                model_latencies.append(time.perf_counter() - m_start)

                input_len = inputs["input_ids"].shape[1]
                
                all_pred_waypoints = []
                all_pred_actions = []
                
                for i in range(num_trajectories):
                    try:
                        raw_output = self.processor.decode(output_ids[i][input_len:], skip_special_tokens=True)

                        if not self.enable_action:
                            # only predict waypoints without action prediction
                            pred_pts = parse_string(raw_output)
                            all_pred_waypoints.append(pred_pts[:, :2].tolist())
                        else:
                            # predict actions, then compute waypoints from predicted actions
                            pred_actions = parse_string(raw_output) # numpy array
                            wp_past = parse_string(data['wp_past'])
                            theta0 = wp_past[-1, 2] # yaw angle
                            pred_pts = compute_trajectory_2(pred_actions, 0, 0, theta0, float(data['vel_val']), 0.5)
                            pred_pts = np.round(pred_pts[1:], 2) # remove the first point

                            all_pred_waypoints.append(pred_pts[:, :2].tolist())
                            all_pred_actions.append(pred_actions.tolist())
                    except Exception as e:
                        print(e)
                        all_pred_waypoints.append([])
                        if self.enable_action: all_pred_actions.append([])
                                        
                gt_pts = parse_string(data['wp_future'])[:, :2]
                
                # Save Record
                record = {
                    "token": data['token'],
                    "gt_waypoints": gt_pts.tolist(),
                    "pred_waypoints": all_pred_waypoints, # [6, N, 2]
                    "gt_actions": parse_string(data['action_future']).tolist() if self.enable_action else None,
                    "pred_actions": all_pred_actions if self.enable_action else None, # [6, N, 2]
                    "reasons": data['reasons'],
                }
                f_out.write(json.dumps(record) + "\n")
                # f_out.flush()
        
        overall_end = time.perf_counter()
        total_time = overall_end - overall_start
        avg_overall_latency = total_time / num_samples
        avg_model_latency = np.mean(model_latencies)

        output_dir = "results"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"results.txt") # save all results to 
        hyper_info = self.hyper_info
        
        with open(output_path, 'a', encoding='utf-8') as f_out:
            # f_out.write("="*100)
            f_out.write(hyper_info)
            f_out.write(f"Total Inference Time: {total_time:.2f} s\n")
            f_out.write(f"Average Latency per Sample: {avg_overall_latency:.2f} s\n")
            f_out.write(f"Average Model Inference Latency: {avg_model_latency:.2f} s\n")

        print(f"Results saved to: {output_path}")

    def inference_once(self, temperature=0.7, top_p=0.8, sample_index=0, is_reason=True):
        eval_path = "data/nuscenes_reasons_val_Qwen_32B.jsonl"
        def get_line(path, index):
            with open(path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i == index:
                        return json.loads(line)
            return None

        data = get_line(eval_path, sample_index)
        token = data['token']
        
        wp_past = filter_to_xy_str(data['wp_past'])
        command_str = f"High-level Command: {data['command']}\n" if self.enable_command else ""
              
        ego_status_prompt = (
            "Current Dynamics:\n"
            f"- Velocity: {data['vel_val']:.2f} m/s\n"
            f"- Yaw Rate: {data['yr_val']:.2f} rad/s\n"
            f"- Acceleration (Longitudinal x, Lateral y): {data['acc_val']}\n"
            f"Past Trajectory (2Hz): {wp_past}\n"
            f"{command_str}\n"
        )
        if is_reason:
            reason = data['reasons'][0] if isinstance(data['reasons'], list) else data['reasons']
        else:
            reason = ""

        full_driver_prompt = (
            # f"Navigator's Analysis and Instructions:\n{reason}\n\n"
            f"{reason}\n\n"
            f"{ego_status_prompt}"
            f"{self.driver_user_prompt}"
        )

        if self.enable_image:
            image_paths = [os.path.join(self.nuscenes_dataroot, p) for p in data['image_paths']]
            if self.image_indices:
                image_paths = [image_paths[i] for i in self.image_indices]
            
            prompt_messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": [{"type": "image", "image": p} for p in image_paths] + 
                                        [{"type": "text", "text": full_driver_prompt}]}
            ]
        else:
            prompt_messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": full_driver_prompt}
            ]

        prompt_text = self.processor.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info([prompt_messages])
        inputs = self.processor(text=[prompt_text], images=image_inputs, padding=True, return_tensors="pt").to(self.model.device)

        print("\n" + "="*30 + " PROMPT " + "="*30)
        print(full_driver_prompt)
        print("="*75 + "\n")

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs, 
                max_new_tokens=1024,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                num_return_sequences=6
            )
        
        outputs = []
        print("\n" + "="*30 + " 6 INDEPENDENT TRAJECTORIES " + "="*30)
        for i in range(len(output_ids)):
            decoded_output = self.processor.decode(
                output_ids[i][inputs["input_ids"].shape[1]:], 
                skip_special_tokens=True
            )
            outputs.append(decoded_output)
            print(f"\n[Trajectory {i+1}]:")
            print(decoded_output)
        
        gt_wp = data["wp_future"]
        command = data["command"] if data["command"] != "None" else None
        
        return outputs, gt_wp, command, token
    
    def get_nusc(self, version="v1.0-trainval"):
        print("Loading NuScenes...")
        self.nusc = NuScenes(version=version, dataroot=self.nuscenes_dataroot, verbose=False)
        return self.nusc
    
    def eval_L2(self, eval_path=None):
        all_results = []
        if not eval_path:
            try:
                eval_path = os.path.join("results", "inference", f"{self.name}.jsonl")
            except:
                pass
        
        if not os.path.exists(eval_path):
            print(f"Error: {eval_path} does not exist.")
            return
            
        with open(eval_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            first_data = json.loads(lines[0])
            is_multi = False
            pk = 'pred_waypoints' if 'pred_waypoints' in first_data else 'predicted_output'
            if isinstance(first_data.get(pk), list) and len(first_data[pk]) > 0 and isinstance(first_data[pk][0][0], list):
                is_multi = True
                print("Detected multi-modal predictions. Using minADE metric.")
                
            for line in lines:
                data = json.loads(line)
                gt = data['gt_waypoints']
                preds = data[pk]
                if is_multi:
                    candidate_metrics = []
                    for single_pred in preds:
                        m = calculate_metrics(gt, single_pred)
                        candidate_metrics.append(m)
                    best_res = min(candidate_metrics, key=lambda x: x['ade'])
                    all_results.append(best_res)
                else:
                    res = calculate_metrics(gt, preds)
                    all_results.append(res)    
                    
        if not all_results:
            print("No valid data processed.")
            return

        avg_metrics = {
            "L2_1s": np.nanmean([r['l2_1s'] for r in all_results]),
            "L2_2s": np.nanmean([r['l2_2s'] for r in all_results]),
            "L2_3s": np.nanmean([r['l2_3s'] for r in all_results]),
            "L2_6s": np.nanmean([r['l2_6s'] for r in all_results]),
            "ADE_3s": np.nanmean([r['ade_3s'] for r in all_results]),
            "ADE_avg": np.mean([r['ade'] for r in all_results]),
            "Failure_Rate": np.mean([r['is_failure'] for r in all_results]) * 100
        }

        output_dir = "results"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"results.txt") # save all results to 
        result_text = format_results(avg_metrics, eval_path, len(all_results), self.cfg["Eval"]["threshold"])
        
        with open(output_path, 'a', encoding='utf-8') as f_out:
            f_out.write(result_text)
            f_out.write("="*100 + "\n")

        print(result_text)
        print(f"Results saved to: {output_path}")
        
    def eval_video(self, eval_path=None, start_idx=0, end_idx=None):
        if not eval_path:
            eval_path = os.path.join("results", "inference", f"{self.name}.jsonl")
        input_file = eval_path
        os.makedirs("results/videos", exist_ok=True)
        nuscenes_version = "v1.0-trainval" if eval_path else "v1.0-mini"
        nusc = self.get_nusc(version=nuscenes_version)
        output_file = os.path.join("results/videos", f"{self.name}.mp4")
        
        with open(input_file, 'r') as f:
            lines = f.readlines() 
        
        selected_lines = lines[start_idx:end_idx]
        print(f"Processing frames {start_idx} to {start_idx + len(selected_lines)}...")
        
        _, _, width, height = render_frame(nusc, selected_lines[0])
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_file, fourcc, self.cfg["Eval"]["video_fps"], (width, height))
        
        for line in tqdm(selected_lines):
            data = json.loads(line)
            gt = data['gt_waypoints']
            pk = 'pred_waypoints' if 'pred_waypoints' in data else 'predicted_output'
            preds = data[pk]
            
            if isinstance(preds, list) and len(preds) > 0 and isinstance(preds[0][0], list):
                best_traj = min(preds, key=lambda p: calculate_metrics(gt, p)['ade'])
            else:
                best_traj = preds            

            vis_img, _, _, _ = render_frame(nusc, line, best_pred=best_traj)
            video_writer.write(vis_img)   
            
        video_writer.release()
        print(f"\nVideo saved successfully to {output_file}")
        
    def eval_images(self, eval_path=None, start_idx=0, end_idx=None):
        input_file = eval_path if eval_path else self.mini_data_path
        os.makedirs("results/images", exist_ok=True)
        nuscenes_version = "v1.0-trainval" if eval_path else "v1.0-mini"
        nusc = self.get_nusc(version=nuscenes_version)
        output_dir = os.path.join("results/images", f"{self.name}")
        os.makedirs(output_dir, exist_ok=True)
        
        with open(input_file, 'r') as f:
            lines = f.readlines()

        selected_lines = lines[start_idx:end_idx]
        
        print(f"Generating images to {output_dir}")
        print(f"Processing frames {start_idx} to {start_idx + len(selected_lines)}...")
        
        for i, line in enumerate(tqdm(selected_lines)):
            img, token, _, _ = render_frame(nusc, line)
            cv2.imwrite(os.path.join(output_dir, f"{start_idx+i:05d}_{token}.jpg"), img)
            
        print(f"\nAll images saved successfully to directory: {output_dir}")
        
    @property
    def model_info(self):
        total_params = self.model.num_parameters()
        trainable_params = self.model.num_parameters(only_trainable=True)
        trainable_ratio = (trainable_params / total_params) * 100
        
        mem_bytes = self.model.get_memory_footprint()
        mem_gb = mem_bytes / (1024 ** 3)
        mem_mb = mem_bytes / (1024 ** 2)
        
        arch_name = self.model.config.architectures[0] if self.model.config.architectures else "Unknown"
        dtype = self.model.dtype

        info = (
            f"\n{'='*80}\n"
            f"Model Summary: {self.name}\n"
            f"{'='*80}\n"
            f"• Architecture:   {arch_name}\n"
            f"• Dtype:          {dtype}\n"
            f"• Total Params:   {total_params:,}\n"
            f"• Trainable:      {trainable_params:,} ({trainable_ratio:.2f}%)\n"
            f"• Memory Size:    {mem_gb:.2f} GB ({mem_mb:.0f} MB)\n"
            f"{'='*80}\n"
            f"• Layers (First 3): {list(self.model.state_dict().keys())[:3]}\n"
            f"{'='*80}"
        )
        return info

    @property
    def hyper_info(self):
        info = (
            f"\n{'='*85}\n"
            f"Training Hyperparameters:\n"
            f"{'='*85}\n"
            f"• Config Name:            {self.name}\n"
            f"• Model:                  {self.model_id}\n"
            f"• Epochs:                 {self.epochs}\n"
            f"• Batch Size:             {self.batch_size}\n"
            f"• Gradient Accumulation:  {self.gradient_accumulation_steps}\n"
            f"• Learning Rate:          {self.learning_rate}\n"
            f"• LR Scheduler:           {self.lr_scheduler_type}\n"
            f"• Optimizer:              {self.optimizer}\n"
            f"• Weight Decay:           {self.weight_decay}\n"
            f"• Log To:                 {self.log_to}\n"
            f"• Max Length:             {self.max_length}\n"
            f"• LoRA:                   {self.enable_quant}\n"
            f"{'='*85}\n"
        )
        return info

class dataCollator():
    def __init__(self, processor, system_prompt, nuscenes_dataroot, enable_image, image_indices=None):
        self.processor = processor
        self.system_prompt = system_prompt
        self.nuscenes_dataroot = nuscenes_dataroot
        self.enable_image = enable_image
        self.image_indices = image_indices
        
    def __call__(self, batch):
        messages_batch = []
        
        for item in batch:
            text_prompt = item['prompt']
            completion = item['completion']
            image_paths = item['image_paths']

            user_content = []
            if self.enable_image:
                if self.image_indices is not None:
                    selected_paths = [image_paths[idx] for idx in self.image_indices if idx < len(image_paths)]
                else:
                    selected_paths = image_paths
                    
                for p in selected_paths:
                    full_path = os.path.join(self.nuscenes_dataroot, p)
                    user_content.append({"type": "image", "image": full_path})
                    
            user_content.append({"type": "text", "text": text_prompt})
                            
            prompt_messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_content}
            ]
            completion_message = [{"role": "assistant", "content": [{"type": "text", "text": f"{completion}."}]}]
            messages_batch.append(prompt_messages + completion_message)     

        texts = [self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=False) for m in messages_batch]
        image_inputs, video_inputs = process_vision_info(messages_batch) # If there is no image, image_inputs is None

        inputs = self.processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )        

        labels = inputs["input_ids"].clone() # All tokens
        
        for i in range(len(batch)):
            prompt_m = messages_batch[i][:-1] # [System, User]
            p_text = self.processor.apply_chat_template(prompt_m, tokenize=False, add_generation_prompt=True)
            p_image_inputs, p_video_inputs = process_vision_info([prompt_m])
            p_inputs = self.processor(
                text=[p_text], 
                images=p_image_inputs,
                videos=p_video_inputs,
                return_tensors="pt"
            )
            prompt_len = p_inputs["input_ids"].shape[1] # Length of prompt
            labels[i, :prompt_len] = -100 # Mask prompt

        labels[inputs["attention_mask"] == 0] = -100
        inputs["labels"] = labels   

        return inputs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a samll-size driver LLM")
    parser.add_argument("--config", type=str, default="default.yaml", help="Path to the configuration YAML file")
    args = parser.parse_args()
    config = load_config(args.config)
    trainer = driverEngine(config)
    trainer.load_model()
    print(trainer.model_info)
    print(trainer.hyper_info)
