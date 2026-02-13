import argparse
import json
import os
import numpy as np
import cv2
import torch
import wandb

from tqdm import tqdm
from datetime import datetime
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import (
    AutoConfig, 
    AutoModelForCausalLM, 
    AutoModelForImageTextToText, 
    AutoProcessor, 
    BitsAndBytesConfig
)
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from nuscenes.nuscenes import NuScenes

from utils.data_utils import preprocess_data, load_config
from utils.caption_utils import reason_generate, parse_waypoints
from utils.results_utils import calculate_metrics, format_results, render_frame

class driverEngine():
    def __init__(self, cfg):
        self.cfg = cfg
        self.name = cfg["Name"]
        self.model_id = cfg["Model"]["model_id"]
        self.attention_type = cfg["Model"]["attention"]
        
        # Quantization & LoRA
        self.enable_quant = cfg["Model"]["Quantization"]["enable_quantization"]
        
        # Data Paths
        self.train_data_path = cfg["Dataset"]["train_data_path"]
        self.mini_data_path = cfg["Dataset"]["mini_data_path"]
        
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

    def init_wandb(self):
        wandb.init(
            project="dllm",
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
            max_pixels=512*28*28, # limit image resolution
            trust_remote_code=True
        )

        if self.processor.tokenizer.pad_token is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
        self.processor.tokenizer.padding_side = "right"
    
    def load_dataset(self):
        print("Loading dataset from:", self.train_data_path)
        raw_dataset = load_dataset("json", data_files=self.train_data_path, split="train")
        self.train_dataset = raw_dataset.map(
            preprocess_data,
            batched=True,
            remove_columns=raw_dataset.column_names
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
        self.load_dataset()
        print(f"Hyperparameters:\n {self.hyper_info}")
        date_str = datetime.now().strftime("%Y%m%d")
        output_dir = os.path.join("checkpoints", f"{self.name}_{date_str}")
        
        # SFTTrainer configuration
        sft_config = SFTConfig(
            output_dir=output_dir,
            per_device_train_batch_size=self.batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            learning_rate=self.learning_rate,
            num_train_epochs=self.epochs,
            lr_scheduler_type=self.lr_scheduler_type,
            optim=self.optimizer,
            weight_decay=self.weight_decay,
            report_to=self.log_to,
            max_length=self.max_length,
            completion_only_loss=True
        )
        
        peft_config = self.get_lora_config() if self.enable_quant else None # When using quantization, we enable LoRA by default to allow fine-tuning
        trainer = SFTTrainer(
            model=self.model,
            args=sft_config,
            train_dataset=self.train_dataset,
            processing_class=self.processor,
            peft_config=peft_config
        )
        
        print("Starting training...")
        trainer.train()
        
        trainer.save_model(output_dir)
        print(f"Model saved to {output_dir}")

    def load_model_from_checkpoint(self, checkpoint_path):
        if not os.path.exists(checkpoint_path):
            print(f"Error: Checkpoint path {checkpoint_path} does not exist.")
            return

        print(f"Loading model from checkpoint: {checkpoint_path}")
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
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation=self.attention_type
        )
        print("Model loaded successfully from checkpoint.")
        
    def inference(self, inference_path=None):
        self.load_model()
        output_dir = os.path.join("results", f"{self.name}_inference.jsonl")
        with open(inference_path or self.mini_data_path, 'r', encoding='utf-8') as f_in, \
             open(output_dir, 'w', encoding='utf-8') as f_out:
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
                    f"{self.driver_user_prompt}"
                )
                
                # Model Inference
                _, output = reason_generate(
                    user=full_driver_prompt,
                    system=self.system_prompt,
                    # images=pil_images,
                    processor=self.processor,
                    model=self.model,
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
    
    def get_nusc(self, version="v1.0-trainval"):
        print("Loading NuScenes...")
        self.nusc = NuScenes(version=version, dataroot=self.cfg["Dataset"]["nuscenes_dataroot"], verbose=False)
        return self.nusc
    
    def eval_L2(self, eval_path=None):
        all_results = []
        if not eval_path:
            eval_path = self.mini_data_path
        
        if not os.path.exists(eval_path):
            print(f"Error: {eval_path} does not exist.")
            return
            
        with open(eval_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                res = calculate_metrics(data['gt_waypoints'], data['pred_waypoints'])
                all_results.append(res)
        
        if not all_results:
            print("No valid data processed.")
            return

        avg_metrics = {
            "L2_1s": np.nanmean([r['l2_1s'] for r in all_results]),
            "L2_2s": np.nanmean([r['l2_2s'] for r in all_results]),
            "L2_3s": np.nanmean([r['l2_3s'] for r in all_results]),
            "L2_6s": np.nanmean([r['l2_6s'] for r in all_results]),
            "ADE_avg": np.mean([r['ade'] for r in all_results]),
            "Failure_Rate": np.mean([r['is_failure'] for r in all_results]) * 100
        }

        output_dir = "results"
        os.makedirs(output_dir, exist_ok=True)
        
        date_str = datetime.now().strftime("%Y%m%d")
        output_path = os.path.join(output_dir, f"{date_str}_results.txt")
        result_text = format_results(avg_metrics, eval_path, len(all_results), self.cfg["Eval"]["threshold"])
        
        with open(output_path, 'a', encoding='utf-8') as f_out:
            f_out.write(result_text)

        print(result_text)
        print(f"Results saved to: {output_path}")
        
    def eval_video(self, eval_path=None, start_idx=0, end_idx=None):
        input_file = eval_path if eval_path else self.mini_data_path
        nuscenes_version = "v1.0-trainval" if eval_path else "v1.0-mini"
        nusc = self.get_nusc(version=nuscenes_version)
        output_file = os.path.join("results", f"{self.name}.mp4")

        with open(input_file, 'r') as f:
            lines = f.readlines() 
        
        selected_lines = lines[start_idx:end_idx]
        print(f"Processing frames {start_idx} to {start_idx + len(selected_lines)}...")
        
        _, _, width, height = render_frame(nusc, selected_lines[0])
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_file, fourcc, self.cfg["Eval"]["video_fps"], (width, height))
        
        for line in tqdm(selected_lines):
            vis_img, _, _, _ = render_frame(nusc, line)
            video_writer.write(vis_img)   
            
        video_writer.release()
        print(f"\nVideo saved successfully to {output_file}")
        
    def eval_images(self, eval_path=None, start_idx=0, end_idx=None):
        input_file = eval_path if eval_path else self.mini_data_path
        nuscenes_version = "v1.0-trainval" if eval_path else "v1.0-mini"
        nusc = self.get_nusc(version=nuscenes_version)
        output_dir = os.path.join("results", f"{self.name}_images")
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
            f"\n{'='*80}\n"
            f"Training Hyperparameters:\n"
            f"{'='*80}\n"
            f"• Epochs:                 {self.epochs}\n"
            f"• Batch Size:             {self.batch_size}\n"
            f"• Gradient Accumulation:  {self.gradient_accumulation_steps}\n"
            f"• Learning Rate:          {self.learning_rate}\n"
            f"• LR Scheduler:           {self.lr_scheduler_type}\n"
            f"• Optimizer:              {self.optimizer}\n"
            f"• Weight Decay:           {self.weight_decay}\n"
            f"• Log To:                {self.log_to}\n"
            f"• Max Length:            {self.max_length}\n"
            f"{'='*80}\n"
        )
        return info

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a samll-size driver LLM")
    parser.add_argument("--config", type=str, default="default.yaml", help="Path to the configuration YAML file")
    parser.add_argument("--eval_path", type=str, help="Path to evaluation dataset (JSONL)")
    
    args = parser.parse_args()
    config = load_config(args.config)
    eval_path = args.eval_path
    trainer = driverEngine(config)
    trainer.train()