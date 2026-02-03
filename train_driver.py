import os
import torch
import json

import wandb
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoProcessor,
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
import copy

MODEL_ID = "Qwen/Qwen3-1.7B"
DATA_FILE = "nusscenes_reasons.jsonl"
OUTPUT_DIR = "./checkpoints/qwen3-1.7b-dllm-sft-0201"
SYSTEM_PROMPT = (
    "You are an expert autonomous driving planning module (Driver). Your goal is to output a safe, smooth, and kinematically feasible future trajectory.\n"
    "Rules:\n"
    "1. Coordinate System: Current ego position is (0,0). X-axis positive is forward, Y-axis positive is left.\n"
    "2. Trajectory Timing: Output exactly 12 waypoints (except origin (0,0)) representing the next 6 seconds (sampled at 2Hz, 0.5s intervals).\n"
    "3. Kinematic Constraints: Ensure the gaps between waypoints are consistent with the current velocity and acceleration. Avoid sudden jumps or unrealistic lateral shifts.\n"
    "4. Safety Alignment: The trajectory must strictly follow the Navigator's safety analysis.\n"
    "5. Output Format: Only output the coordinates: (x1, y1), (x2, y2), ..., (x12, y12)."
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
# eos_token = tokenizer.eos_token
processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct", trust_remote_code=True)
if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
processor.tokenizer.padding_side = "right"

MAX_LENGTH = 1024
def preprocess_function(examples):
    all_prompts = []
    all_completions = []
    
    for i in range(len(examples['token'])):
        ego_status_prompt = (
            f"Current Dynamics:\n"
            f"- Velocity: {examples['vel_val'][i]:.2f} m/s\n"
            f"- Yaw Rate: {examples['yr_val'][i]:.2f} rad/s\n"
            f"- Acceleration (Longitudinal x, Lateral y): {examples['acc_val'][i]}\n"
            f"- Past Trajectory (2Hz): {examples['wp_past'][i]}\n"
            f"- High-level Command: {examples['command'][i]}\n\n"
        )
        
        driver_user_prompt = (
            "Predict the next 12 waypoints. "
        )
        
        reasons_list = examples['reasons'][i]
        future_wp = examples['wp_future'][i]
        
        for reason_text in reasons_list:
            full_driver_prompt = (
                f"Navigator's Analysis and Instructions:\n{reason_text}\n\n"
                f"{ego_status_prompt}\n"
                f"{driver_user_prompt}"
            )
            all_prompts.append([
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": full_driver_prompt}
            ])
            all_completions.append([
                {"role": "assistant", "content": f"Future Waypoints: {future_wp}."}
            ])
            
    return {
        "prompt": all_prompts,
        "completion": all_completions
    }
    # return {"text": all_texts}

def train():
    print("Loading and mapping dataset...")
    raw_dataset = load_dataset("json", data_files=DATA_FILE, split="train")
    train_dataset = raw_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=raw_dataset.column_names
    )
    print(f"Dataset expanded: {len(raw_dataset)} -> {len(train_dataset)} samples.")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

    # SFTTrainer
    sft_config = SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=2e-5,
        num_train_epochs=3,
        lr_scheduler_type="cosine",
        bf16=True,
        gradient_checkpointing=True,
        optim="adamw_torch",
        report_to="wandb",
        max_length=1024,
        # dataset_text_field="messages",
        completion_only_loss=True
    )
    
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        # processing_class=tokenizer,
        processing_class=processor,
    )

    # Training
    print("Starting training...")
    trainer.train()
    
    # Save Model
    trainer.save_model(OUTPUT_DIR)
    print(f"Model saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    wandb.init(
        project="dllm",
        name="qwen-1.7b-full-sft-0202",
        config={
            "model": "Qwen3-1.7B",
            "learning_rate": 2e-5,
            "epochs": 1,
        }
    )    
    train()
