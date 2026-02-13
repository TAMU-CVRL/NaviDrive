import os
import torch

#conda activate /scratch/group/p.cis250376.000/conda_envs/navidrive

import wandb
from datasets import load_dataset
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
    TrainerCallback
)
from qwen_vl_utils import process_vision_info
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

class GPUUsageCallback(TrainerCallback):
    """Print GPU memory usage every 100 steps"""
    def on_step_end(self, args, state, control, **kwargs):
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"Step {state.global_step}: GPU memory - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")

LORA = True

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
# Use precomputed paths: run `python precompute_paths.py` first
DATA_FILE = "nusscenes_reasons_with_paths.jsonl"
OUTPUT_DIR = "./checkpoints/qwen25-7b-dllm-sft-0203"
SYSTEM_PROMPT = (
    "You are an expert autonomous driving planning module (Driver). Your goal is to output a safe, smooth, and kinematically feasible future trajectory.\n"
    "Rules:\n"
    "1. Coordinate System: Current ego position is (0,0). X-axis positive is forward, Y-axis positive is left.\n"
    "2. Trajectory Timing: Output exactly 12 waypoints (except origin (0,0)) representing the next 6 seconds (sampled at 2Hz, 0.5s intervals).\n"
    "3. Kinematic Constraints: Ensure the gaps between waypoints are consistent with the current velocity and acceleration. Avoid sudden jumps or unrealistic lateral shifts.\n"
    "4. Safety Alignment: The trajectory must strictly follow the Navigator's safety analysis.\n"
    "5. Output Format: Only output the coordinates: (x1, y1), (x2, y2), ..., (x12, y12)."
)

# Limit image resolution to reduce sequence length (NuScenes: 1600x900 -> 256x256)
processor = AutoProcessor.from_pretrained(
    MODEL_ID, 
    trust_remote_code=True,
    min_pixels=128*128,      # 16,384 pixels
    max_pixels=256*256,      # 65,536 pixels
)
if processor.tokenizer.pad_token is None:
    processor.tokenizer.pad_token = processor.tokenizer.eos_token
processor.tokenizer.padding_side = "right"

def preprocess_function(examples):
    """Store simple serializable fields - build messages in collate_fn"""
    all_image_paths = []
    all_text_prompts = []
    all_completions = []
    
    for i in range(len(examples['token'])):
        # Use precomputed image paths from JSONL
        image_paths = examples['image_paths'][i]
        
        ego_status_prompt = (
            f"Current Dynamics:\n"
            f"- Velocity: {examples['vel_val'][i]:.2f} m/s\n"
            f"- Yaw Rate: {examples['yr_val'][i]:.2f} rad/s\n"
            f"- Acceleration (Longitudinal x, Lateral y): {examples['acc_val'][i]}\n"
            f"- Past Trajectory (2Hz): {examples['wp_past'][i]}\n"
            # f"- High-level Command: {examples['command'][i]}\n\n" # Current commands are not correct
        )
        
        driver_user_prompt = (
            "Inputs: 6 images (Full Surround View) and Ego-Vehicle Status.\n"
            "1:FRONT_LEFT, 2:FRONT, 3:FRONT_RIGHT, 4:BACK_RIGHT, 5:BACK, 6:BACK_LEFT.\n"
            f"{ego_status_prompt}"
            "Predict the next 12 waypoints."
        )
        
        future_wp = examples['wp_future'][i]

        all_image_paths.append(image_paths)
        all_text_prompts.append(driver_user_prompt)
        all_completions.append(f"Future Waypoints: {future_wp}.")
            
    return {
        "image_paths": all_image_paths,
        "text_prompt": all_text_prompts,
        "completion": all_completions,
    }

def collate_fn(batch):
    """Build full message structure here, not in preprocess"""
    messages_batch = []
    prompt_only_batch = []
    
    for item in batch:
        image_paths = item["image_paths"]
        text_prompt = item["text_prompt"]
        completion = item["completion"]
        
        # Build prompt messages
        prompt_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": (
                [{"type": "image", "image": p} for p in image_paths] +
                [{"type": "text", "text": text_prompt}]
            )}
        ]
        
        # Build completion message
        completion_message = [{"role": "assistant", "content": completion}]
        
        # Full messages = prompt + completion
        messages_batch.append(prompt_messages + completion_message)
        prompt_only_batch.append(prompt_messages)
    
    texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=False) for m in messages_batch]
    prompt_texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in prompt_only_batch]
    image_inputs, video_inputs = process_vision_info(messages_batch)
    
    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    
    prompt_inputs = processor(
        text=prompt_texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    
    input_ids = inputs["input_ids"]
    labels = input_ids.clone()
    
    for i in range(len(batch)):
        prompt_len = prompt_inputs["attention_mask"][i].sum().item()
        labels[i, :prompt_len] = -100
    
    labels[labels == processor.tokenizer.pad_token_id] = -100
    inputs["labels"] = labels
    
    return inputs

def train():
    print("Loading and mapping dataset...")
    raw_dataset = load_dataset("json", data_files=DATA_FILE, split="train")
    train_dataset = raw_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=raw_dataset.column_names
    )
    print(f"Dataset expanded: {len(raw_dataset)} -> {len(train_dataset)} samples.")
    if not LORA:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_ID,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
    else:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_ID,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        
        model = prepare_model_for_kbit_training(model)
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            # target_modules is model-specific; common defaults:
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(model, lora_config)

    # SFTTrainer
    sft_config = SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=16,
        learning_rate=2e-5,
        num_train_epochs=3,
        lr_scheduler_type="cosine",
        bf16=True,
        gradient_checkpointing=True,
        optim="adamw_torch",
        report_to="wandb",
        remove_unused_columns=False,
        dataloader_num_workers=16,
        dataset_kwargs={"skip_prepare_dataset": True},
    )
    
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        data_collator=collate_fn,
        callbacks=[GPUUsageCallback()],
    )

    # Training
    print("Starting training...")
    trainer.train()
    
    # Save Model
    trainer.save_model(OUTPUT_DIR)
    print(f"Model saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    wandb.login(key="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")
    wandb.init(
        project="dllm",
        name="qwen25-7b-vl-sft-0203",
        config={
            "model": "Qwen2.5-VL-7B",
            "learning_rate": 2e-5,
            "epochs": 3,
        }
    )    
    train()
