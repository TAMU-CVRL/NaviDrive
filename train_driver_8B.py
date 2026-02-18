import os
import argparse

# Set cache directories to scratch space (before importing transformers/huggingface)
SCRATCH_DIR = "/scratch/group/p.cis250376.000"
os.environ["HF_HOME"] = f"{SCRATCH_DIR}/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = f"{SCRATCH_DIR}/hf_cache/transformers"
os.environ["HF_DATASETS_CACHE"] = f"{SCRATCH_DIR}/hf_cache/datasets"
os.environ["TORCH_HOME"] = f"{SCRATCH_DIR}/torch_cache"

import torch

# Check GPU availability early
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
else:
    print("WARNING: No GPU detected!")

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
    """Print GPU memory usage every step"""
    def on_step_end(self, args, state, control, **kwargs):
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"Step {state.global_step}: GPU memory - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")

class GenerationCallback(TrainerCallback):
    """Generate and print model predictions every 100 steps"""
    def __init__(self, processor, samples_list):
        self.processor = processor
        self.samples_list = samples_list  # Store multiple samples for generation
    
    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step % 10 == 0 and model is not None:
            model.eval()
            with torch.no_grad():
                print(f"\n{'='*60}")
                print(f"Step {state.global_step} - Testing on {len(self.samples_list)} samples")
                print('='*60)
                
                for idx, sample in enumerate(self.samples_list):
                    image_paths = sample["image_paths"]
                    text_prompt = sample["text_prompt"]
                    
                    # Build prompt messages
                    prompt_messages = [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": (
                            [{"type": "image", "image": p} for p in image_paths] +
                            [{"type": "text", "text": text_prompt}]
                        )}
                    ]
                    
                    prompt_text = self.processor.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
                    image_inputs, video_inputs = process_vision_info([prompt_messages])
                    
                    inputs = self.processor(
                        text=[prompt_text],
                        images=image_inputs,
                        videos=video_inputs,
                        return_tensors="pt",
                    ).to(model.device)
                    
                    output_ids = model.generate(**inputs, max_new_tokens=256)
                    generated_text = self.processor.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                    
                    print(f"\n[Sample {idx}]")
                    print(f"  Predicted:    {generated_text}")
                    print(f"  Ground truth: {sample['completion']}")
                
                print('='*60 + '\n')
            model.train()

LORA = True
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

# Mode-specific configuration
MODE_CONFIG = {
    "waypoints": {
        "data_file": "nusscenes_reasons_with_paths.jsonl",
        "output_dir": "./checkpoints/qwen25-7b-dllm-sft-vislora",
        "system_prompt": (
            "You are an expert autonomous driving planning module (Driver). Your goal is to output a safe, smooth, and kinematically feasible future trajectory.\n"
            "Rules:\n"
            "1. Coordinate System: Current ego position is (0,0). X-axis positive is forward, Y-axis positive is left.\n"
            "2. Trajectory Timing: Output exactly 12 waypoints (except origin (0,0)) representing the next 6 seconds (sampled at 2Hz, 0.5s intervals).\n"
            "3. Kinematic Constraints: Ensure the gaps between waypoints are consistent with the current velocity and acceleration. Avoid sudden jumps or unrealistic lateral shifts.\n"
            "4. Safety Alignment: The trajectory must strictly follow the Navigator's safety analysis.\n"
            "5. Output Format: Only output the coordinates: (x1, y1), (x2, y2), ..., (x12, y12)."
        ),
        "user_instruction": "Predict the next 12 waypoints.",
        "target_key": "wp_future",
        "completion_prefix": "Future Waypoints",
    },
    "actions": {
        "data_file": "data/nuscenes_reasons_Qwen_32B.jsonl",
        "output_dir": "./checkpoints/qwen25-7b-dllm-sft-actions",
        "system_prompt": (
            "You are an expert autonomous driving planning module (Driver). Your goal is to output safe, smooth, and kinematically feasible future control actions.\n"
            "Rules:\n"
            "1. Coordinate System: Current ego position is (0,0). X-axis positive is forward, Y-axis positive is left.\n"
            "2. Action Timing: Output exactly 11 control actions representing the next 5.5 seconds (sampled at 2Hz, 0.5s intervals).\n"
            "3. Kinematic Constraints: Ensure acceleration and curvature values are smooth and consistent with current dynamics.\n"
            "4. Safety Alignment: The actions must strictly follow the Navigator's safety analysis.\n"
            "5. Output Format: Only output the controls: (a1, k1), (a2, k2), ..., (a11, k11)."
        ),
        "user_instruction": "Predict the next 11 control actions (acceleration, curvature).",
        "target_key": "action_future",
        "completion_prefix": "Future Actions",
    },
}

# Limit image resolution to reduce sequence length (NuScenes: 1600x900 -> 256x256)
processor = AutoProcessor.from_pretrained(
    MODEL_ID, 
    trust_remote_code=True,
    min_pixels=384*216,      # ~83k pixels (16:9 aspect ratio)
    max_pixels=512*288,      # ~147k pixels (16:9 aspect ratio)
)
if processor.tokenizer.pad_token is None:
    processor.tokenizer.pad_token = processor.tokenizer.eos_token
processor.tokenizer.padding_side = "right"

def preprocess_function(examples, mode_cfg=None):
    """Store simple serializable fields - build messages in collate_fn"""
    all_image_paths = []
    all_text_prompts = []
    all_completions = []
    
    for i in range(len(examples['token'])):
        image_paths = examples['image_paths'][i]
        
        ego_status_prompt = (
            f"Current Dynamics:\n"
            f"- Velocity: {examples['vel_val'][i]:.2f} m/s\n"
            f"- Yaw Rate: {examples['yr_val'][i]:.2f} rad/s\n"
            f"- Acceleration (Longitudinal x, Lateral y): {examples['acc_val'][i]}\n"
            f"- Past Trajectory (2Hz): {examples['wp_past'][i]}\n"
        )
        
        driver_user_prompt = (
            "Inputs: 6 images (Full Surround View) and Ego-Vehicle Status.\n"
            "1:FRONT_LEFT, 2:FRONT, 3:FRONT_RIGHT, 4:BACK_RIGHT, 5:BACK, 6:BACK_LEFT.\n"
            f"{ego_status_prompt}"
            f"{mode_cfg['user_instruction']}"
        )
        
        target = examples[mode_cfg['target_key']][i]

        all_image_paths.append(image_paths)
        all_text_prompts.append(driver_user_prompt)
        all_completions.append(f"{mode_cfg['completion_prefix']}: {target}.")
            
    return {
        "image_paths": all_image_paths,
        "text_prompt": all_text_prompts,
        "completion": all_completions,
    }

def collate_fn(batch):
    """Build full message structure here, not in preprocess"""
    messages_batch = []
    completions = []
    
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
        completions.append(completion)
    
    texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=False) for m in messages_batch]
    image_inputs, video_inputs = process_vision_info(messages_batch)
    
    # Process images only once
    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    
    input_ids = inputs["input_ids"]
    labels = input_ids.clone()
    
    # First, mask all padding tokens
    labels[labels == processor.tokenizer.pad_token_id] = -100
    
    # Calculate prompt_len by finding where completion starts in the decoded text
    for i in range(len(batch)):
        # Decode the full sequence
        full_text = processor.tokenizer.decode(input_ids[i], skip_special_tokens=False)
        
        # Find where completion starts in the full text
        completion_start_idx = full_text.find(completions[i])
        
        if completion_start_idx != -1:
            # Tokenize just the prefix (everything before the completion)
            prefix_text = full_text[:completion_start_idx]
            prefix_ids = processor.tokenizer(prefix_text, add_special_tokens=False)["input_ids"]
            prompt_len = len(prefix_ids)
        else:
            # Fallback: use the old method
            completion_ids = processor.tokenizer(completions[i], add_special_tokens=False)["input_ids"]
            seq_len = inputs["attention_mask"][i].sum().item()
            prompt_len = seq_len - len(completion_ids)
        
        labels[i, :prompt_len] = -100
    
    inputs["labels"] = labels
    
    return inputs

def train(mode_cfg):
    print(f"Training mode: {mode_cfg['completion_prefix']}")
    print(f"Data file: {mode_cfg['data_file']}")
    print("Loading and mapping dataset...")
    raw_dataset = load_dataset("json", data_files=mode_cfg["data_file"], split="train")
    train_dataset = raw_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=raw_dataset.column_names,
        fn_kwargs={"mode_cfg": mode_cfg},
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
        
        # Print all Linear/Conv module names to verify LoRA targets
        print("\n" + "="*60)
        print("ALL TRAINABLE-ELIGIBLE MODULES (Linear/Conv layers):")
        print("="*60)
        for name, module in model.named_modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)):
                print(f"  {name}: {module.__class__.__name__}({module.in_features if hasattr(module, 'in_features') else '?'}, {module.out_features if hasattr(module, 'out_features') else '?'})")
        print("="*60 + "\n")
        
        lora_config = LoraConfig(
            r=64,
            lora_alpha=128,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            # LLM layers + Vision encoder layers
            # LLM: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
            # Vision: visual.blocks.*.attn.qkv, visual.blocks.*.attn.proj, 
            #         visual.blocks.*.mlp.gate_proj/up_proj/down_proj
            target_modules=[
                # LLM attention & MLP
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
                # Vision encoder attention (fused qkv + output proj)
                "attn.qkv", "attn.proj",
            ],
        )
        model = get_peft_model(model, lora_config)

    # SFTTrainer
    sft_config = SFTConfig(
        output_dir=mode_cfg["output_dir"],
        per_device_train_batch_size=4,
        gradient_accumulation_steps=16,
        learning_rate=1e-5,
        num_train_epochs=3,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        gradient_checkpointing=True,
        optim="adamw_torch",
        report_to="wandb",
        remove_unused_columns=False,
        dataloader_num_workers=16,
        dataset_kwargs={"skip_prepare_dataset": True},
        save_strategy="epoch",
    )
    
    # Get multiple samples for generation callback (different movement patterns)
    # Pick samples at different indices to get variety
    samples_for_gen = [
        train_dataset[0],      # First sample
        train_dataset[100],    # Different scene
        train_dataset[500],    # Another scene
        train_dataset[1000],   # Another scene
    ]
    
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        data_collator=collate_fn,
        callbacks=[GPUUsageCallback(), GenerationCallback(processor, samples_for_gen)],
    )

    # Training
    print("Starting training...")
    trainer.train()
    
    # Save Model
    trainer.save_model(mode_cfg["output_dir"])
    print(f"Model saved to {mode_cfg['output_dir']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--waypoints", action="store_const", const="waypoints", dest="mode")
    group.add_argument("--actions", action="store_const", const="actions", dest="mode")
    args = parser.parse_args()
    
    mode_cfg = MODE_CONFIG[args.mode]
    SYSTEM_PROMPT = mode_cfg["system_prompt"]
    
    wandb.login(key="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")
    wandb.init(
        project="dllm",
        name=f"qwen25-7b-vl-sft-{args.mode}",
        config={
            "model": "Qwen2.5-VL-7B",
            "mode": args.mode,
            "learning_rate": 1e-5,
            "lora_r": 64,
            "lora_alpha": 128,
            "vision_lora": True,
            "warmup_ratio": 0.05,
            "epochs": 3,
        }
    )    
    train(mode_cfg)
