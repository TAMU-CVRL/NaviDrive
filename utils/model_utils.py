import os
import yaml
import torch
import torch.nn as nn
from models.clip.model import CLIP
from models.pointnet2.pointnet2_encoder import PointNet2Encoder
from models.PTv3.ptv3_encoder import PTv3Encoder
from tqdm import tqdm

from transformers import (
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_constant_schedule_with_warmup
)

class CLIPTextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.token_embedding = clip_model.token_embedding
        self.positional_embedding = clip_model.positional_embedding
        self.transformer = clip_model.transformer
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, text_ids):
        x = self.token_embedding(text_ids).type(self.dtype)  # [B,77,D]
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)                              # [77,B,D]
        x = self.transformer(x)
        x = x.permute(1, 0, 2)                              # [B,77,D]
        x = self.ln_final(x)
        x = x[torch.arange(x.shape[0]), text_ids.argmax(dim=-1)] @ self.text_projection
        return x
    
class CLIPImageEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.visual = clip_model.visual
        self.dtype = clip_model.dtype

    def forward(self, images):
        return self.visual(images.type(self.dtype))
        
def build_clip_config(name: str):
    name = name.upper()

    if name == "VIT-B/32":
        return dict(embed_dim=512, image_resolution=224,
                    vision_layers=12, vision_width=768, vision_patch_size=32,
                    context_length=77, vocab_size=49408,
                    transformer_width=512, transformer_heads=8, transformer_layers=12)

    elif name == "VIT-B/16":
        return dict(embed_dim=512, image_resolution=224,
                    vision_layers=12, vision_width=768, vision_patch_size=16,
                    context_length=77, vocab_size=49408,
                    transformer_width=512, transformer_heads=8, transformer_layers=12)

    elif name == "VIT-L/14":
        return dict(embed_dim=768, image_resolution=224,
                    vision_layers=24, vision_width=1024, vision_patch_size=14,
                    context_length=77, vocab_size=49408,
                    transformer_width=768, transformer_heads=12, transformer_layers=12)

    elif name == "VIT-H/14":
        return dict(embed_dim=1024, image_resolution=224,
                    vision_layers=32, vision_width=1280, vision_patch_size=14,
                    context_length=77, vocab_size=49408,
                    transformer_width=1024, transformer_heads=16, transformer_layers=12)

    else:
        raise ValueError(f"Unknown CLIP model name: {name}, please choose from VIT-B/32, VIT-B/16, VIT-L/14, VIT-H/14")

def build_clip_model(name: str):
    config = build_clip_config(name)
    model = CLIP(
        embed_dim=config['embed_dim'],
        image_resolution=config['image_resolution'],
        vision_layers=config['vision_layers'],
        vision_width=config['vision_width'],
        vision_patch_size=config['vision_patch_size'],
        context_length=config['context_length'],
        vocab_size=config['vocab_size'],
        transformer_width=config['transformer_width'],
        transformer_heads=config['transformer_heads'],
        transformer_layers=config['transformer_layers']
    )
    return model
      
def get_clip_encoders(name: str):
    model = build_clip_model(name)
    text_encoder = CLIPTextEncoder(model)
    image_encoder = CLIPImageEncoder(model)
    return text_encoder, image_encoder

def prepare_training():
    if not os.path.exists("checkpoints"):
        os.makedirs("checkpoints", exist_ok=True) # save model checkpoints

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def build_scheduler(optimizer, scheduler_cfg, warmup_steps, total_training_steps):
    sched_type = scheduler_cfg.get("type", "cosine")  # default: cosine

    if sched_type == "cosine":
        return get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_training_steps,
            num_cycles=scheduler_cfg.get("num_cycles", 0.5)
        )
    elif sched_type == "linear":
        return get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_training_steps
        )
    elif sched_type == "constant":
        return get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps
        )
    elif sched_type == "steplr":  # PyTorch’s built-in step decay
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_cfg.get("step_size", 30),
            gamma=scheduler_cfg.get("gamma", 0.1)
        )
    else:
        raise ValueError(f"Unknown scheduler type: {sched_type}")

def evaluate(model, eval_dataloader, tokenize, device):
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        with tqdm(eval_dataloader, desc="Evaluating") as pbar:
            for sample in pbar:
                labels = sample[0]   # string label
                imgs = sample[1].to(device) # [B, 3, H, W]
                points = sample[2].permute(0, 2, 1).to(device)   # tensor of different lengths. [B, N, 3] -> [B, 3, N]
                text_ids = tokenize(labels, truncate=True).to(device)

                text_features, image_features, lidar_features = model(text_ids, imgs, points)
                loss, _ = model.get_loss(text_features, image_features, lidar_features)

                total_loss += loss.item()
                num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    return avg_loss

def _gather_one(feat, with_grad, world_size):
    if world_size < 2:
        return feat
    if with_grad:
        return torch.cat(torch.distributed.nn.all_gather(feat.contiguous()), dim=0)
    else:
        gathered = [torch.zeros_like(feat) for _ in range(world_size)]
        torch.distributed.all_gather(gathered, feat)
        return torch.cat(gathered, dim=0)
    
def gather_features(text_features, image_features, lidar_features, local_loss=False, gather_with_grad=False, rank=0, world_size=1):
    if world_size < 2 or local_loss:
        return text_features, image_features, lidar_features

    all_text_features  = _gather_one(text_features,  gather_with_grad, world_size)
    all_image_features = _gather_one(image_features, gather_with_grad, world_size)
    all_lidar_features = _gather_one(lidar_features, gather_with_grad, world_size)

    return all_text_features, all_image_features, all_lidar_features

def pc_backbone(pc_encoder, device):
    if pc_encoder == "pointnet2":
        # [B, C, N] -> [B, 1024]
        pc_encoder = PointNet2Encoder().to(device)
    elif pc_encoder == "ptv3":
        # [B, C, N] -> [B, 1024]
        pc_encoder = PTv3Encoder().to(device)
    else:
        raise ValueError(f"Unknown lidar encoder: {pc_encoder}")
    return pc_encoder
