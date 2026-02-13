# Environment
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

pip install transformers==5.1.0 accelerate==1.12.0 peft==0.18.1

pip install -U bitsandbytes>=0.46.1

pip install opencv-python==4.11.0.86

pip install open3d

pip install nuscenes-devkit

wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

pip install flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```
# Train
```
python3 train.py --config configs/qwen_vl_2B_sft.yaml 
```
# Inference
```
python3 eval.py --config configs/qwen_vl_2B_sft.yaml --ckpt_path checkpoints/qwen_vl_2B_sft --inference_path data/nusscenes_reasons_mini_0203.jsonl
```
```
python3 eval.py --config configs/qwen_vl_2B_sft.yaml --ckpt_path checkpoints/qwen_vl_2B_sft --inference_path data/nusscenes_reasons_val.jsonl
```
# Evaluation
```
python3 eval.py --config configs/qwen_vl_2B_sft.yaml --ckpt_path checkpoints/qwen_vl_2B_sft --eval_path results/inference/qwen_vl_2B_sft_inference.jsonl --eval_L2 True
```
```
python3 eval.py --config configs/qwen_vl_2B_sft.yaml --ckpt_path checkpoints/qwen_vl_2B_sft --eval_path results/inference/qwen_vl_2B_sft_inference.jsonl --eval_video True
```
```
python3 eval.py --config configs/qwen_vl_2B_sft.yaml --ckpt_path checkpoints/qwen_vl_2B_sft --eval_path results/inference/qwen_vl_2B_sft_inference.jsonl --eval_images True
```