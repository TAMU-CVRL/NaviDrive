#!/bin/bash

set -e

echo "=================================================="
echo ">>> [Step 1/2] Training ..."
echo "=================================================="

python3 train.py \
    --config configs/qwen_vl_2B_sft.yaml

echo ""
echo "=================================================="
echo ">>> [Step 2/2] Inference ..."
echo "=================================================="

python3 eval.py \
    --config configs/qwen_vl_2B_sft.yaml \
    --ckpt_path checkpoints/qwen_vl_2B_sft/ \
    --inference_path data/nuscenes_reasons_val_Qwen_32B.jsonl

echo ""
echo "=================================================="
echo ">>> All tasks finished."
echo "=================================================="

echo "=================================================="
echo ">>> [Step 1/2] Training ..."
echo "=================================================="

python3 train.py \
    --config configs/qwen_1_7B_sft.yaml

echo ""
echo "=================================================="
echo ">>> [Step 2/2] Inference ..."
echo "=================================================="

python3 eval.py \
    --config configs/qwen_1_7B_sft.yaml \
    --ckpt_path checkpoints/qwen_1_7B_sft/ \
    --inference_path data/nuscenes_reasons_val_Qwen_32B.jsonl

echo ""
echo "=================================================="
echo ">>> All tasks finished."
echo "=================================================="
