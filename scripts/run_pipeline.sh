#!/bin/bash

set -e

echo "=================================================="
echo ">>> [Step 1/2] Running naviGen.py ..."
echo "=================================================="

python3 naviGen.py

echo ""
echo "=================================================="
echo ">>> [Step 2/2] naviGen finished. Running eval.py ..."
echo "=================================================="

python3 eval.py \
    --config configs/qwen_1_7B_sft.yaml \
    --ckpt_path checkpoints/qwen3-1.7b-dllm-sft-0203 \
    --inference_path data/nuscenes_reasons_val_0207.jsonl

echo ""
echo "=================================================="
echo ">>> All tasks finished."
echo "=================================================="