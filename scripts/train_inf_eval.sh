#!/bin/bash

set -e

CONFIG_NAME="qwen_vl_2B_sft_cmd_no_res" # Specify the configuration name (without .yaml extension)
CONFIG_FILE="configs/${CONFIG_NAME}.yaml"

echo "=================================================="
echo ">>> [Step 1/3] Training with ${CONFIG_NAME} ..."
echo "=================================================="

python3 train.py \
    --config "${CONFIG_FILE}"

echo ""
echo "=================================================="
echo ">>> [Step 2/3] Inference ..."
echo "=================================================="

python3 eval.py \
    --config "${CONFIG_FILE}" \
    --inference_path data/nuscenes_reasons_val_Qwen_32B.jsonl

echo ""
echo "=================================================="
echo ">>> [Step 3/3] Evaluation ..."
echo "=================================================="

python3 eval.py \
    --config "${CONFIG_FILE}" \
    --eval_L2 True

echo ""
echo "=================================================="
echo ">>> All tasks finished for ${CONFIG_NAME}."
echo "=================================================="
