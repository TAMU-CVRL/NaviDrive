#!/bin/bash

set -e

CONFIG_NAME="qwen_vl_2B_sft_action_cmd" # Specify the configuration name (without .yaml extension)
CONFIG_FILE="configs/${CONFIG_NAME}.yaml"
INFERENCE_PATH="data/nuscenes_reasons_val_Qwen_32B.jsonl"
# data/nuscenes_reasons_val_Qwen_32B.jsonl

echo "=================================================="
echo ">>> [Step 1/2] Inference with ${CONFIG_NAME} ..."
echo "=================================================="

python3 eval.py \
    --config "${CONFIG_FILE}" \
    --inference_path "${INFERENCE_PATH}"

echo ""
echo "=================================================="
echo ">>> [Step 2/2] Evaluation ..."
echo "=================================================="

python3 eval.py \
    --config "${CONFIG_FILE}" \
    --eval_L2 True

echo ""
echo "=================================================="
echo ">>> All tasks finished for ${CONFIG_NAME}."
echo "=================================================="
