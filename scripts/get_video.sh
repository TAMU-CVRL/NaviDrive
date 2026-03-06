#!/bin/bash

set -e

CONFIG_NAME="qwen_vl_2B_sft_cmd_img" # Specify the configuration name (without .yaml extension)
CONFIG_FILE="configs/${CONFIG_NAME}.yaml"

echo "=================================================="
echo ">>> Generating video ..."
echo "=================================================="

python3 eval.py \
    --config "${CONFIG_FILE}" \
    --eval_video True \
    --start_idx 0 \
    --end_idx 2000

echo ""
echo "=================================================="
echo ">>> All tasks finished for ${CONFIG_NAME}."
echo "=================================================="
