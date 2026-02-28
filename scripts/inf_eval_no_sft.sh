#!/bin/bash

set -e

# --- Configuration Section ---
# List of models to process. Format: "MODEL_ID:CONFIG_NAME"
MODELS=(
    "Qwen/Qwen3-VL-2B-Instruct:qwen_vl_2B_cmd_img_6_no_res"
    "Qwen/Qwen3-VL-4B-Instruct:qwen_vl_4B_cmd_img_6_no_res"
    "Qwen/Qwen3-VL-8B-Instruct:qwen_vl_8B_cmd_img_6_no_res"
)

# Shared inference data path
INFERENCE_PATH="data/nuscenes_reasons_val_Qwen_32B.jsonl"

# --- Main Execution Loop ---
for item in "${MODELS[@]}"; do
    # Extract Model ID and Config Name from the array item
    MODEL_ID="${item%%:*}"
    CONFIG_NAME="${item#*:}"
    CONFIG_FILE="configs/${CONFIG_NAME}.yaml"

    echo "=================================================="
    echo ">>> Starting Pipeline for: ${CONFIG_NAME}"
    echo ">>> Model ID: ${MODEL_ID}"
    echo "=================================================="

    # Step 1: Model Inference
    echo ">>> [Step 1/2] Running Inference..."
    python3 eval.py \
        --config "${CONFIG_FILE}" \
        --ckpt_path "${MODEL_ID}" \
        --inference_path "${INFERENCE_PATH}"

    echo ""

    # Step 2: Metrics Evaluation
    echo ">>> [Step 2/2] Running Evaluation (L2)..."
    python3 eval.py \
        --config "${CONFIG_FILE}" \
        --eval_L2 True

    echo ">>> Task Finished: ${CONFIG_NAME}"
    echo "=================================================="
    echo ""
done

echo "All tasks in the queue have been successfully completed."
