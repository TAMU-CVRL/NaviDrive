#!/bin/bash

set -e

echo "========================================================"
echo ">>> [Step 1/2] Running naviGen.py for val dataset..."
echo "========================================================"

python3 naviGen.py \
    --output_file data/nuscenes_reasons_val.jsonl \
    --is_train 1

echo ""
echo "========================================================"
echo ">>> [Step 2/2] Running naviGen.py for train dataset..."
echo "========================================================"

python3 naviGen.py \
    --output_file data/nuscenes_reasons.jsonl \
    --is_train 0

echo ""
echo "========================================================"
echo ">>> All tasks finished."
echo "========================================================"