set -e

echo "========================================================"
echo ">>> [Start] Generating reasons using Gemini..."
echo "========================================================"

python3 naviGen_Gemini.py \
    --model_id gemini-2.5-flash\
    --data_path /home/ximeng/Dataset/nuscenes_full_v1_0/ \
    --output_file data/nuscenes_reasons_Gemini.jsonl \
    --version v1.0-trainval \
    --is_train 0 # 0 for train, 1 for val

echo "========================================================"
echo ">>> [Success] All tasks finished."
echo "========================================================"