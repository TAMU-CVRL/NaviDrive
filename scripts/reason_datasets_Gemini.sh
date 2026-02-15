set -e

echo ">>> [Start] Generating reasons using Gemini..."

python3 naviGen_Gemini.py \
    --model_id gemini-2.5-flash\
    --data_path /home/ximeng/Dataset/nuscenes_full_v1_0/ \
    --output_file data/nuscenes_reasons_val_Gemini.jsonl \
    --version v1.0-trainval \
    --is_train 1

echo ">>> [Success] All tasks finished."
