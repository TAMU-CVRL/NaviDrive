#!/bin/bash
#SBATCH --export=NONE # do not export current env to the job
#SBATCH --time=1-00:00:00 # request 2 day; Format: days-hours:minutes:seconds
#SBATCH --nodes=1 # request 1 node (optional since default=1)
#SBATCH --ntasks-per-node=1 # request 1 task (command) per node
#SBATCH --cpus-per-task=8 # request 1 cpu (core, thread) per task
#SBATCH --mem=56G # request 7.5GB total memory per node; can use 7G but not 7.5G
#SBATCH --gres=gpu:a100:1 # request 2 A100 GPU
#SBATCH --exclude=g031
#SBATCH --output=stdout.%x.%j # save stdout to a file with job name and JobID appended to file name
#SBATCH --error=stderr.%x.%j # save stdout to a file with job name and JobID appended to file name
#SBATCH --mail-user=ximeng@tamu.edu # send email to this address
#SBATCH --mail-type=END # send email per job event: BEGIN, END, FAIL, ALL
#SBATCH --job-name=inference_8B_qlora # specify job name

module purge
module load WebProxy

source /sw/eb/sw/Anaconda3/2024.02-1/etc/profile.d/conda.sh
conda activate dllm

set -e
CONFIG_NAME="qwen_vl_8B_sft_qlora"
CONFIG_FILE="configs/${CONFIG_NAME}.yaml"

echo ">>> [Step 1/2] Inference..."
python3 eval.py \
    --config "${CONFIG_FILE}" \
    --inference_path data/nuscenes_reasons_val_Qwen_32B.jsonl

echo ">>> [Step 2/2] Evaluation..."
python3 eval.py \
    --config "${CONFIG_FILE}" \
    --eval_L2 True

echo ">>> All tasks finished for ${CONFIG_NAME}."