#!/bin/bash
#SBATCH --export=NONE # do not export current env to the job
#SBATCH --time=2-00:00:00 # request 2 day; Format: days-hours:minutes:seconds
#SBATCH --nodes=1 # request 1 node (optional since default=1)
#SBATCH --ntasks-per-node=1 # request 1 task (command) per node
#SBATCH --cpus-per-task=8 # request 1 cpu (core, thread) per task
#SBATCH --mem=56G # request 7.5GB total memory per node; can use 7G but not 7.5G
#SBATCH --gres=gpu:a100:1 # request 2 A100 GPU
#SBATCH --exclude=g031
#SBATCH --output=hprc/stdout.%x.%j # save stdout to a file with job name and JobID appended to file name
#SBATCH --error=hprc/stderr.%x.%j # save stdout to a file with job name and JobID appended to file name
#SBATCH --mail-user=ximeng@tamu.edu # send email to this address
#SBATCH --mail-type=END # send email per job event: BEGIN, END, FAIL, ALL
#SBATCH --job-name=qwen3_8B_train # specify job name

module purge
module load WebProxy

export HF_HOME=/scratch/user/ximeng/hf_cache
export SINGULARITY_CACHEDIR=$TMPDIR/.singularity

CONDA_ENV_PATH=/scratch/user/ximeng/.conda/envs/dllm
SQUASHFS_IMAGE=/scratch/group/cvrl/dataset/nuscenes.squashfs
SIF_IMAGE="/scratch/user/ximeng/dllm/pytorch_24.02.sif"

CONFIG_NAME="qwen_vl_8B_sft_qlora_cmd_img_6_no_res"
CONFIG_FILE="configs/${CONFIG_NAME}.yaml"

set -e

# ==========================================
# STEP 1: TRAINING
# ==========================================
echo ">>> [Step 1/3] Training with ${CONFIG_NAME} ..."
singularity exec --nv \
    --bind ${SQUASHFS_IMAGE}:/mnt/nuscenes:image-src=/ \
    --bind /scratch/user/ximeng:/scratch/user/ximeng \
    "$SIF_IMAGE" \
    ${CONDA_ENV_PATH}/bin/python train.py \
        --config "${CONFIG_FILE}"

# # ==========================================
# # STEP 2: INFERENCE
# # ==========================================
# echo ">>> [Step 2/3] Inference ..."
# singularity exec --nv \
#     --bind ${SQUASHFS_IMAGE}:/mnt/nuscenes:image-src=/ \
#     --bind /scratch/user/ximeng:/scratch/user/ximeng \
#     "$SIF_IMAGE" \
#     ${CONDA_ENV_PATH}/bin/python eval.py \
#         --config "${CONFIG_FILE}" \
#         --inference_path data/nuscenes_reasons_val_Qwen_32B.jsonl

# # ==========================================
# # STEP 3: EVALUATION
# # ==========================================
# echo ">>> [Step 3/3] Evaluation ..."
# singularity exec --nv \
#     --bind ${SQUASHFS_IMAGE}:/mnt/nuscenes:image-src=/ \
#     --bind /scratch/user/ximeng:/scratch/user/ximeng \
#     "$SIF_IMAGE" \
#     ${CONDA_ENV_PATH}/bin/python eval.py \
#         --config "${CONFIG_FILE}" \
#         --eval_L2 True

echo ">>> All tasks finished for ${CONFIG_NAME}."