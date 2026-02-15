#!/bin/bash
#SBATCH --export=NONE # do not export current env to the job
#SBATCH --time=2-00:00:00 # request 2 day; Format: days-hours:minutes:seconds
#SBATCH --nodes=1 # request 1 node (optional since default=1)
#SBATCH --ntasks-per-node=1 # request 1 task (command) per node
#SBATCH --cpus-per-task=16 # request 1 cpu (core, thread) per task
#SBATCH --mem=128G # request 7.5GB total memory per node; can use 7G but not 7.5G
#SBATCH --gres=gpu:a100:2 # request 2 A100 GPU
#SBATCH --exclude=g031
#SBATCH --output=stdout.%x.%j # save stdout to a file with job name and JobID appended to file name
#SBATCH --error=stderr.%x.%j # save stdout to a file with job name and JobID appended to file name
#SBATCH --mail-user=ximeng@tamu.edu # send email to this address
#SBATCH --mail-type=END # send email per job event: BEGIN, END, FAIL, ALL
#SBATCH --job-name=naviGen_32B_0

# unload any modules to start with a clean environment
module purge 
export HF_HOME=/scratch/user/ximeng/hf_cache

export SINGULARITY_CACHEDIR=$TMPDIR/.singularity
module load WebProxy

export HF_TOKEN="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"
CONDA_ENV_PATH=/scratch/user/ximeng/.conda/envs/dllm
SQUASHFS_IMAGE=/scratch/group/cvrl/dataset/nuscenes.squashfs
PYTHON_SCRIPT=/scratch/user/ximeng/dllm/naviGen.py

# SIF_IMAGE="/scratch/data/Singularity/images/pytorch:23.06-py3.sif"
SIF_IMAGE="/scratch/user/ximeng/dllm/pytorch_24.02.sif"

mkdir -p /scratch/user/ximeng/dllm/data

SHARD_ID=0
TOTAL_SHARDS=6

echo "[INFO] Starting Shard ${SHARD_ID} of ${TOTAL_SHARDS}"

singularity exec --nv \
    --bind ${SQUASHFS_IMAGE}:/mnt/nuscenes:image-src=/ \
    --bind /scratch/user/ximeng:/scratch/user/ximeng \
    "$SIF_IMAGE" \
    ${CONDA_ENV_PATH}/bin/python ${PYTHON_SCRIPT} \
        --model_id "Qwen/Qwen3-VL-32B-Instruct" \
        --data_path "/mnt/nuscenes" \
        --output_file "data/nuscenes_reasons_32B_${SHARD_ID}.jsonl" \
        --is_train 0 \
        --total_shards ${TOTAL_SHARDS} \
        --shard_id ${SHARD_ID}    

echo "[INFO] Inference finished."
