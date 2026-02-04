#!/bin/bash
#SBATCH --export=NONE # do not export current env to the job
#SBATCH --time=2-00:00:00 # request 2 day; Format: days-hours:minutes:seconds
#SBATCH --nodes=1 # request 1 node (optional since default=1)
#SBATCH --ntasks-per-node=1 # request 1 task (command) per node
#SBATCH --cpus-per-task=8 # request 1 cpu (core, thread) per task
#SBATCH --mem=80G # request 7.5GB total memory per node; can use 7G but not 7.5G
#SBATCH --gres=gpu:a100:2 # request 2 A100 GPU
#SBATCH --output=stdout.%x.%j # save stdout to a file with job name and JobID appended to file name
#SBATCH --error=stderr.%x.%j # save stdout to a file with job name and JobID appended to file name
#SBATCH --mail-user=ximeng@tamu.edu # send email to this address
#SBATCH --mail-type=ALL # send email per job event: BEGIN, END, FAIL, ALL
#SBATCH --job-name=reasonGen_30B

# unload any modules to start with a clean environment
module purge 
export HF_HOME=/scratch/user/ximeng/hf_cache

export SINGULARITY_CACHEDIR=$TMPDIR/.singularity
module load WebProxy

export HF_TOKEN="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"
CONDA_ENV_PATH=/scratch/user/ximeng/.conda/envs/dllm
SQUASHFS_IMAGE=/scratch/group/cvrl/dataset/nuscenes.squashfs
PYTHON_SCRIPT=/scratch/user/ximeng/dllm/reasonGen_hprc.py

# SIF_IMAGE="/scratch/data/Singularity/images/pytorch:23.06-py3.sif"
SIF_IMAGE="/scratch/user/ximeng/dllm/pytorch_24.02.sif"

echo "[INFO] Starting NaviDrive Inference with Flash Attention 2..."

singularity exec --nv \
    --bind ${SQUASHFS_IMAGE}:/mnt/nuscenes:image-src=/ \
    --bind /scratch/user/ximeng:/scratch/user/ximeng \
    "$SIF_IMAGE" \
    ${CONDA_ENV_PATH}/bin/python ${PYTHON_SCRIPT}

echo "[INFO] Inference finished."
