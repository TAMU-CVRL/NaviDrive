#!/bin/bash
#SBATCH --job-name=pull_nv_img
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=stdout.%x.%j # save stdout to a file with job name and JobID appended to file name
#SBATCH --error=stderr.%x.%j # save stdout to a file with job name and JobID appended to file name

module purge
module load WebProxy

export SINGULARITY_CACHEDIR=$SCRATCH/.singularity
mkdir -p $SINGULARITY_CACHEDIR

TARGET_DIR="/scratch/user/ximeng/dllm"
mkdir -p $TARGET_DIR
cd $TARGET_DIR

echo "[INFO] Job started on node: $(hostname)"
echo "[INFO] Cache dir: $SINGULARITY_CACHEDIR"
echo "[INFO] Downloading NVIDIA PyTorch 24.02..."

singularity pull --force pytorch_24.02.sif docker://nvcr.io/nvidia/pytorch:24.02-py3

if [ $? -eq 0 ]; then
    echo "[SUCCESS] Image downloaded successfully to: $TARGET_DIR/pytorch_24.02.sif"
else
    echo "[ERROR] Download failed! Check pull_err_$SLURM_JOB_ID.txt"
    exit 1
fi
