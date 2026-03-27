#!/bin/bash

set -e
ENV_NAME="navidrive"

echo "Starting to configure the $ENV_NAME environment..."

eval "$(conda shell.bash hook)"

# Check if the conda environment already exists
if conda info --envs | grep -q "^$ENV_NAME "; then
    echo "Environment '$ENV_NAME' already exists. Activating and continuing installation..."
else
    echo "Creating conda environment: $ENV_NAME (Python 3.10)..."
    # -y automatically answers 'yes' to the installation prompts
    conda create -y -n $ENV_NAME python=3.10
fi

# Activate the environment
echo "Activating environment..."
conda activate $ENV_NAME

# Install PyTorch
echo "Installing PyTorch (CUDA 12.6), please refer PyTorch official website for specific GPUs..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# Install other core dependencies
echo "Installing other required libraries..."
pip install transformers==5.1.0 \
            datasets==4.5.0 \
            accelerate==1.12.0 \
            peft==0.18.1 \
            "bitsandbytes>=0.46.1" \
            opencv-python==4.11.0.86 \
            nuscenes-devkit==1.2.0 \
            qwen-vl-utils==0.0.14 \
            beautifulsoup4==4.14.3 \
            typeguard==4.5.1 \
            wandb==0.25.1 \
            tensorboard==2.20.0 \
            ipykernel==7.2.0 \
            ipywidgets==8.1.8 \
            pickleshare==0.7.5 \

echo "Environment setup is complete!"
echo "Please run the following command in your terminal to activate and use it:"
echo "conda activate $ENV_NAME"
