#!/bin/bash
# Setup script for DrupalGym on RunPod (H100 or similar)

set -e

echo "Updating system and installing dependencies..."
apt-get update && apt-get install -y git python3-pip python3-venv

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Install Unsloth for optimized training if possible
# Note: Unsloth installation can be tricky depending on the CUDA version.
# For H100 (sm_90), we use the following:
#pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes

echo "Setup complete. You can now run the pipeline."
echo "Use 'source venv/bin/activate' to enter the environment."
