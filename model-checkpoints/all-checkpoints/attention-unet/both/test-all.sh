#!/bin/bash

# Define directories
TEST_IMGS_DIR="/home/kiss2024/data/space-net-8-bit/"
TEST_MASKS_DIR="/home/kiss2024/data/space-net-8-bit"
MODEL_DIR="$(pwd)"  # Assumes the script is inside the model directory
PYTHON_SCRIPT_DIR="/home/kiss2024/vijit/attention-unet-space-net/"

# Create a directory for logs if it doesn't exist
LOG_DIR="${MODEL_DIR}/logs/"
mkdir -p "$LOG_DIR"

# Navigate to the directory where the python script is
cd "$PYTHON_SCRIPT_DIR"

# Iterate over all .pth files in the model directory
for model_checkpoint in "$MODEL_DIR"/*.pth; do
  model_name=$(basename -- "$model_checkpoint")
  model_name="${model_name%.*}"

  echo "Processing $model_checkpoint for Both datasets..."
  python -u test-model-custom.py --test-imgs "$TEST_IMGS_DIR" --test-masks "$TEST_MASKS_DIR" --load "$model_checkpoint" --dataset both 2>&1 | tee "$LOG_DIR/${model_name}_Both.txt"
  echo "Output for Both datasets saved to ${model_name}_Both.txt"
done

echo "All models processed for Both datasets."
