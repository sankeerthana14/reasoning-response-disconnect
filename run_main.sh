#!/bin/bash
#SBATCH --job-name=run_main
#SBATCH --gres=gpu:1
#SBATCH --partition=NH100q
#SBATCH --nodelist=node07
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

module purge

source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate mech_interp

# === GPU Debug ===
echo "=== GPU Debug ==="
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
python -c "
import torch
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('Device count:', torch.cuda.device_count())
if torch.cuda.is_available():
    print('GPU name:', torch.cuda.get_device_name(0))
"
echo "================="

python /export/home2/sati0004/AAAI/reasoning-response-disconnect/main.py --model "qwen7b" --dataset "truthfulqa" --seed 42 --output_dir /export/home2/sati0004/AAAI/reasoning-response-disconnect/results

