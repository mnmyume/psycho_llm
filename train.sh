#!/bin/bash
#SBATCH --job-name=psy_train
#SBATCH --time=3-00:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

#SBATCH -o logs/JOB%j.out
#SBATCH -e logs/JOB%j-err.out

#SBATCH --mail-user=endavinci808@gmail.com
#SBATCH --mail-type=ALL

source /opt/anaconda3/etc/profile.d/conda.sh
conda activate psyc

mkdir -p logs

echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $SLURMD_NODENAME"
echo "Date: $(date)"

# ============================================================
# Training with YAML config (recommended)
# Override any value via CLI flags
# ============================================================

# --- 8B model on EmoArt-5k ---
# srun python src/train.py \
#     --config recipes/Qwen3-VL-8B.yaml \
#     --run_name "qwen3_vl_8b_emoart_5k_v1"

# --- 32B model on EmoArt-130k ---
srun python src/train.py \
    --config recipes/Qwen3-VL-32B.yaml \
    --run_name "qwen3_vl_32B_emoart_130k_v1"