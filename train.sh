#!/bin/bash
#SBATCH --job-name=psy_30
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

# You need to point your script to the 4-bit bitsandbytes (bnb) version of the model, which is designed for fine-tuning.
MODEL="unsloth/Qwen3-VL-32B-Instruct-unsloth-bnb-4bit"
DATASET="printblue/EmoArt-130k"
RUN_NAME="qwen3_vl_30b_emoart_130k_v1"

srun python src/train.py \
    --model_name "$MODEL" \
    --dataset_name "$DATASET" \
    --run_name "$RUN_NAME"