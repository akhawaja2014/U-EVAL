#!/bin/bash
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --job-name=rce_eval
#SBATCH --output=rce_eval_%j.out
#SBATCH --error=rce_eval_%j.err

set -euo pipefail

module purge
module load Anaconda3/2024.02-1
module load CUDA/12.4.0

source activate rce_env

cd /cluster/home/muhamkh/RCE

echo "====== Environment Information ======"
python --version
nvidia-smi
echo "======================================"

python rce_prob_test1.py \
  --fr-features-root /cluster/home/muhamkh/RCE/data/FR_models \
  --quality-dir /cluster/home/muhamkh/RCE/data/FIQA_models \
  --ca-fiqa-data-root /cluster/home/muhamkh/RCE/data/pairs/pairs \
  --out-root /cluster/home/muhamkh/RCE/output/rce \
  --threshold-method roc \
  --weight-alphas 0,3
