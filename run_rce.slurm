#!/bin/bash
#SBATCH --partition=GPUQ                  # Use the GPU queue
#SBATCH --gres=gpu:a100:1                 # Request 1 A100 GPU
#SBATCH --nodes=1                         # Use 1 node
#SBATCH --ntasks=1                        # One task only
#SBATCH --cpus-per-task=8                 # 8 CPUs per GPU
#SBATCH --mem=32G                         # 32 GB memory (lowered from 200G to queue faster)
#SBATCH --time=02:00:00                   # 2 hours is usually plenty for evaluation
#SBATCH --job-name=rce_eval               # Job name
#SBATCH --output=rce_eval_%j.out          # Standard output log (adds job ID to filename)
#SBATCH --error=rce_eval_%j.err           # Error log

# Load modules
module purge
module load Anaconda3/2024.02-1
module load CUDA/12.4.0

# Activate the RCE Conda environment
source activate rce_env

# Navigate to your cloned repository
cd /cluster/home/muhamkh/RCE

# Print environment information for debugging
echo "====== Environment Information ======"
python --version
nvidia-smi
echo "======================================"

# Execute RCE using the uploaded cluster data.
python rce.py \
	--fr-features-root /cluster/home/muhamkh/RCE/data/FR_models \
	--quality-dir /cluster/home/muhamkh/RCE/data/FIQA_models \
	--ca-fiqa-data-root /cluster/home/muhamkh/RCE/data/pairs/pairs \
	--out-root /cluster/home/muhamkh/RCE/output/rce \
	--threshold-method roc \
	--weight-alphas 0,3
