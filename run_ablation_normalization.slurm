#!/bin/bash
#SBATCH --partition=CPUQ
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --job-name=ablation_norm
#SBATCH --output=ablation_normalization_%j.out
#SBATCH --error=ablation_normalization_%j.err

# Ablation: PURE-FIQA with the normalized error contribution (ablation_normalization.py).
# Needs the counts results in output/protocol_stability (run_protocol_stability.slurm) for the comparison.
# Submit from the repository root:  sbatch run_ablation_normalization.slurm

module purge
module load Anaconda3/2024.02-1
source activate rce_env

# One numpy thread per worker process.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

cd /cluster/home/muhamkh/RCE

python ablation_normalization.py \
	--fr-features-root data/FR_models \
	--quality-dir data/FIQA_models \
	--ca-fiqa-data-root data/pairs/pairs \
	--out-root output/ablation_normalization \
	--counts-root output/protocol_stability \
	--threshold-method roc \
	--rce-alphas 0,3 \
	--lam 0.5 \
	--strength 1.0 \
	--n-samples 400 \
	--workers "${SLURM_CPUS_PER_TASK:-16}"
