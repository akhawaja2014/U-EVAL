#!/bin/bash
#SBATCH --partition=CPUQ
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --job-name=pit_baselines
#SBATCH --output=pit_baselines_%j.out
#SBATCH --error=pit_baselines_%j.err

# Reference rankings for the PURE-FIQA consistency diagnostic (pit_baselines.py).
# Reproduces the F_i / null of output/rce_prob_test1 (same seed, B, threshold rule) and checks it.
# Submit from the repository root:  sbatch run_pit_baselines.slurm

module purge
module load Anaconda3/2024.02-1
source activate rce_env

# One numpy thread per worker process.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

cd /cluster/home/muhamkh/RCE

python pit_baselines.py \
	--fr-features-root data/FR_models \
	--quality-dir data/FIQA_models \
	--ca-fiqa-data-root data/pairs/pairs \
	--out-root output/pit_baselines \
	--reference-csv output/rce_prob_test1/pit_test_all.csv \
	--table-out paper/tables/tab_pit_baselines.tex \
	--threshold-method roc \
	--n-samples 400 \
	--seed 0 \
	--n-random 20 \
	--noise-sd 0.5 \
	--workers "${SLURM_CPUS_PER_TASK:-16}"
