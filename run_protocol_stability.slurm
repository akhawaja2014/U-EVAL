#!/bin/bash
#SBATCH --partition=CPUQ
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --job-name=protocol_stability
#SBATCH --output=protocol_stability_%j.out
#SBATCH --error=protocol_stability_%j.err

# Stability of EDC / RCE FIQA leaderboards under an identity bootstrap (protocol_stability.py).
# Submit from the repository root:  sbatch run_protocol_stability.slurm

module purge
module load Anaconda3/2024.02-1
source activate rce_env

# One numpy thread per worker process.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

cd /cluster/home/muhamkh/RCE

python protocol_stability.py \
	--fr-features-root data/FR_models \
	--quality-dir data/FIQA_models \
	--ca-fiqa-data-root data/pairs/pairs \
	--out-root output/protocol_stability \
	--threshold-method roc \
	--rce-alphas 0,3 \
	--rce-ties margin \
	--resample identities \
	--n-samples 400 \
	--workers "${SLURM_CPUS_PER_TASK:-16}"
