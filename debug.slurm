cd /cluster/home/muhamkh/RCE
module purge
module load Anaconda3/2024.02-1
source activate rce_env

python -m pdb rce.py \
  --fr-features-root /cluster/home/muhamkh/RCE/data/FR_models \
  --quality-dir /cluster/home/muhamkh/RCE/data/FIQA_models \
  --ca-fiqa-data-root /cluster/home/muhamkh/RCE/data/pairs/pairs \
  --out-root /tmp/rce-debug \
  --threshold-method roc \
  --weight-alphas 0