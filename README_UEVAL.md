## U-EVAL: Probabilistic Uncertainty-aware Rank Evaluation of Face Image Quality Scores

_Under review._

## Table of Contents

- [Overview](#overview)
- [What U-EVAL Reports](#what-u-eval-reports)
- [Results](#results)
- [Installation](#installation)
- [Required Input Data](#required-input-data)
- [Running U-EVAL](#running-u-eval)
- [Output Files](#output-files)
- [Repository Structure](#repository-structure)
- [Citation](#citation)
- [License](#license)

## Overview

Face Image Quality Assessment (FIQA) methods are usually evaluated by relating their quality scores to face recognition performance. The standard protocol, the Error-versus-Discard Characteristic (EDC), is sensitive to protocol design choices. Rank Consistency Evaluation ([RCE](https://arxiv.org/pdf/2607.22752)) removes discarding and threshold drift, but it still reports a **single score computed on one fixed set of comparison pairs**. It cannot tell whether the gap between two FIQA methods is real or just an effect of the particular test set.

**U-EVAL** treats the evaluation set as a sample. It:

1. **Resamples the test set by identity** (identity bootstrap), recomputes the recognition-error reference ranking and the evaluation score on every resample, and reports each FIQA method's score with a confidence interval.
2. **Estimates rank probabilities**: how likely each method is to rank first, and how likely it is to beat every other method.
3. **Compares the stability of evaluation protocols.** The same procedure runs unchanged on EDC and RCE, so you can see which protocol gives a more stable leaderboard.
4. **Diagnoses where a FIQA method departs from the recognition errors**, image by image, with a randomized probability integral transform (rank-PIT) goodness-of-fit test.

U-EVAL therefore separates true differences between FIQA methods from differences that fall within the sampling uncertainty of the evaluation set.

## What U-EVAL Reports

For each dataset, face recognition (FR) model and FIQA method:

| Output | Meaning |
|---|---|
| Score + 95% interval | Observed RCE<sub>γ</sub> or EDC pAUC, with the 95% identity-bootstrap percentile interval |
| Rank distribution | Median rank and 90% rank interval of the method across resamples |
| P<sup>top</sup> | Probability that the method ranks first |
| P(m ≻ m′) | Probability that method m outperforms method m′; a pair is *decided* if this is ≥ 0.95 or ≤ 0.05 |
| U | Share of method pairs whose order is undecided (protocol stability, lower is better) |
| W | Mean width of the 90% rank interval (protocol stability, lower is better) |
| Rank-PIT test | KS / Anderson–Darling statistics, Monte Carlo p-value, effect size η, and the share of images ranked much better (φ<sub>low</sub>) or worse (φ<sub>high</sub>) than their errors allow |

Default settings: target FMR = 10<sup>-3</sup> (ROC rule), γ ∈ {0, 3}, B = 400 resamples, EDC discard steps of 5% with pAUC up to 30% discard, fixed random seed.

## Results

**Leaderboard stability (identity bootstrap, B = 400).** U: share of FIQA method pairs whose order is undecided. W: mean width of the 90% rank interval, in ranks. Mean over four FR models (AdaFace, ArcFace, MagFace, SwinFace). Lower is more stable.

| Dataset | U: EDC | U: RCE<sub>0</sub> | U: RCE<sub>3</sub> | W: EDC | W: RCE<sub>0</sub> | W: RCE<sub>3</sub> |
|---|---|---|---|---|---|---|
| LFW   | 56% | **47%** | 95% | 6.8 | **6.0** | 11.7 |
| XQLFW | **28%** | 42% | 59% | **3.6** | 5.5 | 7.2 |
| CALFW | 44% | **8%**  | 21% | 5.6 | **1.3** | 2.9 |
| CPLFW | 16% | **8%**  | 25% | 2.3 | **1.2** | 3.5 |

On LFW, roughly half of all FIQA method pairs cannot be ordered with confidence under either EDC or RCE<sub>0</sub>. A single leaderboard on this dataset hides large uncertainty. With AdaFace on LFW, for example, the top-ranked method (eDifFIQA (L), RCE<sub>0</sub> = 0.143 [0.11, 0.18]) ranks first in only 56% of the resamples.

## Installation

Python 3.10 was used.

```bash
git clone https://github.com/RAIB-group/RCE.git
cd RCE
pip install numpy pandas scipy matplotlib
```

## Required Input Data

U-EVAL uses the same input layout as RCE: face recognition embeddings, FIQA quality scores and verification pairs. The default data root is `data/`.

```text
data/
├── FR_models/
│   └── {dataset}/
│       └── {fr_model}-embeddings.pkl      # {"image.jpg": np.ndarray, ...}
├── FIQA_models/
│   └── {fiqa_method}/
│       └── {dataset}-quality.pkl          # {"image.jpg": float, ...}
└── pairs/
    └── pairs/
        └── {dataset}/
            └── pairs/
                └── pairs.csv              # img1,img2,label  (1 = genuine, 0 = impostor)
```

- **Embeddings:** one vector per image. Pair similarity is the cosine similarity.
- **Quality scores:** one score per image. Images without a score receive quality 0.
- **Pairs CSV:** the columns `img1_id,img2_id` are accepted as well.
- **Identities** are read from the file name: the name without its trailing image number (for example `Aaron_Peirsol_0002.jpg` → `Aaron_Peirsol`). They define the sampling unit of the identity bootstrap.

Defaults used in the paper:

- **Datasets:** LFW, XQLFW, CALFW, CPLFW (CFP-FP is excluded because of a broken image-ID mapping)
- **FR models:** AdaFace, ArcFace, MagFace, SwinFace
- **FIQA methods:** all methods found under `data/FIQA_models/`

## Running U-EVAL

Run all commands from the repository root. Every script accepts `--datasets`, `--fr-models`, `--methods` (comma-separated) and `--workers N` to run dataset / FR model combinations in parallel. `--plot-only` redraws the figures from saved results without recomputing.

### 1. Probabilistic evaluation and protocol stability

Computes the bootstrap intervals, rank distributions, P<sup>top</sup>, pairwise superiority and the stability measures U and W for both EDC and RCE.

```bash
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
    --workers 16
```

### 2. Consistency diagnostic (rank-PIT test)

Builds per-image reference-rank distributions with the joint pair bootstrap. It then tests each FIQA ranking against them.

```bash
python rce_prob_test1.py \
    --fr-features-root data/FR_models \
    --quality-dir data/FIQA_models \
    --ca-fiqa-data-root data/pairs/pairs \
    --out-root output/rce_prob_test1 \
    --threshold-method roc \
    --resample both \
    --n-samples 400

python plot_pit_lines.py          # PIT histograms per FIQA method
python plot_rce_vs_pit.py         # RCE vs. rank-PIT comparison
```

### 3. Reference rankings for the diagnostic (optional)

Runs the rank-PIT test on reference rankings (random, oracle, and oracle + noise) to show what the statistics look like at both extremes. It also checks that step 2 can be reproduced exactly.

```bash
python pit_baselines.py \
    --reference-csv output/rce_prob_test1/pit_test_all.csv \
    --threshold-method roc \
    --n-samples 400 --seed 0 --n-random 20 --noise-sd 0.5 \
    --workers 16
```

### 4. Ablation: normalized error contribution (optional)

Replaces the raw error count of each image with an exposure- and imbalance-normalized relative risk (λ = 0.5, κ = 1). Requires the results of step 1.

```bash
python ablation_normalization.py \
    --counts-root output/protocol_stability \
    --threshold-method roc \
    --rce-alphas 0,3 --lam 0.5 --strength 1.0 \
    --n-samples 400 --workers 16
```

### 5. Paper tables

```bash
python make_results_tables.py     # LaTeX tables in paper/tables/
python make_supplementary.py      # supplementary tables and figures
python validate_supplementary.py  # cross-checks the generated material (exit code 1 on failure)
```

### Running on a SLURM cluster

Ready-made job scripts are included. Adjust the module and environment lines to match your cluster:

```bash
sbatch run_protocol_stability.slurm
sbatch run_pit_baselines.slurm
sbatch run_ablation_normalization.slurm
```

## Output Files

| Script | Folder | Main files |
|---|---|---|
| `protocol_stability.py` | `output/protocol_stability/` | `scores.csv` (per-resample scores), `method_summary.csv` (intervals, ranks, P<sup>top</sup>), `pairwise.csv` (P(m ≻ m′)), `protocol_summary.csv` (U, W), `figures/` |
| `rce_prob_test1.py` | `output/rce_prob_test1/` | `pit_test_all.csv`, and per dataset / FR model: `pit_test_summary.csv`, `pit_values.csv`, `null_statistics.csv` |
| `pit_baselines.py` | `output/pit_baselines/` | `pit_baselines.csv`, `pit_vs_random.csv`, `pit_histograms.csv`, `reproducibility.csv` |
| `ablation_normalization.py` | `output/ablation_normalization/` | `scores.csv`, `method_summary.csv`, `pairwise.csv`, `protocol_summary.csv`, `ablation_summary.csv` |

## Repository Structure

| File | Purpose |
|---|---|
| `protocol_stability.py` | U-EVAL probabilistic evaluation and protocol stability (identity bootstrap) |
| `rce_prob_test1.py` | U-EVAL consistency diagnostic (rank-PIT test, pair bootstrap) |
| `pit_baselines.py` | Reference rankings for the rank-PIT diagnostic |
| `ablation_normalization.py` | Ablation with normalized error contribution |
| `fiqa_protocol_overlap.py`, `new_prob1_test_set.py` | Retained pair-set overlap under discarding (test-set divergence) |
| `rce.py` | RCE metric (Wani et al., IJCB 2026), used by U-EVAL |
| `edc.py` | EDC protocol, used by U-EVAL |
| `make_results_tables.py`, `make_supplementary.py`, `validate_supplementary.py` | Paper tables, supplementary material and checks |
| `plot_pit_lines.py`, `plot_rce_vs_pit.py` | Figures |
| `run_*.slurm` | SLURM job scripts |

## Citation

The citation will be added once the paper is published. If you use U-EVAL, please also cite RCE, which it builds on:

```bibtex
@inproceedings{wani2026rce,
author = {Bhavesh Wani and {\v{Z}}iga Babnik and Vitomir {\v{S}}truc and Philipp Terh{\"o}rst},
title = {Beyond Error-vs-Discard Characteristic: Toward Stable and Reliable Evaluation for Face Image Quality Assessment},
booktitle = {{IEEE} International Joint Conference on Biometrics, {IJCB} 2026, Rome, Italy, September 1--4, 2026},
year = {2026}
}
```

## License

This project is licensed under the Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) license. You may use, modify and redistribute it for non-commercial research purposes, provided appropriate attribution is given. Commercial use requires prior permission from the copyright holder.
