#!/usr/bin/env python3
"""Retained pair-set overlap under discarding (the format of the paper's Fig. 3):
EDC vs the paper's RCE vs our probabilistic harm ranking (rce_prob_test1.py).

Each ranking orders images from least to most harmful. Discarding at rate d drops
the int(d * n) most harmful images, keeps the pairs whose two images both survive,
and the Jaccard overlap (%) of the retained genuine / impostor pair sets is plotted
against d, as in the paper's Fig. 3.

Rankings compared:

  EDC        the discard order EDC itself rewards (EDC-optimal ranking). EDC re-fits
             the FMR threshold on the remaining pairs at every discard step and scores
             FNMR only. Greedy: at each step (--edc-step of the images) re-fit the
             threshold on the remaining impostors, then discard the images with the
             largest first-order FNMR reduction f_i - FNMR * g_i (f_i = false
             non-matches, g_i = genuine pairs of image i among remaining pairs).
             Once EDC's FNMR reaches 0 (after ~1-5% discard) every image has zero gain
             and EDC has no preference; ties are then broken by --edc-tiebreak:
             "margin" (default, the same rule as RCE: smallest remaining
             min genuine sim - max impostor sim first), "min-gen-sim" or "random".
             Images with no genuine pair left (irrelevant to FNMR) are discarded first.
             EDC curves are left blank once fewer than 1/FMR impostor pairs remain
             (where edc.py stops the EDC curve) or no genuine pair remains.
  RCE        the paper's single harm ranking: harm H = C_FNM + C_FM on the observed
             pairs at the fixed FMR threshold, ties broken by margin
             (min genuine sim - max impostor sim), exactly as rce.py writes
             harm_ranking__{fr_model}.csv
  ours       our expected harm ranking: images ordered by their mean rank E[R_e,i]
             over the B bootstrap draws of rce_prob_test1.py
  draws      individual harm rankings R_e^(b): plausible alternatives under
             resampled pairs

Curves:
  EDC vs RCE, EDC vs ours, RCE vs ours   how differently the evaluations order harm
  draw vs draw (grey)    agreement between two plausible rankings (noise ceiling)
  random (grey dashed)   overlap from discarding alone
The CSV also holds EDC / RCE / ours vs individual harm draws (typicality).

R_e samples are regenerated with the same seed / settings as rce_prob_test1.py.

Run:

    python new_prob1_test_set.py \
        --fr-features-root data/FR_models \
        --ca-fiqa-data-root data/pairs/pairs \
        --out-root output/new_prob1_test_set \
        --threshold-method roc --n-samples 400

Outputs, under {out_root}/:
    fig_edc_rce_ours_overlap_{fr_model}.png/.pdf   rows = genuine / impostor, columns = datasets
    edc_rce_ours_overlap.csv                       every curve point (long format)
"""

import argparse
import os

import matplotlib  # type: ignore[reportMissingImports]

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # type: ignore[reportMissingImports]
from matplotlib.lines import Line2D  # type: ignore[reportMissingImports]
import numpy as np  # type: ignore[reportMissingImports]
import pandas as pd

from rce import find_threshold_closest_fmr, find_threshold_for_fmr
from rce_prob_test1 import (
    harm_counts,
    list_jobs,
    load_job,
    rank_random_ties,
    sample_harm_rankings,
    threshold_at_fmr,
)

# Reference palette (categorical slots 1-3) + neutral references.
ORANGE = "#eb6834"   # RCE vs ours
AQUA = "#1baf7a"     # EDC vs RCE
BLUE = "#2a78d6"     # EDC vs ours
GRAY = "#8f8e88"     # draw vs draw, random
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e6e5e0"

DATASET_NAMES = {"calfw": "CALFW", "cplfw": "CPLFW", "lfw": "LFW", "xqlfw": "XQLFW", "cfp-fp": "CFP-FP"}
LABELS = {1: "genuine", 0: "impostor"}
CURVES = [
    # kind, color, linestyle, width, legend label
    ("RCE vs ours", ORANGE, "-", 2.2, "paper RCE vs our probabilistic ranking"),
    ("EDC vs RCE", AQUA, "-", 2.2, "EDC-optimal vs paper RCE ranking"),
    ("EDC vs ours", BLUE, "-", 2.2, "EDC-optimal vs our probabilistic ranking"),
    ("draw vs draw", GRAY, "-", 1.2, "harm draw vs harm draw (noise ceiling)"),
    ("random", GRAY, (0, (4, 2)), 1.2, "random rankings (discarding alone)"),
]


# ---------------------------------------------------------------------------
# Harm rankings
# ---------------------------------------------------------------------------

def rce_harm_ranking(*, n, idx1, idx2, sims, labels, fmr_target, threshold_method) -> np.ndarray:
    """rce.py's harm ranking (rank 1 = least harmful), as in harm_ranking__*.csv."""
    if threshold_method == "roc":
        t, _ = find_threshold_closest_fmr(sims, labels, fmr_target)
    else:
        t, _ = find_threshold_for_fmr(sims, labels, fmr_target)
    harm = (harm_counts(n, idx1, idx2, (labels == 1) & (sims < t))
            + harm_counts(n, idx1, idx2, (labels == 0) & (sims >= t)))

    gen, imp = labels == 1, labels == 0
    min_gen = np.full(n, np.inf)
    max_imp = np.full(n, -np.inf)
    np.minimum.at(min_gen, idx1[gen], sims[gen])
    np.minimum.at(min_gen, idx2[gen], sims[gen])
    np.maximum.at(max_imp, idx1[imp], sims[imp])
    np.maximum.at(max_imp, idx2[imp], sims[imp])
    min_gen = np.where(np.isfinite(min_gen), min_gen, np.nan)
    max_imp = np.where(np.isfinite(max_imp), max_imp, np.nan)
    margin = min_gen - max_imp
    margin = np.where(np.isfinite(margin), margin, -np.inf)

    order = np.lexsort((-margin, harm))
    ranks = np.empty(n, dtype=np.int32)
    ranks[order] = np.arange(1, n + 1, dtype=np.int32)
    return ranks


def edc_harm_ranking(
    *, n, idx1, idx2, sims, labels, fmr_target, threshold_method, step, tiebreak="margin", rng=None
) -> np.ndarray:
    """EDC-optimal discard order (rank 1 = discarded last), see module docstring."""
    rng = rng if rng is not None else np.random.default_rng(0)
    gen, imp = labels == 1, labels == 0
    alive = np.ones(n, dtype=bool)
    order_discard = []
    chunk = max(1, int(round(step * n)))
    t = None
    while alive.any():
        kept = alive[idx1] & alive[idx2]
        imp_k = imp & kept
        if imp_k.any():  # threshold re-fit on the remaining pairs (EDC threshold drift)
            t = threshold_at_fmr(sims[imp_k], fmr_target, threshold_method)
        gen_k = gen & kept
        fnm = gen_k & (sims < t)
        g = harm_counts(n, idx1, idx2, gen_k)
        f = harm_counts(n, idx1, idx2, fnm)
        fnmr = f.sum() / max(g.sum(), 1)
        gain = f - fnmr * g
        if tiebreak == "random":
            key = rng.random(n)
        else:
            min_gen = np.full(n, np.inf)
            np.minimum.at(min_gen, idx1[gen_k], sims[gen_k])
            np.minimum.at(min_gen, idx2[gen_k], sims[gen_k])
            key = min_gen
            if tiebreak == "margin":
                max_imp = np.full(n, -np.inf)
                np.maximum.at(max_imp, idx1[imp_k], sims[imp_k])
                np.maximum.at(max_imp, idx2[imp_k], sims[imp_k])
                key = min_gen - max_imp
            # No remaining genuine pair: irrelevant to EDC's FNMR -> discard first, so the
            # images that still carry genuine pairs stay as long as possible.
            key = np.where(np.isfinite(key), key, -np.inf)

        cand = np.flatnonzero(alive)
        # Largest FNMR gain first, then the tie-break key (smallest first).
        pick = cand[np.lexsort((key[cand], -gain[cand]))][:chunk]
        order_discard.extend(pick.tolist())
        alive[pick] = False

    ranks = np.empty(n, dtype=np.int32)
    ranks[np.array(order_discard)] = np.arange(n, 0, -1, dtype=np.int32)  # first discarded = worst rank
    return ranks


# ---------------------------------------------------------------------------
# Overlap of retained pair sets
# ---------------------------------------------------------------------------

def kept_pairs(ranks: np.ndarray, discard_rate: float, idx1: np.ndarray, idx2: np.ndarray) -> np.ndarray:
    """Boolean mask of pairs whose two images survive discarding the int(d * n) most harmful."""
    n = ranks.size
    keep = ranks <= n - min(int(discard_rate * n), n)
    return keep[idx1] & keep[idx2]


def jaccard(kept_a: np.ndarray, kept_b: np.ndarray, label_mask: np.ndarray) -> float:
    union = np.count_nonzero((kept_a | kept_b) & label_mask)
    if union == 0:
        return np.nan
    return 100.0 * np.count_nonzero(kept_a & kept_b & label_mask) / union


def run_one(
    *,
    dataset: str,
    fr_model: str,
    emb_pkl: str,
    pairs_csv: str,
    n_samples: int,
    n_draw_pairs: int,
    resample: str,
    fmr_target: float,
    threshold_method: str,
    discard_rates: list[float],
    edc_step: float,
    edc_tiebreak: str,
    seed: int,
) -> pd.DataFrame:
    # Same rng sequence as rce_prob_test1.run_one -> identical R_e samples.
    rng = np.random.default_rng(seed)
    img_ids, idx1, idx2, labels, sims = load_job(emb_pkl, pairs_csv)
    n = len(img_ids)
    R_e, _, _ = sample_harm_rankings(
        n=n, idx1=idx1, idx2=idx2, sims=sims, labels=labels,
        n_samples=n_samples, resample=resample, fmr_target=fmr_target,
        threshold_method=threshold_method, rng=rng,
    )

    rng2 = np.random.default_rng(seed + 1)
    rankings = {
        "rce": rce_harm_ranking(n=n, idx1=idx1, idx2=idx2, sims=sims, labels=labels,
                                fmr_target=fmr_target, threshold_method=threshold_method),
        "ours": rank_random_ties(R_e.mean(axis=0), rng2),
        "edc": edc_harm_ranking(n=n, idx1=idx1, idx2=idx2, sims=sims, labels=labels,
                                fmr_target=fmr_target, threshold_method=threshold_method, step=edc_step,
                                tiebreak=edc_tiebreak, rng=np.random.default_rng(seed + 2)),
    }
    k = min(n_draw_pairs, n_samples // 2)
    draws = rng2.permutation(n_samples)[: 2 * k]
    for j, b in enumerate(draws):
        rankings[f"h:{j}"] = R_e[b]
        rankings[f"r:{j}"] = rng2.permutation(n).astype(np.int32) + 1

    comparisons = [("RCE vs ours", "rce", "ours"), ("EDC vs RCE", "edc", "rce"), ("EDC vs ours", "edc", "ours")]
    for name in ("edc", "rce", "ours"):
        comparisons += [(f"{name.upper() if name != 'ours' else 'ours'} vs draws", name, f"h:{j}")
                        for j in range(2 * k)]
    comparisons += [("draw vs draw", f"h:{2 * j}", f"h:{2 * j + 1}") for j in range(k)]
    comparisons += [("random", f"r:{2 * j}", f"r:{2 * j + 1}") for j in range(k)]

    min_impostors = int(np.ceil(1.0 / fmr_target))  # edc.py stops the EDC curve below this
    rows = []
    for d in discard_rates:
        kept = {name: kept_pairs(r, d, idx1, idx2) for name, r in rankings.items()}
        edc_defined = (np.count_nonzero(kept["edc"] & (labels == 0)) >= min_impostors
                       and np.count_nonzero(kept["edc"] & (labels == 1)) > 0)  # else FNMR undefined
        for lbl in (1, 0):
            mask = labels == lbl
            for kind, a, b in comparisons:
                rows.append({"dataset": dataset, "fr_model": fr_model, "kind": kind, "a": a, "b": b,
                             "label": LABELS[lbl], "discard_rate": d,
                             "overlap": jaccard(kept[a], kept[b], mask)
                             if (edc_defined or "edc" not in (a, b)) else np.nan})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def style_axes(ax):
    ax.grid(True, color=GRID, linewidth=0.6, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)
        ax.spines[side].set_linewidth(0.6)
    ax.tick_params(colors=INK_2, labelsize=7, length=2)


def plot(df: pd.DataFrame, datasets: list[str], fr_model: str, out_path: str):
    fig, axes = plt.subplots(2, len(datasets), figsize=(3.3 * len(datasets), 5.6), sharex=True, sharey=True)
    axes = np.asarray(axes).reshape(2, len(datasets))
    for j, ds in enumerate(datasets):
        for i, lbl in enumerate(("genuine", "impostor")):
            ax = axes[i, j]
            style_axes(ax)
            g = df[(df.dataset == ds) & (df.label == lbl)]
            for z, (kind, color, ls, lw, _) in enumerate(reversed(CURVES)):
                s = g[g.kind == kind].groupby("discard_rate")["overlap"]
                if s.ngroups == 0:
                    continue
                x = np.array(s.mean().index) * 100
                ax.plot(x, s.mean(), color=color, linestyle=ls, linewidth=lw, zorder=2 + z)
            ax.set_ylim(0, 102)
            ax.set_xlim(0, max(df.discard_rate) * 100)
            ax.set_title(f"{DATASET_NAMES.get(ds, ds)} · {lbl} pairs", fontsize=8.5, color=INK, loc="left")
            if i == 1:
                ax.set_xlabel("Discard rate (%)", fontsize=8, color=INK_2)
            if j == 0:
                ax.set_ylabel("Pair overlap (Jaccard, %)", fontsize=8, color=INK_2)

    handles = [Line2D([0], [0], color=c, linestyle=ls, linewidth=lw, label=lab) for _, c, ls, lw, lab in CURVES]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle(f"Retained pair-set overlap of the harm rankings implied by EDC, RCE and our probabilistic method ({fr_model})",
                 fontsize=9.5, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(f"{out_path}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}.png/.pdf")


def main() -> None:
    p = argparse.ArgumentParser(description="Paper Fig. 3 format: EDC vs RCE vs our probabilistic harm ranking")
    p.add_argument("--fr-features-root", default="data/FR_models")
    p.add_argument("--ca-fiqa-data-root", default="data/pairs/pairs")
    p.add_argument("--out-root", default=os.path.join("output", "new_prob1_test_set"))
    p.add_argument("--datasets", default="lfw,xqlfw,calfw,cplfw",
                   help="Datasets (columns), in plot order; cfp-fp is broken (10 images)")
    p.add_argument("--fr-models", default="adaface,arcface_o,magface,swinface", help="One figure per FR model")
    p.add_argument("--discard-rates", default=",".join(str(i / 10) for i in range(10)),
                   help="Comma-separated discard rates in [0, 1)")
    p.add_argument("--edc-step", type=float, default=0.005,
                   help="Fraction of images discarded per EDC greedy step (threshold re-fit each step)")
    p.add_argument("--edc-tiebreak", choices=["margin", "min-gen-sim", "random"], default="margin",
                   help="EDC order once its FNMR is 0 (default: same margin rule as RCE)")
    p.add_argument("--n-draw-pairs", type=int, default=50, help="Harm-draw / random pairs per curve")
    # Must match the rce_prob_test1.py run to reproduce its R_e samples.
    p.add_argument("--fmr", type=float, default=1e-3)
    p.add_argument("--threshold-method", choices=["percentile", "roc"], default="percentile")
    p.add_argument("--resample", choices=["imp", "gen", "both"], default="both")
    p.add_argument("--n-samples", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    split = lambda s: [x.strip() for x in str(s).split(",") if x.strip()]
    datasets = split(args.datasets)
    discard_rates = [float(x) for x in split(args.discard_rates)]

    jobs = {(d, m): (pkl, pairs) for d, m, _, pkl, pairs in
            list_jobs(args.fr_features_root, args.ca_fiqa_data_root, run_all=True)}

    frames = []
    for fr_model in split(args.fr_models):
        for ds in datasets:
            if (ds, fr_model) not in jobs:
                print(f"[skip] {ds}/{fr_model}: no embeddings or pairs")
                continue
            emb_pkl, pairs_csv = jobs[(ds, fr_model)]
            print(f"[run] {ds}/{fr_model}: B={args.n_samples} ({args.resample}, {args.threshold_method})")
            frames.append(run_one(
                dataset=ds, fr_model=fr_model, emb_pkl=emb_pkl, pairs_csv=pairs_csv,
                n_samples=args.n_samples, n_draw_pairs=args.n_draw_pairs, resample=args.resample,
                fmr_target=args.fmr, threshold_method=args.threshold_method,
                discard_rates=discard_rates, edc_step=args.edc_step,
                edc_tiebreak=args.edc_tiebreak, seed=args.seed,
            ))
    if not frames:
        return

    df = pd.concat(frames, ignore_index=True)
    os.makedirs(args.out_root, exist_ok=True)
    csv = os.path.join(args.out_root, "edc_rce_ours_overlap.csv")
    df.to_csv(csv, index=False)
    print(f"saved {csv}")
    for fr_model, g in df.groupby("fr_model"):
        plot(g, [d for d in datasets if d in set(g.dataset)], fr_model,
             os.path.join(args.out_root, f"fig_edc_rce_ours_overlap_{fr_model}"))


if __name__ == "__main__":
    main()
