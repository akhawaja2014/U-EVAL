#!/usr/bin/env python3
"""Stability of FIQA leaderboards under resampled evaluation pairs:
the paper's EDC, the paper's RCE, and our probabilistic summary.

Each dataset / FR model's evaluation set is resampled B times. On every resample, and
on the original pairs (sample = -1), each protocol is run unchanged.

Resampling (--resample):
  identities (default)  cluster bootstrap over identities. Pairs sharing an image or an
        identity are not independent (one LFW identity has 530 images), so identities
        are the sampling unit: K identities are drawn with replacement from the K in
        the set. An identity drawn c times contributes c copies of each of its images;
        each copy keeps its genuine pairs (c copies of every genuine pair), and an
        impostor pair between identities drawn c_a and c_b times appears c_a * c_b
        times (once for every combination of copies). Image copies are separate
        samples for both protocols (EDC discards copies, RCE ranks copies). The
        identity is the file name without the trailing _<number>.<ext>; names that do
        not match count as their own identity.
  gen / imp / both      pair bootstrap as in rce_prob_test1.py (same seed and
        random-number sequence): genuine and/or impostor pairs drawn with replacement,
        images fixed. Ignores the dependence between pairs and understates variance.

Protocols:

  EDC   edc.py: discard the lowest-quality images in steps of 5 %, re-fit the FMR
        threshold on the remaining pairs at every step, FNMR of the remaining genuine
        pairs, pAUC of the FNMR curve up to 30 % discard (lower is better). The curve
        stops when fewer than 1/FMR impostor pairs remain, as in edc.py.
  RCE   per-image harm E_i = false non-matches + false matches at a fixed FMR threshold,
        harm rank R_E vs quality rank R_Q (average ranks), weighted Spearman rho_w with
        gamma in --rce-alphas (higher is better). --rce-ties margin (default): ties in E_i
        are broken by the margin m(i) = min genuine sim - max impostor sim, as defined in
        the RCE paper and used by rce.py for harm_ranking__*.csv; --rce-ties midrank:
        average ranks, as in rce.py's score, which inflates RCE_0 (a random FIQA ranking
        scores about +0.3 when ~85 % of images have zero harm).

From the B scores of every FIQA method:
  - the rank of each method among all FIQA methods in every sample (1 = best),
  - P(A beats B) = share of samples in which method A scores better than method B,
  - ours: the probabilistic RCE summary, E[rho_w] with a 95 % interval and P(A beats B).

Leaderboard stability measures per protocol:
  - undecided pairs: share of method pairs with 0.05 < P(A beats B) < 0.95,
  - rank interval: width of the 5-95 % interval of a method's rank.

Run (on a compute node, see run_protocol_stability.slurm):

    python protocol_stability.py \
        --fr-features-root data/FR_models \
        --quality-dir data/FIQA_models \
        --ca-fiqa-data-root data/pairs/pairs \
        --out-root output/protocol_stability \
        --threshold-method roc --rce-alphas 0,3 --n-samples 400 --workers 16

Redraw figures from the saved scores without recomputing:

    python protocol_stability.py --out-root output/protocol_stability --plot-only

Outputs, under {out_root}/:
  scores.csv              score of every method in every sample (sample -1 = original pairs)
  method_summary.csv      per method: observed score/rank, mean, 95 % interval, rank interval
  pairwise.csv            P(A beats B) per protocol
  protocol_summary.csv    undecided pairs and rank-interval width per protocol
  figures/{dataset}/{fr_model}/   one figure per file (PDF + PNG):
      rank_intervals_g{gamma}       rank of each FIQA method: observed (EDC, RCE) and PURE-FIQA 90 % interval
      pure_fiqa_leaderboard_g{gamma}  PURE-FIQA mean and 95 % interval of RCE_gamma, RCE value marked
      pairwise_EDC, pairwise_RCE_g{gamma}   P(row method beats column method)
  figures/summary/                one figure per file:
      undecided_pairs_g{gamma}, rank_interval_width_g{gamma}
"""

import argparse
import os
import re
from multiprocessing import Pool

import matplotlib  # type: ignore[reportMissingImports]

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # type: ignore[reportMissingImports]
from matplotlib.colors import LinearSegmentedColormap  # type: ignore[reportMissingImports]
from matplotlib.legend_handler import HandlerBase, HandlerTuple  # type: ignore[reportMissingImports]
from matplotlib.lines import Line2D  # type: ignore[reportMissingImports]
import numpy as np  # type: ignore[reportMissingImports]
import pandas as pd

from rce import (
    _quality_pkl_for_dataset,
    _quality_scores_for_ids,
    _rank_ascending,
    _rank_descending,
    _weighted_spearman,
    load_quality_scores_pkl,
)
from rce_prob_test1 import harm_counts, list_jobs, load_job

EDC_STEPS = np.arange(0, 0.98, 0.05)  # edc.py default reject steps
EDC_MAX_REJECT = 0.3                  # edc.py pAUC range

DATASET_NAMES = {"calfw": "CALFW", "cplfw": "CPLFW", "lfw": "LFW", "xqlfw": "XQLFW", "cfp-fp": "CFP-FP",
                 "adience": "Adience", "agedb": "AgeDB"}
FR_NAMES = {"adaface": "AdaFace", "arcface_o": "ArcFace", "magface": "MagFace", "swinface": "SwinFace"}
METHOD_NAMES = {
    "clib-fiqa": "CLIB-FIQA", "cr-fiqa": "CR-FIQA (L)", "diffiqa_r": "DifFIQA (R)",
    "ediffiqa_l": "eDifFIQA (L)", "faceqan": "FaceQAN", "faceqgen": "FaceQGen",
    "faceqnet": "FaceQnet", "froq_ada": "FROQ", "grafiqs": "GraFIQs",
    "lightqnet": "LightQNet", "magface": "MagFace", "pcnet": "PCNet", "pfe": "PFE",
    "sdd-fiqa": "SDD-FIQA", "ser-fiq": "SER-FIQA", "vit-fiqa": "ViT-FIQA",
    "vitnt-fiqa": "ViTNT-FIQA",
}

# Reference palette: categorical slots 1-3, diverging pair with a neutral midpoint.
C_EDC = "#1baf7a"
C_RCE = "#eb6834"
C_OURS = "#2a78d6"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e6e5e0"
# P(row beats column), following the decided/undecided rule of the paper (0.05 / 0.95):
# red = decided, row loses; green = undecided, monotonic from pale (just above 0.05) to dark
# (just below 0.95), so every value has its own colour; blue = decided, row wins.
P_BEATS_CMAP = LinearSegmentedColormap.from_list("p_beats", [
    (0.0, "#b2221f"), (0.05, "#e8817f"),
    (0.0501, "#e3f2b8"), (0.5, "#7cc46f"), (0.9499, "#1b6e33"),
    (0.95, "#8fb8e6"), (1.0, "#1f5fa8")])
P_BEATS_CMAP.set_bad("#5f5f5f")  # diagonal (method vs itself): dark grey

COLUMN_WIDTH = 3.3  # inches, single column of a two-column paper (CVPR / IEEE)


# ---------------------------------------------------------------------------
# Threshold at target FMR (exactly rce.py / edc.py rules, fast)
# ---------------------------------------------------------------------------

def fmr_threshold(imp_sims: np.ndarray, fmr_target: float, method: str) -> float:
    """Same value as find_threshold_for_fmr / find_threshold_closest_fmr in rce.py and
    edc.py. "roc" only sorts the top scores: every candidate whose FMR can be closest
    to the target lies there."""
    if method == "percentile":
        return float(np.percentile(imp_sims, (1.0 - fmr_target) * 100.0))
    n = imp_sims.size
    m = min(n, int(np.ceil(4 * fmr_target * n)) + 10)
    floor = np.partition(imp_sims, n - m)[n - m]
    top = np.sort(imp_sims[imp_sims >= floor])
    cand = np.unique(top)
    fmr = (top.size - np.searchsorted(top, cand, side="left")) / n
    return float(cand[np.argmin(np.abs(fmr - fmr_target))])


# ---------------------------------------------------------------------------
# Identity bootstrap
# ---------------------------------------------------------------------------

_IDENTITY = re.compile(r"^(.+)_\d+\.[A-Za-z]+$")


def image_identities(img_ids: list[str]) -> np.ndarray:
    """Integer identity code per image: file name without the trailing _<number>.<ext>."""
    names = []
    for img in img_ids:
        m = _IDENTITY.match(os.path.basename(str(img)))
        names.append(m.group(1) if m else f"__own__{img}")
    return pd.factorize(pd.Series(names))[0]


def identity_resample(identity: np.ndarray, idx1: np.ndarray, idx2: np.ndarray,
                      labels: np.ndarray, rng: np.random.Generator):
    """One identity-bootstrap draw, expanded to image copies.

    Returns (obs_img, i1, i2, pidx): the source image of every copy, the copy indices of
    both sides of every resampled pair, and the original pair index of every resampled
    pair (for its similarity and label).
    """
    k = int(identity.max()) + 1
    c = np.bincount(rng.integers(0, k, k), minlength=k)       # copies per identity
    w = c[identity]                                            # copies per image
    obs_img = np.repeat(np.arange(identity.size), w)
    offset = np.concatenate([[0], np.cumsum(w)[:-1]])          # first copy of each image

    ca, cb = c[identity[idx1]], c[identity[idx2]]
    genuine = labels == 1
    size = np.where(genuine, ca, ca * cb)                      # copies of every pair
    pidx = np.repeat(np.arange(labels.size), size)
    t = np.arange(pidx.size) - np.repeat(np.cumsum(size) - size, size)
    cb_p = cb[pidx]
    gen_p = genuine[pidx]
    k1 = np.where(gen_p, t, t // np.maximum(cb_p, 1))          # copy of the first image
    k2 = np.where(gen_p, t, t % np.maximum(cb_p, 1))           # copy of the second image
    return obs_img, offset[idx1[pidx]] + k1, offset[idx2[pidx]] + k2, pidx


# ---------------------------------------------------------------------------
# Protocols on one pair set
# ---------------------------------------------------------------------------

def edc_pauc(keep_steps, steps, i1, i2, sims, labels, fmr_target, threshold_method) -> float:
    """edc.py error_vs_reject + compute_pauc for one FIQA method on one pair set."""
    min_impostors = int(np.ceil(1.0 / fmr_target))
    is_imp = labels == 0
    rates, fnmrs = [], []
    for rate, keep in zip(steps, keep_steps):
        kept = keep[i1] & keep[i2]
        imp_k = kept & is_imp
        if np.count_nonzero(imp_k) < min_impostors:
            break
        thr = fmr_threshold(sims[imp_k], fmr_target, threshold_method)
        gen_sims = sims[kept & ~is_imp]
        rates.append(rate)
        fnmrs.append(np.mean(gen_sims < thr) if gen_sims.size else np.nan)
    rates, fnmrs = np.array(rates), np.array(fnmrs)
    mask = np.round(rates, 5) <= EDC_MAX_REJECT
    if mask.sum() < 2:
        return np.nan
    trap = getattr(np, "trapezoid", None) or np.trapz
    return float(trap(fnmrs[mask], rates[mask]))


def margin_harm_rank(harm, n, i1, i2, sims, labels) -> np.ndarray:
    """RCE harm ranking with margin tie-breaking, as defined in [1] (Eqs. 3-4) and written
    by rce.py to harm_ranking__*.csv: ascending harm, and among equal harm the larger
    margin m(i) = min genuine sim - max impostor sim ranks better (rank 1 = best). Images
    without a genuine or an impostor pair get m(i) = -inf. Remaining exact ties (equal
    harm and margin, e.g. copies of one image) receive their average rank."""
    gen, imp = labels == 1, labels == 0
    min_gen = np.full(n, np.inf)
    max_imp = np.full(n, -np.inf)
    np.minimum.at(min_gen, i1[gen], sims[gen])
    np.minimum.at(min_gen, i2[gen], sims[gen])
    np.maximum.at(max_imp, i1[imp], sims[imp])
    np.maximum.at(max_imp, i2[imp], sims[imp])
    margin = min_gen - max_imp
    margin = np.where(np.isfinite(min_gen) & np.isfinite(max_imp), margin, -np.inf)

    order = np.lexsort((-margin, harm))
    h, m = harm[order], margin[order]
    new_group = np.ones(n, dtype=bool)
    new_group[1:] = (h[1:] != h[:-1]) | (m[1:] != m[:-1])
    group = np.cumsum(new_group) - 1
    pos = np.arange(1, n + 1, dtype=float)
    mean_pos = np.bincount(group, weights=pos) / np.bincount(group)
    ranks = np.empty(n, dtype=float)
    ranks[order] = mean_pos[group]
    return ranks


def rce_scores(quality_ranks, n, i1, i2, sims, labels, fmr_target, threshold_method, alphas,
               ties="margin", harm_fn=None) -> dict:
    """Weighted RCE (Eq. 13) of every FIQA method on one pair set. ties = "margin": harm
    ranking with margin tie-breaking as in [1]; "midrank": average ranks as in rce.py's
    score (inflates RCE_0 when most images have zero harm). harm_fn: optional error
    contribution harm_fn(n, i1, i2, sims, labels, thr) -> (n,) array, e.g. the normalized
    one of ablation_normalization.py; default: the counts E_i = c_FN(i) + c_FA(i) of [1]."""
    thr = fmr_threshold(sims[labels == 0], fmr_target, threshold_method)
    if harm_fn is None:
        fail = ((labels == 1) & (sims < thr)) | ((labels == 0) & (sims >= thr))
        harm = harm_counts(n, i1, i2, fail)
    else:
        harm = harm_fn(n, i1, i2, sims, labels, thr)
    if ties == "margin":
        harm_rank = margin_harm_rank(harm, n, i1, i2, sims, labels)
    else:
        harm_rank = _rank_ascending(harm)
    return {(m, a): _weighted_spearman(harm_rank, rq, alpha=a)[0]
            for m, rq in quality_ranks.items() for a in alphas}


def run_job(job: dict) -> pd.DataFrame:
    """All samples of one dataset / FR model -> long table of scores."""
    dataset, fr_model = job["dataset"], job["fr_model"]
    fmr, thr_method, alphas = job["fmr"], job["threshold_method"], job["alphas"]

    # Same rng sequence as rce_prob_test1.run_one -> identical resampled pair sets.
    rng = np.random.default_rng(job["seed"])
    img_ids, idx1, idx2, labels, sims = load_job(job["emb_pkl"], job["pairs_csv"])
    n = len(img_ids)

    steps = [s for s in EDC_STEPS if np.round(s, 5) <= EDC_MAX_REJECT]
    quality = {}
    for m in job["methods"]:
        qpath = _quality_pkl_for_dataset(os.path.join(job["quality_dir"], m), dataset)
        if os.path.exists(qpath):
            quality[m] = _quality_scores_for_ids(load_quality_scores_pkl(qpath), img_ids)

    gen = np.flatnonzero(labels == 1)
    imp = np.flatnonzero(labels == 0)
    identity = image_identities(img_ids)
    rows = []

    def evaluate(sample: int, obs_img: np.ndarray, i1: np.ndarray, i2: np.ndarray, pidx: np.ndarray):
        """Both protocols on one sample: obs_img = source image of every sample image,
        i1 / i2 = sample-image indices of every pair, pidx = original pair index."""
        s, lab = sims[pidx], labels[pidx]
        n_obs = obs_img.size
        quality_ranks = {}
        for m, q_img in quality.items():
            q = q_img[obs_img]
            quality_ranks[m] = _rank_descending(q)          # rce.py R_Q
            order = np.argsort(q)                           # edc.py: low -> high quality
            keep_steps = []
            for rate in steps:
                keep = np.ones(n_obs, dtype=bool)
                keep[order[:int(rate * n_obs)]] = False
                keep_steps.append(keep)
            rows.append((dataset, fr_model, "EDC", np.nan, m, sample,
                         edc_pauc(keep_steps, steps, i1, i2, s, lab, fmr, thr_method)))
        for (m, a), v in rce_scores(quality_ranks, n_obs, i1, i2, s, lab, fmr, thr_method, alphas,
                                    ties=job["rce_ties"], harm_fn=job.get("harm_fn")).items():
            rows.append((dataset, fr_model, "RCE", a, m, sample, v))

    all_pairs = np.arange(labels.size)
    evaluate(-1, np.arange(n), idx1, idx2, all_pairs)
    for b in range(job["n_samples"]):
        if job["resample"] == "identities":
            obs_img, i1, i2, pidx = identity_resample(identity, idx1, idx2, labels, rng)
            evaluate(b, obs_img, i1, i2, pidx)
        else:
            g = gen[rng.integers(0, gen.size, gen.size)] if job["resample"] in ("gen", "both") else gen
            mi = imp[rng.integers(0, imp.size, imp.size)] if job["resample"] in ("imp", "both") else imp
            rng.random(n)  # tie-breaking draw of rce_prob_test1, keeps the sequences aligned
            pidx = np.concatenate([g, mi])
            evaluate(b, np.arange(n), idx1[pidx], idx2[pidx], pidx)
        if (b + 1) % 50 == 0:
            print(f"[{dataset}/{fr_model}] {b + 1}/{job['n_samples']}", flush=True)

    return pd.DataFrame(rows, columns=["dataset", "fr_model", "protocol", "alpha", "method", "sample", "score"])


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def protocol_key(protocol: str, alpha) -> str:
    return "EDC" if protocol == "EDC" else f"RCE_g{format(float(alpha), 'g')}"


def higher_is_better(key: str) -> bool:
    return key.startswith("RCE")


def statistics(scores: pd.DataFrame):
    scores = scores.copy()
    scores["key"] = [protocol_key(p, a) for p, a in zip(scores.protocol, scores.alpha)]
    method_rows, pair_rows, proto_rows = [], [], []
    for (ds, fr, key), g in scores.groupby(["dataset", "fr_model", "key"]):
        wide = g.pivot(index="sample", columns="method", values="score")
        better_sign = 1.0 if higher_is_better(key) else -1.0
        ranks = (-better_sign * wide).rank(axis=1, method="average")      # 1 = best
        obs, res = wide.loc[-1], wide.drop(index=-1)
        r_obs, r_res = ranks.loc[-1], ranks.drop(index=-1)
        for m in wide.columns:
            method_rows.append({
                "dataset": ds, "fr_model": fr, "protocol": key, "method": m,
                "observed": obs[m], "observed_rank": r_obs[m],
                "mean": res[m].mean(), "q025": res[m].quantile(0.025), "q975": res[m].quantile(0.975),
                "rank_median": r_res[m].median(), "rank_q05": r_res[m].quantile(0.05),
                "rank_q95": r_res[m].quantile(0.95), "p_rank1": float((r_res[m] == 1).mean()),
            })
        methods = list(wide.columns)
        s = better_sign * res.to_numpy()
        undecided, n_pairs = 0, 0
        for i, a in enumerate(methods):
            for j, b in enumerate(methods):
                if i == j:
                    continue
                p = float(np.mean(s[:, i] > s[:, j]) + 0.5 * np.mean(s[:, i] == s[:, j]))
                pair_rows.append({"dataset": ds, "fr_model": fr, "protocol": key, "a": a, "b": b, "p_a_beats_b": p})
                if i < j:
                    n_pairs += 1
                    undecided += 0.05 < p < 0.95
        width = (r_res.quantile(0.95) - r_res.quantile(0.05)).mean()
        proto_rows.append({"dataset": ds, "fr_model": fr, "protocol": key, "n_methods": len(methods),
                           "n_samples": len(res), "undecided_pairs_pct": 100.0 * undecided / max(n_pairs, 1),
                           "mean_rank_interval_90": float(width)})
    return pd.DataFrame(method_rows), pd.DataFrame(pair_rows), pd.DataFrame(proto_rows)


# ---------------------------------------------------------------------------
# Figures (one plot per file, publication style)
# ---------------------------------------------------------------------------

def set_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 7, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "axes.edgecolor": INK_2,
        "xtick.color": INK_2, "ytick.color": INK_2, "axes.labelcolor": INK,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300,
    })


def clean_axes(ax, grid_axis="x"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if grid_axis:
        ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)


def save(fig, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path + ".pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(path + ".png", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def gamma_label(key: str) -> str:
    return key.split("_g")[1]


def rce_label(key: str) -> str:
    """RCE score name as in the paper, e.g. RCE_0."""
    return rf"$\mathrm{{RCE}}_{{{gamma_label(key)}}}$"


class HandlerCappedInterval(HandlerBase):
    """Legend icon of a capped interval as drawn in the plot: a line with an end cap on each side."""

    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        color = orig_handle.get_color()
        x0, x1 = -xdescent, -xdescent + width
        y, h = height / 2 - ydescent, height * 0.6
        artists = [Line2D([x0, x1], [y, y], color=color, linewidth=1.6, alpha=0.55, solid_capstyle="butt"),
                   Line2D([x0, x0], [y - h, y + h], color=color, linewidth=1.0),
                   Line2D([x1, x1], [y - h, y + h], color=color, linewidth=1.0)]
        for a in artists:
            a.set_transform(trans)
        return artists


def fig_rank_intervals(ms: pd.DataFrame, rce_key: str, path: str):
    """Rank of every FIQA method under EDC and RCE: dot = rank on the original test set, capped
    line = PURE-FIQA 90% rank interval. A zero-width interval (same rank in >= 90% of the
    resamples) shows as a vertical cap through the dot."""
    edc, rce = ms[ms.protocol == "EDC"].set_index("method"), ms[ms.protocol == rce_key].set_index("method")
    order = rce.sort_values(["observed_rank"]).index.tolist()[::-1]
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 0.23 * len(order) + 0.95))
    clean_axes(ax)
    offset, cap = 0.22, 0.19         # row offset of the two protocols, half-height of the end caps
    for k, m in enumerate(order):
        for d, (tab, color) in zip((offset, -offset), ((edc, C_EDC), (rce, C_RCE))):
            r = tab.loc[m]
            y = k + d
            ax.plot([r.rank_q05, r.rank_q95], [y, y], color=color, linewidth=1.6,
                    solid_capstyle="butt", alpha=0.55, zorder=2)
            for x in (r.rank_q05, r.rank_q95):
                ax.plot([x, x], [y - cap, y + cap], color=color, linewidth=1.0, zorder=2)
            ax.plot(r.observed_rank, y, "o", markersize=3.4, color=color,
                    markeredgecolor="white", markeredgewidth=0.5, zorder=3)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([METHOD_NAMES.get(m, m) for m in order])
    ax.set_xlim(0.5, len(order) + 0.5)
    ax.set_xticks([1] + list(range(5, len(order) + 1, 5)))
    ax.set_ylim(-0.6, len(order) - 0.4)

    # Separate entries for the dot and the interval of each protocol, drawn like the plot marks.
    def dot(color):
        return Line2D([0], [0], color=color, marker="o", markersize=3.4, markeredgecolor="white",
                      markeredgewidth=0.5, linestyle="")

    rce_name = rce_label(rce_key)
    intervals = [Line2D([0], [0], color=C_EDC), Line2D([0], [0], color=C_RCE)]
    handles = [dot(C_EDC), dot(C_RCE)] + intervals
    labels = ["EDC: rank", f"{rce_name}: rank", "EDC: PURE-FIQA 90% interval", f"{rce_name}: PURE-FIQA 90% interval"]
    ax.legend(handles, labels, handler_map={h: HandlerCappedInterval() for h in intervals},
              loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False,
              handlelength=1.8, handletextpad=0.5, columnspacing=1.2, fontsize=6.5)
    save(fig, path)


def fig_ours_leaderboard(ms: pd.DataFrame, rce_key: str, path: str):
    r = ms[ms.protocol == rce_key].sort_values("mean")
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 0.17 * len(r) + 0.75))
    clean_axes(ax)
    y = np.arange(len(r))
    ax.hlines(y, r.q025, r.q975, color=C_OURS, linewidth=1.6, alpha=0.55, zorder=2)
    ax.plot(r["mean"], y, "o", markersize=3.6, color=C_OURS, markeredgecolor="white", markeredgewidth=0.5, zorder=3)
    ax.plot(r.observed, y, "D", markersize=3.0, markerfacecolor="none", markeredgecolor=C_RCE,
            markeredgewidth=0.8, zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels([METHOD_NAMES.get(m, m) for m in r.method])
    ax.set_ylim(-0.6, len(r) - 0.4)
    ax.set_xlabel(rce_label(rce_key))
    # One entry per mark: blue dot = bootstrap mean, blue line = 95% interval, orange diamond = RCE score.
    handles = [Line2D([0], [0], color=C_OURS, marker="o", markersize=3.6, markeredgecolor="white",
                      markeredgewidth=0.5, linestyle="", label="PURE-FIQA mean"),
               Line2D([0], [0], color=C_OURS, linewidth=1.6, alpha=0.55, solid_capstyle="round",
                      label="PURE-FIQA 95% interval"),
               Line2D([0], [0], color=C_RCE, marker="D", markersize=3.0, markerfacecolor="none",
                      markeredgewidth=0.8, linestyle="", label=rce_label(rce_key))]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False,
              handlelength=1.4, handletextpad=0.4, columnspacing=0.9)
    save(fig, path)


def fig_pairwise(pw: pd.DataFrame, ms: pd.DataFrame, key: str, path: str):
    """P(row method beats column method) as a heat map; methods in observed-rank order."""
    order = ms[ms.protocol == key].sort_values("observed_rank").method.tolist()
    mat = pw[pw.protocol == key].pivot(index="a", columns="b", values="p_a_beats_b").reindex(index=order, columns=order)
    values = np.ma.masked_invalid(mat.to_numpy(dtype=float))   # diagonal is NaN -> grey
    labels = [METHOD_NAMES.get(m, m) for m in order]
    n = len(order)
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, COLUMN_WIDTH * 0.92))
    # Vector cells with a white border around each cell.
    mesh = ax.pcolormesh(np.arange(n + 1), np.arange(n + 1), values, cmap=P_BEATS_CMAP, vmin=0, vmax=1,
                         edgecolors="white", linewidth=0.6)
    ax.set_xlim(0, n)
    ax.set_ylim(n, 0)
    ax.set_aspect("equal")
    ax.set_xticks(np.arange(n) + 0.5)
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_yticks(np.arange(n) + 0.5)
    ax.set_yticklabels(labels, fontsize=6)
    ax.tick_params(length=0, pad=2)
    for side in ax.spines.values():
        side.set_visible(False)
    cb = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.03, ticks=[0, 0.05, 0.5, 0.95, 1])
    cb.ax.tick_params(labelsize=6, length=2)
    cb.outline.set_linewidth(0.4)
    save(fig, path)


# FR models in the per-dataset summary figures: own colour and marker each (colours differ from
# the protocol colours used elsewhere; the marker is a redundant cue for print / colour vision).
FR_STYLE = {"adaface": ("#2a78d6", "o"), "arcface_o": ("#e87ba4", "s"),
            "magface": ("#eda100", "^"), "swinface": ("#6250d6", "D")}


def fr_summary_figure(groups, column: str, ylabel: str, path: str, datasets=None, label_size=6.5,
                      dataset_y=-0.12):
    """Per dataset, one cluster per group (protocol); in every cluster one marker per FR model
    (colour + shape, see FR_STYLE) and a black bar for the mean over the FR models.

    groups: list of (summary table, protocol key, cluster label)."""
    tables = pd.concat([t for t, _, _ in groups])
    if datasets is None:
        datasets = [d for d in ["lfw", "xqlfw", "calfw", "cplfw", "cfp-fp", "adience", "agedb"]
                    if d in set(tables.dataset)]
    fr_models = [f for f in FR_STYLE if f in set(tables.fr_model)]
    fr_models += sorted(set(tables.fr_model) - set(fr_models))
    n_g = len(groups)
    spread = 0.8 / n_g                                   # cluster spacing inside a dataset
    cluster_off = (np.arange(n_g) - (n_g - 1) / 2) * spread
    fr_off = (np.arange(len(fr_models)) - (len(fr_models) - 1) / 2) * spread * 0.18

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 2.5))
    clean_axes(ax, grid_axis="y")
    ticks, tick_labels = [], []
    for x, ds in enumerate(datasets):
        for c_off, (table, key, label) in zip(cluster_off, groups):
            g = table[(table.dataset == ds) & (table.protocol == key)].set_index("fr_model")[column]
            if g.empty:
                continue
            for f_off, fr in zip(fr_off, fr_models):
                if fr in g.index:
                    color, marker = FR_STYLE.get(fr, (INK_2, "o"))
                    ax.plot(x + c_off + f_off, g[fr], marker=marker, linestyle="", markersize=3.4,
                            color=color, markeredgecolor="white", markeredgewidth=0.4, zorder=3)
            half = spread * 0.32
            ax.plot([x + c_off - half, x + c_off + half], [g.mean()] * 2, color=INK, linewidth=1.3, zorder=4)
            ticks.append(x + c_off)
            tick_labels.append(label)
        if x > 0:
            ax.axvline(x - 0.5, color=GRID, linewidth=0.6, zorder=0)
    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels, fontsize=label_size, linespacing=0.9)
    ax.tick_params(axis="x", length=0, pad=2)
    for x, ds in enumerate(datasets):                    # dataset names below the cluster labels
        ax.text(x, dataset_y, DATASET_NAMES.get(ds, ds), transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=8, color=INK)
    ax.set_xlim(-0.5, len(datasets) - 0.5)
    ax.set_ylim(bottom=0)
    ax.set_ylabel(ylabel)
    handles = [Line2D([0], [0], color=FR_STYLE.get(fr, (INK_2, "o"))[0], marker=FR_STYLE.get(fr, (INK_2, "o"))[1],
                      linestyle="", markersize=3.6, markeredgecolor="white", markeredgewidth=0.4)
               for fr in fr_models]
    handles.append(Line2D([0], [0], color=INK, linewidth=1.3))
    labels = [FR_NAMES.get(fr, fr) for fr in fr_models] + ["mean over FR models"]
    ax.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=5, frameon=False,
              handlelength=1.0, handletextpad=0.3, columnspacing=0.8, fontsize=6.5)
    save(fig, path)


def fig_protocol_summary(ps: pd.DataFrame, rce_key: str, column: str, ylabel: str, path: str):
    fr_summary_figure([(ps, "EDC", "EDC"), (ps, rce_key, rce_label(rce_key))], column, ylabel, path)


def make_figures(out_root: str, ms: pd.DataFrame, pw: pd.DataFrame, ps: pd.DataFrame, plot_alphas: list[float]):
    set_style()
    fig_root = os.path.join(out_root, "figures")
    rce_keys = [protocol_key("RCE", a) for a in plot_alphas if protocol_key("RCE", a) in set(ms.protocol)]
    for (ds, fr), g in ms.groupby(["dataset", "fr_model"]):
        base = os.path.join(fig_root, ds, fr)
        p = pw[(pw.dataset == ds) & (pw.fr_model == fr)]
        fig_pairwise(p, g, "EDC", os.path.join(base, "pairwise_EDC"))
        for key in rce_keys:
            tag = "g" + gamma_label(key)
            fig_rank_intervals(g, key, os.path.join(base, f"rank_intervals_{tag}"))
            fig_ours_leaderboard(g, key, os.path.join(base, f"pure_fiqa_leaderboard_{tag}"))
            fig_pairwise(p, g, key, os.path.join(base, f"pairwise_RCE_{tag}"))
    for key in rce_keys:
        tag = "g" + gamma_label(key)
        fig_protocol_summary(ps, key, "undecided_pairs_pct", "Undecided FIQA method pairs (%)",
                             os.path.join(fig_root, "summary", f"undecided_pairs_{tag}"))
        fig_protocol_summary(ps, key, "mean_rank_interval_90", "Mean 90% rank interval (ranks)",
                             os.path.join(fig_root, "summary", f"rank_interval_width_{tag}"))
    print(f"figures written under {fig_root}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Stability of EDC / RCE FIQA leaderboards under resampling")
    p.add_argument("--fr-features-root", default="data/FR_models")
    p.add_argument("--quality-dir", default="data/FIQA_models")
    p.add_argument("--ca-fiqa-data-root", default="data/pairs/pairs")
    p.add_argument("--out-root", default=os.path.join("output", "protocol_stability"))
    p.add_argument("--datasets", default="lfw,xqlfw,calfw,cplfw",
                   help="cfp-fp is excluded by default (broken image-ID mapping, 10 images)")
    p.add_argument("--fr-models", default="adaface,arcface_o,magface,swinface")
    p.add_argument("--methods", default="", help="FIQA methods (default: all found)")
    p.add_argument("--rce-alphas", default="0,3", help="RCE gamma values (paper: 0 and 3)")
    p.add_argument("--rce-ties", choices=["margin", "midrank"], default="margin",
                   help="Ties in the RCE harm ranking: margin tie-breaking as in the RCE paper "
                        "(default) or average ranks as in rce.py's score")
    p.add_argument("--plot-alphas", default="", help="RCE gammas to plot (default: all computed)")
    p.add_argument("--fmr", type=float, default=1e-3)
    p.add_argument("--threshold-method", choices=["percentile", "roc"], default="percentile")
    p.add_argument("--resample", choices=["identities", "imp", "gen", "both"], default="identities",
                   help="identities = cluster bootstrap over identities (default); "
                        "gen/imp/both = pair bootstrap of rce_prob_test1.py")
    p.add_argument("--n-samples", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=1, help="Parallel dataset / FR-model jobs")
    p.add_argument("--plot-only", action="store_true", help="Redraw figures from {out_root}/scores.csv")
    args = p.parse_args()

    split = lambda s: [x.strip() for x in str(s).split(",") if x.strip()]
    alphas = [float(a) for a in split(args.rce_alphas)]
    plot_alphas = [float(a) for a in split(args.plot_alphas)] or alphas
    scores_csv = os.path.join(args.out_root, "scores.csv")

    if args.plot_only:
        scores = pd.read_csv(scores_csv)
    else:
        methods = split(args.methods) or sorted(
            d for d in os.listdir(args.quality_dir)
            if os.path.isdir(os.path.join(args.quality_dir, d)) and not d.startswith(".")
        )
        found = {(d, m): (pkl, pairs) for d, m, _, pkl, pairs in
                 list_jobs(args.fr_features_root, args.ca_fiqa_data_root, run_all=True)}
        jobs = []
        for ds in split(args.datasets):
            for fr in split(args.fr_models):
                if (ds, fr) not in found:
                    print(f"[skip] {ds}/{fr}: no embeddings or pairs")
                    continue
                jobs.append({"dataset": ds, "fr_model": fr, "emb_pkl": found[(ds, fr)][0],
                             "pairs_csv": found[(ds, fr)][1], "quality_dir": args.quality_dir,
                             "methods": methods, "alphas": alphas, "rce_ties": args.rce_ties, "fmr": args.fmr,
                             "threshold_method": args.threshold_method, "resample": args.resample,
                             "n_samples": args.n_samples, "seed": args.seed})
        print(f"{len(jobs)} jobs, B={args.n_samples}, {args.workers} workers", flush=True)
        if args.workers > 1:
            with Pool(args.workers) as pool:
                frames = pool.map(run_job, jobs, chunksize=1)
        else:
            frames = [run_job(j) for j in jobs]
        scores = pd.concat(frames, ignore_index=True)
        os.makedirs(args.out_root, exist_ok=True)
        scores.to_csv(scores_csv, index=False)
        print(f"saved {scores_csv}")

    ms, pw, ps = statistics(scores)
    ms.to_csv(os.path.join(args.out_root, "method_summary.csv"), index=False)
    pw.to_csv(os.path.join(args.out_root, "pairwise.csv"), index=False)
    ps.to_csv(os.path.join(args.out_root, "protocol_summary.csv"), index=False)
    print(ps.groupby(["dataset", "protocol"])[["undecided_pairs_pct", "mean_rank_interval_90"]].mean().round(1))
    make_figures(args.out_root, ms, pw, ps, plot_alphas)


if __name__ == "__main__":
    main()
