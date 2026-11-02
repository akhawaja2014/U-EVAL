#!/usr/bin/env python3
"""Compare rce.py Spearman with the rank-PIT KS test (rce_prob_test1.py).

Figures (PNG + PDF) written to {results_root}/figures/:

  1. fig_scatter_rho_vs_ks   per dataset: each FIQA method's rce.py Spearman (x, higher
                             is better) vs its KS distance to the harm-rank distribution
                             (y, lower is better, axis inverted so "up" = better).
                             Points are means over FR models; whiskers span the FR models.
  2. fig_rank_slope          per dataset: method rank under rce.py Spearman (left) vs rank
                             under KS (right). Crossing lines = the two metrics disagree.
  3. fig_pit_hist            per dataset: PIT histograms of the best and worst method (by KS)
                             for one FR model. Flat at 1 = consistent with the harm
                             distribution; mass at 0 = quality ranks the image better than its
                             harm allows, mass at 1 = worse.

Run:

    python plot_rce_vs_pit.py --results-root output/rce_prob_test1
"""

import argparse
import os

import matplotlib  # type: ignore[reportMissingImports]

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # type: ignore[reportMissingImports]
import numpy as np  # type: ignore[reportMissingImports]
import pandas as pd
from scipy.stats import spearmanr  # type: ignore[reportMissingImports]

# Reference palette (categorical slots 1-2) + neutral inks.
BLUE = "#2a78d6"
ORANGE = "#eb6834"
GRAY = "#b9b8b2"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e6e5e0"

DATASET_NAMES = {"calfw": "CALFW", "cplfw": "CPLFW", "lfw": "LFW", "xqlfw": "XQLFW", "cfp-fp": "CFP-FP"}
METHOD_NAMES = {
    "clib-fiqa": "CLIB-FIQA", "cr-fiqa": "CR-FIQA", "diffiqa_r": "DifFIQA(R)",
    "ediffiqa_l": "eDifFIQA(L)", "faceqan": "FaceQAN", "faceqgen": "FaceQGen",
    "faceqnet": "FaceQnet", "froq_ada": "FROQ", "grafiqs": "GraFIQs",
    "lightqnet": "LightQNet", "magface": "MagFace", "pcnet": "PCNet", "pfe": "PFE",
    "sdd-fiqa": "SDD-FIQA", "ser-fiq": "SER-FIQ", "vit-fiqa": "ViT-FIQA",
    "vitnt-fiqa": "ViTNT-FIQA",
}


def style_axes(ax):
    ax.grid(True, color=GRID, linewidth=0.6, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)
        ax.spines[side].set_linewidth(0.6)
    ax.tick_params(colors=INK_2, labelsize=7, length=2)


def save(fig, out_dir, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"{name}.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {os.path.join(out_dir, name)}.png/.pdf")


def per_method_means(df: pd.DataFrame) -> pd.DataFrame:
    """Mean / min / max over FR models for each (dataset, method)."""
    g = df.groupby(["dataset", "method"])
    out = g[["rce_spearman", "ks_stat"]].mean()
    out["rho_lo"], out["rho_hi"] = g["rce_spearman"].min(), g["rce_spearman"].max()
    out["ks_lo"], out["ks_hi"] = g["ks_stat"].min(), g["ks_stat"].max()
    return out.reset_index()


def fig_scatter(m: pd.DataFrame, datasets: list[str], out_dir: str):
    fig, axes = plt.subplots(1, len(datasets), figsize=(3.2 * len(datasets), 3.4))
    axes = np.atleast_1d(axes)
    for ax, ds in zip(axes, datasets):
        d = m[m.dataset == ds]
        style_axes(ax)
        ax.errorbar(d.rce_spearman, d.ks_stat,
                    xerr=[d.rce_spearman - d.rho_lo, d.rho_hi - d.rce_spearman],
                    yerr=[d.ks_stat - d.ks_lo, d.ks_hi - d.ks_stat],
                    fmt="none", ecolor=GRAY, elinewidth=0.6, zorder=2)
        ax.scatter(d.rce_spearman, d.ks_stat, s=22, color=BLUE,
                   edgecolor="white", linewidth=0.8, zorder=3)
        # Label the best and worst method under each metric only.
        label_ids = {d.rce_spearman.idxmax(), d.rce_spearman.idxmin(), d.ks_stat.idxmin(), d.ks_stat.idxmax()}
        for i in label_ids:
            r = d.loc[i]
            ax.annotate(METHOD_NAMES.get(r.method, r.method), (r.rce_spearman, r.ks_stat),
                        xytext=(4, 3), textcoords="offset points", fontsize=6.5, color=INK)
        rho = spearmanr(d.rce_spearman, -d.ks_stat).correlation
        ax.set_title(f"{DATASET_NAMES.get(ds, ds)}   agreement ρ = {rho:+.2f}", fontsize=9, color=INK, loc="left")
        ax.invert_yaxis()  # up = better on both axes
        ax.set_xlabel("rce.py Spearman ρ  (→ better)", fontsize=8, color=INK_2)
    axes[0].set_ylabel("Rank-PIT KS distance  (↑ better)", fontsize=8, color=INK_2)
    fig.suptitle("FIQA methods: fixed-pair Spearman vs. distance to the harm-rank distribution "
                 "(mean over FR models, whiskers = FR-model range)", fontsize=9, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    save(fig, out_dir, "fig_scatter_rho_vs_ks")


def fig_slope(m: pd.DataFrame, datasets: list[str], out_dir: str, highlight_shift: int):
    fig, axes = plt.subplots(1, len(datasets), figsize=(3.0 * len(datasets), 5.2))
    axes = np.atleast_1d(axes)
    for ax, ds in zip(axes, datasets):
        d = m[m.dataset == ds].copy()
        n = len(d)
        d["r_rho"] = d.rce_spearman.rank(ascending=False, method="first")
        d["r_ks"] = d.ks_stat.rank(ascending=True, method="first")
        moved = (d.r_rho - d.r_ks).abs() >= highlight_shift
        for _, r in d.sort_values("r_rho").iterrows():
            big = abs(r.r_rho - r.r_ks) >= highlight_shift
            ax.plot([0, 1], [r.r_rho, r.r_ks], color=ORANGE if big else GRAY,
                    linewidth=2 if big else 1.2, zorder=3 if big else 2, solid_capstyle="round")
            name = METHOD_NAMES.get(r.method, r.method)
            ax.text(-0.04, r.r_rho, name, ha="right", va="center", fontsize=6.5, color=INK if big else INK_2)
            ax.text(1.04, r.r_ks, name, ha="left", va="center", fontsize=6.5, color=INK if big else INK_2)
        ax.set_xlim(-0.9, 1.9)
        ax.set_ylim(n + 0.7, 0.3)
        ax.axis("off")
        ax.text(0, 0, "rce.py ρ", ha="center", va="bottom", fontsize=7.5, color=INK_2)
        ax.text(1, 0, "Rank-PIT KS", ha="center", va="bottom", fontsize=7.5, color=INK_2)
        rho = spearmanr(d.r_rho, d.r_ks).correlation
        ax.set_title(f"{DATASET_NAMES.get(ds, ds)}   ρ = {rho:+.2f},  {int(moved.sum())} moved ≥{highlight_shift}",
                     fontsize=9, color=INK, pad=14)
    fig.suptitle(f"Method rank (1 = best) under each metric, mean over FR models; "
                 f"orange = rank change ≥ {highlight_shift}", fontsize=9, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    save(fig, out_dir, "fig_rank_slope")


def fig_pit(df: pd.DataFrame, results_root: str, datasets: list[str], fr_model: str, out_dir: str, bins: int):
    fig, axes = plt.subplots(2, len(datasets), figsize=(3.0 * len(datasets), 4.4), sharey="row")
    axes = np.asarray(axes).reshape(2, len(datasets))
    edges = np.linspace(0, 1, bins + 1)
    for j, ds in enumerate(datasets):
        rows = df[(df.dataset == ds) & (df.fr_model == fr_model)].sort_values("ks_stat")
        pit = pd.read_csv(os.path.join(results_root, ds, fr_model, "pit_values.csv"))
        for i, (label, r) in enumerate([("best", rows.iloc[0]), ("worst", rows.iloc[-1])]):
            ax = axes[i, j]
            style_axes(ax)
            dens, _ = np.histogram(pit[r.method], bins=edges, density=True)
            ax.bar(edges[:-1], dens, width=np.diff(edges), align="edge", color=BLUE,
                   edgecolor="white", linewidth=0.8, zorder=2)
            ax.axhline(1.0, color=INK_2, linewidth=0.8, linestyle=(0, (3, 2)), zorder=3)
            ax.set_xlim(0, 1)
            ax.set_title(f"{DATASET_NAMES.get(ds, ds)} · {label}: {METHOD_NAMES.get(r.method, r.method)}"
                         f"  (KS {r.ks_stat:.3f})", fontsize=7.5, color=INK, loc="left")
            if i == 1:
                ax.set_xlabel("PIT u  (0 = ranked better than harm allows)", fontsize=7, color=INK_2)
        axes[0, 0].set_ylabel("density", fontsize=8, color=INK_2)
        axes[1, 0].set_ylabel("density", fontsize=8, color=INK_2)
    fig.suptitle(f"PIT of the quality ranking under the harm-rank distribution ({fr_model}); "
                 "dashed line = perfectly consistent", fontsize=9, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    save(fig, out_dir, "fig_pit_hist")


def main():
    p = argparse.ArgumentParser(description="Plot rce.py Spearman vs rank-PIT KS results")
    p.add_argument("--results-root", default=os.path.join("output", "rce_prob_test1"))
    p.add_argument("--datasets", default="calfw,cplfw,lfw,xqlfw",
                   help="Datasets to plot (cfp-fp excluded by default: broken image-ID mapping)")
    p.add_argument("--pit-fr-model", default="adaface", help="FR model for the PIT histograms")
    p.add_argument("--highlight-shift", type=int, default=5, help="Rank change that counts as a big move")
    p.add_argument("--bins", type=int, default=20)
    args = p.parse_args()

    df = pd.read_csv(os.path.join(args.results_root, "pit_test_all.csv"))
    datasets = [d for d in args.datasets.split(",") if d in set(df.dataset)]
    out_dir = os.path.join(args.results_root, "figures")
    os.makedirs(out_dir, exist_ok=True)

    m = per_method_means(df)
    fig_scatter(m, datasets, out_dir)
    fig_slope(m, datasets, out_dir, args.highlight_shift)
    fig_pit(df, args.results_root, datasets, args.pit_fr_model, out_dir, args.bins)


if __name__ == "__main__":
    main()
