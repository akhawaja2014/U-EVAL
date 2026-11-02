#!/usr/bin/env python3
"""PIT histograms of all FIQA methods for the PURE-FIQA consistency diagnostic.

One figure per dataset and FR model (one plot per file, PDF + PNG): the density of the PIT
values u_i(R_Q) of every FIQA method, drawn as one line per method (histogram with 20 bins of
width 0.05, plotted at the bin centres). Under H0 (R_Q a plausible draw of the reference
ranking) every line would be flat at density 1 (dashed grey line). The shaded bands are the
tails u < 0.05 and u > 0.95 that define phi_low and phi_high in the paper.

Input:  output/rce_prob_test1/{dataset}/{fr_model}/pit_values.csv  (rce_prob_test1.py, joint pair bootstrap)
Output: output/rce_prob_test1/figures_pit/{fr_model}/pit_{dataset}.pdf/.png

Run:
    python plot_pit_lines.py
    python plot_pit_lines.py --fr-models adaface --datasets lfw,xqlfw
"""

import argparse
import os

import matplotlib  # type: ignore[reportMissingImports]

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # type: ignore[reportMissingImports]
from matplotlib.lines import Line2D  # type: ignore[reportMissingImports]
from matplotlib.patches import Patch  # type: ignore[reportMissingImports]
import numpy as np  # type: ignore[reportMissingImports]
import pandas as pd

from protocol_stability import COLUMN_WIDTH, GRID, INK_2, METHOD_NAMES, clean_axes, save, set_style

# 9 distinct hues (reference palette slots 1-8 + dark grey), each used solid and dashed:
# 18 distinguishable styles for the 17 methods.
HUES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948", "#52514e"]
STYLES = [(h, "-") for h in HUES] + [(h, (0, (3.2, 1.6))) for h in HUES]
BINS = np.linspace(0.0, 1.0, 21)  # width 0.05, aligned with the 0.05 / 0.95 tails


def plot_dataset(pit: pd.DataFrame, methods: list[str], path: str):
    centers = 0.5 * (BINS[:-1] + BINS[1:])
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 3.55))
    clean_axes(ax, grid_axis="y")
    ax.axvspan(0.0, 0.05, color=GRID, alpha=0.8, linewidth=0, zorder=0)
    ax.axvspan(0.95, 1.0, color=GRID, alpha=0.8, linewidth=0, zorder=0)
    ax.axhline(1.0, color=INK_2, linewidth=0.9, linestyle=(0, (4, 2)), zorder=1)

    handles = []
    for (color, ls), m in zip(STYLES, methods):
        density, _ = np.histogram(pit[m].dropna(), bins=BINS, density=True)
        ax.plot(centers, density, color=color, linestyle=ls, linewidth=1.0, zorder=2)
        handles.append(Line2D([0], [0], color=color, linestyle=ls, linewidth=1.0, label=METHOD_NAMES.get(m, m)))
    handles.append(Line2D([0], [0], color=INK_2, linestyle=(0, (4, 2)), linewidth=0.9, label=r"uniform ($H_0$)"))
    handles.append(Patch(facecolor=GRID, edgecolor="none", label=r"tails $u<0.05$, $u>0.95$"))

    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xticklabels(["0", "0.25", "0.5", "0.75", "1"])
    ax.set_xlabel(r"PIT value $u_i$")
    ax.set_ylabel("Density")
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3, frameon=False,
              fontsize=6, handlelength=2.2, handletextpad=0.5, columnspacing=0.9, labelspacing=0.35)
    save(fig, path)


def main():
    p = argparse.ArgumentParser(description="PIT histograms (one line per FIQA method), one figure per dataset")
    p.add_argument("--results-root", default=os.path.join("output", "rce_prob_test1"))
    p.add_argument("--datasets", default="lfw,xqlfw,calfw,cplfw")
    p.add_argument("--fr-models", default="adaface,arcface_o,magface,swinface")
    p.add_argument("--out-dir", default=os.path.join("output", "rce_prob_test1", "figures_pit"))
    args = p.parse_args()

    set_style()
    split = lambda s: [x.strip() for x in s.split(",") if x.strip()]
    for fr in split(args.fr_models):
        for ds in split(args.datasets):
            src = os.path.join(args.results_root, ds, fr, "pit_values.csv")
            if not os.path.exists(src):
                print(f"[skip] {src}")
                continue
            pit = pd.read_csv(src)
            methods = sorted((c for c in pit.columns if c in METHOD_NAMES), key=lambda m: METHOD_NAMES[m].lower())
            out = os.path.join(args.out_dir, fr, f"pit_{ds}")
            plot_dataset(pit, methods, out)
            print(f"saved {out}.pdf/.png")


if __name__ == "__main__":
    main()
