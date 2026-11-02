#!/usr/bin/env python3
"""Ablation: PURE-FIQA with a normalized error contribution E_i.

The error contribution of RCE [1] adds raw counts, E_i = c_FN(i) + c_FA(i). It ignores (a) how
many comparisons an image takes part in (exposure: 3 errors in 20 pairs count as much as 3 in
200) and (b) that false matches and false non-matches occur at very different overall rates
(FMR is fixed at 1e-3, FNMR is far larger). This ablation replaces E_i by a normalized error
contribution and keeps every other step of PURE-FIQA unchanged: identity bootstrap (same seed,
hence the same resamples), RCE with margin tie-breaking, EDC, and all statistics.

Normalized error contribution (Steps 1-3), computed on every evaluation set (observed pairs and
every resample, with the pair multiplicities n_p of that set):

  Step 1  rates instead of counts: exposures n_G(i), n_I(i) = genuine / impostor pairs of image i
          (counted with multiplicity), error rates c_FN(i)/n_G(i) and c_FA(i)/n_I(i).
  Step 2  balance of the two error types: each rate is divided by the overall rate of its type
          on the same set, FNMR = sum c_FN / sum n_G and FMR = sum c_FA / sum n_I, giving a
          relative risk (1 = average image).
  Step 3  shrinkage of small counts: alpha virtual comparisons at the overall rate are added,
          r~(i) = (c(i) + alpha * rate) / (n(i) + alpha) with alpha = s / rate, i.e. the prior is
          worth s expected errors (default s = 1). Equivalently, the relative risk is
              RR(i) = r~(i) / rate = (c(i) + s) / (n(i) * rate + s)
          = (observed errors + s) / (expected errors + s). An image without comparisons of a
          type gets RR = 1 for that type; if a set contains no error of a type, RR = 1 for all.

  E_norm(i) = lambda * RR_FN(i) + (1 - lambda) * RR_FA(i),   lambda = 0.5 by default
  (equal weight of both error types, as the lambda = 0.5 of FT-EDC in [1]).

Remaining exact ties in E_norm are broken by the margin, exactly as for the counts.

Run (on a compute node, see run_ablation_normalization.slurm):

    python ablation_normalization.py --threshold-method roc --rce-alphas 0,3 --n-samples 400 --workers 16

Redraw figures / tables from the saved scores without recomputing:

    python ablation_normalization.py --plot-only

Outputs, under {out_root} (default output/ablation_normalization/):
  scores.csv, method_summary.csv, pairwise.csv, protocol_summary.csv
                          same content and format as output/protocol_stability/, normalized E_i
  ablation_summary.csv    counts vs normalized per dataset / FR model / gamma
  figures/{dataset}/{fr_model}/   the PURE-FIQA figures of protocol_stability.py for normalized E_i
  figures/ablation/       counts vs normalized comparison, one plot per file (PDF + PNG):
      undecided_pairs_g{gamma}, rank_interval_width_g{gamma}   stability per dataset
      {dataset}/{fr_model}/scores_g{gamma}                     RCE_gamma per FIQA method
and paper/tables/tab_ablation_normalization.tex.

[1] B. Wani et al., Beyond error-vs-discard characteristic, arXiv:2607.22752, 2026.
"""

import argparse
import os
from functools import partial
from multiprocessing import Pool

import matplotlib  # type: ignore[reportMissingImports]

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # type: ignore[reportMissingImports]
from matplotlib.legend_handler import HandlerTuple  # type: ignore[reportMissingImports]
from matplotlib.lines import Line2D  # type: ignore[reportMissingImports]
import numpy as np  # type: ignore[reportMissingImports]
import pandas as pd
from scipy.stats import spearmanr  # type: ignore[reportMissingImports]

import protocol_stability as PS
from rce_prob_test1 import harm_counts, list_jobs

# Reference palette, categorical slots 1-3 (validated for all pairs).
C_EDC = "#1baf7a"
C_COUNTS = "#eb6834"
C_NORM = "#2a78d6"
DATASETS = ["lfw", "xqlfw", "calfw", "cplfw"]


# ---------------------------------------------------------------------------
# Normalized error contribution (Steps 1-3)
# ---------------------------------------------------------------------------

def relative_risk(errors: np.ndarray, exposure: np.ndarray, strength: float) -> np.ndarray:
    """Shrunk relative error risk (c + s) / (n * rate + s) of every image (Steps 1-3).

    errors[i]   = errors of image i (with multiplicity), exposure[i] = its comparisons of the
    same type. rate = overall error rate of that type on the same set; strength s = prior worth
    in expected errors (alpha = s / rate virtual comparisons)."""
    total_errors, total_exposure = errors.sum(), exposure.sum()
    if total_errors == 0 or total_exposure == 0:
        return np.ones(errors.size)            # no error of this type in the set: no information
    rate = total_errors / total_exposure
    return (errors + strength) / (exposure * rate + strength)


def normalized_harm(n, i1, i2, sims, labels, thr, lam=0.5, strength=1.0) -> np.ndarray:
    """E_norm(i) = lam * RR_FN(i) + (1 - lam) * RR_FA(i) at threshold thr (one evaluation set)."""
    gen, imp = labels == 1, labels == 0
    c_fn = harm_counts(n, i1, i2, gen & (sims < thr))
    c_fa = harm_counts(n, i1, i2, imp & (sims >= thr))
    n_g = harm_counts(n, i1, i2, gen)
    n_i = harm_counts(n, i1, i2, imp)
    return lam * relative_risk(c_fn, n_g, strength) + (1.0 - lam) * relative_risk(c_fa, n_i, strength)


# ---------------------------------------------------------------------------
# Ablation statistics and table
# ---------------------------------------------------------------------------

def ablation_summary(ms_c, ms_n, ps_c, ps_n, keys) -> pd.DataFrame:
    rows = []
    for (ds, fr), _ in ms_n.groupby(["dataset", "fr_model"]):
        for key in keys:
            c = ms_c[(ms_c.dataset == ds) & (ms_c.fr_model == fr) & (ms_c.protocol == key)].set_index("method")
            m = ms_n[(ms_n.dataset == ds) & (ms_n.fr_model == fr) & (ms_n.protocol == key)].set_index("method")
            if c.empty or m.empty:
                continue
            common = m.index.intersection(c.index)
            pc = ps_c[(ps_c.dataset == ds) & (ps_c.fr_model == fr) & (ps_c.protocol == key)].iloc[0]
            pn = ps_n[(ps_n.dataset == ds) & (ps_n.fr_model == fr) & (ps_n.protocol == key)].iloc[0]
            top_c = c.observed.idxmax()
            top_n = m.observed.idxmax()
            rows.append({
                "dataset": ds, "fr_model": fr, "protocol": key,
                "order_agreement": spearmanr(c.loc[common, "observed"], m.loc[common, "observed"])[0],
                "observed_min_counts": c.observed.min(), "observed_max_counts": c.observed.max(),
                "observed_min_norm": m.observed.min(), "observed_max_norm": m.observed.max(),
                "U_counts": pc.undecided_pairs_pct, "U_norm": pn.undecided_pairs_pct,
                "W_counts": pc.mean_rank_interval_90, "W_norm": pn.mean_rank_interval_90,
                "top_counts": top_c, "p_top_counts": c.loc[top_c, "p_rank1"],
                "top_norm": top_n, "p_top_norm": m.loc[top_n, "p_rank1"],
            })
    return pd.DataFrame(rows)


def write_table(summary: pd.DataFrame, path: str):
    """paper table: stability and leaderboard change, counts vs normalized E_i."""
    rows = []
    for ds in [d for d in DATASETS if d in set(summary.dataset)]:
        cells = []
        for key in ("RCE_g0", "RCE_g3"):
            g = summary[(summary.dataset == ds) & (summary.protocol == key)]
            cells.append(f"{g.U_counts.mean():.0f} & {g.U_norm.mean():.0f}")
        for key in ("RCE_g0", "RCE_g3"):
            g = summary[(summary.dataset == ds) & (summary.protocol == key)]
            cells.append(f"{g.order_agreement.mean():+.2f}")
        g = summary[(summary.dataset == ds) & (summary.protocol == "RCE_g0")]
        lead = g.top_norm.value_counts()
        name = PS.METHOD_NAMES.get(lead.index[0], lead.index[0])
        p = g[g.top_norm == lead.index[0]].p_top_norm
        cells.append(f"{name} ({lead.iloc[0]}/{len(g)}) & {p.min():.2f}--{p.max():.2f}")
        rows.append(f"{PS.DATASET_NAMES.get(ds, ds)} & " + " & ".join(cells) + r" \\")
    body = "\n".join("  " + r for r in rows)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(rf"""\begin{{table*}}[t]
\caption{{Ablation: raw error counts (C) vs.\ the normalized error contribution (N) in PURE-FIQA
(identity bootstrap, $B = 400$, mean over the four FR models). $U$: share of undecided FIQA method
pairs (\%). Agreement: Spearman correlation between the leaderboards on the observed pairs under C
and N. Last columns: most frequent first place under N (number of FR models) and its
$P^{{\mathrm{{top}}}}$ range.}}
\label{{tab:ablation-normalization}}
\centering
\begin{{tabular}}{{lcccccccc}}
  \toprule
  & \multicolumn{{2}}{{c}}{{$U$, $\mathrm{{RCE}}_0$}} & \multicolumn{{2}}{{c}}{{$U$, $\mathrm{{RCE}}_3$}}
  & \multicolumn{{2}}{{c}}{{Agreement C vs.\ N}} & \multicolumn{{2}}{{c}}{{Top method, $\mathrm{{RCE}}_0$ (N)}} \\
  \cmidrule(lr){{2-3}} \cmidrule(lr){{4-5}} \cmidrule(lr){{6-7}} \cmidrule(lr){{8-9}}
  Dataset & C & N & C & N & $\mathrm{{RCE}}_0$ & $\mathrm{{RCE}}_3$ & Method & $P^{{\mathrm{{top}}}}$ \\
  \midrule
{body}
  \bottomrule
\end{{tabular}}
\end{{table*}}
""")
    print(f"wrote {path}")


# ---------------------------------------------------------------------------
# Comparison figures (one plot per file)
# ---------------------------------------------------------------------------

def mark(color):
    """Legend handle drawn like the plot: interval line with the dot on it."""
    return (Line2D([0], [0], color=color, linewidth=1.6, alpha=0.55, solid_capstyle="round"),
            Line2D([0], [0], color=color, marker="o", markersize=3.6, markeredgecolor="white",
                   markeredgewidth=0.5, linestyle=""))


def fig_stability(ps_c, ps_n, key, column, ylabel, path):
    """Per dataset: EDC, RCE_gamma with counts and with normalized E_i; one marker per FR model."""
    rce = PS.rce_label(key)
    PS.fr_summary_figure([(ps_c, "EDC", "EDC"), (ps_c, key, f"{rce}\ncount"), (ps_n, key, f"{rce}\nnorm.")],
                         column, ylabel, path, datasets=[d for d in DATASETS if d in set(ps_n.dataset)],
                         label_size=5, dataset_y=-0.135)


def fig_scores(ms_c, ms_n, key, path):
    """PURE-FIQA mean and 95% interval of RCE_gamma per FIQA method, counts vs normalized."""
    c = ms_c[ms_c.protocol == key].set_index("method")
    m = ms_n[ms_n.protocol == key].set_index("method")
    order = m.sort_values("mean").index.tolist()
    fig, ax = plt.subplots(figsize=(PS.COLUMN_WIDTH, 0.19 * len(order) + 0.75))
    PS.clean_axes(ax)
    for k, meth in enumerate(order):
        for d, (tab, color) in zip((0.17, -0.17), ((c, C_COUNTS), (m, C_NORM))):
            if meth not in tab.index:
                continue
            r = tab.loc[meth]
            ax.plot([r.q025, r.q975], [k + d, k + d], color=color, linewidth=1.6, alpha=0.55,
                    solid_capstyle="round", zorder=2)
            ax.plot(r["mean"], k + d, "o", markersize=3.6, color=color, markeredgecolor="white",
                    markeredgewidth=0.5, zorder=3)
    ax.axvline(0, color=PS.INK_2, linewidth=0.6, zorder=1)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([PS.METHOD_NAMES.get(x, x) for x in order])
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.set_xlabel(PS.rce_label(key))
    ax.legend([mark(C_COUNTS), mark(C_NORM)], ["counts", "normalized"],
              handler_map={tuple: HandlerTuple(ndivide=1, pad=0)}, loc="lower center",
              bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, handlelength=1.8, columnspacing=1.6)
    PS.save(fig, path)


def make_ablation_figures(out_root, ms_c, ms_n, ps_c, ps_n, keys):
    PS.set_style()
    base = os.path.join(out_root, "figures", "ablation")
    for key in keys:
        tag = "g" + PS.gamma_label(key)
        fig_stability(ps_c, ps_n, key, "undecided_pairs_pct", "Undecided FIQA method pairs (%)",
                      os.path.join(base, f"undecided_pairs_{tag}"))
        fig_stability(ps_c, ps_n, key, "mean_rank_interval_90", "Mean 90% rank interval (ranks)",
                      os.path.join(base, f"rank_interval_width_{tag}"))
        for (ds, fr), g in ms_n.groupby(["dataset", "fr_model"]):
            gc = ms_c[(ms_c.dataset == ds) & (ms_c.fr_model == fr)]
            fig_scores(gc, g, key, os.path.join(base, ds, fr, f"scores_{tag}"))
    print(f"ablation figures written under {base}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Ablation: PURE-FIQA with normalized error contribution")
    p.add_argument("--fr-features-root", default="data/FR_models")
    p.add_argument("--quality-dir", default="data/FIQA_models")
    p.add_argument("--ca-fiqa-data-root", default="data/pairs/pairs")
    p.add_argument("--out-root", default=os.path.join("output", "ablation_normalization"))
    p.add_argument("--counts-root", default=os.path.join("output", "protocol_stability"),
                   help="PURE-FIQA results with raw counts (protocol_stability.py) for the comparison")
    p.add_argument("--table-out", default=os.path.join("paper", "tables", "tab_ablation_normalization.tex"))
    p.add_argument("--datasets", default=",".join(DATASETS))
    p.add_argument("--fr-models", default="adaface,arcface_o,magface,swinface")
    p.add_argument("--methods", default="", help="FIQA methods (default: all found)")
    p.add_argument("--rce-alphas", default="0,3", help="RCE gamma values")
    p.add_argument("--lam", type=float, default=0.5, help="weight of the false non-match term (lambda)")
    p.add_argument("--strength", type=float, default=1.0,
                   help="prior strength s in expected errors (alpha = s / rate virtual comparisons)")
    p.add_argument("--fmr", type=float, default=1e-3)
    p.add_argument("--threshold-method", choices=["percentile", "roc"], default="roc")
    p.add_argument("--n-samples", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--plot-only", action="store_true", help="Redraw from {out_root}/scores.csv")
    args = p.parse_args()

    split = lambda s: [x.strip() for x in str(s).split(",") if x.strip()]
    alphas = [float(a) for a in split(args.rce_alphas)]
    keys = [PS.protocol_key("RCE", a) for a in alphas]
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
        harm_fn = partial(normalized_harm, lam=args.lam, strength=args.strength)
        jobs = []
        for ds in split(args.datasets):
            for fr in split(args.fr_models):
                if (ds, fr) not in found:
                    print(f"[skip] {ds}/{fr}: no embeddings or pairs")
                    continue
                # Same job as protocol_stability.py (identity bootstrap, margin ties, same seed),
                # only the error contribution differs.
                jobs.append({"dataset": ds, "fr_model": fr, "emb_pkl": found[(ds, fr)][0],
                             "pairs_csv": found[(ds, fr)][1], "quality_dir": args.quality_dir,
                             "methods": methods, "alphas": alphas, "rce_ties": "margin",
                             "harm_fn": harm_fn, "fmr": args.fmr, "threshold_method": args.threshold_method,
                             "resample": "identities", "n_samples": args.n_samples, "seed": args.seed})
        print(f"{len(jobs)} jobs, B={args.n_samples}, lambda={args.lam}, s={args.strength}, "
              f"{args.workers} workers", flush=True)
        if args.workers > 1:
            with Pool(args.workers) as pool:
                frames = pool.map(PS.run_job, jobs, chunksize=1)
        else:
            frames = [PS.run_job(j) for j in jobs]
        scores = pd.concat(frames, ignore_index=True)
        os.makedirs(args.out_root, exist_ok=True)
        scores.to_csv(scores_csv, index=False)
        print(f"saved {scores_csv}")

    ms, pw, ps = PS.statistics(scores)
    ms.to_csv(os.path.join(args.out_root, "method_summary.csv"), index=False)
    pw.to_csv(os.path.join(args.out_root, "pairwise.csv"), index=False)
    ps.to_csv(os.path.join(args.out_root, "protocol_summary.csv"), index=False)
    PS.make_figures(args.out_root, ms, pw, ps, alphas)

    # Comparison with the counts run (same resamples).
    ms_c = pd.read_csv(os.path.join(args.counts_root, "method_summary.csv"))
    ps_c = pd.read_csv(os.path.join(args.counts_root, "protocol_summary.csv"))
    counts_scores = os.path.join(args.counts_root, "scores.csv")
    if os.path.exists(counts_scores):
        sc = pd.read_csv(counts_scores)
        k = ["dataset", "fr_model", "method", "sample"]
        e = scores[scores.protocol == "EDC"].merge(sc[sc.protocol == "EDC"], on=k, suffixes=("", "_c"))
        print(f"check: EDC scores identical to the counts run (same resamples): "
              f"{len(e)} rows, max abs diff {np.nanmax(np.abs(e.score - e.score_c)):.2e}")
    summary = ablation_summary(ms_c, ms, ps_c, ps, keys)
    summary.to_csv(os.path.join(args.out_root, "ablation_summary.csv"), index=False)
    print(summary.groupby(["dataset", "protocol"])[["U_counts", "U_norm", "order_agreement"]].mean().round(2))
    write_table(summary, args.table_out)
    make_ablation_figures(args.out_root, ms_c, ms, ps_c, ps, keys)


if __name__ == "__main__":
    main()
