#!/usr/bin/env python3
"""Reference rankings for the PURE-FIQA consistency diagnostic (rank-PIT test of rce_prob_test1.py).

The PIT statistics of a FIQA ranking (phi_low, phi_high, KS distance D) can only be interpreted
against reference rankings evaluated with the same per-image rank distributions F_i and the same
calibration null. For every dataset / FR model this script

  1. replays the random-number sequence of rce_prob_test1.run_one exactly (same seed, joint pair
     bootstrap, same B), so F_i (reference half A) and the null statistics (calibration half C)
     are identical to output/rce_prob_test1;
  2. recomputes the PIT statistics of every FIQA method in the same order, which reproduces
     output/rce_prob_test1/pit_test_all.csv exactly (checked and reported);
  3. evaluates reference rankings with a separate generator (seed + 1), so they do not depend on
     the number of FIQA methods:
       random        uniformly random permutation of the images, averaged over --n-random draws;
                     carries no information about the errors
       oracle        ranking by the expected error contribution E_bar(i) = mean_b E^(b)(i) over
                     all B resamples, random tie-breaking; uses the errors themselves, so it is not
                     attainable by a FIQA method
       oracle+noise  the oracle with Gaussian noise of standard deviation --noise-sd (in units of
                     E_bar) added before ranking; shows that a noisier ranking can obtain a smaller
                     D than the oracle (the test measures consistency, not accuracy)
  4. computes, on the original evaluation set (threshold at FMR_target with the same rule), the
     AUC of every ranking for flagging the images that cause at least one error:
         AUC = P[Q(i) < Q(j) | E(i) > 0, E(j) = 0] + 1/2 P[Q(i) = Q(j) | ...]
     (0.5 = chance). For reference rankings, Q = -rank (oracle: Q = -E_bar).

Run (on a compute node, see run_pit_baselines.slurm):

    python pit_baselines.py --threshold-method roc --n-samples 400 --workers 16

Redraw figures and the table from the saved CSVs:

    python pit_baselines.py --plot-only

Outputs, under {out_root} (default output/pit_baselines/):
  pit_baselines.csv     one row per dataset / FR model / ranking (FIQA methods, random, oracle,
                        oracle+noise): phi_low, phi_high, ks_stat, ks_pvalue, ks_null_median, auc
                        (random: mean over the draws, *_sd = spread over the draws)
  pit_vs_random.csv     per dataset / FR model: how many FIQA methods have a smaller D / larger
                        AUC than the random ranking (beyond 2 sd of its draws)
  pit_histograms.csv    PIT density per bin (20 bins of width 0.05) of every ranking
  reproducibility.csv   recomputed FIQA statistics vs output/rce_prob_test1/pit_test_all.csv
  figures/{fr_model}/pit_{dataset}.pdf/.png   one plot per file
and paper/tables/tab_pit_baselines.tex.
"""

import argparse
import os
from multiprocessing import Pool

import matplotlib  # type: ignore[reportMissingImports]

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # type: ignore[reportMissingImports]
from matplotlib.lines import Line2D  # type: ignore[reportMissingImports]
from matplotlib.patches import Patch  # type: ignore[reportMissingImports]
import numpy as np  # type: ignore[reportMissingImports]
import pandas as pd
from scipy.stats import rankdata  # type: ignore[reportMissingImports]

import protocol_stability as PS
from rce import _quality_pkl_for_dataset, _quality_scores_for_ids, load_quality_scores_pkl
from rce_prob_test1 import (
    RankCDF,
    harm_counts,
    ks_uniform,
    list_jobs,
    load_job,
    mc_pvalue,
    rank_random_ties,
    sample_harm_rankings,
    threshold_at_fmr,
)

DATASETS = ["lfw", "xqlfw", "calfw", "cplfw"]
BINS = np.linspace(0.0, 1.0, 21)
C_RANDOM = "#eb6834"
C_ORACLE = "#2a78d6"
C_NOISY = "#1baf7a"
C_BAND = "#c9c8c2"
REFERENCE_NAMES = {"random": "Random ranking", "oracle": "Oracle ranking", "oracle_noise": "Oracle + noise"}


# ---------------------------------------------------------------------------
# Statistics of one ranking
# ---------------------------------------------------------------------------

def auc_low_quality_flags_errors(quality: np.ndarray, has_error: np.ndarray) -> float:
    """P[Q(i) < Q(j) | i causes an error, j does not] + 1/2 P[tie]; 0.5 = chance."""
    pos, neg = -quality[has_error], -quality[~has_error]   # low quality = high value
    if pos.size == 0 or neg.size == 0:
        return np.nan
    r = rankdata(np.concatenate([pos, neg]))
    return float((r[:pos.size].sum() - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))


def pit_stats(u: np.ndarray, d_null: np.ndarray) -> dict:
    d = ks_uniform(u)
    return {"phi_low": float(np.mean(u < 0.05)), "phi_high": float(np.mean(u > 0.95)),
            "ks_stat": d, "ks_pvalue": mc_pvalue(d, d_null)}


def density(u: np.ndarray) -> np.ndarray:
    return np.histogram(u, bins=BINS, density=True)[0]


# ---------------------------------------------------------------------------
# One dataset / FR model
# ---------------------------------------------------------------------------

def run_job(job: dict):
    dataset, fr_model, seed, B = job["dataset"], job["fr_model"], job["seed"], job["n_samples"]

    # --- exactly the random-number sequence of rce_prob_test1.run_one ---------------------------
    rng = np.random.default_rng(seed)
    img_ids, idx1, idx2, labels, sims = load_job(job["emb_pkl"], job["pairs_csv"])
    n = len(img_ids)
    R_e, _, mean_harm = sample_harm_rankings(
        n=n, idx1=idx1, idx2=idx2, sims=sims, labels=labels, n_samples=B, resample="both",
        fmr_target=job["fmr"], threshold_method=job["threshold_method"], rng=rng,
    )
    perm = rng.permutation(B)
    A, C = perm[: B // 2], perm[B // 2:]
    cdf = RankCDF(R_e[A])
    d_null = np.array([ks_uniform(cdf.pit(R_e[b], rng)) for b in C])
    d_null_median = float(np.median(d_null))

    # Errors on the original evaluation set (for the AUC).
    thr = threshold_at_fmr(sims[labels == 0], job["fmr"], job["threshold_method"])
    has_error = harm_counts(n, idx1, idx2, ((labels == 1) & (sims < thr)) | ((labels == 0) & (sims >= thr))) > 0

    rows, hist = [], []

    def record(kind, name, u, quality, draw=0):
        row = {"dataset": dataset, "fr_model": fr_model, "kind": kind, "ranking": name, "draw": draw,
               **pit_stats(u, d_null), "ks_null_median": d_null_median,
               "auc": auc_low_quality_flags_errors(quality, has_error),
               "error_images_pct": 100.0 * has_error.mean()}
        rows.append(row)
        hist.append({"dataset": dataset, "fr_model": fr_model, "kind": kind, "ranking": name, "draw": draw,
                     **{f"bin{k:02d}": v for k, v in enumerate(density(u))}})

    # FIQA methods, same order and generator as rce_prob_test1 (reproduces its results).
    for method in job["methods"]:
        qpath = _quality_pkl_for_dataset(os.path.join(job["quality_dir"], method), dataset)
        if not os.path.exists(qpath):
            continue
        q = _quality_scores_for_ids(load_quality_scores_pkl(qpath), img_ids)
        R_q = rank_random_ties(q, rng, descending=True)
        record("method", method, cdf.pit(R_q, rng), q)

    # Reference rankings, own generator.
    ref = np.random.default_rng(seed + 1)
    for k in range(job["n_random"]):
        R = ref.permutation(n).astype(np.int32) + 1
        record("reference", "random", cdf.pit(R, ref), -R.astype(float), draw=k)
    R = rank_random_ties(mean_harm, ref)
    record("reference", "oracle", cdf.pit(R, ref), -mean_harm)
    noisy = mean_harm + ref.normal(0.0, job["noise_sd"], n)
    R = rank_random_ties(noisy, ref)
    record("reference", "oracle_noise", cdf.pit(R, ref), -noisy)

    print(f"[{dataset}/{fr_model}] done", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(hist)


# ---------------------------------------------------------------------------
# Aggregation, check, table, figures
# ---------------------------------------------------------------------------

def collapse_random(df: pd.DataFrame) -> pd.DataFrame:
    """Average the random-ranking draws (p-value: the largest over the draws)."""
    num = ["phi_low", "phi_high", "ks_stat", "ks_null_median", "auc", "error_images_pct"]
    keys = ["dataset", "fr_model", "kind", "ranking"]
    rand = df[df.ranking == "random"]
    agg = rand.groupby(keys)[num].mean()
    agg["ks_pvalue"] = rand.groupby(keys)["ks_pvalue"].max()
    for col in ("phi_low", "phi_high", "ks_stat", "auc"):          # spread over the draws
        agg[f"{col}_sd"] = rand.groupby(keys)[col].std()
    agg["draws"] = rand.groupby(keys).size()
    rest = df[df.ranking != "random"].drop(columns="draw").assign(draws=1)
    return pd.concat([rest, agg.reset_index()], ignore_index=True)


def reproducibility_check(stats: pd.DataFrame, reference_csv: str) -> pd.DataFrame:
    if not os.path.exists(reference_csv):
        print(f"[check] {reference_csv} not found, skipped")
        return pd.DataFrame()
    ref = pd.read_csv(reference_csv).rename(columns={"method": "ranking"})
    m = stats[stats.kind == "method"].merge(ref, on=["dataset", "fr_model", "ranking"], suffixes=("", "_ref"))
    out = pd.DataFrame({
        "dataset": m.dataset, "fr_model": m.fr_model, "method": m.ranking,
        "ks_diff": (m.ks_stat - m.ks_stat_ref).abs(),
        "phi_low_diff": (m.phi_low - m["frac_u_below_0.05"]).abs(),
        "phi_high_diff": (m.phi_high - m["frac_u_above_0.95"]).abs(),
        "pvalue_diff": (m.ks_pvalue - m.ks_pvalue_ref).abs(),
    })
    worst = out[["ks_diff", "phi_low_diff", "phi_high_diff", "pvalue_diff"]].to_numpy().max()
    print(f"[check] FIQA statistics vs {reference_csv}: {len(out)} rows, max abs diff {worst:.2e}")
    return out


def compare_with_random(stats: pd.DataFrame) -> pd.DataFrame:
    """Per dataset / FR model: the random ranking (mean +- sd over its draws) vs the FIQA methods.
    A method counts as better than random if its D is below the random mean by more than 2 sd of
    the random draws (D: smaller is closer to H0), resp. its AUC is above 0.5 + 2 sd."""
    rows = []
    for (ds, fr), g in stats.groupby(["dataset", "fr_model"]):
        r = g[g.ranking == "random"].iloc[0]
        m = g[g.kind == "method"]
        rows.append({
            "dataset": ds, "fr_model": fr, "n_methods": len(m),
            "random_ks": r.ks_stat, "random_ks_sd": r.ks_stat_sd,
            "methods_ks_min": m.ks_stat.min(), "methods_ks_median": m.ks_stat.median(), "methods_ks_max": m.ks_stat.max(),
            "n_ks_below_random": int((m.ks_stat < r.ks_stat - 2 * r.ks_stat_sd).sum()),
            "n_ks_above_random": int((m.ks_stat > r.ks_stat + 2 * r.ks_stat_sd).sum()),
            "random_phi_high": r.phi_high, "methods_phi_high_min": m.phi_high.min(), "methods_phi_high_max": m.phi_high.max(),
            "random_auc_sd": r.auc_sd, "methods_auc_min": m.auc.min(), "methods_auc_max": m.auc.max(),
            "n_auc_above_chance": int((m.auc > 0.5 + 2 * r.auc_sd).sum()),
            "n_auc_below_chance": int((m.auc < 0.5 - 2 * r.auc_sd).sum()),
        })
    return pd.DataFrame(rows)


def write_table(stats: pd.DataFrame, path: str):
    """Per dataset: FIQA methods (range over methods and FR models), random, oracle, oracle+noise
    (mean over FR models)."""
    lines = []
    for ds in [d for d in DATASETS if d in set(stats.dataset)]:
        g = stats[stats.dataset == ds]
        meth = g[g.kind == "method"]
        rng = lambda col, f, s=1.0: f"{s * meth[col].min():{f}}--{s * meth[col].max():{f}}"
        lines.append(rf"\multicolumn{{6}}{{l}}{{\textit{{{PS.DATASET_NAMES.get(ds, ds)}}} "
                     rf"($D_{{\mathrm{{null}}}} = {g.ks_null_median.mean():.4f}$, "
                     rf"{g.error_images_pct.mean():.1f}\% error-causing images)}} \\")
        lines.append(f"FIQA methods & {rng('phi_low', '.1f', 100)} & {rng('phi_high', '.1f', 100)} & "
                     f"{rng('ks_stat', '.3f')} & {meth.ks_pvalue.max():.3f} & {rng('auc', '.2f')}" + r" \\")
        for name in ("random", "oracle", "oracle_noise"):
            r = g[g.ranking == name]
            lines.append(f"{REFERENCE_NAMES[name]} & {100 * r.phi_low.mean():.1f} & {100 * r.phi_high.mean():.1f} & "
                         f"{r.ks_stat.mean():.3f} & {r.ks_pvalue.max():.3f} & {r.auc.mean():.2f}" + r" \\")
        lines.append(r"\midrule")
    lines = lines[:-1]
    body = "\n".join("  " + l for l in lines)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(rf"""\begin{{table}}[t]
\caption{{Consistency diagnostic with reference rankings (joint pair bootstrap, $B = 400$,
$|\Cs| = 200$). FIQA methods: range over the 17 methods and four FR models; reference rankings:
mean over the FR models (random: 20 draws each). $\phi_{{\mathrm{{low}}}}$, $\phi_{{\mathrm{{high}}}}$:
share (\%) of images with $u_i < 0.05$ and $u_i > 0.95$ ($5\%$ each under $H_0$); $D$: KS distance;
$\hat{{p}}$: largest p-value; AUC: probability that an image causing an error on the original
evaluation set is ranked lower than an error-free image ($0.5$ = chance).}}
\label{{tab:pit-baselines}}
\centering
\setlength{{\tabcolsep}}{{3pt}}
\resizebox{{\columnwidth}}{{!}}{{%
\begin{{tabular}}{{lccccc}}
  \toprule
  Ranking & $\phi_{{\mathrm{{low}}}}$ & $\phi_{{\mathrm{{high}}}}$ & $D$ & $\hat{{p}}$ & AUC \\
  \midrule
{body}
  \bottomrule
\end{{tabular}}}}
\end{{table}}
""")
    print(f"wrote {path}")


def fig_pit(hist: pd.DataFrame, path: str):
    """FIQA methods as a band (min--max over methods per bin) with their median, reference
    rankings as lines, uniform density under H0 dashed."""
    cols = [c for c in hist.columns if c.startswith("bin")]
    centers = 0.5 * (BINS[:-1] + BINS[1:])
    meth = hist[hist.kind == "method"][cols].to_numpy()
    fig, ax = plt.subplots(figsize=(PS.COLUMN_WIDTH, 2.9))
    PS.clean_axes(ax, grid_axis="y")
    ax.axvspan(0.0, 0.05, color=PS.GRID, alpha=0.8, linewidth=0, zorder=0)
    ax.axvspan(0.95, 1.0, color=PS.GRID, alpha=0.8, linewidth=0, zorder=0)
    ax.fill_between(centers, meth.min(0), meth.max(0), color=C_BAND, linewidth=0, zorder=1)
    ax.plot(centers, np.median(meth, axis=0), color=PS.INK_2, linewidth=1.0, zorder=3)
    ax.axhline(1.0, color=PS.INK, linewidth=0.8, linestyle=(0, (4, 2)), zorder=2)
    ref_style = {"random": (C_RANDOM, "-"), "oracle": (C_ORACLE, "-"), "oracle_noise": (C_NOISY, (0, (1.5, 1.2)))}
    for name, (color, ls) in ref_style.items():
        d = hist[hist.ranking == name][cols].to_numpy().mean(axis=0)
        ax.plot(centers, d, color=color, linestyle=ls, linewidth=1.3, zorder=4)
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xticklabels(["0", "0.25", "0.5", "0.75", "1"])
    ax.set_xlabel(r"PIT value $u_i$")
    ax.set_ylabel("Density")
    handles = [Patch(facecolor=C_BAND, edgecolor="none"),
               Line2D([0], [0], color=PS.INK_2, linewidth=1.0)]
    labels = ["FIQA methods: range", "FIQA methods: median"]
    for name, (color, ls) in ref_style.items():
        handles.append(Line2D([0], [0], color=color, linestyle=ls, linewidth=1.3))
        labels.append(REFERENCE_NAMES[name])
    handles += [Line2D([0], [0], color=PS.INK, linewidth=0.8, linestyle=(0, (4, 2))),
                Patch(facecolor=PS.GRID, edgecolor="none")]
    labels += [r"uniform ($H_0$)", r"tails $u<0.05$, $u>0.95$"]
    ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False,
              fontsize=6.5, handlelength=2.0, columnspacing=1.2, labelspacing=0.4)
    PS.save(fig, path)


def main():
    p = argparse.ArgumentParser(description="Reference rankings for the PURE-FIQA consistency diagnostic")
    p.add_argument("--fr-features-root", default="data/FR_models")
    p.add_argument("--quality-dir", default="data/FIQA_models")
    p.add_argument("--ca-fiqa-data-root", default="data/pairs/pairs")
    p.add_argument("--out-root", default=os.path.join("output", "pit_baselines"))
    p.add_argument("--reference-csv", default=os.path.join("output", "rce_prob_test1", "pit_test_all.csv"),
                   help="rce_prob_test1.py results to reproduce (check)")
    p.add_argument("--table-out", default=os.path.join("paper", "tables", "tab_pit_baselines.tex"))
    p.add_argument("--datasets", default=",".join(DATASETS))
    p.add_argument("--fr-models", default="adaface,arcface_o,magface,swinface")
    # Must match the rce_prob_test1.py run (output/rce_prob_test1) to reproduce F_i and the null.
    p.add_argument("--fmr", type=float, default=1e-3)
    p.add_argument("--threshold-method", choices=["percentile", "roc"], default="roc")
    p.add_argument("--n-samples", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-random", type=int, default=20, help="random-ranking draws per dataset / FR model")
    p.add_argument("--noise-sd", type=float, default=0.5, help="noise of the oracle+noise ranking (units of E_bar)")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--plot-only", action="store_true")
    args = p.parse_args()

    split = lambda s: [x.strip() for x in str(s).split(",") if x.strip()]
    stats_csv = os.path.join(args.out_root, "pit_baselines.csv")
    hist_csv = os.path.join(args.out_root, "pit_histograms.csv")

    if args.plot_only:
        stats, hist = pd.read_csv(stats_csv), pd.read_csv(hist_csv)
    else:
        methods = sorted(d for d in os.listdir(args.quality_dir)
                         if os.path.isdir(os.path.join(args.quality_dir, d)) and not d.startswith("."))
        found = {(d, m): (pkl, pairs) for d, m, _, pkl, pairs in
                 list_jobs(args.fr_features_root, args.ca_fiqa_data_root, run_all=True)}
        jobs = [{"dataset": ds, "fr_model": fr, "emb_pkl": found[(ds, fr)][0], "pairs_csv": found[(ds, fr)][1],
                 "quality_dir": args.quality_dir, "methods": methods, "fmr": args.fmr,
                 "threshold_method": args.threshold_method, "n_samples": args.n_samples, "seed": args.seed,
                 "n_random": args.n_random, "noise_sd": args.noise_sd}
                for ds in split(args.datasets) for fr in split(args.fr_models) if (ds, fr) in found]
        print(f"{len(jobs)} jobs, B={args.n_samples}, {args.n_random} random draws, {args.workers} workers", flush=True)
        if args.workers > 1:
            with Pool(args.workers) as pool:
                results = pool.map(run_job, jobs, chunksize=1)
        else:
            results = [run_job(j) for j in jobs]
        raw = pd.concat([r[0] for r in results], ignore_index=True)
        hist = pd.concat([r[1] for r in results], ignore_index=True)
        os.makedirs(args.out_root, exist_ok=True)
        stats = collapse_random(raw)
        stats.to_csv(stats_csv, index=False)
        hist.to_csv(hist_csv, index=False)
        check = reproducibility_check(raw, args.reference_csv)
        check.to_csv(os.path.join(args.out_root, "reproducibility.csv"), index=False)
        print(f"saved {stats_csv}, {hist_csv}")

    comp = compare_with_random(stats)
    comp.to_csv(os.path.join(args.out_root, "pit_vs_random.csv"), index=False)
    print(comp[["dataset", "fr_model", "random_ks", "random_ks_sd", "methods_ks_min", "methods_ks_max",
                "n_ks_below_random", "n_ks_above_random", "n_auc_above_chance", "n_auc_below_chance"]].round(4).to_string(index=False))
    group = np.where(stats.kind == "method", "FIQA methods", stats.ranking)
    summary = stats.assign(group=group).groupby(["dataset", "group"])[["phi_low", "phi_high", "ks_stat", "auc"]].agg(["min", "max"])
    print(summary.round(3).to_string())
    write_table(stats, args.table_out)
    PS.set_style()
    for (fr, ds), h in hist.groupby(["fr_model", "dataset"]):
        fig_pit(h, os.path.join(args.out_root, "figures", fr, f"pit_{ds}"))
    print(f"figures written under {os.path.join(args.out_root, 'figures')}")


if __name__ == "__main__":
    main()
