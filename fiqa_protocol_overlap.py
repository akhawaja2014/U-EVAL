#!/usr/bin/env python3
"""Retained pair-set overlap (the format of the paper's Fig. 3) for FIQA methods under
three evaluation protocols: the paper's EDC, the paper's RCE and our probabilistic
method (rce_prob_test1.py).

Discarding at rate d drops the int(d * n) worst images of a ranking (lowest quality for
a FIQA method, most harmful for a harm ranking) and keeps the pairs whose two images
both survive. Overlap = Jaccard (%) of two retained pair sets.

Figure 1 - test-set divergence between FIQA methods (the paper's Fig. 3, per protocol)
    EDC    each FIQA method is evaluated on the pairs left after discarding its own
           lowest-quality images -> overlap between the test sets of methods A and B
           (the paper's curves).
    RCE    every FIQA method is evaluated on the full pair set -> 100% at every d.
    ours   every FIQA method is evaluated on the same B resampled pair sets -> 100%
           between methods; the band shows how much the test set itself varies between
           two resampled pair sets (Jaccard of their distinct pairs).

Figure 2 - FIQA method vs protocol reference
    Overlap between the pairs a FIQA method keeps and the pairs the protocol's own
    reference harm ranking keeps at the same d:
    EDC    EDC-optimal ranking (greedy FNMR reduction with threshold re-fit per step,
           see new_prob1_test_set.py); left blank once fewer than 1/FMR impostor
           pairs remain (where edc.py stops the EDC curve) or no genuine pair remains
    RCE    rce.py harm ranking (C_FNM + C_FM at the fixed threshold, margin tie-break)
    ours   individual harm draws R_e^(b) of rce_prob_test1.py (mean over draws; the
           summary also gives the 5-95% range)
    One thin line per FIQA method, thick line = mean over methods; dashed = random.

Figure 3 - summary per FIQA method
    Mean overlap over d in --summary-rates (default 10-30%, the paper's pAUC range) on
    all pairs, one dot per protocol (ours with its 5-95% range over draws).

R_e samples are regenerated with the same seed / settings as rce_prob_test1.py.

Run:

    python fiqa_protocol_overlap.py \
        --fr-features-root data/FR_models \
        --quality-dir data/FIQA_models \
        --ca-fiqa-data-root data/pairs/pairs \
        --out-root output/fiqa_protocol_overlap \
        --threshold-method roc --n-samples 400

Outputs, under {out_root}/:
    fig1_test_set_divergence_{fr_model}.png/.pdf
    fig2_fiqa_vs_reference_{label}_{fr_model}.png/.pdf   (label = genuine / impostor)
    fig3_fiqa_overlap_summary_{fr_model}.png/.pdf
    overlap_curves.csv, overlap_summary.csv
"""

import argparse
import itertools
import os

import matplotlib  # type: ignore[reportMissingImports]

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # type: ignore[reportMissingImports]
from matplotlib.lines import Line2D  # type: ignore[reportMissingImports]
import numpy as np  # type: ignore[reportMissingImports]
import pandas as pd

from new_prob1_test_set import edc_harm_ranking, jaccard, kept_pairs, rce_harm_ranking, style_axes
from rce import _quality_pkl_for_dataset, _quality_scores_for_ids, load_quality_scores_pkl
from rce_prob_test1 import list_jobs, load_job, rank_random_ties, sample_harm_rankings

# Reference palette (categorical slots 1-3) + neutral inks.
PROTOCOLS = {"EDC": "#1baf7a", "RCE": "#eb6834", "ours": "#2a78d6"}
PROTOCOL_LABELS = {"EDC": "paper EDC", "RCE": "paper RCE", "ours": "ours (probabilistic)"}
GRAY = "#8f8e88"
THIN = "#c9c8c2"
INK = "#0b0b0b"
INK_2 = "#52514e"

DATASET_NAMES = {"calfw": "CALFW", "cplfw": "CPLFW", "lfw": "LFW", "xqlfw": "XQLFW", "cfp-fp": "CFP-FP"}
METHOD_NAMES = {
    "clib-fiqa": "CLIB-FIQA", "cr-fiqa": "CR-FIQA (L)", "diffiqa_r": "DifFIQA (R)",
    "ediffiqa_l": "eDifFIQA (L)", "faceqan": "FaceQAN", "faceqgen": "FaceQGen",
    "faceqnet": "FaceQnet", "froq_ada": "FROQ", "grafiqs": "GraFIQs",
    "lightqnet": "LightQNet", "magface": "MagFace", "pcnet": "PCNet", "pfe": "PFE",
    "sdd-fiqa": "SDD-FIQA", "ser-fiq": "SER-FIQA", "vit-fiqa": "ViT-FIQA",
    "vitnt-fiqa": "ViTNT-FIQA",
}
LABEL_MASKS = ("genuine", "impostor", "all")


def mname(m: str) -> str:
    return METHOD_NAMES.get(m, m)


# ---------------------------------------------------------------------------
# Computation per dataset / FR model
# ---------------------------------------------------------------------------

def resampled_pair_set_overlap(labels: np.ndarray, resample: str, n_pairs: int, rng) -> pd.DataFrame:
    """Jaccard (%) of the distinct pairs in two independent bootstrap pair sets."""
    rows = []
    groups = {"genuine": np.flatnonzero(labels == 1), "impostor": np.flatnonzero(labels == 0)}
    masks = {"genuine": labels == 1, "impostor": labels == 0, "all": np.ones(labels.size, dtype=bool)}
    for _ in range(n_pairs):
        sets = []
        for _draw in range(2):
            chosen = np.zeros(labels.size, dtype=bool)
            for name, idx in groups.items():
                if resample == "both" or resample[:3] == name[:3]:
                    chosen[idx[rng.integers(0, idx.size, idx.size)]] = True
                else:
                    chosen[idx] = True
            sets.append(chosen)
        for lbl, mask in masks.items():
            rows.append({"label": lbl, "overlap": jaccard(sets[0], sets[1], mask)})
    return pd.DataFrame(rows)


def run_one(
    *,
    dataset: str,
    fr_model: str,
    emb_pkl: str,
    pairs_csv: str,
    quality_dir: str,
    methods: list[str],
    fig1_methods: list[str],
    n_samples: int,
    n_draws: int,
    resample: str,
    fmr_target: float,
    threshold_method: str,
    discard_rates: list[float],
    edc_step: float,
    edc_tiebreak: str,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    min_impostors = int(np.ceil(1.0 / fmr_target))  # edc.py stops the EDC curve below this
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

    quality = {}
    for m in methods:
        qpath = _quality_pkl_for_dataset(os.path.join(quality_dir, m), dataset)
        if os.path.exists(qpath):
            q = _quality_scores_for_ids(load_quality_scores_pkl(qpath), img_ids)
            quality[m] = rank_random_ties(q, rng2, descending=True)
    refs = {
        "EDC": edc_harm_ranking(n=n, idx1=idx1, idx2=idx2, sims=sims, labels=labels, fmr_target=fmr_target,
                                threshold_method=threshold_method, step=edc_step, tiebreak=edc_tiebreak,
                                rng=np.random.default_rng(seed + 2)),
        "RCE": rce_harm_ranking(n=n, idx1=idx1, idx2=idx2, sims=sims, labels=labels,
                                fmr_target=fmr_target, threshold_method=threshold_method),
    }
    draws = R_e[rng2.permutation(n_samples)[:n_draws]]
    randoms = [rng2.permutation(n).astype(np.int32) + 1 for _ in range(2 * n_draws)]
    masks = {"genuine": labels == 1, "impostor": labels == 0, "all": np.ones(labels.size, dtype=bool)}
    fig1_pairs = list(itertools.combinations([m for m in fig1_methods if m in quality], 2))

    rows = []

    def add(kind, protocol, a, b, d, lbl, value):
        rows.append({"dataset": dataset, "fr_model": fr_model, "kind": kind, "protocol": protocol,
                     "a": a, "b": b, "discard_rate": d, "label": lbl, "overlap": value})

    for d in discard_rates:
        kept_q = {m: kept_pairs(r, d, idx1, idx2) for m, r in quality.items()}
        kept_ref = {p: kept_pairs(r, d, idx1, idx2) for p, r in refs.items()}
        edc_defined = (np.count_nonzero(kept_ref["EDC"] & (labels == 0)) >= min_impostors
                       and np.count_nonzero(kept_ref["EDC"] & (labels == 1)) > 0)  # else FNMR undefined
        kept_draw = [kept_pairs(r, d, idx1, idx2) for r in draws]
        kept_rand = [kept_pairs(r, d, idx1, idx2) for r in randoms]
        for lbl, mask in masks.items():
            # Figure 1: test-set divergence between FIQA methods.
            for a, b in fig1_pairs:
                add("divergence", "EDC", a, b, d, lbl, jaccard(kept_q[a], kept_q[b], mask))
                add("divergence", "RCE", a, b, d, lbl, 100.0)
                add("divergence", "ours", a, b, d, lbl, 100.0)
            # Figure 2/3: FIQA method vs protocol reference.
            for m, km in kept_q.items():
                for p, kr in kept_ref.items():
                    value = jaccard(km, kr, mask) if (p != "EDC" or edc_defined) else np.nan
                    add("vs_reference", p, m, p, d, lbl, value)
                for j, kd in enumerate(kept_draw):
                    add("vs_reference", "ours", m, f"draw{j}", d, lbl, jaccard(km, kd, mask))
            for j in range(n_draws):
                add("random", "random", f"r{2 * j}", f"r{2 * j + 1}", d, lbl,
                    jaccard(kept_rand[2 * j], kept_rand[2 * j + 1], mask))

    between_draws = resampled_pair_set_overlap(labels, resample, n_draws, rng2)
    between_draws.insert(0, "dataset", dataset)
    between_draws.insert(1, "fr_model", fr_model)
    return pd.DataFrame(rows), between_draws


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def save(fig, path):
    for ext in ("png", "pdf"):
        fig.savefig(f"{path}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}.png/.pdf")


def fig1(curves, between, datasets, fr_model, path):
    c = curves[curves.kind == "divergence"]
    fig, axes = plt.subplots(2, len(datasets), figsize=(3.3 * len(datasets), 5.6), sharex=True, sharey=True)
    axes = np.asarray(axes).reshape(2, len(datasets))
    for j, ds in enumerate(datasets):
        for i, lbl in enumerate(("genuine", "impostor")):
            ax = axes[i, j]
            style_axes(ax)
            g = c[(c.dataset == ds) & (c.label == lbl)]
            x = np.sort(g.discard_rate.unique()) * 100
            # ours: test sets differ between resampled pair sets (same for every FIQA method).
            b = between[(between.dataset == ds) & (between.label == lbl)].overlap
            if len(b):
                ax.fill_between(x, b.quantile(0.05), b.quantile(0.95), color=PROTOCOLS["ours"],
                                alpha=0.15, linewidth=0, zorder=1)
                ax.plot(x, np.full(x.size, b.mean()), color=PROTOCOLS["ours"], linewidth=1.2,
                        linestyle=(0, (4, 2)), zorder=2)
            e = g[g.protocol == "EDC"]
            for _, cc in e.groupby(["a", "b"]):
                ax.plot(cc.discard_rate * 100, cc.overlap, color=PROTOCOLS["EDC"], linewidth=0.7, alpha=0.4, zorder=2)
            ax.plot(x, e.groupby("discard_rate").overlap.mean(), color=PROTOCOLS["EDC"], linewidth=2.2, zorder=4)
            ax.plot(x, np.full(x.size, 100.0), color=PROTOCOLS["RCE"], linewidth=2.2, zorder=5)
            ax.plot(x, np.full(x.size, 100.0), color=PROTOCOLS["ours"], linewidth=1.4,
                    linestyle=(0, (1.5, 1.5)), zorder=6)
            ax.set_ylim(0, 104)
            ax.set_xlim(x.min(), x.max())
            ax.set_title(f"{DATASET_NAMES.get(ds, ds)} · {lbl} pairs", fontsize=8.5, color=INK, loc="left")
            if i == 1:
                ax.set_xlabel("Discard rate (%)", fontsize=8, color=INK_2)
            if j == 0:
                ax.set_ylabel("Overlap between FIQA test sets (%)", fontsize=8, color=INK_2)
    handles = [
        Line2D([0], [0], color=PROTOCOLS["EDC"], linewidth=2.2, label="paper EDC: mean over FIQA pairs (thin = per pair)"),
        Line2D([0], [0], color=PROTOCOLS["RCE"], linewidth=2.2, label="paper RCE: same full test set (100%)"),
        Line2D([0], [0], color=PROTOCOLS["ours"], linewidth=1.4, linestyle=(0, (1.5, 1.5)),
               label="ours: same resampled test sets for all methods (100%)"),
        Line2D([0], [0], color=PROTOCOLS["ours"], linewidth=1.2, linestyle=(0, (4, 2)),
               label="ours: overlap between two resampled pair sets (5–95% band)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle(f"Test-set divergence between FIQA methods under each protocol ({fr_model})",
                 fontsize=9.5, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save(fig, path)


def fig2(curves, datasets, fr_model, label, path):
    c = curves[(curves.label == label)]
    fig, axes = plt.subplots(3, len(datasets), figsize=(3.3 * len(datasets), 7.6), sharex=True, sharey=True)
    axes = np.asarray(axes).reshape(3, len(datasets))
    for i, p in enumerate(PROTOCOLS):
        for j, ds in enumerate(datasets):
            ax = axes[i, j]
            style_axes(ax)
            g = c[(c.dataset == ds) & (c.kind == "vs_reference") & (c.protocol == p)]
            per_method = g.groupby(["a", "discard_rate"]).overlap.mean().reset_index()
            for _, cc in per_method.groupby("a"):
                ax.plot(cc.discard_rate * 100, cc.overlap, color=THIN, linewidth=0.8, zorder=2)
            mean = per_method.groupby("discard_rate").overlap.mean()
            ax.plot(mean.index * 100, mean, color=PROTOCOLS[p], linewidth=2.2, zorder=4)
            r = c[(c.dataset == ds) & (c.kind == "random")].groupby("discard_rate").overlap.mean()
            ax.plot(r.index * 100, r, color=GRAY, linewidth=1.2, linestyle=(0, (4, 2)), zorder=3)
            ax.set_ylim(0, 102)
            ax.set_xlim(r.index.min() * 100, r.index.max() * 100)
            ax.set_title(f"{DATASET_NAMES.get(ds, ds)} · {PROTOCOL_LABELS[p]}", fontsize=8.5, color=INK, loc="left")
            if i == 2:
                ax.set_xlabel("Discard rate (%)", fontsize=8, color=INK_2)
            if j == 0:
                ax.set_ylabel("Overlap: FIQA vs reference (%)", fontsize=8, color=INK_2)
    handles = [Line2D([0], [0], color=c_, linewidth=2.2, label=f"{PROTOCOL_LABELS[p]}: mean over FIQA methods")
               for p, c_ in PROTOCOLS.items()]
    handles += [Line2D([0], [0], color=THIN, linewidth=0.8, label="one FIQA method"),
                Line2D([0], [0], color=GRAY, linewidth=1.2, linestyle=(0, (4, 2)), label="random rankings")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle(f"Pairs kept by each FIQA method vs pairs kept by the protocol's reference ranking "
                 f"({label} pairs, {fr_model})", fontsize=9.5, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, path)


def summarize(curves: pd.DataFrame, summary_rates: list[float]) -> pd.DataFrame:
    c = curves[(curves.kind == "vs_reference") & curves.discard_rate.isin(summary_rates)]
    # Per (protocol, method, reference/draw): mean over the summary discard rates.
    per_ref = c.groupby(["dataset", "fr_model", "label", "protocol", "a", "b"]).overlap.mean().reset_index()
    s = per_ref.groupby(["dataset", "fr_model", "label", "protocol", "a"]).overlap
    out = pd.DataFrame({"overlap": s.mean(), "q05": s.quantile(0.05), "q95": s.quantile(0.95)}).reset_index()
    return out.rename(columns={"a": "method"})


def fig3(summary, datasets, fr_model, path):
    s = summary[(summary.label == "all") & (summary.fr_model == fr_model)]
    fig, axes = plt.subplots(1, len(datasets), figsize=(3.3 * len(datasets), 5.4), sharey=False)
    axes = np.atleast_1d(axes)
    offsets = {"EDC": -0.22, "RCE": 0.0, "ours": 0.22}
    for ax, ds in zip(axes, datasets):
        style_axes(ax)
        g = s[s.dataset == ds]
        order = g[g.protocol == "RCE"].sort_values("overlap").method.tolist()
        y = {m: k for k, m in enumerate(order)}
        for p, color in PROTOCOLS.items():
            gp = g[g.protocol == p].set_index("method").loc[order]
            yy = np.array([y[m] for m in order]) + offsets[p]
            if p == "ours":
                ax.hlines(yy, gp.q05, gp.q95, color=color, linewidth=1.2, alpha=0.6, zorder=2)
            ax.scatter(gp.overlap, yy, s=20, color=color, edgecolor="white", linewidth=0.7, zorder=3)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([mname(m) for m in order], fontsize=7, color=INK_2)
        ax.set_ylim(-0.7, len(order) - 0.3)
        ax.set_title(DATASET_NAMES.get(ds, ds), fontsize=9, color=INK, loc="left")
        ax.set_xlabel("Mean overlap with reference (%)", fontsize=8, color=INK_2)
    handles = [Line2D([0], [0], marker="o", linestyle="", markersize=5, color=c_, label=PROTOCOL_LABELS[p])
               for p, c_ in PROTOCOLS.items()]
    handles.append(Line2D([0], [0], color=PROTOCOLS["ours"], linewidth=1.2, label="ours: 5–95% over harm draws"))
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle(f"Per FIQA method: overlap of its kept pairs with each protocol's reference, "
                 f"mean over 10–30% discard, all pairs ({fr_model}); sorted by RCE",
                 fontsize=9.5, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, path)


def main() -> None:
    p = argparse.ArgumentParser(description="Fig. 3-style pair overlap for FIQA methods under EDC, RCE and ours")
    p.add_argument("--fr-features-root", default="data/FR_models")
    p.add_argument("--quality-dir", default="data/FIQA_models")
    p.add_argument("--ca-fiqa-data-root", default="data/pairs/pairs")
    p.add_argument("--out-root", default=os.path.join("output", "fiqa_protocol_overlap"))
    p.add_argument("--datasets", default="lfw,xqlfw,calfw,cplfw",
                   help="Datasets (columns), in plot order; cfp-fp is broken (10 images)")
    p.add_argument("--fr-models", default="adaface,arcface_o,magface,swinface", help="One figure set per FR model")
    p.add_argument("--methods", default="", help="FIQA methods for figures 2/3 (default: all found)")
    p.add_argument("--fig1-methods", default="faceqnet,ser-fiq,cr-fiqa,ediffiqa_l",
                   help="FIQA methods for figure 1 (default: the four in the paper's Fig. 3)")
    p.add_argument("--discard-rates", default=",".join(str(i / 10) for i in range(10)))
    p.add_argument("--summary-rates", default="0.1,0.2,0.3", help="Discard rates averaged in figure 3")
    p.add_argument("--n-draws", type=int, default=20, help="Harm draws / random pairs per curve")
    p.add_argument("--edc-step", type=float, default=0.005)
    p.add_argument("--edc-tiebreak", choices=["margin", "min-gen-sim", "random"], default="margin")
    # Must match the rce_prob_test1.py run to reproduce its R_e samples.
    p.add_argument("--fmr", type=float, default=1e-3)
    p.add_argument("--threshold-method", choices=["percentile", "roc"], default="percentile")
    p.add_argument("--resample", choices=["imp", "gen", "both"], default="both")
    p.add_argument("--n-samples", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    split = lambda s: [x.strip() for x in str(s).split(",") if x.strip()]
    datasets = split(args.datasets)
    fig1_methods = split(args.fig1_methods)
    methods = split(args.methods) or sorted(
        d for d in os.listdir(args.quality_dir)
        if os.path.isdir(os.path.join(args.quality_dir, d)) and not d.startswith(".")
    )
    methods = sorted(set(methods) | set(fig1_methods))
    discard_rates = [float(x) for x in split(args.discard_rates)]
    summary_rates = [float(x) for x in split(args.summary_rates)]

    jobs = {(d, m): (pkl, pairs) for d, m, _, pkl, pairs in
            list_jobs(args.fr_features_root, args.ca_fiqa_data_root, run_all=True)}

    curves, betweens = [], []
    for fr_model in split(args.fr_models):
        for ds in datasets:
            if (ds, fr_model) not in jobs:
                print(f"[skip] {ds}/{fr_model}: no embeddings or pairs")
                continue
            emb_pkl, pairs_csv = jobs[(ds, fr_model)]
            print(f"[run] {ds}/{fr_model}")
            c, b = run_one(
                dataset=ds, fr_model=fr_model, emb_pkl=emb_pkl, pairs_csv=pairs_csv,
                quality_dir=args.quality_dir, methods=methods, fig1_methods=fig1_methods,
                n_samples=args.n_samples, n_draws=args.n_draws, resample=args.resample,
                fmr_target=args.fmr, threshold_method=args.threshold_method,
                discard_rates=discard_rates, edc_step=args.edc_step, edc_tiebreak=args.edc_tiebreak,
                seed=args.seed,
            )
            curves.append(c)
            betweens.append(b)
    if not curves:
        return

    curves = pd.concat(curves, ignore_index=True)
    between = pd.concat(betweens, ignore_index=True)
    summary = summarize(curves, summary_rates)
    os.makedirs(args.out_root, exist_ok=True)
    curves.to_csv(os.path.join(args.out_root, "overlap_curves.csv"), index=False)
    between.to_csv(os.path.join(args.out_root, "between_resampled_pair_sets.csv"), index=False)
    summary.to_csv(os.path.join(args.out_root, "overlap_summary.csv"), index=False)

    for fr_model in curves.fr_model.unique():
        c = curves[curves.fr_model == fr_model]
        b = between[between.fr_model == fr_model]
        ds_here = [d for d in datasets if d in set(c.dataset)]
        fig1(c, b, ds_here, fr_model, os.path.join(args.out_root, f"fig1_test_set_divergence_{fr_model}"))
        for label in ("genuine", "impostor"):
            fig2(c, ds_here, fr_model, label, os.path.join(args.out_root, f"fig2_fiqa_vs_reference_{label}_{fr_model}"))
        fig3(summary, ds_here, fr_model, os.path.join(args.out_root, f"fig3_fiqa_overlap_summary_{fr_model}"))


if __name__ == "__main__":
    main()
