#!/usr/bin/env python3
"""Supplementary material of the PURE-FIQA paper: tables and figure floats generated from saved results.

No experiment is rerun. Inputs (saved results):
    output/protocol_stability/                 identity bootstrap, margin RCE (main results)
    output/protocol_stability_pair_bootstrap/  joint pair bootstrap, margin RCE
    output/protocol_stability_midrank/         identity bootstrap, mid-rank RCE (same resamples as the main run)
    output/pit_baselines/                      consistency diagnostic with reference rankings
    output/ablation_normalization/             normalized error contribution (same resamples as the main run)
The dataset tables (S1) also read the pair lists and embeddings, to count images, identities and pairs
and the errors on the original evaluation set (FMR_target = 1e-3, ROC rule, as in all experiments).
These counts are saved to output/supplementary/dataset_stats.csv.

Outputs:
    paper/supplementary/tables/*.tex      one file per table, \\input by paper/supplementary.tex
    paper/supplementary/sec_figures.tex   figure floats of Section S8 (all FR models)
    paper/supplementary/figures/{adaface,arcface,magface,swinface}/   copies of the figures of one FR model
    paper/supplementary/figures/summary/  copies of the figures that show all FR models in one plot
    paper/supplementary/figures/manifest.csv   source of every copy (the outputs are not changed)
    output/supplementary/dataset_stats.csv

Run from the repository root:
    python make_supplementary.py
    python make_supplementary.py --skip-data      # reuse output/supplementary/dataset_stats.csv
Check everything with:
    python validate_supplementary.py
"""

import argparse
import os
import shutil

import numpy as np  # type: ignore[reportMissingImports]
import pandas as pd

from protocol_stability import DATASET_NAMES, FR_NAMES, METHOD_NAMES, fmr_threshold, image_identities
from rce_prob_test1 import harm_counts, list_jobs, load_job

DATASETS = ["lfw", "xqlfw", "calfw", "cplfw"]
FR_MODELS = ["adaface", "arcface_o", "magface", "swinface"]
PROTOCOLS = ["EDC", "RCE_g0", "RCE_g3"]
RCE_KEYS = ["RCE_g0", "RCE_g3"]
PROTO_TEX = {"EDC": "EDC", "RCE_g0": r"$\mathrm{RCE}_0$", "RCE_g3": r"$\mathrm{RCE}_3$"}
PROTO_LABEL = {"EDC": "edc", "RCE_g0": "rce0", "RCE_g3": "rce3"}
PTOP = r"$P^{\mathrm{top}}$"
FMR_TARGET = 1e-3
THRESHOLD_RULE = "roc"
REFERENCES = ["random", "oracle", "oracle_noise"]
REFERENCE_NAMES = {"random": "Random ranking", "oracle": "Oracle ranking", "oracle_noise": "Oracle + noise"}
EDC_SCALE = 1e3                         # EDC pAUC is printed multiplied by 10^3


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def name(method: str) -> str:
    return METHOD_NAMES.get(method, method)


def num(value: float, fmt: str) -> str:
    """Formatted number with a typeset minus sign."""
    s = format(value, fmt)
    return "$-$" + s[1:] if s.startswith("-") else s


def method_order_alpha(methods) -> list[str]:
    return sorted(methods, key=lambda m: name(m).lower())


def edc_decimals(observed_scaled: pd.Series) -> int:
    """Decimals for EDC pAUC x 10^3: four significant digits for the largest score of the table
    (methods can differ by less than 0.1 x 10^-3)."""
    mx = float(np.nanmax(np.abs(observed_scaled)))
    return int(max(1, 3 - np.floor(np.log10(mx)))) if mx > 0 else 3


def write(path: str, text: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)
    print(f"wrote {path}")


def table(path: str, *, caption: str, label: str, colspec: str, head: list[str], rows: list[str],
          wide: bool = True, resize: str = "", tabcolsep: str = "", size: str = ""):
    """Write one booktabs table float. rows may contain group separators such as \\midrule."""
    env, place = ("table*", "!tp") if wide else ("table", "!tbp")   # table* allows only t and p
    out = [rf"\begin{{{env}}}[{place}]", rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\centering"]
    if size:
        out.append(size)
    if tabcolsep:
        out.append(rf"\setlength{{\tabcolsep}}{{{tabcolsep}}}")
    if resize:
        out.append(rf"\resizebox{{{resize}}}{{!}}{{%")
    out.append(rf"\begin{{tabular}}{{{colspec}}}")
    out.append(r"  \toprule")
    out += ["  " + h for h in head]
    out.append(r"  \midrule")
    out += ["  " + r for r in rows]
    out.append(r"  \bottomrule")
    out.append(r"\end{tabular}" + ("}" if resize else ""))
    out.append(rf"\end{{{env}}}")
    write(path, "\n".join(out) + "\n")


def fr_header(sub: list[str], first: str) -> list[str]:
    """Two header lines: FR model names over len(sub) columns each, then the sub-column names."""
    k = len(sub)
    top = " & ".join(rf"\multicolumn{{{k}}}{{c}}{{{FR_NAMES[f]}}}" for f in FR_MODELS)
    cmid = " ".join(rf"\cmidrule(lr){{{2 + k * j}-{1 + k * (j + 1)}}}" for j in range(len(FR_MODELS)))
    return [f"& {top} \\\\", cmid, f"{first} & " + " & ".join(sub * len(FR_MODELS)) + r" \\"]


# ---------------------------------------------------------------------------
# S1: datasets
# ---------------------------------------------------------------------------

def dataset_statistics(fr_root: str, pairs_root: str) -> pd.DataFrame:
    """Counts per dataset / FR model on the original evaluation set (no resampling)."""
    found = {(d, m): (pkl, pairs) for d, m, _, pkl, pairs in list_jobs(fr_root, pairs_root, run_all=True)}
    rows = []
    for ds in DATASETS:
        for fr in FR_MODELS:
            pkl, pairs = found[(ds, fr)]
            img_ids, i1, i2, labels, sims = load_job(pkl, pairs)
            n = len(img_ids)
            per_identity = np.bincount(image_identities(img_ids))
            thr = fmr_threshold(sims[labels == 0], FMR_TARGET, THRESHOLD_RULE)
            fail = ((labels == 1) & (sims < thr)) | ((labels == 0) & (sims >= thr))
            harm = harm_counts(n, i1, i2, fail)
            rows.append({
                "dataset": ds, "fr_model": fr, "images": n, "identities": per_identity.size,
                "max_images_per_identity": int(per_identity.max()),
                "single_image_identities": int((per_identity == 1).sum()),
                "genuine_pairs": int((labels == 1).sum()), "impostor_pairs": int((labels == 0).sum()),
                "threshold": thr, "fnmr_pct": 100.0 * float(np.mean(sims[labels == 1] < thr)),
                "fmr_pct": 100.0 * float(np.mean(sims[labels == 0] >= thr)),
                "error_images_pct": 100.0 * float(np.mean(harm > 0)),
            })
            print(f"[S1] {ds}/{fr}: {n} images, {per_identity.size} identities")
    return pd.DataFrame(rows)


def tables_datasets(st: pd.DataFrame, out: str):
    cols = ["images", "identities", "max_images_per_identity", "single_image_identities",
            "genuine_pairs", "impostor_pairs"]
    rows = []
    for ds in DATASETS:
        g = st[st.dataset == ds]
        if (g[cols].nunique() > 1).any():
            raise ValueError(f"{ds}: dataset counts differ between FR models")
        r = g.iloc[0]
        rows.append(f"{DATASET_NAMES[ds]} & " + " & ".join(f"{int(r[c]):,}" for c in cols) + r" \\")
    table(os.path.join(out, "tab_s_datasets.tex"),
          caption=r"Evaluation sets. Identities are derived from the image file names; every image takes "
                  r"part in at least one pair, and no pair is dropped. LFW and XQLFW share the same pair "
                  r"list. Identities with one image have no genuine pair.",
          label="tab:s-datasets", colspec="lrrrrrr",
          head=[r"Dataset & Images & Identities & \shortstack{Max.\ images\\per identity} & "
                r"\shortstack{Identities\\with one image} & \shortstack{Genuine\\pairs} & "
                r"\shortstack{Impostor\\pairs} \\"],
          rows=rows)

    rows = []
    for ds in DATASETS:
        g = st[st.dataset == ds].set_index("fr_model")
        cells = [f"{g.loc[f, 'fnmr_pct']:.2f} & {g.loc[f, 'error_images_pct']:.1f}" for f in FR_MODELS]
        rows.append(f"{DATASET_NAMES[ds]} & " + " & ".join(cells) + r" \\")
    table(os.path.join(out, "tab_s_errors.tex"),
          caption=r"Recognition errors on the original evaluation set at $\mathrm{FMR}_{\mathrm{target}} = "
                  r"10^{-3}$ (ROC rule). FNMR: false non-match rate (\%); Err.: share of images (\%) that "
                  r"take part in at least one false non-match or false match, i.e.\ with $E(i) > 0$.",
          label="tab:s-errors", colspec="l" + "cc" * len(FR_MODELS),
          head=fr_header(["FNMR", "Err."], "Dataset"), rows=rows)


# ---------------------------------------------------------------------------
# S2: full leaderboards
# ---------------------------------------------------------------------------

def tables_leaderboards(ms: pd.DataFrame, out: str) -> list[str]:
    files = []
    for ds in DATASETS:
        for key in PROTOCOLS:
            g = ms[(ms.dataset == ds) & (ms.protocol == key)]
            mean_rank = g.groupby("method").observed_rank.mean()
            order = sorted(mean_rank.index, key=lambda m: (mean_rank[m], name(m).lower()))
            if key == "EDC":
                scale, dec = EDC_SCALE, edc_decimals(g.observed * EDC_SCALE)
                fmt_v, fmt_i = f".{dec}f", f".{dec}f"
            else:
                scale, fmt_v, fmt_i = 1.0, ".3f", ".2f"
            rows = []
            for m in order:
                cells = []
                for fr in FR_MODELS:
                    r = g[(g.fr_model == fr) & (g.method == m)]
                    if r.empty:
                        cells.append("-- & --")
                        continue
                    r = r.iloc[0]
                    v = num(scale * r.observed, fmt_v)
                    v = rf"\textbf{{{v}}}" if r.observed_rank == 1 else v
                    cells.append(rf"{v} {{\scriptsize[{num(scale * r.q025, fmt_i)}, {num(scale * r.q975, fmt_i)}]}}"
                                 rf" & {r.p_rank1:.2f}")
                rows.append(f"{name(m)} & " + " & ".join(cells) + r" \\")
            if key == "EDC":
                what = (r"EDC: partial area under the EDC curve up to $30\%$ discard, multiplied by $10^{3}$ "
                        r"(lower is better)")
                score = "pAUC"
            else:
                what = rf"{PROTO_TEX[key]} (higher is better)"
                score = PROTO_TEX[key]
            fname = f"tab_s_lb_{ds}_{PROTO_LABEL[key]}.tex"
            table(os.path.join(out, fname),
                  caption=rf"{DATASET_NAMES[ds]}, {what}: score on the original evaluation set with its 95\% "
                          rf"identity-bootstrap interval, and the probability of ranking first, {PTOP}, for every "
                          rf"FR model ($B = 400$). Bold: first place on the original evaluation set. Methods "
                          rf"sorted by their mean rank on the original evaluation set over the four FR models.",
                  label=f"tab:s-lb-{ds}-{PROTO_LABEL[key]}", colspec="l" + "cc" * len(FR_MODELS),
                  head=fr_header([rf"{score} [95\% int.]", PTOP], "Method"), rows=rows,
                  resize=r"\textwidth", tabcolsep="3pt")
            files.append(fname)
    return files


# ---------------------------------------------------------------------------
# S3: conclusions of RCE per FR model
# ---------------------------------------------------------------------------

def table_claims(ms: pd.DataFrame, pw: pd.DataFrame, out: str):
    rows = []
    for ds in DATASETS:
        for j, fr in enumerate(FR_MODELS):
            cells = []
            for key in RCE_KEYS:
                g = ms[(ms.dataset == ds) & (ms.fr_model == fr) & (ms.protocol == key)]
                top = g[g.observed_rank == 1]
                if len(top) != 1:
                    raise ValueError(f"{ds}/{fr}/{key}: {len(top)} methods with observed rank 1")
                top = top.iloc[0]
                cells.append(f"{name(top.method)} & {top.p_rank1:.2f}")
            for key in RCE_KEYS:
                p = pw[(pw.dataset == ds) & (pw.fr_model == fr) & (pw.protocol == key)
                       & (pw.a == "ediffiqa_l") & (pw.b == "froq_ada")].p_a_beats_b
                if len(p) != 1:
                    raise ValueError(f"{ds}/{fr}/{key}: P(eDifFIQA > FROQ) not unique")
                cells.append(f"{p.iloc[0]:.2f}")
            label = DATASET_NAMES[ds] if j == 0 else ""
            rows.append(f"{label} & {FR_NAMES[fr]} & " + " & ".join(cells) + r" \\")
        rows.append(r"\midrule")
    table(os.path.join(out, "tab_s_claims.tex"),
          caption=r"Conclusions of RCE~\cite{wani2026beyond} per FR model (identity bootstrap, $B = 400$). "
                  rf"Top method: first place on the original evaluation set, with its probability of ranking "
                  rf"first, {PTOP}. Last two columns: probability that eDifFIQA (L) obtains a higher score than "
                  r"FROQ. The main paper reports the range of these values over the FR models.",
          label="tab:s-claims", colspec="llcccccc",
          head=[r"& & \multicolumn{2}{c}{$\mathrm{RCE}_0$} & \multicolumn{2}{c}{$\mathrm{RCE}_3$} "
                r"& \multicolumn{2}{c}{$P(\text{eDifFIQA (L)} \succ \text{FROQ})$} \\",
                r"\cmidrule(lr){3-4} \cmidrule(lr){5-6} \cmidrule(lr){7-8}",
                rf"Dataset & FR model & Top method & {PTOP} & Top method & {PTOP} & $\mathrm{{RCE}}_0$ & "
                r"$\mathrm{RCE}_3$ \\"],
          rows=rows[:-1])


# ---------------------------------------------------------------------------
# S4 / S5: stability per FR model, pair vs identity bootstrap
# ---------------------------------------------------------------------------

def per_fr_rows(make_cells) -> list[str]:
    rows = []
    for ds in DATASETS:
        for j, fr in enumerate(FR_MODELS):
            label = DATASET_NAMES[ds] if j == 0 else ""
            rows.append(f"{label} & {FR_NAMES[fr]} & " + " & ".join(make_cells(ds, fr)) + r" \\")
        rows.append(r"\midrule")
    return rows[:-1]


def table_stability(ps: pd.DataFrame, out: str):
    def cells(ds, fr):
        g = ps[(ps.dataset == ds) & (ps.fr_model == fr)].set_index("protocol")
        return ([f"{g.loc[k, 'undecided_pairs_pct']:.0f}" for k in PROTOCOLS]
                + [f"{g.loc[k, 'mean_rank_interval_90']:.1f}" for k in PROTOCOLS])

    head_p = " & ".join(PROTO_TEX[k] for k in PROTOCOLS)
    table(os.path.join(out, "tab_s_stability.tex"),
          caption=r"Leaderboard stability per FR model (identity bootstrap, $B = 400$). $U$: share of "
                  r"FIQA method pairs whose order is undecided ($0.05 < P(m \succ m') < 0.95$); $W$: mean "
                  r"width of the 90\% rank interval, in ranks. Lower is more stable. The main paper reports "
                  r"the mean and range over the FR models.",
          label="tab:s-stability", colspec="llcccccc",
          head=[r"& & \multicolumn{3}{c}{$U$ (\%)} & \multicolumn{3}{c}{$W$ (ranks)} \\",
                r"\cmidrule(lr){3-5} \cmidrule(lr){6-8}",
                f"Dataset & FR model & {head_p} & {head_p} \\\\"],
          rows=per_fr_rows(cells))


def table_bootstrap(ps_id: pd.DataFrame, ps_pair: pd.DataFrame, out: str):
    def cells(ds, fr):
        a = ps_pair[(ps_pair.dataset == ds) & (ps_pair.fr_model == fr)].set_index("protocol")
        b = ps_id[(ps_id.dataset == ds) & (ps_id.fr_model == fr)].set_index("protocol")
        return [f"{a.loc[k, 'undecided_pairs_pct']:.0f} & {b.loc[k, 'undecided_pairs_pct']:.0f}" for k in PROTOCOLS]

    head = " & ".join(rf"\multicolumn{{2}}{{c}}{{{PROTO_TEX[k]}}}" for k in PROTOCOLS)
    table(os.path.join(out, "tab_s_bootstrap.tex"),
          caption=r"Share of undecided FIQA method pairs $U$ (\%) per FR model under the joint pair "
                  r"bootstrap (Pair) and the identity bootstrap (Id.), $B = 400$ each. The original "
                  r"evaluation set and all other settings are the same. The main paper reports the mean "
                  r"over the FR models.",
          label="tab:s-bootstrap", colspec="llcccccc",
          head=[f"& & {head} \\\\", r"\cmidrule(lr){3-4} \cmidrule(lr){5-6} \cmidrule(lr){7-8}",
                r"Dataset & FR model & Pair & Id. & Pair & Id. & Pair & Id. \\"],
          rows=per_fr_rows(cells), wide=False, tabcolsep="4pt")


# ---------------------------------------------------------------------------
# S6: consistency diagnostic
# ---------------------------------------------------------------------------

def tables_pit(stats: pd.DataFrame, comp: pd.DataFrame, out: str) -> list[str]:
    files = []
    for ds in DATASETS:
        g = stats[stats.dataset == ds]
        mean = g.groupby("ranking")[["phi_low", "phi_high", "ks_stat", "auc"]].mean()
        n_fr = g.groupby("ranking").fr_model.nunique()
        if (n_fr != len(FR_MODELS)).any():
            raise ValueError(f"{ds}: PIT results missing for some FR models")

        def row(r, label):
            m = mean.loc[r]
            return (f"{label} & {100 * m.phi_low:.1f} & {100 * m.phi_high:.1f} & {m.ks_stat:.3f} & "
                    f"{m.auc:.3f}" + r" \\")

        methods = method_order_alpha(g[g.kind == "method"].ranking.unique())
        rows = [row(m, name(m)) for m in methods] + [r"\midrule"] + [row(r, REFERENCE_NAMES[r]) for r in REFERENCES]
        fname = f"tab_s_pit_{ds}.tex"
        table(os.path.join(out, fname),
              caption=rf"Consistency diagnostic on {DATASET_NAMES[ds]} per ranking (joint pair bootstrap, "
                      r"$B = 400$, $|\mathcal{C}| = 200$), mean over the four FR models; random ranking: mean "
                      r"over 20 draws per FR model. $\phi_{\mathrm{low}}$, $\phi_{\mathrm{high}}$: share (\%) of "
                      r"images with $u_i < 0.05$ and $u_i > 0.95$ ($5\%$ each under $H_0$); $D$: KS distance; "
                      r"AUC: probability that an image causing an error on the original evaluation set is ranked "
                      r"lower than an error-free image ($0.5$ = chance). Methods in alphabetical order.",
              label=f"tab:s-pit-{ds}", colspec="lcccc",
              head=[r"Ranking & $\phi_{\mathrm{low}}$ & $\phi_{\mathrm{high}}$ & $D$ & AUC \\"],
              rows=rows, wide=False, tabcolsep="4pt")
        files.append(fname)

    def cells(ds, fr):
        r = comp[(comp.dataset == ds) & (comp.fr_model == fr)].iloc[0]
        return [rf"{r.random_ks:.3f} $\pm$ {r.random_ks_sd:.3f}",
                f"{r.methods_ks_min:.3f}--{r.methods_ks_max:.3f}",
                f"{int(r.n_ks_below_random)}", f"{int(r.n_ks_above_random)}",
                f"{r.methods_auc_min:.3f}--{r.methods_auc_max:.3f}",
                f"{int(r.n_auc_above_chance)}", f"{int(r.n_auc_below_chance)}"]

    table(os.path.join(out, "tab_s_pit_random.tex"),
          caption=r"FIQA methods relative to the random ranking, per FR model. Random $D$: mean $\pm$ "
                  r"standard deviation over 20 draws. Closer / Further: number of the 17 FIQA methods whose "
                  r"$D$ lies more than two standard deviations below / above the random mean. AUC above / "
                  r"below: number of methods whose AUC lies more than two standard deviations of the random "
                  r"draws above / below $0.5$.",
          label="tab:s-pit-random", colspec="llccccccc",
          head=[r"& & \multicolumn{4}{c}{KS distance $D$} & \multicolumn{3}{c}{AUC} \\",
                r"\cmidrule(lr){3-6} \cmidrule(lr){7-9}",
                r"Dataset & FR model & Random & FIQA methods & Closer & Further & FIQA methods & Above & Below \\"],
          rows=per_fr_rows(cells))
    files.append("tab_s_pit_random.tex")
    return files


# ---------------------------------------------------------------------------
# S7: ablation, normalized error contribution
# ---------------------------------------------------------------------------

def tables_ablation(ab: pd.DataFrame, out: str) -> list[str]:
    files = []
    for key in RCE_KEYS:
        def cells(ds, fr):
            r = ab[(ab.dataset == ds) & (ab.fr_model == fr) & (ab.protocol == key)].iloc[0]
            return [f"{r.U_counts:.0f}", f"{r.U_norm:.0f}", f"{r.W_counts:.1f}", f"{r.W_norm:.1f}",
                    num(r.order_agreement, "+.2f"),
                    f"{name(r.top_counts)} & {r.p_top_counts:.2f}", f"{name(r.top_norm)} & {r.p_top_norm:.2f}"]

        g = key.split("_g")[1]
        fname = f"tab_s_ablation_{PROTO_LABEL[key]}.tex"
        table(os.path.join(out, fname),
              caption=rf"Ablation per FR model, $\mathrm{{RCE}}_{g}$: error counts (C) vs.\ the normalized error "
                      rf"contribution (N), identity bootstrap with the same $B = 400$ resamples. $U$: share of "
                      rf"undecided FIQA method pairs (\%); $W$: mean width of the 90\% rank interval (ranks); "
                      rf"Agreement: Spearman correlation between the leaderboards on the original evaluation "
                      rf"set under C and N; Top: first place on the original evaluation set and its {PTOP}.",
              label=f"tab:s-ablation-{PROTO_LABEL[key]}", colspec="llccccccccc",
              head=[r"& & \multicolumn{2}{c}{$U$ (\%)} & \multicolumn{2}{c}{$W$ (ranks)} & & "
                    r"\multicolumn{2}{c}{Top (C)} & \multicolumn{2}{c}{Top (N)} \\",
                    r"\cmidrule(lr){3-4} \cmidrule(lr){5-6} \cmidrule(lr){8-9} \cmidrule(lr){10-11}",
                    rf"Dataset & FR model & C & N & C & N & Agreement & Method & {PTOP} & Method & {PTOP} \\"],
              rows=per_fr_rows(cells), tabcolsep="4pt")
        files.append(fname)
    return files


# ---------------------------------------------------------------------------
# S9: mid-rank vs margin tie-breaking
# ---------------------------------------------------------------------------

def tie_summary(ms_mar, ms_mid, ps_mar, ps_mid) -> pd.DataFrame:
    from scipy.stats import spearmanr  # type: ignore[reportMissingImports]
    rows = []
    for ds in DATASETS:
        for fr in FR_MODELS:
            for key in RCE_KEYS:
                a = ms_mar[(ms_mar.dataset == ds) & (ms_mar.fr_model == fr) & (ms_mar.protocol == key)].set_index("method")
                b = ms_mid[(ms_mid.dataset == ds) & (ms_mid.fr_model == fr) & (ms_mid.protocol == key)].set_index("method")
                b = b.loc[a.index]
                pa = ps_mar[(ps_mar.dataset == ds) & (ps_mar.fr_model == fr) & (ps_mar.protocol == key)].iloc[0]
                pb = ps_mid[(ps_mid.dataset == ds) & (ps_mid.fr_model == fr) & (ps_mid.protocol == key)].iloc[0]
                rows.append({"dataset": ds, "fr_model": fr, "protocol": key,
                             "min_mid": b.observed.min(), "max_mid": b.observed.max(),
                             "min_margin": a.observed.min(), "max_margin": a.observed.max(),
                             "U_mid": pb.undecided_pairs_pct, "U_margin": pa.undecided_pairs_pct,
                             "W_mid": pb.mean_rank_interval_90, "W_margin": pa.mean_rank_interval_90,
                             "agreement": spearmanr(a.observed, b.observed)[0],
                             "top_mid": b.observed.idxmax(), "top_margin": a.observed.idxmax()})
    return pd.DataFrame(rows)


def table_ties(ts: pd.DataFrame, out: str):
    def most_frequent(s: pd.Series) -> str:
        c = s.value_counts()
        return f"{name(c.index[0])} ({c.iloc[0]}/{len(s)})"

    rows = []
    for key in RCE_KEYS:
        rows.append(rf"\multicolumn{{10}}{{l}}{{\textit{{{PROTO_TEX[key]}}}}} \\")
        for ds in DATASETS:
            g = ts[(ts.dataset == ds) & (ts.protocol == key)]
            rows.append(
                f"{DATASET_NAMES[ds]} & {num(g.min_mid.min(), '.2f')} to {num(g.max_mid.max(), '.2f')} & "
                f"{num(g.min_margin.min(), '.2f')} to {num(g.max_margin.max(), '.2f')} & "
                f"{g.U_mid.mean():.0f} & {g.U_margin.mean():.0f} & {g.W_mid.mean():.1f} & {g.W_margin.mean():.1f} & "
                f"{num(g.agreement.mean(), '+.2f')} & {most_frequent(g.top_mid)} & {most_frequent(g.top_margin)}" + r" \\")
        rows.append(r"\midrule")
    table(os.path.join(out, "tab_s_ties.tex"),
          caption=r"Ties in the error ranking: mid-ranks, i.e.\ average ranks for tied error contributions "
                  r"(Mid), vs.\ margin tie-breaking (Margin) as defined in~\cite{wani2026beyond} and used "
                  r"throughout the paper. "
                  r"Both use the same $B = 400$ identity-bootstrap resamples. Range: lowest to highest score "
                  r"on the original evaluation set over the 17 FIQA methods and four FR models; $U$ (\%) and $W$ "
                  r"(ranks): mean over the FR models; Agreement: Spearman correlation between the two "
                  r"leaderboards on the original evaluation set, mean over the FR models; Top: most frequent "
                  r"first place on the original evaluation set (number of FR models).",
          label="tab:s-ties", colspec="lccccccccc",
          head=[r"& \multicolumn{2}{c}{Range} & \multicolumn{2}{c}{$U$ (\%)} & \multicolumn{2}{c}{$W$ (ranks)} "
                r"& & \multicolumn{2}{c}{Top} \\",
                r"\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7} \cmidrule(lr){9-10}",
                r"Dataset & Mid & Margin & Mid & Margin & Mid & Margin & Agreement & Mid & Margin \\"],
          rows=rows[:-1], resize=r"\textwidth", tabcolsep="4pt")


# ---------------------------------------------------------------------------
# S8: figures, copied into one folder per FR model (+ one for the summaries over all FR models)
# ---------------------------------------------------------------------------

PS_FIG = os.path.join("output", "protocol_stability", "figures")
PIT_FIG = os.path.join("output", "pit_baselines", "figures")
AB_FIG = os.path.join("output", "ablation_normalization", "figures", "ablation")
FR_FOLDER = {"adaface": "adaface", "arcface_o": "arcface", "magface": "magface", "swinface": "swinface"}
SUMMARY_FOLDER = "summary"          # figures that show all four FR models in one plot
PS_KINDS = ["pure_fiqa_leaderboard_g0", "pure_fiqa_leaderboard_g3", "rank_intervals_g0", "rank_intervals_g3",
            "pairwise_EDC", "pairwise_RCE_g0", "pairwise_RCE_g3"]
SUMMARY_KINDS = ["undecided_pairs_g0", "undecided_pairs_g3", "rank_interval_width_g0", "rank_interval_width_g3"]
# Figures of the main paper: copied into the folders, but not repeated in the supplementary document.
IN_MAIN_PAPER = {("adaface", "lfw_pure_fiqa_leaderboard_g0.pdf"), ("adaface", "lfw_rank_intervals_g0.pdf"),
                 ("adaface", "lfw_pairwise_RCE_g0.pdf"), ("adaface", "lfw_pit.pdf"),
                 (SUMMARY_FOLDER, "undecided_pairs_g0.pdf")}

P_BEATS_TEXT = (r"Red: the column method is reliably better ($P \le 0.05$); blue: the row method is reliably "
                r"better ($P \ge 0.95$); green: undecided ($0.05 < P < 0.95$), from pale ($P$ just above $0.05$) "
                r"to dark ($P$ just below $0.95$). Methods sorted by their rank on the original evaluation set; "
                r"dark grey: a method compared with itself.")


def rce_tex(g: str) -> str:
    return rf"$\mathrm{{RCE}}_{g}$"


def figure_caption(kind: str, ds: str, fr: str) -> str:
    where = f"{DATASET_NAMES[ds]} ({FR_NAMES[fr]})"
    if kind.startswith("pure_fiqa_leaderboard_g"):
        g = kind[-1]
        return (rf"{rce_tex(g)} of every FIQA method on {where}. Blue dot: PURE-FIQA mean over $B = 400$ "
                rf"identity-bootstrap resamples; blue line: PURE-FIQA 95\% interval; orange diamond: {rce_tex(g)} "
                r"on the original evaluation set. Methods sorted by the PURE-FIQA mean, best at the top.")
    if kind.startswith("rank_intervals_g"):
        g = kind[-1]
        return (rf"Rank of every FIQA method among the 17 methods (1 = best) on {where} under EDC (green) and "
                rf"{rce_tex(g)} (orange). Dots: rank on the original evaluation set; capped lines: PURE-FIQA 90\% "
                r"rank interval over $B = 400$ identity-bootstrap resamples. A cap through a dot without a line "
                rf"marks a zero-width interval. Methods sorted by their {rce_tex(g)} rank on the original "
                r"evaluation set, best at the top.")
    if kind == "pairwise_EDC":
        return (rf"PURE-FIQA pairwise superiority under EDC on {where}. Cell $(m, m')$ shows $P(m \succ m')$, the "
                r"share of the $B = 400$ identity-bootstrap resamples in which the row method $m$ obtains a lower "
                r"pAUC than the column method $m'$. " + P_BEATS_TEXT)
    if kind.startswith("pairwise_RCE_g"):
        g = kind[-1]
        return (rf"PURE-FIQA pairwise superiority under {rce_tex(g)} on {where}. Cell $(m, m')$ shows "
                r"$P(m \succ m')$, the share of the $B = 400$ identity-bootstrap resamples in which the row "
                rf"method $m$ obtains a higher {rce_tex(g)} than the column method $m'$. " + P_BEATS_TEXT)
    raise ValueError(kind)


def pit_caption(ds: str, fr: str) -> str:
    return (rf"PIT density of the FIQA rankings on {DATASET_NAMES[ds]} ({FR_NAMES[fr]}), joint pair bootstrap "
            r"($B = 400$, $|\mathcal{C}| = 200$): range (grey band) and median (grey line) over the 17 FIQA "
            r"methods, the random ranking (mean over 20 draws), the oracle ranking and the oracle perturbed by "
            r"noise. Density: share of images per bin of width $0.05$ divided by $0.05$; under $H_0$, the density "
            r"is uniform (dashed). Shaded: the tails $u_i < 0.05$ and $u_i > 0.95$.")


def ablation_scores_caption(ds: str, fr: str, g: str) -> str:
    return (rf"Ablation on {DATASET_NAMES[ds]} ({FR_NAMES[fr]}): PURE-FIQA mean (dot) and 95\% interval (line) of "
            rf"{rce_tex(g)} for every FIQA method with error counts (orange) and with the normalized error "
            r"contribution (blue). Methods sorted by the mean under normalization, best at the top.")


def summary_caption(kind: str, ablation: bool) -> str:
    g = kind[-1]
    if kind.startswith("undecided_pairs"):
        what = r"share of undecided FIQA method pairs $U$"
    else:
        what = r"mean width $W$ of the 90\% rank interval, in ranks"
    if ablation:
        return (rf"Ablation, {what}: EDC, {rce_tex(g)} with error counts (count) and {rce_tex(g)} with the "
                r"normalized error contribution (norm.), same $B = 400$ identity-bootstrap resamples. EDC does not "
                r"use the error contribution and is unchanged. Each marker is one FR model; each black bar is the "
                r"mean over the four FR models.")
    return (rf"{what[0].upper() + what[1:]} under EDC and {rce_tex(g)} ($B = 400$ identity-bootstrap resamples). "
            r"Each marker is one FR model, i.e.\ the value over all 17 FIQA methods for one dataset and FR model; "
            r"each black bar is the mean over the four FR models. Lower is more stable.")


def figure_items() -> list[dict]:
    """Every figure: target folder and file name, source PDF, caption, label."""
    items = []

    def add(folder, name, src, caption):
        label = f"fig:s-{folder}-{name[:-4].replace('_', '-')}"
        items.append({"folder": folder, "file": name, "source": src, "caption": caption, "label": label,
                      "in_main_paper": (folder, name) in IN_MAIN_PAPER})

    for kind in SUMMARY_KINDS:
        add(SUMMARY_FOLDER, f"{kind}.pdf", os.path.join(PS_FIG, "summary", f"{kind}.pdf"), summary_caption(kind, False))
    for kind in SUMMARY_KINDS:
        add(SUMMARY_FOLDER, f"ablation_{kind}.pdf", os.path.join(AB_FIG, f"{kind}.pdf"), summary_caption(kind, True))
    for fr in FR_MODELS:
        for ds in DATASETS:
            for kind in PS_KINDS:
                add(FR_FOLDER[fr], f"{ds}_{kind}.pdf", os.path.join(PS_FIG, ds, fr, f"{kind}.pdf"),
                    figure_caption(kind, ds, fr))
            add(FR_FOLDER[fr], f"{ds}_pit.pdf", os.path.join(PIT_FIG, fr, f"pit_{ds}.pdf"), pit_caption(ds, fr))
            for g in ("0", "3"):
                add(FR_FOLDER[fr], f"{ds}_ablation_scores_g{g}.pdf", os.path.join(AB_FIG, ds, fr, f"scores_g{g}.pdf"),
                    ablation_scores_caption(ds, fr, g))
    return items


def copy_figures(items: list[dict], fig_dir: str):
    """Copy every figure (unchanged, with its time stamp) into fig_dir/<folder>/ and write a manifest."""
    for it in items:
        if not os.path.exists(it["source"]):
            raise FileNotFoundError(it["source"])
        dst = os.path.join(fig_dir, it["folder"], it["file"])
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(it["source"], dst)
    manifest = pd.DataFrame([{k: it[k] for k in ("folder", "file", "source", "in_main_paper")} for it in items])
    manifest.to_csv(os.path.join(fig_dir, "manifest.csv"), index=False)
    counts = manifest.groupby("folder").size()
    print(f"copied {len(items)} figures to {fig_dir}: " + ", ".join(f"{k} {v}" for k, v in counts.items()))


def figure_float(it: dict) -> str:
    return "\n".join([r"\begin{figure}[!htbp]", r"  \centering",
                      rf"  \includegraphics[width=\columnwidth]{{\suppfigdir/{it['folder']}/{it['file']}}}",
                      rf"  \caption{{{it['caption']}}}", rf"  \label{{{it['label']}}}", r"\end{figure}", ""])


def figures_tex(items: list[dict], path: str):
    out = ["% Generated by make_supplementary.py -- do not edit by hand.",
           "% Figures of the main paper are copied into the folders but not repeated here.", ""]
    shown_items = [it for it in items if not it["in_main_paper"]]
    out += [r"\subsection{Stability summaries over all FR models}", r"\label{sec:s-fig-summary}", ""]
    out += [figure_float(it) for it in shown_items if it["folder"] == SUMMARY_FOLDER]
    out += [r"\clearpage", ""]
    for fr in FR_MODELS:
        folder = FR_FOLDER[fr]
        out += [rf"\subsection{{{FR_NAMES[fr]}}}", rf"\label{{sec:s-fig-{folder}}}", ""]
        for ds in DATASETS:
            out += [figure_float(it) for it in shown_items if it["folder"] == folder and it["file"].startswith(f"{ds}_")]
            out += [r"\clearpage", ""]
    write(path, "\n".join(out))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Tables and figure floats of the PURE-FIQA supplementary material")
    p.add_argument("--main-root", default=os.path.join("output", "protocol_stability"))
    p.add_argument("--pair-root", default=os.path.join("output", "protocol_stability_pair_bootstrap"))
    p.add_argument("--midrank-root", default=os.path.join("output", "protocol_stability_midrank"))
    p.add_argument("--pit-root", default=os.path.join("output", "pit_baselines"))
    p.add_argument("--ablation-root", default=os.path.join("output", "ablation_normalization"))
    p.add_argument("--fr-features-root", default="data/FR_models")
    p.add_argument("--ca-fiqa-data-root", default="data/pairs/pairs")
    p.add_argument("--stats-csv", default=os.path.join("output", "supplementary", "dataset_stats.csv"))
    p.add_argument("--out-dir", default=os.path.join("paper", "supplementary"))
    p.add_argument("--skip-data", action="store_true", help="reuse --stats-csv instead of reading the data")
    args = p.parse_args()

    read = lambda root, f: pd.read_csv(os.path.join(root, f))
    ms, pw, ps = (read(args.main_root, f) for f in ("method_summary.csv", "pairwise.csv", "protocol_summary.csv"))
    tab_dir = os.path.join(args.out_dir, "tables")

    if args.skip_data and os.path.exists(args.stats_csv):
        st = pd.read_csv(args.stats_csv)
    else:
        st = dataset_statistics(args.fr_features_root, args.ca_fiqa_data_root)
        os.makedirs(os.path.dirname(args.stats_csv), exist_ok=True)
        st.to_csv(args.stats_csv, index=False)
        print(f"wrote {args.stats_csv}")

    tables_datasets(st, tab_dir)                                                    # S1
    tables_leaderboards(ms, tab_dir)                                                # S2
    table_claims(ms, pw, tab_dir)                                                   # S3
    table_stability(ps, tab_dir)                                                    # S4
    table_bootstrap(ps, read(args.pair_root, "protocol_summary.csv"), tab_dir)      # S5
    tables_pit(read(args.pit_root, "pit_baselines.csv"), read(args.pit_root, "pit_vs_random.csv"), tab_dir)  # S6
    tables_ablation(read(args.ablation_root, "ablation_summary.csv"), tab_dir)      # S7
    items = figure_items()                                                          # S8
    copy_figures(items, os.path.join(args.out_dir, "figures"))
    figures_tex(items, os.path.join(args.out_dir, "sec_figures.tex"))
    ts = tie_summary(ms, read(args.midrank_root, "method_summary.csv"), ps,
                     read(args.midrank_root, "protocol_summary.csv"))
    table_ties(ts, tab_dir)                                                         # S9


if __name__ == "__main__":
    main()
