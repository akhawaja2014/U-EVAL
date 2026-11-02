#!/usr/bin/env python3
"""LaTeX tables for the PURE-FIQA Results section, generated from the result CSVs.

Inputs:
    output/protocol_stability/                 identity bootstrap, margin RCE (main results)
    output/protocol_stability_pair_bootstrap/  pair bootstrap, margin RCE (ablation)
    output/rce_prob_test1/pit_test_all.csv     consistency diagnostic (pair bootstrap)

Outputs (paper/tables/):
    tab_stability.tex     undecided pairs U and rank-interval width W: EDC vs RCE_0 vs RCE_3
    tab_leaderboard.tex   RCE_gamma on the observed pairs with 95% interval and P(top), one FR model
    tab_claims.tex        top method and P(top), P(eDifFIQA (L) beats FROQ)
    tab_bootstrap.tex     U under the identity vs the pair bootstrap
    tab_pit.tex           consistency diagnostic: phi_low, phi_high, KS distance

Run:
    python make_results_tables.py
    python make_results_tables.py --leaderboard-fr adaface --leaderboard-gamma 0
"""

import argparse
import os

import numpy as np  # type: ignore[reportMissingImports]
import pandas as pd

DATASETS = ["lfw", "xqlfw", "calfw", "cplfw"]
DATASET_NAMES = {"lfw": "LFW", "xqlfw": "XQLFW", "calfw": "CALFW", "cplfw": "CPLFW"}
FR_NAMES = {"adaface": "AdaFace", "arcface_o": "ArcFace", "magface": "MagFace", "swinface": "SwinFace"}
METHOD_NAMES = {
    "clib-fiqa": "CLIB-FIQA", "cr-fiqa": "CR-FIQA (L)", "diffiqa_r": "DifFIQA (R)",
    "ediffiqa_l": "eDifFIQA (L)", "faceqan": "FaceQAN", "faceqgen": "FaceQGen",
    "faceqnet": "FaceQnet", "froq_ada": "FROQ", "grafiqs": "GraFIQs",
    "lightqnet": "LightQNet", "magface": "MagFace", "pcnet": "PCNet", "pfe": "PFE",
    "sdd-fiqa": "SDD-FIQA", "ser-fiq": "SER-FIQA", "vit-fiqa": "ViT-FIQA",
    "vitnt-fiqa": "ViTNT-FIQA",
}
PROTOCOLS = [("EDC", "EDC"), ("RCE_g0", r"$\mathrm{RCE}_0$"), ("RCE_g3", r"$\mathrm{RCE}_3$")]


def mean_range(v: pd.Series, fmt: str = "{:.0f}") -> str:
    """mean (min--max) over FR models."""
    return f"{fmt.format(v.mean())} {{\\scriptsize({fmt.format(v.min())}--{fmt.format(v.max())})}}"


def write(path: str, body: str):
    with open(path, "w") as f:
        f.write(body)
    print(f"wrote {path}")


def table_stability(ps: pd.DataFrame, out: str):
    rows = []
    for ds in DATASETS:
        g = ps[ps.dataset == ds]
        cells = [mean_range(g[g.protocol == k].undecided_pairs_pct) for k, _ in PROTOCOLS]
        cells += [mean_range(g[g.protocol == k].mean_rank_interval_90, "{:.1f}") for k, _ in PROTOCOLS]
        rows.append(f"{DATASET_NAMES[ds]} & " + " & ".join(cells) + r" \\")
    head = " & ".join(n for _, n in PROTOCOLS)
    write(out, rf"""\begin{{table*}}[t]
\caption{{Leaderboard stability under the identity bootstrap ($B = 400$). $U$: share of FIQA method
pairs whose order is undecided ($0.05 < P(m \succ m') < 0.95$), Eq.~\eqref{{eq:undecided}};
$W$: mean width of the 90\% rank interval in ranks, Eq.~\eqref{{eq:width}}. Mean over the four
FR models, range in parentheses. Lower is more stable.}}
\label{{tab:stability}}
\centering
\begin{{tabular}}{{lcccccc}}
  \toprule
  & \multicolumn{{3}}{{c}}{{$U$ (\%)}} & \multicolumn{{3}}{{c}}{{$W$ (ranks)}} \\
  \cmidrule(lr){{2-4}} \cmidrule(lr){{5-7}}
  Dataset & {head} & {head} \\
  \midrule
  {chr(10).join("  " + r for r in rows).strip()}
  \bottomrule
\end{{tabular}}
\end{{table*}}
""")


def table_leaderboard(ms: pd.DataFrame, fr: str, gamma: str, out: str):
    key = f"RCE_g{gamma}"
    m = ms[(ms.protocol == key) & (ms.fr_model == fr)]
    order = m[m.dataset == "lfw"].sort_values("observed", ascending=False).method.tolist()
    rows = []
    for meth in order:
        cells = []
        for ds in DATASETS:
            r = m[(m.dataset == ds) & (m.method == meth)]
            if r.empty:
                cells.append("--")
                continue
            r = r.iloc[0]
            best = r.observed_rank == 1
            val = f"{r.observed:.3f}"
            val = rf"\textbf{{{val}}}" if best else val
            cells.append(rf"{val} {{\scriptsize[{r.q025:.2f}, {r.q975:.2f}]}} & {r.p_rank1:.2f}")
        rows.append(f"{METHOD_NAMES.get(meth, meth)} & " + " & ".join(cells) + r" \\")
    head = " & ".join(rf"\multicolumn{{2}}{{c}}{{{DATASET_NAMES[d]}}}" for d in DATASETS)
    sub = " & ".join(r"$\mathrm{RCE}_" + gamma + r"$ [95\% int.] & $P^{\mathrm{top}}$" for _ in DATASETS)
    cmid = " ".join(rf"\cmidrule(lr){{{2 + 2 * k}-{3 + 2 * k}}}" for k in range(len(DATASETS)))
    write(out, rf"""\begin{{table*}}[t]
\caption{{$\mathrm{{RCE}}_{gamma}$ on the observed pairs with its 95\% identity-bootstrap interval and
the probability of ranking first, $P^{{\mathrm{{top}}}}$ (Eq.~\eqref{{eq:ptop}}), {FR_NAMES.get(fr, fr)}.
Bold: first place on the observed pairs. Methods sorted by $\mathrm{{RCE}}_{gamma}$ on LFW.}}
\label{{tab:leaderboard}}
\centering
\setlength{{\tabcolsep}}{{3pt}}
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{l{"cc" * len(DATASETS)}}}
  \toprule
  & {head} \\
  {cmid}
  Method & {sub} \\
  \midrule
{chr(10).join("  " + r for r in rows)}
  \bottomrule
\end{{tabular}}}}
\end{{table*}}
""")


def table_claims(ms: pd.DataFrame, pw: pd.DataFrame, out: str):
    rows = []
    for ds in DATASETS:
        cells = []
        for key in ("RCE_g0", "RCE_g3"):
            top = ms[(ms.protocol == key) & (ms.dataset == ds) & (ms.observed_rank == 1)]
            names = top.method.value_counts()
            lead = names.index[0]
            p = top[top.method == lead].p_rank1
            n_fr = f"{names.iloc[0]}/{top.fr_model.nunique()}"
            cells.append(rf"{METHOD_NAMES.get(lead, lead)} ({n_fr}) & {p.min():.2f}--{p.max():.2f}")
        for key in ("RCE_g0", "RCE_g3"):
            q = pw[(pw.protocol == key) & (pw.dataset == ds) & (pw.a == "ediffiqa_l") & (pw.b == "froq_ada")]
            cells.append(f"{q.p_a_beats_b.min():.2f}--{q.p_a_beats_b.max():.2f}")
        rows.append(f"{DATASET_NAMES[ds]} & " + " & ".join(cells) + r" \\")
    write(out, rf"""\begin{{table*}}[t]
\caption{{Conclusions of RCE~\cite{{wani2026beyond}} under the identity bootstrap. Top method: most
frequent first place on the observed pairs (number of FR models in parentheses) and its
$P^{{\mathrm{{top}}}}$ range over these FR models. Last two columns: range of
$P(\text{{eDifFIQA (L)}} \succ \text{{FROQ}})$ over the four FR models.}}
\label{{tab:claims}}
\centering
\begin{{tabular}}{{lcccccc}}
  \toprule
  & \multicolumn{{2}}{{c}}{{$\mathrm{{RCE}}_0$}} & \multicolumn{{2}}{{c}}{{$\mathrm{{RCE}}_3$}}
  & \multicolumn{{2}}{{c}}{{$P(\text{{eDifFIQA (L)}} \succ \text{{FROQ}})$}} \\
  \cmidrule(lr){{2-3}} \cmidrule(lr){{4-5}} \cmidrule(lr){{6-7}}
  Dataset & Top method & $P^{{\mathrm{{top}}}}$ & Top method & $P^{{\mathrm{{top}}}}$
  & $\mathrm{{RCE}}_0$ & $\mathrm{{RCE}}_3$ \\
  \midrule
{chr(10).join("  " + r for r in rows)}
  \bottomrule
\end{{tabular}}
\end{{table*}}
""")


def table_bootstrap(ps_id: pd.DataFrame, ps_pair: pd.DataFrame, out: str):
    rows = []
    for ds in DATASETS:
        cells = []
        for k, _ in PROTOCOLS:
            a = ps_pair[(ps_pair.dataset == ds) & (ps_pair.protocol == k)].undecided_pairs_pct.mean()
            b = ps_id[(ps_id.dataset == ds) & (ps_id.protocol == k)].undecided_pairs_pct.mean()
            cells.append(f"{a:.0f} & {b:.0f}")
        rows.append(f"{DATASET_NAMES[ds]} & " + " & ".join(cells) + r" \\")
    head = " & ".join(rf"\multicolumn{{2}}{{c}}{{{n}}}" for _, n in PROTOCOLS)
    write(out, rf"""\begin{{table}}[t]
\caption{{Share of undecided FIQA method pairs $U$ (\%) under the pair bootstrap and the identity
bootstrap (mean over FR models). The pair bootstrap ignores the dependence between pairs of the
same identity and understates the uncertainty.}}
\label{{tab:bootstrap}}
\centering
\setlength{{\tabcolsep}}{{4pt}}
\begin{{tabular}}{{lcccccc}}
  \toprule
  & {head} \\
  \cmidrule(lr){{2-3}} \cmidrule(lr){{4-5}} \cmidrule(lr){{6-7}}
  Dataset & Pair & Id. & Pair & Id. & Pair & Id. \\
  \midrule
{chr(10).join("  " + r for r in rows)}
  \bottomrule
\end{{tabular}}
\end{{table}}
""")


def table_pit(pit: pd.DataFrame, out: str):
    rows = []
    for ds in DATASETS:
        g = pit[pit.dataset == ds]
        rows.append(
            f"{DATASET_NAMES[ds]} & {100 * g['frac_u_below_0.05'].mean():.1f} & "
            f"{100 * g['frac_u_above_0.95'].mean():.1f} & {g.ks_stat.min():.3f}--{g.ks_stat.max():.3f} & "
            f"{g.ks_null_median.mean():.4f} & {g.ks_pvalue.max():.3f}" + r" \\"
        )
    write(out, rf"""\begin{{table}}[t]
\caption{{Consistency diagnostic (pair bootstrap, $B = 400$, $|\mathcal{{C}}| = 200$), all FIQA methods
and FR models. $\phi_{{\mathrm{{low}}}}$, $\phi_{{\mathrm{{high}}}}$: mean share (\%) of images with
$u_i < 0.05$ and $u_i > 0.95$; expected $5\%$ each under $H_0$. $D$: range of the KS distance over
methods; $D_{{\mathrm{{null}}}}$: null median; $\max \hat{{p}}$: largest p-value.}}
\label{{tab:pit}}
\centering
\setlength{{\tabcolsep}}{{4pt}}
\begin{{tabular}}{{lccccc}}
  \toprule
  Dataset & $\phi_{{\mathrm{{low}}}}$ & $\phi_{{\mathrm{{high}}}}$ & $D$ & $D_{{\mathrm{{null}}}}$ & $\max \hat{{p}}$ \\
  \midrule
{chr(10).join("  " + r for r in rows)}
  \bottomrule
\end{{tabular}}
\end{{table}}
""")


def main():
    p = argparse.ArgumentParser(description="LaTeX tables for the PURE-FIQA Results section")
    p.add_argument("--identity-root", default=os.path.join("output", "protocol_stability"))
    p.add_argument("--pair-root", default=os.path.join("output", "protocol_stability_pair_bootstrap"))
    p.add_argument("--pit-csv", default=os.path.join("output", "rce_prob_test1", "pit_test_all.csv"))
    p.add_argument("--out-dir", default=os.path.join("paper", "tables"))
    p.add_argument("--leaderboard-fr", default="adaface")
    p.add_argument("--leaderboard-gamma", default="0")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    ps = pd.read_csv(os.path.join(args.identity_root, "protocol_summary.csv"))
    ms = pd.read_csv(os.path.join(args.identity_root, "method_summary.csv"))
    pw = pd.read_csv(os.path.join(args.identity_root, "pairwise.csv"))
    table_stability(ps, os.path.join(args.out_dir, "tab_stability.tex"))
    table_leaderboard(ms, args.leaderboard_fr, args.leaderboard_gamma, os.path.join(args.out_dir, "tab_leaderboard.tex"))
    table_claims(ms, pw, os.path.join(args.out_dir, "tab_claims.tex"))

    pair_summary = os.path.join(args.pair_root, "protocol_summary.csv")
    if os.path.exists(pair_summary):
        table_bootstrap(ps, pd.read_csv(pair_summary), os.path.join(args.out_dir, "tab_bootstrap.tex"))
    else:
        print(f"[skip] tab_bootstrap.tex: {pair_summary} not found")

    pit = pd.read_csv(args.pit_csv)
    table_pit(pit[pit.dataset.isin(DATASETS)], os.path.join(args.out_dir, "tab_pit.tex"))


if __name__ == "__main__":
    main()
