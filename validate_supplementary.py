#!/usr/bin/env python3
"""Cross-validation of the PURE-FIQA supplementary material and of the main-paper tables.

Independent checks of what make_supplementary.py wrote (no file is modified):
  1. summaries   method_summary / pairwise / protocol_summary of every run, recomputed from the
                 per-resample scores (scores.csv) with an independent implementation
  2. runs        shared resamples: EDC identical in every resample of the main, mid-rank and
                 ablation runs; identical scores on the original evaluation set (main vs pair bootstrap)
  3. datasets    counts recomputed from the pair lists with pandas only (not the experiment code);
                 errors recomputed with the threshold function of the PIT code path
  4. tables      every number of every supplementary table parsed back from the .tex file and
                 compared with values recomputed here from the result CSVs
  5. main paper  paper/tables/*.tex against the same recomputed values
  6. text        every factual statement of paper/supplementary.tex
  7. latex       braces, environments, $ signs, column counts, \\input and \\includegraphics files,
                 unique labels, resolved references; figure folders (one per FR model + summary): every copy
                 identical to its source, every figure shown once (except the main-paper ones),
                 captions matching their files, no figure older than its results
Report only (not a failure): statements of the main paper that the per-FR values contradict.

Run from the repository root:   python validate_supplementary.py
Exit code 1 if any check fails.
"""

import glob
import hashlib
import os
import re
import sys

import numpy as np  # type: ignore[reportMissingImports]
import pandas as pd
from scipy.stats import rankdata, spearmanr  # type: ignore[reportMissingImports]

from protocol_stability import DATASET_NAMES, FR_NAMES, METHOD_NAMES
from rce_prob_test1 import harm_counts, list_jobs, load_job, threshold_at_fmr

DATASETS = ["lfw", "xqlfw", "calfw", "cplfw"]
FR_MODELS = ["adaface", "arcface_o", "magface", "swinface"]
PROTOCOLS = ["EDC", "RCE_g0", "RCE_g3"]
RUNS = {"main": "output/protocol_stability", "pair": "output/protocol_stability_pair_bootstrap",
        "midrank": "output/protocol_stability_midrank", "ablation": "output/ablation_normalization"}
PAPER = "paper"
SUPP = os.path.join(PAPER, "supplementary")
TAB = os.path.join(SUPP, "tables")
NAME_TO_KEY = {v: k for k, v in METHOD_NAMES.items()}
DS_TO_KEY = {v: k for k, v in DATASET_NAMES.items() if k in DATASETS}
FR_TO_KEY = {v: k for k, v in FR_NAMES.items()}
REF_NAMES = {"Random ranking": "random", "Oracle ranking": "oracle", "Oracle + noise": "oracle_noise"}

RESULTS = []          # (section, check, ok, detail)
REPORT = []           # main-paper statements contradicted by the per-FR values


def check(section: str, what: str, ok: bool, detail: str = ""):
    RESULTS.append((section, what, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {section}: {what}" + (f"  ({detail})" if detail else ""), flush=True)


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def to_float(s: str) -> float:
    s = s.replace("$-$", "-").replace(r"\textbf{", "").replace("}", "").replace("{", "").strip()
    return float(s)


def decimals(s: str) -> int:
    s = s.replace("$-$", "-").replace(r"\textbf{", "").replace("}", "").strip()
    return len(s.split(".")[1]) if "." in s else 0


def shown(s: str, value: float, scale: float = 1.0) -> bool:
    """Does the displayed string s show value * scale, rounded to its number of decimals?"""
    return abs(to_float(s) - scale * value) <= 0.5 * 10 ** (-decimals(s)) + 1e-9


def split_cells(line: str) -> list[str]:
    cells, depth, cur, i = [], 0, "", 0
    while i < len(line):
        ch = line[i]
        if ch == "\\" and i + 1 < len(line):
            cur += line[i:i + 2]
            i += 2
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        if ch == "&" and depth == 0:
            cells.append(cur.strip())
            cur = ""
        else:
            cur += ch
        i += 1
    cells.append(cur.strip())
    return cells


def table_rows(path: str):
    """Data rows of the (only) tabular in a table file: list of cell lists; a group header such as
    \\multicolumn{10}{l}{...} is returned as ['#group', text]."""
    text = open(path).read()
    lines = [l.strip() for l in text.split("\n")]
    start = lines.index(r"\toprule")
    first_mid = start + lines[start:].index(r"\midrule")
    end = lines.index(r"\bottomrule")
    rows = []
    for l in lines[first_mid + 1:end]:
        if not l or l.startswith(r"\midrule"):
            continue
        if not l.endswith(r"\\"):
            raise ValueError(f"{path}: unexpected line {l!r}")
        l = l[:-2].strip()
        cells = split_cells(l)
        if len(cells) == 1 and cells[0].startswith(r"\multicolumn"):
            rows.append(["#group", cells[0]])
        else:
            rows.append(cells)
    return rows


def carry_dataset(rows):
    """Rows of per-FR tables: the dataset name is only given on the first row of a group."""
    out, ds = [], None
    for r in rows:
        if r[0] == "#group":
            out.append(r)
            continue
        if r[0]:
            ds = DS_TO_KEY[r[0]]
        out.append([ds] + r[1:])
    return out


SCORE_CELL = re.compile(r"^(\\textbf\{)?(\S+?)\}? \{\\scriptsize\[(\S+), (\S+)\]\}$")


def parse_score_cell(cell: str):
    m = SCORE_CELL.match(cell)
    if not m:
        raise ValueError(f"bad score cell {cell!r}")
    return m.group(2), m.group(3), m.group(4), bool(m.group(1))


# ---------------------------------------------------------------------------
# 1. Summaries recomputed from the per-resample scores
# ---------------------------------------------------------------------------

def recompute_summaries(scores: pd.DataFrame):
    scores = scores.copy()
    scores["key"] = ["EDC" if p == "EDC" else f"RCE_g{a:g}" for p, a in zip(scores.protocol, scores.alpha)]
    ms, pw, ps = [], [], []
    for (ds, fr, key), g in scores.groupby(["dataset", "fr_model", "key"]):
        wide = g.pivot(index="sample", columns="method", values="score").sort_index()
        methods = list(wide.columns)
        sign = 1.0 if key.startswith("RCE") else -1.0               # EDC: lower pAUC is better
        better = sign * wide.to_numpy()
        ranks = rankdata(-better, axis=1, method="average")          # 1 = best
        obs_i = list(wide.index).index(-1)
        res = np.delete(np.arange(len(wide)), obs_i)
        S, R = wide.to_numpy()[res], ranks[res]
        for j, m in enumerate(methods):
            ms.append({"dataset": ds, "fr_model": fr, "protocol": key, "method": m,
                       "observed": wide.to_numpy()[obs_i, j], "observed_rank": ranks[obs_i, j],
                       "mean": S[:, j].mean(), "q025": np.quantile(S[:, j], 0.025),
                       "q975": np.quantile(S[:, j], 0.975), "rank_median": np.median(R[:, j]),
                       "rank_q05": np.quantile(R[:, j], 0.05), "rank_q95": np.quantile(R[:, j], 0.95),
                       "p_rank1": float(np.mean(R[:, j] == 1))})
        B = better[res]
        und = 0
        for i in range(len(methods)):
            for j in range(len(methods)):
                if i == j:
                    continue
                p = np.mean(B[:, i] > B[:, j]) + 0.5 * np.mean(B[:, i] == B[:, j])
                pw.append({"dataset": ds, "fr_model": fr, "protocol": key, "a": methods[i], "b": methods[j],
                           "p_a_beats_b": p})
                if i < j and 0.05 < p < 0.95:
                    und += 1
        n_pairs = len(methods) * (len(methods) - 1) // 2
        width = np.mean(np.quantile(R, 0.95, axis=0) - np.quantile(R, 0.05, axis=0))
        ps.append({"dataset": ds, "fr_model": fr, "protocol": key, "n_methods": len(methods),
                   "n_samples": len(res), "undecided_pairs_pct": 100.0 * und / n_pairs,
                   "mean_rank_interval_90": width})
    return pd.DataFrame(ms), pd.DataFrame(pw), pd.DataFrame(ps)


def compare_frames(sec, what, saved: pd.DataFrame, mine: pd.DataFrame, keys, cols, tol=1e-9):
    m = saved.merge(mine, on=keys, suffixes=("_saved", "_mine"), how="outer", indicator=True)
    unmatched = int((m._merge != "both").sum())
    worst = max(float(np.nanmax(np.abs(m[f"{c}_saved"] - m[f"{c}_mine"]))) for c in cols)
    check(sec, what, unmatched == 0 and worst <= tol, f"{len(m)} rows, unmatched {unmatched}, max |diff| {worst:.1e}")


def part_summaries():
    data = {}
    for run, root in RUNS.items():
        scores = pd.read_csv(os.path.join(root, "scores.csv"))
        ms, pw, ps = recompute_summaries(scores)
        k = ["dataset", "fr_model", "protocol"]
        compare_frames("1 summaries", f"{run}: method_summary.csv", pd.read_csv(os.path.join(root, "method_summary.csv")),
                       ms, k + ["method"], ["observed", "observed_rank", "mean", "q025", "q975", "rank_median",
                                            "rank_q05", "rank_q95", "p_rank1"])
        compare_frames("1 summaries", f"{run}: pairwise.csv", pd.read_csv(os.path.join(root, "pairwise.csv")),
                       pw, k + ["a", "b"], ["p_a_beats_b"])
        compare_frames("1 summaries", f"{run}: protocol_summary.csv", pd.read_csv(os.path.join(root, "protocol_summary.csv")),
                       ps, k, ["undecided_pairs_pct", "mean_rank_interval_90", "n_samples"])
        check("1 summaries", f"{run}: complete grid", scores.groupby(["dataset", "fr_model"]).ngroups == 16
              and scores.method.nunique() == 17 and scores["sample"].nunique() == 401
              and set(ms.protocol) == set(PROTOCOLS), "4 datasets x 4 FR models x 17 methods x (1 + 400) samples")
        data[run] = {"scores": scores, "ms": ms, "pw": pw, "ps": ps}
    return data


# ---------------------------------------------------------------------------
# 2. Runs share resamples
# ---------------------------------------------------------------------------

def part_runs(data):
    key = ["dataset", "fr_model", "protocol", "alpha", "method", "sample"]
    main = data["main"]["scores"]
    for run in ("midrank", "ablation"):
        x = main[main.protocol == "EDC"].merge(data[run]["scores"].query("protocol == 'EDC'"), on=key)
        check("2 runs", f"EDC identical in all 401 samples, main vs {run} (same identity resamples)",
              len(x) == 17 * 16 * 401 and float((x.score_x - x.score_y).abs().max()) == 0.0, f"{len(x)} rows")
    o = main[main["sample"] == -1].merge(data["pair"]["scores"].query("sample == -1"), on=key)
    check("2 runs", "main vs pair bootstrap: identical scores on the original evaluation set",
          len(o) == 17 * 16 * 3 and float((o.score_x - o.score_y).abs().max()) == 0.0, f"{len(o)} rows")
    o = main[(main["sample"] == -1) & (main.protocol == "RCE")].merge(
        data["midrank"]["scores"].query("sample == -1 and protocol == 'RCE'"), on=key)
    check("2 runs", "main vs mid-rank: RCE differs on the original evaluation set (other tie rule)",
          float((o.score_x - o.score_y).abs().min()) > 0)


# ---------------------------------------------------------------------------
# 3. Dataset statistics
# ---------------------------------------------------------------------------

def part_datasets():
    st = pd.read_csv("output/supplementary/dataset_stats.csv")
    ident_re = re.compile(r"^(.+)_\d+\.[A-Za-z]+$")
    mine, hashes = [], {}
    for ds in DATASETS:
        path = f"data/pairs/pairs/{ds}/pairs/pairs.csv"
        p = pd.read_csv(path)
        lo, hi = p.img1.where(p.img1 < p.img2, p.img2), p.img2.where(p.img1 < p.img2, p.img1)
        hashes[ds] = hashlib.md5("\n".join(sorted(lo + "|" + hi + "|" + p.label.astype(str))).encode()).hexdigest()
        imgs = pd.Index(pd.concat([p.img1, p.img2]).unique())
        ids = imgs.map(lambda s: (ident_re.match(os.path.basename(s)) or [None, "__own__" + s])[1])
        per = pd.Series(ids).value_counts()
        mine.append({"dataset": ds, "images": len(imgs), "identities": len(per), "max_images_per_identity": per.max(),
                     "single_image_identities": int((per == 1).sum()), "genuine_pairs": int((p.label == 1).sum()),
                     "impostor_pairs": int((p.label == 0).sum())})
    mine = pd.DataFrame(mine).set_index("dataset")
    cols = list(mine.columns)
    ok = all((st[st.dataset == ds][cols] == mine.loc[ds, cols].values).all(axis=None) for ds in DATASETS)
    check("3 datasets", "counts from the pair lists equal dataset_stats.csv (all FR models)", ok)

    rows = {r[0]: r[1:] for r in table_rows(os.path.join(TAB, "tab_s_datasets.tex"))}
    ok = all([int(c.replace(",", "")) for c in rows[DATASET_NAMES[ds]]] == [int(mine.loc[ds, c]) for c in cols]
             for ds in DATASETS)
    check("3 datasets", "Table tab_s_datasets shows these counts", ok)

    # Errors with the threshold function of the PIT code path (rce_prob_test1.threshold_at_fmr).
    found = {(d, m): (pkl, pairs) for d, m, _, pkl, pairs in list_jobs("data/FR_models", "data/pairs/pairs", run_all=True)}
    pit = pd.read_csv("output/pit_baselines/pit_baselines.csv")
    err = []
    for ds in DATASETS:
        for fr in FR_MODELS:
            ids, i1, i2, lab, sims = load_job(*found[(ds, fr)])
            thr = threshold_at_fmr(sims[lab == 0], 1e-3, "roc")
            fail = ((lab == 1) & (sims < thr)) | ((lab == 0) & (sims >= thr))
            err.append({"dataset": ds, "fr_model": fr, "fnmr": 100 * np.mean(sims[lab == 1] < thr),
                        "err": 100 * np.mean(harm_counts(len(ids), i1, i2, fail) > 0), "n": len(ids)})
    err = pd.DataFrame(err)
    x = err.merge(st, on=["dataset", "fr_model"])
    check("3 datasets", "FNMR and error-causing images: independent threshold function = dataset_stats.csv",
          np.allclose(x.fnmr, x.fnmr_pct, atol=1e-12) and np.allclose(x.err, x.error_images_pct, atol=1e-12)
          and (x.n == x.images).all())
    pe = pit.groupby(["dataset", "fr_model"]).error_images_pct.agg(["min", "max"]).reset_index().merge(err)
    check("3 datasets", "error-causing images = pit_baselines.csv (PIT run)",
          np.allclose(pe["min"], pe.err, atol=1e-9) and np.allclose(pe["max"], pe.err, atol=1e-9))
    rows = {r[0]: r[1:] for r in table_rows(os.path.join(TAB, "tab_s_errors.tex"))}
    ok = all(shown(rows[DATASET_NAMES[ds]][2 * j], err.query("dataset == @ds and fr_model == @fr").fnmr.iloc[0])
             and shown(rows[DATASET_NAMES[ds]][2 * j + 1], err.query("dataset == @ds and fr_model == @fr").err.iloc[0])
             for ds in DATASETS for j, fr in enumerate(FR_MODELS))
    check("3 datasets", "Table tab_s_errors shows these values", ok)
    return mine, err, hashes


# ---------------------------------------------------------------------------
# 4. Supplementary tables
# ---------------------------------------------------------------------------

def part_tables(data, pit_stats):
    ms, pw, ps = data["main"]["ms"], data["main"]["pw"], data["main"]["ps"]

    # S2 leaderboards
    n_cells, bad = 0, []
    for ds in DATASETS:
        for key, tag in (("EDC", "edc"), ("RCE_g0", "rce0"), ("RCE_g3", "rce3")):
            rows = table_rows(os.path.join(TAB, f"tab_s_lb_{ds}_{tag}.tex"))
            g = ms[(ms.dataset == ds) & (ms.protocol == key)]
            scale = 1e3 if key == "EDC" else 1.0
            methods = [NAME_TO_KEY[r[0]] for r in rows]
            mean_rank = g.groupby("method").observed_rank.mean()
            expected_order = sorted(mean_rank.index, key=lambda m: (mean_rank[m], METHOD_NAMES[m].lower()))
            if methods != expected_order or len(methods) != 17:
                bad.append(f"{ds}/{key}: row order")
            for r in rows:
                m = NAME_TO_KEY[r[0]]
                for j, fr in enumerate(FR_MODELS):
                    e = g[(g.fr_model == fr) & (g.method == m)].iloc[0]
                    v, lo, hi, bold = parse_score_cell(r[1 + 2 * j])
                    ok = (shown(v, e.observed, scale) and shown(lo, e.q025, scale) and shown(hi, e.q975, scale)
                          and shown(r[2 + 2 * j], e.p_rank1) and bold == (e.observed_rank == 1))
                    n_cells += 1
                    if not ok:
                        bad.append(f"{ds}/{key}/{m}/{fr}")
    check("4 tables", "S2 leaderboards: every score, interval, P^top, bold mark and row order", not bad,
          f"{n_cells} cells checked" + (f", wrong: {bad[:5]}" if bad else ""))

    # S3 claims
    bad = []
    rows = carry_dataset(table_rows(os.path.join(TAB, "tab_s_claims.tex")))
    for r in rows:
        ds, fr = r[0], FR_TO_KEY[r[1]]
        for k, key in enumerate(("RCE_g0", "RCE_g3")):
            g = ms[(ms.dataset == ds) & (ms.fr_model == fr) & (ms.protocol == key)]
            top = g.loc[g.observed_rank.idxmin()]
            if NAME_TO_KEY[r[2 + 2 * k]] != top.method or not shown(r[3 + 2 * k], top.p_rank1) or top.observed_rank != 1:
                bad.append(f"{ds}/{fr}/{key} top")
            p = pw[(pw.dataset == ds) & (pw.fr_model == fr) & (pw.protocol == key) & (pw.a == "ediffiqa_l")
                   & (pw.b == "froq_ada")].p_a_beats_b.iloc[0]
            if not shown(r[6 + k], p):
                bad.append(f"{ds}/{fr}/{key} P")
    check("4 tables", "S3 claims per FR model", not bad and len(rows) == 16, f"{len(rows)} rows" + (f", wrong: {bad}" if bad else ""))

    # S4 stability
    bad = []
    rows = carry_dataset(table_rows(os.path.join(TAB, "tab_s_stability.tex")))
    for r in rows:
        ds, fr = r[0], FR_TO_KEY[r[1]]
        g = ps[(ps.dataset == ds) & (ps.fr_model == fr)].set_index("protocol")
        for k, key in enumerate(PROTOCOLS):
            if not (shown(r[2 + k], g.loc[key, "undecided_pairs_pct"]) and shown(r[5 + k], g.loc[key, "mean_rank_interval_90"])):
                bad.append(f"{ds}/{fr}/{key}")
    check("4 tables", "S4 stability per FR model", not bad and len(rows) == 16, f"{len(rows)} rows" + (f", wrong: {bad}" if bad else ""))

    # S5 bootstrap
    bad = []
    pp = data["pair"]["ps"]
    rows = carry_dataset(table_rows(os.path.join(TAB, "tab_s_bootstrap.tex")))
    for r in rows:
        ds, fr = r[0], FR_TO_KEY[r[1]]
        a = pp[(pp.dataset == ds) & (pp.fr_model == fr)].set_index("protocol")
        b = ps[(ps.dataset == ds) & (ps.fr_model == fr)].set_index("protocol")
        for k, key in enumerate(PROTOCOLS):
            if not (shown(r[2 + 2 * k], a.loc[key, "undecided_pairs_pct"]) and shown(r[3 + 2 * k], b.loc[key, "undecided_pairs_pct"])):
                bad.append(f"{ds}/{fr}/{key}")
    check("4 tables", "S5 pair vs identity bootstrap per FR model", not bad and len(rows) == 16,
          f"{len(rows)} rows" + (f", wrong: {bad}" if bad else ""))

    # S6 PIT per dataset (mean over FR models)
    bad, n = [], 0
    for ds in DATASETS:
        rows = table_rows(os.path.join(TAB, f"tab_s_pit_{ds}.tex"))
        g = pit_stats[pit_stats.dataset == ds]
        names = [r[0] for r in rows]
        meth_names = [x for x in names if x not in REF_NAMES]
        if meth_names != sorted(meth_names, key=str.lower) or len(meth_names) != 17 or names[17:] != list(REF_NAMES):
            bad.append(f"{ds}: row order")
        for r in rows:
            rk = REF_NAMES.get(r[0]) or NAME_TO_KEY[r[0]]
            h = g[g.ranking == rk]
            if h.fr_model.nunique() != 4 or len(h) != 4:
                bad.append(f"{ds}/{rk}: {len(h)} FR rows")
            vals = [100 * h.phi_low.mean(), 100 * h.phi_high.mean(), h.ks_stat.mean(), h.auc.mean()]
            n += 1
            if not all(shown(c, v) for c, v in zip(r[1:], vals)):
                bad.append(f"{ds}/{rk}")
    check("4 tables", "S6 PIT per method and reference ranking (mean over FR models)", not bad,
          f"{n} rows" + (f", wrong: {bad[:5]}" if bad else ""))

    # S6 vs random: 2-sd rule recomputed here from pit_baselines.csv
    bad = []
    comp_saved = pd.read_csv("output/pit_baselines/pit_vs_random.csv")
    rows = carry_dataset(table_rows(os.path.join(TAB, "tab_s_pit_random.tex")))
    mine = {}
    for r in rows:
        ds, fr = r[0], FR_TO_KEY[r[1]]
        g = pit_stats[(pit_stats.dataset == ds) & (pit_stats.fr_model == fr)]
        rnd = g[g.ranking == "random"].iloc[0]
        m = g[g.kind == "method"]
        exp = {"closer": int((m.ks_stat < rnd.ks_stat - 2 * rnd.ks_stat_sd).sum()),
               "further": int((m.ks_stat > rnd.ks_stat + 2 * rnd.ks_stat_sd).sum()),
               "above": int((m.auc > 0.5 + 2 * rnd.auc_sd).sum()), "below": int((m.auc < 0.5 - 2 * rnd.auc_sd).sum())}
        mine[(ds, fr)] = exp
        mean_sd = r[2].split(r" $\pm$ ")
        dmin, dmax = r[3].split("--")
        amin, amax = r[6].split("--")
        ok = (shown(mean_sd[0], rnd.ks_stat) and shown(mean_sd[1], rnd.ks_stat_sd) and shown(dmin, m.ks_stat.min())
              and shown(dmax, m.ks_stat.max()) and shown(amin, m.auc.min()) and shown(amax, m.auc.max())
              and [int(r[4]), int(r[5]), int(r[7]), int(r[8])] == [exp["closer"], exp["further"], exp["above"], exp["below"]])
        s = comp_saved[(comp_saved.dataset == ds) & (comp_saved.fr_model == fr)].iloc[0]
        ok &= [s.n_ks_below_random, s.n_ks_above_random, s.n_auc_above_chance, s.n_auc_below_chance] == list(exp.values())
        if not ok:
            bad.append(f"{ds}/{fr}")
    check("4 tables", "S6 FIQA methods vs random ranking (counts recomputed with the 2-sd rule)", not bad and len(rows) == 16,
          f"{len(rows)} rows" + (f", wrong: {bad}" if bad else ""))

    # S7 ablation: recomputed from the method/protocol summaries of both runs
    bad = []
    msa, psa = data["ablation"]["ms"], data["ablation"]["ps"]
    ab_saved = pd.read_csv("output/ablation_normalization/ablation_summary.csv")
    for key, tag in (("RCE_g0", "rce0"), ("RCE_g3", "rce3")):
        rows = carry_dataset(table_rows(os.path.join(TAB, f"tab_s_ablation_{tag}.tex")))
        for r in rows:
            ds, fr = r[0], FR_TO_KEY[r[1]]
            c = ms[(ms.dataset == ds) & (ms.fr_model == fr) & (ms.protocol == key)].set_index("method")
            nn = msa[(msa.dataset == ds) & (msa.fr_model == fr) & (msa.protocol == key)].set_index("method").loc[c.index]
            pc = ps[(ps.dataset == ds) & (ps.fr_model == fr) & (ps.protocol == key)].iloc[0]
            pn = psa[(psa.dataset == ds) & (psa.fr_model == fr) & (psa.protocol == key)].iloc[0]
            agr = spearmanr(c.observed, nn.observed)[0]
            tc, tn = c.observed.idxmax(), nn.observed.idxmax()
            ok = (shown(r[2], pc.undecided_pairs_pct) and shown(r[3], pn.undecided_pairs_pct)
                  and shown(r[4], pc.mean_rank_interval_90) and shown(r[5], pn.mean_rank_interval_90)
                  and shown(r[6], agr) and NAME_TO_KEY[r[7]] == tc and shown(r[8], c.loc[tc, "p_rank1"])
                  and NAME_TO_KEY[r[9]] == tn and shown(r[10], nn.loc[tn, "p_rank1"]))
            s = ab_saved[(ab_saved.dataset == ds) & (ab_saved.fr_model == fr) & (ab_saved.protocol == key)].iloc[0]
            ok &= (abs(s.order_agreement - agr) < 1e-12 and s.top_counts == tc and s.top_norm == tn
                   and abs(s.U_norm - pn.undecided_pairs_pct) < 1e-9 and abs(s.W_norm - pn.mean_rank_interval_90) < 1e-9)
            if not ok:
                bad.append(f"{ds}/{fr}/{key}")
        if len(rows) != 16:
            bad.append(f"{key}: {len(rows)} rows")
    check("4 tables", "S7 ablation per FR model (recomputed from both runs)", not bad, f"wrong: {bad}" if bad else "32 rows")

    # S9 ties
    bad = []
    msm, psm = data["midrank"]["ms"], data["midrank"]["ps"]
    rows = table_rows(os.path.join(TAB, "tab_s_ties.tex"))
    key = None
    n = 0
    for r in rows:
        if r[0] == "#group":
            key = "RCE_g0" if "RCE}_0" in r[1] else "RCE_g3"
            continue
        ds = DS_TO_KEY[r[0]]
        a = ms[(ms.dataset == ds) & (ms.protocol == key)]
        b = msm[(msm.dataset == ds) & (msm.protocol == key)]
        pa = ps[(ps.dataset == ds) & (ps.protocol == key)]
        pb = psm[(psm.dataset == ds) & (psm.protocol == key)]
        agr = np.mean([spearmanr(a[a.fr_model == f].set_index("method").observed,
                                 b[b.fr_model == f].set_index("method").observed.loc[a[a.fr_model == f].method])[0]
                       for f in FR_MODELS])

        def top(t):
            c = t.loc[t.groupby("fr_model")["observed"].idxmax(), "method"].value_counts()
            return f"{METHOD_NAMES[c.index[0]]} ({c.iloc[0]}/4)"

        lo_b, hi_b = r[1].split(" to ")
        lo_a, hi_a = r[2].split(" to ")
        ok = (shown(lo_b, b.observed.min()) and shown(hi_b, b.observed.max()) and shown(lo_a, a.observed.min())
              and shown(hi_a, a.observed.max()) and shown(r[3], pb.undecided_pairs_pct.mean())
              and shown(r[4], pa.undecided_pairs_pct.mean()) and shown(r[5], pb.mean_rank_interval_90.mean())
              and shown(r[6], pa.mean_rank_interval_90.mean()) and shown(r[7], agr)
              and r[8] == top(b) and r[9] == top(a))
        n += 1
        if not ok:
            bad.append(f"{ds}/{key}")
    check("4 tables", "S9 mid-rank vs margin (recomputed from both runs)", not bad and n == 8, f"{n} rows" + (f", wrong: {bad}" if bad else ""))


# ---------------------------------------------------------------------------
# 5. Main-paper tables
# ---------------------------------------------------------------------------

def part_main_tables(data, pit_stats):
    ms, pw, ps, pp = data["main"]["ms"], data["main"]["pw"], data["main"]["ps"], data["pair"]["ps"]
    T = os.path.join(PAPER, "tables")

    bad = []
    for r in table_rows(os.path.join(T, "tab_leaderboard.tex")):
        m = NAME_TO_KEY[r[0]]
        for j, ds in enumerate(DATASETS):
            e = ms[(ms.dataset == ds) & (ms.fr_model == "adaface") & (ms.protocol == "RCE_g0") & (ms.method == m)].iloc[0]
            v, lo, hi, bold = parse_score_cell(r[1 + 2 * j])
            if not (shown(v, e.observed) and shown(lo, e.q025) and shown(hi, e.q975) and shown(r[2 + 2 * j], e.p_rank1)
                    and bold == (e.observed_rank == 1)):
                bad.append(f"{ds}/{m}")
    check("5 main paper", "tab_leaderboard (AdaFace, RCE_0)", not bad, f"wrong: {bad[:5]}" if bad else "")

    def rng(s):
        a, b = s.split("--")
        return a, b

    bad = []
    for r in table_rows(os.path.join(T, "tab_claims.tex")):
        ds = DS_TO_KEY[r[0]]
        for k, key in enumerate(("RCE_g0", "RCE_g3")):
            top = ms[(ms.dataset == ds) & (ms.protocol == key) & (ms.observed_rank == 1)]
            c = top.method.value_counts()
            lead = c.index[0]
            p = top[top.method == lead].p_rank1
            lo, hi = rng(r[2 + 2 * k])
            if r[1 + 2 * k] != f"{METHOD_NAMES[lead]} ({c.iloc[0]}/4)" or not (shown(lo, p.min()) and shown(hi, p.max())):
                bad.append(f"{ds}/{key} top")
            q = pw[(pw.dataset == ds) & (pw.protocol == key) & (pw.a == "ediffiqa_l") & (pw.b == "froq_ada")].p_a_beats_b
            lo, hi = rng(r[5 + k])
            if not (shown(lo, q.min()) and shown(hi, q.max())):
                bad.append(f"{ds}/{key} P")
    check("5 main paper", "tab_claims", not bad, f"wrong: {bad}" if bad else "")

    bad = []
    cell = re.compile(r"^(\S+) \{\\scriptsize\((\S+)--(\S+)\)\}$")
    for r in table_rows(os.path.join(T, "tab_stability.tex")):
        ds = DS_TO_KEY[r[0]]
        for k, (key, col) in enumerate([(p, "undecided_pairs_pct") for p in PROTOCOLS]
                                       + [(p, "mean_rank_interval_90") for p in PROTOCOLS]):
            v = ps[(ps.dataset == ds) & (ps.protocol == key)][col]
            m = cell.match(r[1 + k])
            if not (m and shown(m.group(1), v.mean()) and shown(m.group(2), v.min()) and shown(m.group(3), v.max())):
                bad.append(f"{ds}/{key}/{col}")
    check("5 main paper", "tab_stability (mean and range over FR models)", not bad, f"wrong: {bad}" if bad else "")

    bad = []
    for r in table_rows(os.path.join(T, "tab_bootstrap.tex")):
        ds = DS_TO_KEY[r[0]]
        for k, key in enumerate(PROTOCOLS):
            a = pp[(pp.dataset == ds) & (pp.protocol == key)].undecided_pairs_pct.mean()
            b = ps[(ps.dataset == ds) & (ps.protocol == key)].undecided_pairs_pct.mean()
            if not (shown(r[1 + 2 * k], a) and shown(r[2 + 2 * k], b)):
                bad.append(f"{ds}/{key}")
    check("5 main paper", "tab_bootstrap", not bad, f"wrong: {bad}" if bad else "")

    bad = []
    ds = None
    head = re.compile(r"\\textit\{(\w+)\} \(\$D_\{\\mathrm\{null\}\} = (\S+)\$, (\S+)\\% error-causing images\)")
    for r in table_rows(os.path.join(T, "tab_pit_baselines.tex")):
        if r[0] == "#group":
            m = head.search(r[1])
            ds = DS_TO_KEY[m.group(1)]
            g = pit_stats[pit_stats.dataset == ds]
            if not (shown(m.group(2), g.ks_null_median.mean()) and shown(m.group(3), g.error_images_pct.mean())):
                bad.append(f"{ds} header")
            continue
        g = pit_stats[pit_stats.dataset == ds]
        if r[0] == "FIQA methods":
            mm = g[g.kind == "method"]
            for c, col, s in ((1, "phi_low", 100), (2, "phi_high", 100), (3, "ks_stat", 1), (5, "auc", 1)):
                lo, hi = rng(r[c])
                if not (shown(lo, s * mm[col].min()) and shown(hi, s * mm[col].max())):
                    bad.append(f"{ds}/methods/{col}")
            if not shown(r[4], mm.ks_pvalue.max()):
                bad.append(f"{ds}/methods/p")
        else:
            h = g[g.ranking == REF_NAMES[r[0]]]
            vals = [100 * h.phi_low.mean(), 100 * h.phi_high.mean(), h.ks_stat.mean(), h.ks_pvalue.max(), h.auc.mean()]
            if not all(shown(c, v) for c, v in zip(r[1:], vals)):
                bad.append(f"{ds}/{r[0]}")
    check("5 main paper", "tab_pit_baselines", not bad, f"wrong: {bad}" if bad else "")

    bad = []
    msa, psa = data["ablation"]["ms"], data["ablation"]["ps"]
    for r in table_rows(os.path.join(T, "tab_ablation_normalization.tex")):
        ds = DS_TO_KEY[r[0]]
        for k, key in enumerate(("RCE_g0", "RCE_g3")):
            uc = ps[(ps.dataset == ds) & (ps.protocol == key)].undecided_pairs_pct.mean()
            un = psa[(psa.dataset == ds) & (psa.protocol == key)].undecided_pairs_pct.mean()
            agr = np.mean([spearmanr(ms[(ms.dataset == ds) & (ms.fr_model == f) & (ms.protocol == key)].set_index("method").observed,
                                     msa[(msa.dataset == ds) & (msa.fr_model == f) & (msa.protocol == key)].set_index("method").observed
                                     .loc[ms[(ms.dataset == ds) & (ms.fr_model == f) & (ms.protocol == key)].method])[0]
                           for f in FR_MODELS])
            if not (shown(r[1 + 2 * k], uc) and shown(r[2 + 2 * k], un) and shown(r[5 + k], agr)):
                bad.append(f"{ds}/{key}")
        tops = msa[(msa.dataset == ds) & (msa.protocol == "RCE_g0")]
        tops = tops.loc[tops.groupby("fr_model")["observed"].idxmax()]
        c = tops.method.value_counts()
        p = tops[tops.method == c.index[0]].p_rank1
        lo, hi = rng(r[8])
        if r[7] != f"{METHOD_NAMES[c.index[0]]} ({c.iloc[0]}/4)" or not (shown(lo, p.min()) and shown(hi, p.max())):
            bad.append(f"{ds}/top")
    check("5 main paper", "tab_ablation_normalization", not bad, f"wrong: {bad}" if bad else "")


# ---------------------------------------------------------------------------
# 6. Statements of the supplementary text
# ---------------------------------------------------------------------------

def part_text(data, pit_stats, counts, err, hashes):
    ms, pw, ps, pp = data["main"]["ms"], data["main"]["pw"], data["main"]["ps"], data["pair"]["ps"]
    msa, psa, msm, psm = data["ablation"]["ms"], data["ablation"]["ps"], data["midrank"]["ms"], data["midrank"]["ps"]
    S = "6 text"
    U = lambda t, ds, fr, key: t[(t.dataset == ds) & (t.fr_model == fr) & (t.protocol == key)].undecided_pairs_pct.iloc[0]

    # S1
    check(S, "S1: LFW and XQLFW have the same pair list (same labelled pairs; row order differs) and counts",
          hashes["lfw"] == hashes["xqlfw"] and (counts.loc["lfw"] == counts.loc["xqlfw"]).all())
    check(S, "S1: up to 530 images per identity on LFW/XQLFW, at most 4 on CALFW and 3 on CPLFW",
          list(counts.max_images_per_identity) == [530, 530, 4, 3], str(list(counts.max_images_per_identity)))
    xq, other = err[err.dataset == "xqlfw"].err, err[err.dataset != "xqlfw"].err
    check(S, "S1: error-causing images about half on XQLFW, below a quarter elsewhere",
          xq.between(40, 60).all() and (other < 25).all(), f"XQLFW {xq.min():.1f}-{xq.max():.1f}%, others max {other.max():.1f}%")
    l = err[err.dataset == "lfw"]
    check(S, "S1: on LFW, the error rates hardly depend on the FR model",
          l.fnmr.max() - l.fnmr.min() < 0.01 and l.err.max() - l.err.min() < 0.5,
          f"FNMR {l.fnmr.min():.3f}-{l.fnmr.max():.3f}%, error images {l.err.min():.2f}-{l.err.max():.2f}%")

    # S3
    p = {fr: pw[(pw.dataset == "calfw") & (pw.fr_model == fr) & (pw.protocol == "RCE_g0") & (pw.a == "ediffiqa_l")
                & (pw.b == "froq_ada")].p_a_beats_b.iloc[0] for fr in FR_MODELS}
    tops = {fr: ms[(ms.dataset == "calfw") & (ms.fr_model == fr) & (ms.protocol == "RCE_g0") & (ms.observed_rank == 1)].method.iloc[0]
            for fr in FR_MODELS}
    check(S, "S3: CALFW, RCE_0: P(eDifFIQA > FROQ) lowest for AdaFace, the only FR model with PFE (not MagFace) first",
          min(p, key=p.get) == "adaface" and tops["adaface"] == "pfe" and all(tops[f] == "magface" for f in FR_MODELS[1:]),
          f"P {', '.join(f'{f} {v:.2f}' for f, v in p.items())}; top {tops}")

    # S4
    ok = all((U(ps, ds, fr, "RCE_g0") < U(ps, ds, fr, "EDC")) == (ds != "xqlfw") for ds in DATASETS for fr in FR_MODELS)
    ok &= all(U(ps, "xqlfw", fr, "RCE_g0") > U(ps, "xqlfw", fr, "EDC") for fr in FR_MODELS)
    check(S, "S4: RCE_0 fewer undecided pairs than EDC on LFW/CALFW/CPLFW, more on XQLFW, for every FR model", ok)
    check(S, "S4: RCE_3 more undecided pairs than RCE_0 on every dataset and FR model",
          all(U(ps, ds, fr, "RCE_g3") > U(ps, ds, fr, "RCE_g0") for ds in DATASETS for fr in FR_MODELS))

    # S5
    ratios = [U(pp, ds, fr, k) / U(ps, ds, fr, k) for ds in ("lfw", "xqlfw") for fr in FR_MODELS for k in PROTOCOLS]
    check(S, "S5: on LFW and XQLFW, the pair bootstrap reports at most half as many undecided pairs (every FR model, protocol)",
          max(ratios) <= 0.5, f"largest ratio pair/identity {max(ratios):.2f}")
    d_big = min(U(ps, ds, fr, k) - U(pp, ds, fr, k) for ds in ("lfw", "xqlfw") for fr in FR_MODELS for k in PROTOCOLS)
    d_small = max(abs(U(ps, ds, fr, k) - U(pp, ds, fr, k)) for ds in ("calfw", "cplfw") for fr in FR_MODELS for k in PROTOCOLS)
    check(S, "S5: on CALFW and CPLFW, the two schemes differ much less", d_small < d_big,
          f"largest |difference| on CALFW/CPLFW {d_small:.0f} points, smallest difference on LFW/XQLFW {d_big:.0f} points")

    # S6
    check(S, "S6: every ranking rejected with p = 1/201 for every dataset and FR model",
          np.allclose(pit_stats.ks_pvalue, 1 / 201, atol=1e-12), f"p-values in [{pit_stats.ks_pvalue.min():.6f}, {pit_stats.ks_pvalue.max():.6f}]")
    closer = {}
    for (ds, fr), g in pit_stats.groupby(["dataset", "fr_model"]):
        rnd = g[g.ranking == "random"].iloc[0]
        m = g[g.kind == "method"]
        closer[(ds, fr)] = set(m[m.ks_stat < rnd.ks_stat - 2 * rnd.ks_stat_sd].ranking)
    check(S, "S6: number of methods closer to H0 than random is the same for all FR models on every dataset",
          all(len({len(closer[(ds, fr)]) for fr in FR_MODELS}) == 1 for ds in DATASETS),
          "; ".join(f"{ds} {sorted({len(closer[(ds, fr)]) for fr in FR_MODELS})}" for ds in DATASETS))
    all_m = set(pit_stats[pit_stats.kind == "method"].ranking)
    check(S, "S6: on CPLFW, the only method not closer is PCNet (every FR model)",
          all(all_m - closer[("cplfw", fr)] == {"pcnet"} for fr in FR_MODELS),
          str({fr: sorted(all_m - closer[("cplfw", fr)]) for fr in FR_MODELS}))

    # S7
    check(S, "S7: normalization lowers U on XQLFW for both gamma and every FR model",
          all(U(psa, "xqlfw", fr, k) < U(ps, "xqlfw", fr, k) for fr in FR_MODELS for k in ("RCE_g0", "RCE_g3")))
    check(S, "S7: normalization raises U under RCE_0 on LFW, CALFW and CPLFW for every FR model",
          all(U(psa, ds, fr, "RCE_g0") > U(ps, ds, fr, "RCE_g0") for ds in ("lfw", "calfw", "cplfw") for fr in FR_MODELS))

    # S9
    raised = {}
    for ds in DATASETS:
        for fr in FR_MODELS:
            a = ms[(ms.dataset == ds) & (ms.fr_model == fr) & (ms.protocol == "RCE_g0")].set_index("method").observed
            b = msm[(msm.dataset == ds) & (msm.fr_model == fr) & (msm.protocol == "RCE_g0")].set_index("method").observed.loc[a.index]
            raised[(ds, fr)] = (b > a)
    check(S, "S9: mid-ranks raise RCE_0 of every method on LFW, CALFW and CPLFW (every FR model)",
          all(raised[(ds, fr)].all() for ds in ("lfw", "calfw", "cplfw") for fr in FR_MODELS))
    check(S, "S9: ... but not on XQLFW (for every FR model, some methods are not raised)",
          all(not raised[("xqlfw", fr)].all() for fr in FR_MODELS),
          ", ".join(f"{fr}: {int(raised[('xqlfw', fr)].sum())}/17 raised" for fr in FR_MODELS))
    mid0 = msm[msm.protocol == "RCE_g0"]
    v = mid0[mid0.dataset.isin(["lfw", "calfw"])].observed
    check(S, "S9: on LFW and CALFW, every method scores about 0.3 with mid-ranks", v.between(0.25, 0.36).all(),
          f"{v.min():.3f}-{v.max():.3f}")
    r = [U(ps, ds, fr, "RCE_g0") / U(psm, ds, fr, "RCE_g0") for ds in ("lfw", "calfw", "cplfw") for fr in FR_MODELS]
    check(S, "S9: margin tie-breaking leaves far fewer undecided pairs under RCE_0 on LFW/CALFW/CPLFW, "
             "about half as many or fewer, for every FR model",
          max(r) <= 0.6, f"ratio margin/mid {min(r):.2f}-{max(r):.2f}")
    agr = [spearmanr(ms[(ms.dataset == "lfw") & (ms.fr_model == fr) & (ms.protocol == "RCE_g0")].set_index("method").observed,
                     msm[(msm.dataset == "lfw") & (msm.fr_model == fr) & (msm.protocol == "RCE_g0")].set_index("method").observed
                     .loc[ms[(ms.dataset == "lfw") & (ms.fr_model == fr) & (ms.protocol == "RCE_g0")].method])[0]
           for fr in FR_MODELS]
    check(S, "S9: on LFW, the two leaderboards are unrelated", max(abs(a) for a in agr) < 0.2,
          "Spearman " + ", ".join(f"{a:+.2f}" for a in agr))
    return closer


# ---------------------------------------------------------------------------
# 7. LaTeX checks
# ---------------------------------------------------------------------------

def strip_comments(text: str) -> str:
    return "\n".join(re.sub(r"(?<!\\)%.*$", "", l) for l in text.split("\n"))


def part_latex():
    L = "7 latex"
    main = os.path.join(PAPER, "supplementary.tex")
    files = [main, os.path.join(SUPP, "sec_figures.tex")] + sorted(glob.glob(os.path.join(TAB, "*.tex")))
    texts = {f: strip_comments(open(f).read()) for f in files}

    # \input targets: every input file exists, and every generated table is used exactly once
    inputs = re.findall(r"\\input\{([^}]+)\}", texts[main])
    missing = [i for i in inputs if not os.path.exists(os.path.join(PAPER, i + ".tex"))]
    used = sorted(os.path.join(PAPER, i + ".tex") for i in inputs if "tables/" in i)
    unused = sorted(set(glob.glob(os.path.join(TAB, "*.tex"))) - set(used))
    check(L, "every \\input file exists; every table is input exactly once", not missing and not unused
          and len(used) == len(set(used)), f"{len(inputs)} inputs" + (f", missing {missing}" if missing else "")
          + (f", unused {unused}" if unused else ""))

    bad = []
    for f, t in texts.items():
        t2 = t.replace("\\\\", "").replace(r"\{", "").replace(r"\}", "")
        depth = 0
        for ch in t2:
            depth += ch == "{"
            depth -= ch == "}"
            if depth < 0:
                break
        if depth != 0:
            bad.append(f"{f}: braces {depth:+d}")
        stack = []
        for kind, env in re.findall(r"\\(begin|end)\{([^}]+)\}", t):
            if kind == "begin":
                stack.append(env)
            elif not stack or stack.pop() != env:
                bad.append(f"{f}: \\end{{{env}}} unmatched")
                break
        if stack:
            bad.append(f"{f}: unclosed {stack}")
        for i, l in enumerate(t.split("\n")):
            if l.replace(r"\$", "").count("$") % 2:
                bad.append(f"{os.path.basename(f)}:{i + 1}: odd number of $")
    check(L, "balanced braces, environments and $ signs in all files", not bad, "; ".join(bad[:5]))

    # column counts
    bad, n_rows = [], 0
    for f in files:
        for m in re.finditer(r"\\begin\{tabular\}\{([^}]*)\}(.*?)\\end\{tabular\}", texts[f], re.S):
            ncol = len(re.findall(r"[lcr]", m.group(1)))
            for line in m.group(2).split("\n"):
                line = line.strip()
                if not line.endswith(r"\\"):
                    continue
                cells = split_cells(line[:-2])
                width = 0
                for c in cells:
                    mc = re.match(r"\\multicolumn\{(\d+)\}", c)
                    width += int(mc.group(1)) if mc else 1
                n_rows += 1
                if width != ncol:
                    bad.append(f"{os.path.basename(f)}: {width} != {ncol}: {line[:60]}")
    check(L, "every table row has as many cells as its column specification", not bad,
          f"{n_rows} rows" + (f"; {bad[:3]}" if bad else ""))

    # figures: folders, copies, inclusion, captions
    fig_tex = os.path.join(SUPP, "sec_figures.tex")
    macros = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}", texts[main]))
    floats = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}\s*\\caption\{(.*?)\}\s*\\label", texts[fig_tex], re.S)
    graphics = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", texts[fig_tex])
    paths = []
    for g in graphics:
        for k, v in macros.items():
            g = g.replace("\\" + k, v)
        paths.append(os.path.normpath(os.path.join(PAPER, g)))
    missing = [p for p in paths if not os.path.exists(p)]
    check(L, "every \\includegraphics file exists", not missing and len(floats) == len(graphics),
          f"{len(paths)} figures" + (f", missing {missing[:3]}" if missing else ""))

    # expected folder contents, built here from the output layout
    fig_dir = os.path.join(SUPP, "figures")
    folder_of = {"adaface": "adaface", "arcface_o": "arcface", "magface": "magface", "swinface": "swinface"}
    ps_fig, pit_fig, ab_fig = "output/protocol_stability/figures", "output/pit_baselines/figures", \
        "output/ablation_normalization/figures/ablation"
    kinds = ["pure_fiqa_leaderboard_g0", "pure_fiqa_leaderboard_g3", "rank_intervals_g0", "rank_intervals_g3",
             "pairwise_EDC", "pairwise_RCE_g0", "pairwise_RCE_g3"]
    expected = {}
    for k in ("undecided_pairs_g0", "undecided_pairs_g3", "rank_interval_width_g0", "rank_interval_width_g3"):
        expected[("summary", f"{k}.pdf")] = f"{ps_fig}/summary/{k}.pdf"
        expected[("summary", f"ablation_{k}.pdf")] = f"{ab_fig}/{k}.pdf"
    for fr, folder in folder_of.items():
        for ds in DATASETS:
            for k in kinds:
                expected[(folder, f"{ds}_{k}.pdf")] = f"{ps_fig}/{ds}/{fr}/{k}.pdf"
            expected[(folder, f"{ds}_pit.pdf")] = f"{pit_fig}/{fr}/pit_{ds}.pdf"
            for g in ("0", "3"):
                expected[(folder, f"{ds}_ablation_scores_g{g}.pdf")] = f"{ab_fig}/{ds}/{fr}/scores_g{g}.pdf"
    present = {(os.path.basename(os.path.dirname(f)), os.path.basename(f))
               for f in glob.glob(os.path.join(fig_dir, "*", "*"))}
    md5 = lambda f: hashlib.md5(open(f, "rb").read()).hexdigest()
    differ = [k for k, src in expected.items() if k in present and md5(os.path.join(fig_dir, *k)) != md5(src)]
    manifest = pd.read_csv(os.path.join(fig_dir, "manifest.csv"))
    man = {(r.folder, r.file): r.source for r in manifest.itertuples()}
    counts = pd.Series([k[0] for k in present]).value_counts().to_dict()
    check(L, "figure folders: exactly the expected files, each identical to its source in output/",
          present == set(expected) and not differ and man == expected,
          f"{len(present)} files: {counts}" + (f"; missing {sorted(set(expected) - present)[:3]}" if set(expected) - present else "")
          + (f"; extra {sorted(present - set(expected))[:3]}" if present - set(expected) else "")
          + (f"; differ {differ[:3]}" if differ else "") + ("" if man == expected else "; manifest differs"))

    main_figs = {("adaface", "lfw_pure_fiqa_leaderboard_g0.pdf"), ("adaface", "lfw_rank_intervals_g0.pdf"),
                 ("adaface", "lfw_pairwise_RCE_g0.pdf"), ("adaface", "lfw_pit.pdf"), ("summary", "undecided_pairs_g0.pdf")}
    included = [(os.path.basename(os.path.dirname(p)), os.path.basename(p)) for p in paths]
    ok = (sorted(included) == sorted(set(expected) - main_figs)) and len(included) == len(set(included))
    check(L, "the document shows every figure once, except the 5 figures of the main paper", ok,
          f"{len(included)} of {len(expected)} figures")

    # every caption names the dataset, FR model and protocol of its file
    bad, current = [], None
    sections = re.split(r"\\subsection\{([^}]*)\}", texts[fig_tex])
    for title, body in zip(sections[1::2], sections[2::2]):
        for g, cap in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}\s*\\caption\{(.*?)\}\s*\\label", body, re.S):
            folder, name = g.split("/")[-2], g.split("/")[-1][:-4]
            want = []
            if folder != "summary":
                fr = {v: k for k, v in folder_of.items()}[folder]
                ds = name.split("_")[0]
                want.append(f"{DATASET_NAMES[ds]} ({FR_NAMES[fr]})")
                if title != FR_NAMES[fr]:
                    bad.append(f"{folder}/{name} under subsection {title!r}")
            elif "all FR models" not in title:
                bad.append(f"{folder}/{name} under subsection {title!r}")
            if name.endswith(("_g0", "_RCE_g0")):
                want.append(r"$\mathrm{RCE}_0$")
            if name.endswith(("_g3", "_RCE_g3")):
                want.append(r"$\mathrm{RCE}_3$")
            if "pairwise_EDC" in name:
                want.append("under EDC on")
            if name.endswith("_pit"):
                want.append("PIT density")
            if "ablation" in name:
                want.append("Ablation")
            if "undecided_pairs" in name:
                want.append("undecided")
            if "rank_interval_width" in name:
                want.append("rank interval")
            if "leaderboard" in name:
                want.append("PURE-FIQA mean")
            if "rank_intervals" in name:
                want.append("90\\% rank interval")
            miss = [w for w in want if w not in cap]
            if miss:
                bad.append(f"{folder}/{name}: caption lacks {miss}")
    check(L, "every figure caption names its dataset, FR model and protocol; figures sit in their FR subsection",
          not bad, "; ".join(bad[:4]) if bad else f"{len(floats)} captions")

    # figures not older than the results they show (sources and copies)
    results = {ps_fig: "output/protocol_stability/scores.csv", pit_fig: "output/pit_baselines/pit_histograms.csv",
               ab_fig: "output/ablation_normalization/scores.csv"}
    stale = [src for src in expected.values() for k, res in results.items()
             if src.startswith(k) and os.path.getmtime(src) < os.path.getmtime(res)]
    stale += [os.path.join(fig_dir, *k) for k, src in expected.items()
              if os.path.getmtime(os.path.join(fig_dir, *k)) < os.path.getmtime(src)]
    check(L, "no figure is older than the results it shows", not stale, f"stale: {stale[:3]}" if stale else "")

    # labels and references
    labels = [l for t in texts.values() for l in re.findall(r"\\label\{([^}]+)\}", t)]
    dup = sorted({l for l in labels if labels.count(l) > 1})
    refs = {r for t in texts.values() for r in re.findall(r"\\ref\{([^}]+)\}", t)}
    unresolved = sorted(refs - set(labels))
    check(L, "labels unique and every \\ref resolved", not dup and not unresolved,
          f"{len(labels)} labels, {len(refs)} refs" + (f", duplicate {dup}" if dup else "")
          + (f", unresolved {unresolved}" if unresolved else ""))
    cites = {c for t in texts.values() for c in re.findall(r"\\cite\{([^}]+)\}", t)}
    main_cites = set(re.findall(r"\\cite\{([^}]+)\}", open(os.path.join(PAPER, "pure_fiqa_fg2027.tex")).read()))
    check(L, "every \\cite key is also cited by the main paper (same egbib.bib)", cites <= main_cites, str(sorted(cites)))

    # commands: everything used is standard or defined in the preamble
    known = set("""begin end documentclass usepackage newcommand renewcommand def title author maketitle section
        label ref cite input caption centering includegraphics textbf textit scriptsize small LARGE Large bf
        toprule midrule bottomrule cmidrule multicolumn resizebox setlength tabcolsep shortstack textwidth
        columnwidth clearpage mathrm mathcal text succ le ge pm gamma phi thesection thetable thefigure arabic
        IEEEoverridecommandlockouts overrideIEEEmargins FGPaperID ifFGfinal else fi thispagestyle pagestyle
        bibliographystyle bibliography subsection FGfinalcopy""".split())
    defined = set(macros)
    used_cmds = {c for t in texts.values() for c in re.findall(r"(?<!\\)\\([A-Za-z]+)", t.replace("\\\\", " "))}
    unknown = sorted(used_cmds - known - defined)
    check(L, "no undefined command (all standard, from the loaded packages, or defined in the preamble)", not unknown,
          f"unknown: {unknown}" if unknown else f"{len(used_cmds)} distinct commands")

    # number of floats per \clearpage block (LaTeX keeps at most 18 floats in its queue)
    body = texts[main]
    for i in inputs:
        body = body.replace("\\input{" + i + "}", texts.get(os.path.join(PAPER, i + ".tex"), ""))
    blocks = body.split(r"\clearpage")
    worst = max(len(re.findall(r"\\begin\{(table\*?|figure\*?)\}", b)) for b in blocks)
    check(L, "at most 18 floats between two \\clearpage", worst <= 18, f"largest block: {worst} floats")
    n_tab = len(re.findall(r"\\begin\{table\*?\}", body))
    n_fig = len(re.findall(r"\\begin\{figure\*?\}", body))
    print(f"        supplementary: {n_tab} tables, {n_fig} figures")
    return n_tab, n_fig


# ---------------------------------------------------------------------------
# Report: main-paper statements contradicted by the per-FR values
# ---------------------------------------------------------------------------

def report_main_paper(data, err):
    ps, psa = data["main"]["ps"], data["ablation"]["ps"]
    free = 100 - err.err
    lo, hi = free.min(), free.max()
    per_ds = err.assign(free=100 - err.err).groupby("dataset").free.agg(["min", "max"]).round(1)
    if lo < 84 or hi > 87:
        REPORT.append("pure_fiqa_fg2027.tex (margin tie-breaking paragraph): 'between 84% and 87% of the images of the "
                      f"evaluated datasets cause no error' -- error-free images per dataset (min/max over FR models): "
                      + ", ".join(f"{ds} {per_ds.loc[ds, 'min']}-{per_ds.loc[ds, 'max']}%" for ds in DATASETS))
    lfw = ps[ps.dataset == "lfw"].groupby("protocol").undecided_pairs_pct.mean()
    if lfw.max() > 70:
        REPORT.append("results_section.tex (stability): 'On LFW, about half of the method pairs remain undecided under "
                      "every protocol' -- LFW U: " + ", ".join(f"{k} {v:.0f}%" for k, v in lfw.items())
                      + " (at least about half)")
    m_c = ps[ps.protocol.str.startswith("RCE")].groupby(["dataset", "protocol"]).undecided_pairs_pct.mean()
    m_n = psa[psa.protocol.str.startswith("RCE")].groupby(["dataset", "protocol"]).undecided_pairs_pct.mean()
    better = [f"{ds} {k}: {m_c[(ds, k)]:.0f} -> {m_n[(ds, k)]:.0f}%" for ds, k in m_c.index
              if ds != "xqlfw" and m_n[(ds, k)] < m_c[(ds, k)]]
    if better:
        REPORT.append("results_section.tex (ablation): 'Normalization improves stability only on XQLFW' -- U also "
                      "decreases for " + "; ".join(better))
    cp = err[err.dataset == "cplfw"].err
    if cp.max() > 21:
        REPORT.append("results_section.tex (ablation): error-causing images 'at most about one fifth on the other "
                      f"datasets' -- true for the means in the PIT table, but CPLFW reaches {cp.max():.1f}% (MagFace)")


# ---------------------------------------------------------------------------

def main():
    print("== 1. summaries recomputed from scores.csv")
    data = part_summaries()
    print("== 2. shared resamples")
    part_runs(data)
    print("== 3. dataset statistics")
    counts, err, hashes = part_datasets()
    pit_stats = pd.read_csv("output/pit_baselines/pit_baselines.csv")
    print("== 4. supplementary tables")
    part_tables(data, pit_stats)
    print("== 5. main-paper tables")
    part_main_tables(data, pit_stats)
    print("== 6. statements of the supplementary text")
    part_text(data, pit_stats, counts, err, hashes)
    print("== 7. LaTeX")
    part_latex()
    report_main_paper(data, err)

    n_fail = sum(not ok for _, _, ok, _ in RESULTS)
    print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} checks passed")
    if REPORT:
        print("\nMain paper -- statements contradicted by the per-FR values (not changed):")
        for r in REPORT:
            print("  - " + r)
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
