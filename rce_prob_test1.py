#!/usr/bin/env python3
"""Rank-PIT KS test: is a FIQA quality ranking R_q a plausible draw from the
distribution of harm rankings R_e?

R_e is random because the evaluation pairs are a sample: drawing a different set
of impostor (and/or genuine) pairs moves the FMR threshold t* and changes the
per-image counts c_fn, c_fa, hence E_i = c_fn + c_fa and its rank R_e.

Procedure (per dataset / FR model):

  1. Bootstrap the pairs B times (impostors, genuines or both), re-derive t* at the
     target FMR each time, recompute E_i and rank it with random tie-breaking, so
     every sample R_e^(b) is a permutation of 1..n (rank 1 = lowest harm).
  2. Split the samples in half: A builds the per-image rank CDFs F_i, C calibrates
     the test.
  3. Randomized PIT of a ranking R:  u_i = F_i(R_i - 1) + V_i * [F_i(R_i) - F_i(R_i - 1)].
     If R is a draw from R_e, the u_i are Uniform(0, 1).
  4. Statistic T(R) = KS (or Anderson-Darling) distance of {u_i} from Uniform(0, 1).
  5. Monte Carlo p-value: p = (1 + #{b in C: T(R_e^(b)) >= T(R_q)}) / (|C| + 1).
     The usual KS tables do not apply: ranks within a permutation are dependent and
     the shared threshold moves all images together.

For each FIQA method, R_q = rank of quality descending (rank 1 = best quality),
also with random tie-breaking.

With n ~ 10^4 images the test rejects for essentially any real quality method, so
also report the effect size T(R_q) / median(T_null) to compare methods.

Run (same datasets / FR models / threshold rule as rce.py; --run-all also supported):

    python rce_prob_test1.py \
        --fr-features-root data/FR_models \
        --quality-dir data/FIQA_models \
        --ca-fiqa-data-root data/pairs/pairs \
        --out-root output/rce_prob_test1 \
        --threshold-method roc \
        --weight-alphas 0,3 \
        --resample both --n-samples 400

Each summary row also carries rce.py's Spearman / weighted Spearman on the observed
(non-resampled) harm, so the two evaluations can be compared side by side.

Outputs, under {out_root}/{dataset}/{fr_model}/:
    pit_test_summary.csv      one row per FIQA method (statistics, p-values, effect sizes)
    pit_values.csv            per-image PIT u_i for every method (for PIT histograms)
    null_statistics.csv       T(R_e^(b)) for the calibration samples
and {out_root}/pit_test_all.csv across all runs.
"""

import argparse
import os

import numpy as np  # type: ignore[reportMissingImports]
import pandas as pd

from rce import (
    DEFAULT_DATASET_ALLOWLIST,
    DEFAULT_FR_MODEL_ALLOWLIST,
    DEFAULT_WEIGHT_ALPHAS,
    _alpha_tag,
    _derive_embedding_name,
    _list_embeddings_pkls,
    _quality_pkl_for_dataset,
    _quality_scores_for_ids,
    _rank_ascending,
    _rank_descending,
    _spearman_from_ranks,
    _weighted_spearman,
    find_threshold_closest_fmr,
    find_threshold_for_fmr,
    load_embeddings_pkl,
    load_pairs_csv,
    load_quality_scores_pkl,
)


# ---------------------------------------------------------------------------
# Threshold at target FMR (vectorized versions of the rce.py rules)
# ---------------------------------------------------------------------------

def threshold_at_fmr(imp_sims: np.ndarray, fmr_target: float, method: str) -> float:
    """Similarity threshold t with FMR(t) = mean(imp_sims >= t) at/near fmr_target."""
    if method == "percentile":
        return float(np.percentile(imp_sims, (1.0 - fmr_target) * 100.0))
    # "roc": unique impostor score whose achieved FMR is closest to the target
    # (ties -> smallest threshold, matching find_threshold_closest_fmr in rce.py).
    s = np.sort(imp_sims)
    cand = np.unique(s)
    fmr = (s.size - np.searchsorted(s, cand, side="left")) / s.size
    return float(cand[np.argmin(np.abs(fmr - fmr_target))])


# ---------------------------------------------------------------------------
# Sampling harm rankings
# ---------------------------------------------------------------------------

def rank_random_ties(values: np.ndarray, rng: np.random.Generator, descending: bool = False) -> np.ndarray:
    """1-based ranks with ties broken uniformly at random (result is a permutation)."""
    v = -values if descending else values
    order = np.lexsort((rng.random(v.size), v))
    ranks = np.empty(v.size, dtype=np.int32)
    ranks[order] = np.arange(1, v.size + 1, dtype=np.int32)
    return ranks


def harm_counts(n: int, idx1: np.ndarray, idx2: np.ndarray, sel: np.ndarray) -> np.ndarray:
    """Per-image count of the selected (failed) pairs; each pair counts for both images."""
    return np.bincount(idx1[sel], minlength=n) + np.bincount(idx2[sel], minlength=n)


def sample_harm_rankings(
    *,
    n: int,
    idx1: np.ndarray,
    idx2: np.ndarray,
    sims: np.ndarray,
    labels: np.ndarray,
    n_samples: int,
    resample: str,
    fmr_target: float,
    threshold_method: str,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bootstrap pairs and return (R_e samples (B, n), thresholds (B,), mean harm (n,))."""
    gen = np.flatnonzero(labels == 1)
    imp = np.flatnonzero(labels == 0)

    ranks = np.empty((n_samples, n), dtype=np.int32)
    thresholds = np.empty(n_samples, dtype=float)
    harm_sum = np.zeros(n, dtype=float)
    for b in range(n_samples):
        g = gen[rng.integers(0, gen.size, gen.size)] if resample in ("gen", "both") else gen
        m = imp[rng.integers(0, imp.size, imp.size)] if resample in ("imp", "both") else imp

        t = threshold_at_fmr(sims[m], fmr_target, threshold_method)
        c_fn = harm_counts(n, idx1, idx2, g[sims[g] < t])
        c_fa = harm_counts(n, idx1, idx2, m[sims[m] >= t])
        harm = c_fn + c_fa

        ranks[b] = rank_random_ties(harm, rng)
        thresholds[b] = t
        harm_sum += harm
    return ranks, thresholds, harm_sum / n_samples


# ---------------------------------------------------------------------------
# Rank-PIT and goodness-of-fit statistics
# ---------------------------------------------------------------------------

class RankCDF:
    """Per-image empirical CDFs of R_e,i built from reference samples (|A|, n)."""

    def __init__(self, ref_ranks: np.ndarray):
        n_ref, n = ref_ranks.shape
        self.n_ref = n_ref
        # Rows sorted per image, laid out with a per-image offset so a single
        # searchsorted over the flat array answers all n per-image queries.
        self.offset = np.arange(n, dtype=np.int64) * (n + 2)
        self.flat = (np.sort(ref_ranks.T, axis=1).astype(np.int64) + self.offset[:, None]).ravel()
        self.base = np.arange(n, dtype=np.int64) * n_ref

    def pit(self, ranks: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Randomized PIT u_i = F_i(R_i - 1) + V_i * P(R_e,i = R_i)."""
        q = ranks.astype(np.int64) + self.offset
        lo = (np.searchsorted(self.flat, q, side="left") - self.base) / self.n_ref
        hi = (np.searchsorted(self.flat, q, side="right") - self.base) / self.n_ref
        return lo + rng.random(ranks.size) * (hi - lo)


def ks_uniform(u: np.ndarray) -> float:
    """One-sample KS distance of u from Uniform(0, 1)."""
    u = np.sort(u)
    k = np.arange(1, u.size + 1)
    return float(max(np.max(k / u.size - u), np.max(u - (k - 1) / u.size)))


def ad_uniform(u: np.ndarray, eps: float = 1e-10) -> float:
    """Anderson-Darling A^2 of u against Uniform(0, 1) (tail-sensitive)."""
    u = np.clip(np.sort(u), eps, 1.0 - eps)
    n = u.size
    k = np.arange(1, n + 1)
    return float(-n - np.mean((2 * k - 1) * (np.log(u) + np.log1p(-u[::-1]))))


STATISTICS = {"ks": ks_uniform, "ad": ad_uniform}


def mc_pvalue(t_obs: float, t_null: np.ndarray) -> float:
    return float((1 + np.sum(t_null >= t_obs)) / (t_null.size + 1))


# ---------------------------------------------------------------------------
# Per dataset / FR model
# ---------------------------------------------------------------------------

def load_job(emb_pkl: str, pairs_csv: str):
    """Load one dataset / FR model: (img_ids, idx1, idx2, labels, sims).

    Images get a compact index 0..n-1 aligned with img_ids (independent of how
    the embeddings pickle numbered its rows); pairs with unknown images are dropped.
    """
    embeddings, id_to_idx = load_embeddings_pkl(emb_pkl)
    pairs_df = load_pairs_csv(pairs_csv)
    valid = pairs_df["img1_id"].isin(id_to_idx) & pairs_df["img2_id"].isin(id_to_idx)
    pairs_df = pairs_df[valid].reset_index(drop=True)

    img_ids = list(id_to_idx.keys())
    pos = {img_id: i for i, img_id in enumerate(img_ids)}
    emb_rows = np.array([id_to_idx[i] for i in img_ids], dtype=int)

    idx1 = pairs_df["img1_id"].map(pos).to_numpy(dtype=np.int64)
    idx2 = pairs_df["img2_id"].map(pos).to_numpy(dtype=np.int64)
    labels = pairs_df["label"].astype(int).to_numpy()
    sims = pair_similarity(embeddings[emb_rows], idx1, idx2)
    return img_ids, idx1, idx2, labels, sims


def pair_similarity(embeddings: np.ndarray, idx1: np.ndarray, idx2: np.ndarray, chunk: int = 20000) -> np.ndarray:
    """Cosine similarity per pair, same arithmetic as rce.compute_similarity but in
    chunks (bounded memory instead of two (n_pairs, dim) copies)."""
    norms = np.linalg.norm(embeddings, axis=-1, keepdims=True)
    if not np.allclose(norms, 1.0, atol=1e-6) and np.all(norms > 0):
        embeddings = embeddings / norms
    out = np.empty(idx1.size, dtype=embeddings.dtype)
    for s in range(0, idx1.size, chunk):
        out[s:s + chunk] = np.sum(embeddings[idx1[s:s + chunk]] * embeddings[idx2[s:s + chunk]], axis=1)
    return np.clip(out, -1.0, 1.0)


def run_one(
    *,
    dataset: str,
    fr_model: str,
    emb_tag: str,
    emb_pkl: str,
    pairs_csv: str,
    quality_dir: str,
    methods: list[str] | None,
    out_dir: str,
    n_samples: int,
    resample: str,
    fmr_target: float,
    threshold_method: str,
    weight_alphas: list[float],
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    img_ids, idx1, idx2, labels, sims = load_job(emb_pkl, pairs_csv)
    n = len(img_ids)

    # Observed harm on the full pair set, with rce.py's own threshold rule, so the
    # rce.py Spearman values can be reported next to the test for the same run.
    if threshold_method == "roc":
        t_obs, _ = find_threshold_closest_fmr(sims, labels, fmr_target)
    else:
        t_obs, _ = find_threshold_for_fmr(sims, labels, fmr_target)
    obs_harm = (harm_counts(n, idx1, idx2, (labels == 1) & (sims < t_obs))
                + harm_counts(n, idx1, idx2, (labels == 0) & (sims >= t_obs)))
    obs_harm_rank = _rank_ascending(obs_harm)

    print(f"[{dataset}/{fr_model}] n_images={n} gen_pairs={int((labels == 1).sum())} "
          f"imp_pairs={int((labels == 0).sum())}; sampling B={n_samples} ({resample})")
    R_e, thresholds, mean_harm = sample_harm_rankings(
        n=n, idx1=idx1, idx2=idx2, sims=sims, labels=labels,
        n_samples=n_samples, resample=resample, fmr_target=fmr_target,
        threshold_method=threshold_method, rng=rng,
    )

    # Split into reference (A) and calibration (C) halves.
    perm = rng.permutation(n_samples)
    A, C = perm[: n_samples // 2], perm[n_samples // 2:]
    cdf = RankCDF(R_e[A])

    # Null distribution of each statistic from genuine draws of R_e.
    null_u = [cdf.pit(R_e[b], rng) for b in C]
    t_null = {name: np.array([f(u) for u in null_u]) for name, f in STATISTICS.items()}

    # Observed statistic per FIQA method.
    method_names = sorted(
        d for d in os.listdir(quality_dir)
        if os.path.isdir(os.path.join(quality_dir, d)) and not d.startswith(".")
    )
    if methods:
        method_names = [m for m in method_names if m in set(methods)]

    rows, pit_cols = [], {"img": img_ids, "mean_harm": mean_harm}
    for method in method_names:
        qpath = _quality_pkl_for_dataset(os.path.join(quality_dir, method), dataset)
        if not os.path.exists(qpath):
            continue
        qdf = load_quality_scores_pkl(qpath)
        q = _quality_scores_for_ids(qdf, img_ids)
        have_q = set(qdf["image_id"])
        n_missing = sum(i not in have_q for i in img_ids)
        R_q = rank_random_ties(q, rng, descending=True)

        u = cdf.pit(R_q, rng)
        pit_cols[method] = u
        row = {
            "dataset": dataset,
            "fr_model": fr_model,
            "emb_tag": emb_tag,
            "method": method,
            "n_images": n,
            "n_missing_quality": n_missing,
            "n_samples": n_samples,
            "n_reference": A.size,
            "n_calibration": C.size,
            "resample": resample,
            "fmr_target": fmr_target,
            "threshold_mean": float(thresholds.mean()),
            "threshold_std": float(thresholds.std()),
            "frac_u_below_0.05": float(np.mean(u < 0.05)),
            "frac_u_above_0.95": float(np.mean(u > 0.95)),
        }
        # rce.py metrics on the observed (non-resampled) harm, average-rank ties.
        quality_rank = _rank_descending(q)
        row["threshold_observed"] = float(t_obs)
        row["rce_spearman"] = _spearman_from_ranks(quality_rank, obs_harm_rank)
        for alpha in weight_alphas:
            row[f"rce_weighted_spearman_{_alpha_tag(alpha)}"] = _weighted_spearman(
                obs_harm_rank, quality_rank, alpha=alpha
            )[0]
        for name, f in STATISTICS.items():
            t_stat = f(u)
            med = float(np.median(t_null[name]))
            row[f"{name}_stat"] = t_stat
            row[f"{name}_pvalue"] = mc_pvalue(t_stat, t_null[name])
            row[f"{name}_null_median"] = med
            row[f"{name}_null_q95"] = float(np.quantile(t_null[name], 0.95))
            row[f"{name}_effect"] = t_stat / med if med > 0 else np.nan
        rows.append(row)
        print(f"  {method:14s} KS={row['ks_stat']:.4f} (x{row['ks_effect']:.1f} null, p={row['ks_pvalue']:.3g})  "
              f"AD={row['ad_stat']:.1f} (p={row['ad_pvalue']:.3g})  rce_spearman={row['rce_spearman']:.4f}")

    os.makedirs(out_dir, exist_ok=True)
    summary = pd.DataFrame(rows)
    if len(summary):
        summary = summary.sort_values("ks_effect").reset_index(drop=True)
    summary.to_csv(os.path.join(out_dir, "pit_test_summary.csv"), index=False)
    pd.DataFrame(pit_cols).to_csv(os.path.join(out_dir, "pit_values.csv"), index=False)
    pd.DataFrame({f"{k}_null": v for k, v in t_null.items()}).to_csv(
        os.path.join(out_dir, "null_statistics.csv"), index=False
    )
    return summary


def list_jobs(fr_features_root: str, ca_fiqa_data_root: str, run_all: bool) -> list[tuple[str, str, str, str, str]]:
    """(dataset, fr_model, emb_tag, emb_pkl, pairs_csv) jobs, discovered exactly as in rce.py."""
    datasets = sorted(
        d for d in os.listdir(fr_features_root)
        if os.path.isdir(os.path.join(fr_features_root, d)) and not d.startswith(".")
    )
    if not run_all:
        datasets = [d for d in datasets if d in set(DEFAULT_DATASET_ALLOWLIST)]

    jobs = []
    for dataset in datasets:
        dataset_dir = os.path.join(fr_features_root, dataset)
        pairs_csv = os.path.join(ca_fiqa_data_root, dataset, "pairs", "pairs.csv")
        if not os.path.exists(pairs_csv):
            print(f"[skip] Missing pairs: {pairs_csv}")
            continue

        model_dirs = [
            os.path.join(dataset_dir, d)
            for d in sorted(os.listdir(dataset_dir))
            if os.path.isdir(os.path.join(dataset_dir, d)) and not d.startswith(".")
        ]
        if not run_all:
            model_dirs = [md for md in model_dirs if os.path.basename(md) in set(DEFAULT_FR_MODEL_ALLOWLIST)]
        dataset_pkls = [
            os.path.join(dataset_dir, f)
            for f in sorted(os.listdir(dataset_dir))
            if f.endswith(".pkl") and not f.startswith(".")
        ]

        if len(model_dirs) > 0:
            for md in model_dirs:
                for emb_pkl in _list_embeddings_pkls(md):
                    jobs.append((dataset, os.path.basename(md), _derive_embedding_name(emb_pkl), emb_pkl, pairs_csv))
        else:
            for emb_pkl in dataset_pkls:
                fr_model = _derive_embedding_name(emb_pkl)
                if (not run_all) and fr_model not in set(DEFAULT_FR_MODEL_ALLOWLIST):
                    continue
                jobs.append((dataset, fr_model, fr_model, emb_pkl, pairs_csv))
    return jobs


def main() -> None:
    p = argparse.ArgumentParser(description="Rank-PIT KS test of FIQA rankings against bootstrapped harm rankings")
    p.add_argument("--fr-features-root", default="data/FR_models",
                   help="Root folder containing dataset subfolders (same layout as rce.py)")
    p.add_argument("--quality-dir", default="data/FIQA_models",
                   help="Directory containing FIQA method subfolders")
    p.add_argument("--ca-fiqa-data-root", default="data/pairs/pairs",
                   help="Root folder containing {dataset}/pairs/pairs.csv")
    p.add_argument("--out-root", default=os.path.join("output", "rce_prob_test1"))
    p.add_argument("--run-all", action="store_true",
                   help="Ignore rce.py allowlists and run all datasets / all FR models.")
    p.add_argument("--datasets", default="", help="Optional extra filter: comma-separated dataset names")
    p.add_argument("--fr-models", default="", help="Optional extra filter: comma-separated FR models")
    p.add_argument("--methods", default="", help="Comma-separated FIQA methods (default: all found)")
    p.add_argument("--fmr", type=float, default=1e-3, help="Target FMR")
    p.add_argument("--threshold-method", choices=["percentile", "roc"], default="percentile",
                   help="FMR threshold method (same default as rce.py)")
    p.add_argument("--weight-alphas", default=",".join(format(a, "g") for a in DEFAULT_WEIGHT_ALPHAS),
                   help="Alphas for the rce.py weighted Spearman columns")
    p.add_argument("--resample", choices=["imp", "gen", "both"], default="both",
                   help="Which pairs to bootstrap when sampling R_e")
    p.add_argument("--n-samples", type=int, default=400, help="Bootstrap samples B (half reference, half calibration)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    split = lambda s: [x.strip() for x in str(s).split(",") if x.strip()]
    methods = split(args.methods) or None
    only_datasets = set(split(args.datasets))
    only_models = set(split(args.fr_models))
    weight_alphas = [float(a) for a in split(args.weight_alphas)] or list(DEFAULT_WEIGHT_ALPHAS)

    jobs = list_jobs(args.fr_features_root, args.ca_fiqa_data_root, bool(args.run_all))
    jobs = [j for j in jobs if (not only_datasets or j[0] in only_datasets)
            and (not only_models or j[1] in only_models)]
    print(f"{len(jobs)} jobs: " + ", ".join(f"{d}/{m}" for d, m, *_ in jobs))

    all_rows = []
    for dataset, fr_model, emb_tag, emb_pkl, pairs_csv in jobs:
        try:
            summary = run_one(
                dataset=dataset,
                fr_model=fr_model,
                emb_tag=emb_tag,
                emb_pkl=emb_pkl,
                pairs_csv=pairs_csv,
                quality_dir=args.quality_dir,
                methods=methods,
                out_dir=os.path.join(args.out_root, dataset, fr_model),
                n_samples=args.n_samples,
                resample=args.resample,
                fmr_target=args.fmr,
                threshold_method=args.threshold_method,
                weight_alphas=weight_alphas,
                seed=args.seed,
            )
            all_rows.append(summary)
        except Exception as e:
            print(f"[fail] {dataset}/{fr_model}/{emb_tag}: {e}")

    if all_rows:
        os.makedirs(args.out_root, exist_ok=True)
        out = os.path.join(args.out_root, "pit_test_all.csv")
        pd.concat(all_rows, ignore_index=True).to_csv(out, index=False)
        print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
