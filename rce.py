#!/usr/bin/env python3
"""Rank Consistency Evaluation (RCE) (fixed threshold at FMR=0.001).

Run:

    python rce.py \
        --fr-features-root /home/bw/FIQA/fiq_baselines/fr_features \
        --quality-dir /home/bw/FIQA/fiq_baselines/quality_scores \
        --ca-fiqa-data-root /home/bw/FIQA/ca-fiqa/data \
        --out-root output/sol4_rank_sample \
        --threshold-method roc \
        --weight-alphas 0,3

To run everything (all datasets / all FR models found under `--fr-features-root`):

    python rce.py --run-all --fmr 0.001 --threshold-method roc --weight-alphas 0,1,2

This script scans all datasets and FR models under:

    /home/bw/FIQA/fiq_baselines/fr_features/*

For each dataset it loads the verification pairs from:

    /home/bw/FIQA/ca-fiqa/data/{dataset_name}/pairs/pairs.csv

For each FR model inside that dataset it computes per-sample harm H(s) at a fixed
verification threshold (FMR=0.001) and evaluates FIQA quality scores against harm.

    /home/bw/FIQA/fiq_baselines/quality_scores/<method>/{dataset_name}-quality.pkl

Outputs are written to:

    output/rce/{dataset_name}/{fr_model}/
"""

import os
import pickle

import numpy as np  # type: ignore[reportMissingImports]
import pandas as pd

from prob1_test_set import run_prob1_test_set_batch  # type: ignore[reportMissingImports]


DEFAULT_FR_FEATURES_ROOT = "/home/bw/FIQA/fiq_baselines/fr_features"
DEFAULT_QUALITY_DIR = "/home/bw/FIQA/fiq_baselines/quality_scores"
DEFAULT_CA_FIQA_DATA_ROOT = "/home/bw/FIQA/ca-fiqa/data"
DEFAULT_OUT_ROOT = os.path.join("output", "rce")
DEFAULT_WEIGHT_ALPHAS = [0.0, 1.0, 2.0, 3.0]

# Default allowlists (used unless --run-all is passed)
DEFAULT_DATASET_ALLOWLIST = ["adience", "lfw", "cfp-fp", "agedb", "calfw", "cplfw", "xqlfw"]
DEFAULT_FR_MODEL_ALLOWLIST = ["adaface", "arcface_o", "magface", "swinface"]

DATASET_DISPLAY_NAMES = {
    "adience": "Adience",
    "lfw": "LFW",
    "cfp-fp": "CFP-FP",
    "agedb": "AgeDB",
    "calfw": "CALFW",
    "cplfw": "CPLFW",
    "xqlfw": "XQLFW",
}

FR_MODEL_DISPLAY_NAMES = {
    "adaface": "AdaFace",
    "arcface_o": "ArcFace",
    "magface": "MagFace",
    "swinface": "SwinFace",
}

METHOD_DISPLAY_NAMES = {
    "ediffiqa(L)": "eDifFIQA (L)",
    "diffiqa(R)": "DifFIQA (R)",
    "cr-fiqa(L)": "CR-FIQA (L)",
    "clib-fiqa": "CLIB-FIQA",
    "grafiqs": "GraFIQs",
    "froqAda": "FROQ",
    "magface": "MagFace",
    "ser-fiq": "SER-FIQA",
    "faceqan": "FaceQAN",
    "faceqnet": "FaceQnet",
    "sdd-fiqa": "SDD-FIQA",
    "faceqgen": "FaceQGen",
    "lightqnet": "LightQNet",
    "vit-fiqa": "ViT-FIQA",
    "pfe": "PFE",
    "pcnet": "PCNet",
    "diffiqa": "DifFIQA",
    "ediffiqa(M)": "eDifFIQA (M)",
    "ediffiqa(S)": "eDifFIQA (S)",
}


def _display_dataset_name(dataset_name: str) -> str:
    return DATASET_DISPLAY_NAMES.get(str(dataset_name), str(dataset_name))


def _display_fr_model_name(fr_model_name: str) -> str:
    return FR_MODEL_DISPLAY_NAMES.get(str(fr_model_name), str(fr_model_name))


def _display_method_name(method_name: str) -> str:
    return METHOD_DISPLAY_NAMES.get(str(method_name), str(method_name))


def _norm_image_id(x):
    s = str(x)
    return os.path.basename(s)


def load_embeddings_pkl(path):
    """Load embeddings dict {img_id: embedding_vector} and return (embeddings, id_to_idx)."""
    with open(path, "rb") as f:
        obj = pickle.load(f)

    if isinstance(obj, dict):
        emb_dict = {_norm_image_id(k): np.asarray(v) for k, v in obj.items()}
        img_ids = list(emb_dict.keys())
        embeddings = np.vstack([emb_dict[i] for i in img_ids])
        id_to_idx = {img_id: idx for idx, img_id in enumerate(img_ids)}
        return embeddings, id_to_idx

    # Some pipelines save (embeddings, img_ids) or (embeddings, id_to_idx)
    if isinstance(obj, (tuple, list)) and len(obj) == 2:
        embeddings, ids = obj
        embeddings = np.asarray(embeddings)
        if isinstance(ids, dict):
            id_to_idx = {_norm_image_id(k): int(v) for k, v in ids.items()}
            return embeddings, id_to_idx
        img_ids = [_norm_image_id(x) for x in list(ids)]
        if len(img_ids) != embeddings.shape[0]:
            raise ValueError(
                f"Embedding count mismatch: embeddings has {embeddings.shape[0]} rows but ids has {len(img_ids)}"
            )
        id_to_idx = {img_id: idx for idx, img_id in enumerate(img_ids)}
        return embeddings, id_to_idx

    raise TypeError(
        f"Unsupported embeddings PKL type: {type(obj)!r}. Expected dict or (embeddings, ids)."
    )


def load_pairs_csv(path):
    df = pd.read_csv(path)
    if "img1" in df.columns and "img2" in df.columns:
        df = df.rename(columns={"img1": "img1_id", "img2": "img2_id"})
    assert {"img1_id", "img2_id", "label"}.issubset(df.columns)
    df["img1_id"] = df["img1_id"].map(_norm_image_id)
    df["img2_id"] = df["img2_id"].map(_norm_image_id)
    return df


def load_quality_scores_pkl(path):
    """Load quality scores from PKL and return DataFrame [image_id, score]."""
    with open(path, "rb") as f:
        q = pickle.load(f)
    if isinstance(q, pd.DataFrame):
        df = q.copy()
        if "image_path" in df.columns and "quality_score" in df.columns:
            df = df.rename(columns={"image_path": "image_id", "quality_score": "score"})
        assert {"image_id", "score"}.issubset(df.columns)
        df["image_id"] = df["image_id"].map(_norm_image_id)
        return df[["image_id", "score"]]
    if isinstance(q, dict):
        return pd.DataFrame(
            [{"image_id": _norm_image_id(k), "score": float(v)} for k, v in q.items()]
        )
    raise TypeError(f"Unsupported quality PKL type: {type(q)!r}")


def is_normalized(embedding, atol=1e-6):
    """Check if an embedding is normalized (L2 norm is approximately 1)."""
    return np.isclose(np.linalg.norm(embedding), 1.0, atol=atol)


def l2_normalize(v):
    """L2 normalize a vector."""
    v = np.array(v)
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / norm if np.all(norm > 0) else v


def compute_similarity(embeddings, pairs):
    """Compute cosine similarity between embedding pairs."""
    emb_i = embeddings[pairs[:, 0]]
    emb_j = embeddings[pairs[:, 1]]
    
    # Check if all embeddings are normalized (sample up to 32 randomly)
    idx = np.random.choice(len(embeddings), size=min(32, len(embeddings)), replace=False)
    all_normalized = np.all([is_normalized(embeddings[i]) for i in idx])
    
    if all_normalized:
        # If normalized, simple dot product
        sims = np.sum(emb_i * emb_j, axis=1)
    else:
        # If not normalized, L2 normalize then compute dot product
        emb_i_norm = l2_normalize(emb_i)
        emb_j_norm = l2_normalize(emb_j)
        sims = np.sum(emb_i_norm * emb_j_norm, axis=1)
    
    return np.clip(sims, -1.0, 1.0)


def find_threshold_for_fmr_percentile(similarities, labels, target_fmr):
    """
    Find similarity threshold corresponding to target FMR using percentile-based selection.
    
    FMR = (# of impostor pairs with sim >= threshold) / (# of impostor pairs)
    
    Returns:
        threshold: Similarity threshold
        actual_fmr: Actual FMR achieved at this threshold
        n_impostor: Total number of impostor pairs
    """
    impostor_sims = similarities[labels == 0]
    if len(impostor_sims) == 0:
        return None, np.nan, 0
    
    # Use percentile to find threshold
    # If target_fmr = 0.001, we want the 99.9th percentile of impostor similarities
    percentile = (1 - target_fmr) * 100
    threshold = np.percentile(impostor_sims, percentile)
    
    # Compute actual FMR achieved
    actual_fmr = np.mean(impostor_sims >= threshold)
    
    return threshold, actual_fmr, len(impostor_sims)


def find_threshold_for_fmr(similarities, labels, target_fmr):
    """
    Find similarity threshold corresponding to target FMR using percentile-based method.
    
    Args:
        similarities: Array of similarity scores
        labels: Array of pair labels (0=impostor, 1=genuine)
        target_fmr: Target false match rate (e.g., 1e-3)
    
    Returns:
        threshold: Similarity threshold
        actual_fmr: Actual FMR achieved at this threshold
    """
    threshold, actual_fmr, _ = find_threshold_for_fmr_percentile(similarities, labels, target_fmr)
    return threshold, actual_fmr


def find_threshold_closest_fmr(similarities, labels, target_fmr):
    """
    Find threshold whose achieved FMR is closest to target_fmr.

    FMR(threshold) = fraction of impostor scores >= threshold
    """
    impostor_sims = np.asarray(similarities[labels == 0], dtype=float)

    if impostor_sims.size == 0:
        return None, np.nan

    sorted_imp = np.sort(impostor_sims)
    thresholds = np.unique(sorted_imp)
    n = sorted_imp.size

    best_thr = None
    best_fmr = None
    best_diff = np.inf

    for thr in thresholds:
        idx = np.searchsorted(sorted_imp, thr, side="left")
        fmr = (n - idx) / n
        diff = abs(fmr - target_fmr)

        if diff < best_diff:
            best_diff = diff
            best_thr = thr
            best_fmr = fmr

    return float(best_thr), float(best_fmr)


def _quality_pkl_for_dataset(method_dir: str, dataset_name: str) -> str:
    return os.path.join(method_dir, f"{dataset_name}-quality.pkl")


def compute_harm_scores(
    *,
    embeddings: np.ndarray,
    pairs_df: pd.DataFrame,
    id_to_idx: dict,
    fmr_target: float = 1e-3,
    threshold_method: str = "percentile",
):
    """Compute per-sample harm H(s) at a fixed threshold t* (FMR=target).

    Returns:
        img_ids, harm, c_fn, c_fa, c_gen_above, c_imp_below, min_gen_sim, max_imp_sim, threshold, base_fmr
    """
    # Ensure pairs reference known embeddings.
    valid_mask = pairs_df["img1_id"].isin(id_to_idx) & pairs_df["img2_id"].isin(id_to_idx)
    pairs_df = pairs_df[valid_mask].reset_index(drop=True)

    pairs_idx = np.array(
        [
            [id_to_idx[i1], id_to_idx[i2], lbl]
            for i1, i2, lbl in zip(pairs_df["img1_id"], pairs_df["img2_id"], pairs_df["label"])
        ],
        dtype=int,
    )

    # Compute similarity scores for all pairs once.
    sims = compute_similarity(embeddings, pairs_idx)
    labels = pairs_idx[:, 2].astype(int)

    # Find fixed threshold t* at target FMR on the full set.
    if threshold_method == "roc":
        threshold, base_fmr = find_threshold_closest_fmr(sims, labels, fmr_target)
    else:
        threshold, base_fmr = find_threshold_for_fmr(sims, labels, fmr_target)

    if threshold is None:
        return None

    img_ids = list(id_to_idx.keys())
    n = len(img_ids)
    c_fn = np.zeros(n, dtype=int)
    c_fa = np.zeros(n, dtype=int)
    c_gen_above = np.zeros(n, dtype=int)
    c_imp_below = np.zeros(n, dtype=int)
    min_gen_sim = np.full(n, np.inf, dtype=float)
    max_imp_sim = np.full(n, -np.inf, dtype=float)

    idx1 = pairs_idx[:, 0]
    idx2 = pairs_idx[:, 1]

    # Track per-sample min similarity for genuine pairs and max similarity for impostor pairs.
    gen_mask = labels == 1
    imp_mask = labels == 0
    np.minimum.at(min_gen_sim, idx1[gen_mask], sims[gen_mask])
    np.minimum.at(min_gen_sim, idx2[gen_mask], sims[gen_mask])
    np.maximum.at(max_imp_sim, idx1[imp_mask], sims[imp_mask])
    np.maximum.at(max_imp_sim, idx2[imp_mask], sims[imp_mask])
    min_gen_sim = np.where(np.isfinite(min_gen_sim), min_gen_sim, np.nan)
    max_imp_sim = np.where(np.isfinite(max_imp_sim), max_imp_sim, np.nan)

    # Count false non-matches per sample (genuine pairs below threshold).
    fn_mask = (labels == 1) & (sims < threshold)
    np.add.at(c_fn, idx1[fn_mask], 1)
    np.add.at(c_fn, idx2[fn_mask], 1)

    # Count genuine pairs above threshold (correct accepts) per sample.
    gen_above_mask = (labels == 1) & (sims >= threshold)
    np.add.at(c_gen_above, idx1[gen_above_mask], 1)
    np.add.at(c_gen_above, idx2[gen_above_mask], 1)

    # Count false accepts per sample (impostor pairs above threshold).
    fa_mask = (labels == 0) & (sims >= threshold)
    np.add.at(c_fa, idx1[fa_mask], 1)
    np.add.at(c_fa, idx2[fa_mask], 1)

    # Count impostor pairs below threshold (correct rejects) per sample.
    imp_below_mask = (labels == 0) & (sims < threshold)
    np.add.at(c_imp_below, idx1[imp_below_mask], 1)
    np.add.at(c_imp_below, idx2[imp_below_mask], 1)

    harm = c_fn + c_fa
    return (
        img_ids,
        harm,
        c_fn,
        c_fa,
        c_gen_above,
        c_imp_below,
        min_gen_sim,
        max_imp_sim,
        float(threshold),
        float(base_fmr),
    )


def _quality_scores_for_ids(quality_df: pd.DataFrame, img_ids: list[str]) -> np.ndarray:
    """Align quality scores Q(s) to the same image-id order as img_ids."""
    quality_map = dict(zip(quality_df["image_id"], quality_df["score"]))
    return np.array([float(quality_map.get(img_id, 0.0)) for img_id in img_ids], dtype=float)


def _spearman_from_ranks(rank_q: np.ndarray, rank_h: np.ndarray) -> float:
    """Compute Spearman correlation from rank arrays."""
    if len(rank_q) < 2:
        return np.nan
    return float(np.corrcoef(rank_q.astype(float), rank_h.astype(float))[0, 1])


def _rank_ascending(values: np.ndarray) -> np.ndarray:
    """Return 1-based average ranks (smaller value = better rank)."""
    s = pd.Series(np.asarray(values, dtype=float))
    return s.rank(method="average", ascending=True).to_numpy(dtype=float)


def _rank_descending(values: np.ndarray) -> np.ndarray:
    """Return 1-based average ranks (larger value = better rank)."""
    s = pd.Series(np.asarray(values, dtype=float))
    return s.rank(method="average", ascending=False).to_numpy(dtype=float)


def _weighted_rank_error(rank_h: np.ndarray, rank_q: np.ndarray, alpha: float) -> tuple[float, float, np.ndarray]:
    """Compute E_w, W and per-sample weights w(s)."""
    r = float(len(rank_h))
    if r == 0:
        return np.nan, np.nan, np.array([], dtype=float)

    weights = np.power(rank_h / r, float(alpha))
    diff = rank_h - rank_q
    ew = float(np.sum(weights * np.square(diff)))
    w_total = float(np.sum(weights))
    return ew, w_total, weights


def _weighted_spearman(rank_h: np.ndarray, rank_q: np.ndarray, alpha: float) -> tuple[float, float, float, np.ndarray]:
    """Compute weighted Spearman rho_w from rank arrays."""
    r = float(len(rank_h))
    if r < 2:
        return np.nan, np.nan, np.nan, np.array([], dtype=float)

    ew, w_total, weights = _weighted_rank_error(rank_h, rank_q, alpha)
    if not np.isfinite(ew) or not np.isfinite(w_total) or w_total <= 0:
        return np.nan, ew, w_total, weights

    denom = (r * r - 1.0) * w_total
    if denom <= 0:
        return np.nan, ew, w_total, weights

    rho_w = 1.0 - (6.0 * ew) / denom
    return float(rho_w), ew, w_total, weights


def _kendall_tau_b(rank_q: np.ndarray, rank_h: np.ndarray) -> float:
    """Compute Kendall tau-b; uses scipy if available, else returns NaN."""
    try:
        from scipy.stats import kendalltau  # type: ignore[reportMissingImports]
    except Exception:
        return np.nan
    return float(kendalltau(rank_q, rank_h, variant="b").correlation)


def save_spearman_leaderboard(out_dir: str) -> None:
    """Scan out_dir/*/harm_final_score.csv and save a single combined final-score CSV."""

    rows = []
    for method_name in sorted(os.listdir(out_dir)):
        method_dir = os.path.join(out_dir, method_name)
        if not os.path.isdir(method_dir) or method_name.startswith("."):
            continue
        summary_path = os.path.join(method_dir, "harm_final_score.csv")
        if not os.path.exists(summary_path):
            continue
        try:
            df = pd.read_csv(summary_path)
        except Exception:
            continue
        if len(df) == 0:
            continue
        row = df.iloc[0].to_dict()
        rows.append({
            "method": method_name,
            "weighted_spearman": float(row.get("weighted_spearman_rho_w", np.nan)),
            "weight_alpha": float(row.get("weight_alpha", np.nan)),
            "weighted_rank_error": float(row.get("weighted_rank_error_ew", np.nan)),
            "total_weight": float(row.get("total_weight_w", np.nan)),
        })

    if len(rows) == 0:
        return

    leaderboard = pd.DataFrame(rows).sort_values("weighted_spearman", ascending=False)
    leaderboard_out = os.path.join(out_dir, "rank_metrics_all_methods.csv")
    leaderboard.to_csv(leaderboard_out, index=False)


def save_spearman_leaderboard_for_alpha(out_dir: str, alpha: float) -> None:
    """Save a combined final-score CSV across methods for a specific alpha."""
    alpha_tag = _alpha_tag(alpha)
    rows = []
    for method_name in sorted(os.listdir(out_dir)):
        method_dir = os.path.join(out_dir, method_name)
        if not os.path.isdir(method_dir) or method_name.startswith("."):
            continue
        summary_path = os.path.join(method_dir, f"harm_final_score_{alpha_tag}.csv")
        if not os.path.exists(summary_path):
            continue
        try:
            df = pd.read_csv(summary_path)
        except Exception:
            continue
        if len(df) == 0:
            continue
        row = df.iloc[0].to_dict()
        rows.append({
            "method": method_name,
            "weighted_spearman": float(row.get("weighted_spearman_rho_w", np.nan)),
            "weight_alpha": float(row.get("weight_alpha", np.nan)),
            "weighted_rank_error": float(row.get("weighted_rank_error_ew", np.nan)),
            "total_weight": float(row.get("total_weight_w", np.nan)),
        })

    if len(rows) == 0:
        return

    leaderboard = pd.DataFrame(rows).sort_values("weighted_spearman", ascending=False)
    leaderboard["rank"] = leaderboard["weighted_spearman"].rank(method="min", ascending=False).astype(int)
    leaderboard = leaderboard[["rank", "method", "weighted_spearman", "weight_alpha", "weighted_rank_error", "total_weight"]]
    leaderboard_out = os.path.join(out_dir, f"rank_metrics_all_methods_{alpha_tag}.csv")
    leaderboard.to_csv(leaderboard_out, index=False)


def save_comprehensive_results(out_root: str, weight_alphas: list[float]) -> None:
    """Save global comprehensive CSVs across dataset/fr_model/FIQA methods and alphas."""
    alpha_tags = [(_alpha_tag(a), a) for a in weight_alphas]
    long_rows = []

    for dataset_name in sorted(os.listdir(out_root)):
        dataset_dir = os.path.join(out_root, dataset_name)
        if not os.path.isdir(dataset_dir) or dataset_name.startswith("."):
            continue
        for fr_model in sorted(os.listdir(dataset_dir)):
            fr_model_dir = os.path.join(dataset_dir, fr_model)
            if not os.path.isdir(fr_model_dir) or fr_model.startswith("."):
                continue
            for fiqa in sorted(os.listdir(fr_model_dir)):
                fiqa_dir = os.path.join(fr_model_dir, fiqa)
                if not os.path.isdir(fiqa_dir) or fiqa.startswith("."):
                    continue

                for alpha_tag, alpha in alpha_tags:
                    fp = os.path.join(fiqa_dir, f"harm_final_score_{alpha_tag}.csv")
                    if not os.path.exists(fp):
                        continue
                    try:
                        df = pd.read_csv(fp)
                    except Exception:
                        continue
                    if len(df) == 0:
                        continue
                    row = df.iloc[0].to_dict()
                    long_rows.append(
                        {
                            "dataset": _display_dataset_name(dataset_name),
                            "fr_model": _display_fr_model_name(fr_model),
                            "fiqa": _display_method_name(row.get("m_n", row.get("method", ""))),
                            "alpha": float(alpha),
                            "spearman": float(row.get("spearman_rank_q_vs_rank_h", np.nan)),
                            "weighted_spearman": float(row.get("weighted_spearman_rho_w", np.nan)),
                            "weighted_rank_error": float(row.get("weighted_rank_error_ew", np.nan)),
                            "total_weight": float(row.get("total_weight_w", np.nan)),
                            "threshold": float(row.get("threshold", np.nan)),
                            "fmr_target": float(row.get("fmr_target", np.nan)),
                            "base_fmr": float(row.get("base_fmr", np.nan)),
                            "total_genuine_pairs": float(row.get("total_genuine_pairs", np.nan)),
                            "total_impostor_pairs": float(row.get("total_impostor_pairs", np.nan)),
                            "genuine_above_thr": float(row.get("genuine_above_thr", np.nan)),
                            "genuine_below_thr": float(row.get("genuine_below_thr", np.nan)),
                            "impostor_above_thr": float(row.get("impostor_above_thr", np.nan)),
                            "impostor_below_thr": float(row.get("impostor_below_thr", np.nan)),
                            "n_samples": float(row.get("n_samples", np.nan)),
                        }
                    )

    if len(long_rows) == 0:
        return

    long_df = pd.DataFrame(long_rows)
    long_df = long_df.sort_values(["dataset", "fr_model", "fiqa", "alpha"])

    # Higher weighted_spearman is better: rank 1 is the best FIQA method
    # within each (dataset, fr_model, alpha) group.
    long_df["rank_in_dataset_fr_model_alpha"] = (
        long_df.groupby(["dataset", "fr_model", "alpha"])["weighted_spearman"]
        .rank(method="min", ascending=False)
        .astype(int)
    )

    long_df.to_csv(os.path.join(out_root, "comprehensive_results_long.csv"), index=False)

    # Requested wide view: fr_model, fiqa, alpha=0, alpha=1, alpha=2, alpha=3
    wide = (
        long_df.pivot_table(
            index=["dataset", "fr_model", "fiqa"],
            columns="alpha",
            values="weighted_spearman",
            aggfunc="first",
        )
        .reset_index()
    )
    wide_rank = (
        long_df.pivot_table(
            index=["dataset", "fr_model", "fiqa"],
            columns="alpha",
            values="rank_in_dataset_fr_model_alpha",
            aggfunc="first",
        )
        .reset_index()
    )
    for a in weight_alphas:
        if a not in wide.columns:
            wide[a] = np.nan
        if a not in wide_rank.columns:
            wide_rank[a] = np.nan
    rename_map = {a: f"alpha={format(float(a), 'g')}" for a in weight_alphas}
    wide = wide.rename(columns=rename_map)
    rank_rename_map = {a: f"rank_alpha={format(float(a), 'g')}" for a in weight_alphas}
    wide_rank = wide_rank.rename(columns=rank_rename_map)
    wide = wide.merge(wide_rank, on=["dataset", "fr_model", "fiqa"], how="left")
    ordered_cols = ["dataset", "fr_model", "fiqa"] + [
        item
        for a in weight_alphas
        for item in (f"alpha={format(float(a), 'g')}", f"rank_alpha={format(float(a), 'g')}")
    ]
    wide = wide[ordered_cols]
    wide.to_csv(os.path.join(out_root, "comprehensive_results.csv"), index=False)


def _run_harm_evaluation_for_method(
    *,
    method_name: str,
    out_dir: str,
    img_ids: list[str],
    harm: np.ndarray,
    c_fn: np.ndarray,
    c_fa: np.ndarray,
    min_gen_sim: np.ndarray,
    max_imp_sim: np.ndarray,
    quality_df: pd.DataFrame,
    threshold: float,
    fmr_target: float,
    base_fmr: float,
    total_genuine_pairs: int,
    total_impostor_pairs: int,
    genuine_above_thr: int,
    genuine_below_thr: int,
    impostor_above_thr: int,
    impostor_below_thr: int,
    weight_alpha: float = 0.0,
    alpha_tag: str | None = None,
):
    """Generate only ranking and final-score outputs for a single FIQA method."""
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(method_name))
    method_out_dir = os.path.join(out_dir, safe)
    os.makedirs(method_out_dir, exist_ok=True)
    prefix = os.path.join(method_out_dir, "harm")

    # Align quality scores to the same image order as harm.
    q_scores = _quality_scores_for_ids(quality_df, img_ids)

    # Formal rankings: R_H(s)=rank(H(s)), R_Q(s)=rank(Q(s)).
    margin = min_gen_sim - max_imp_sim
    harm_rank = _rank_ascending(harm)
    quality_rank = _rank_descending(q_scores)
    rank_diff = harm_rank - quality_rank

    # Rank-based Spearman and Kendall tau-b.
    spearman = _spearman_from_ranks(quality_rank, harm_rank)
    kendall = _kendall_tau_b(quality_rank, harm_rank)
    weighted_spearman, weighted_error, total_weight, weights = _weighted_spearman(
        harm_rank,
        quality_rank,
        alpha=weight_alpha,
    )

    ranking_df = pd.DataFrame({
        "img": img_ids,
        "R_H": harm_rank,
        "R_Q": quality_rank,
        "d": rank_diff,
        "weight": weights,
        "Q": q_scores,
        "C_FNM": c_fn,
        "C_FM": c_fa,
        "H": harm,
        "min_gen_sim": min_gen_sim,
        "max_imp_sim": max_imp_sim,
        "margin": margin,
    })
    suffix = "" if not alpha_tag else f"_{alpha_tag}"
    ranking_df.to_csv(prefix + f"_ranking{suffix}.csv", index=False)

    # Final score CSV.
    summary_df = pd.DataFrame([
        {
            "method": method_name,
            "threshold": threshold,
            "fmr_target": fmr_target,
            "base_fmr": base_fmr,
            "total_genuine_pairs": int(total_genuine_pairs),
            "total_impostor_pairs": int(total_impostor_pairs),
            "genuine_above_thr": int(genuine_above_thr),
            "genuine_below_thr": int(genuine_below_thr),
            "impostor_above_thr": int(impostor_above_thr),
            "impostor_below_thr": int(impostor_below_thr),
            "spearman_rank_q_vs_rank_h": spearman,
            "kendall_tau_b": kendall,
            "weighted_spearman_rho_w": weighted_spearman,
            "weighted_rank_error_ew": weighted_error,
            "total_weight_w": total_weight,
            "weight_alpha": float(weight_alpha),
            "n_samples": int(len(img_ids)),
        }
    ])
    summary_df.to_csv(prefix + f"_final_score{suffix}.csv", index=False)

    print(f"[harm] {method_name}: saved {prefix}_ranking{suffix}.csv, {prefix}_final_score{suffix}.csv")


def _derive_embedding_name(path: str) -> str:
    """Derive a stable short name from an embeddings file path."""
    base = os.path.basename(str(path))
    name = os.path.splitext(base)[0]
    for suffix in ("-embeddings", "_embeddings", "_embeddings_dict", "_embedding", "-embedding"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return name


def _alpha_tag(alpha: float) -> str:
    """Stable filename-safe alpha tag, e.g. alpha1, alpha2_5."""
    a = format(float(alpha), "g").replace("-", "m").replace(".", "_")
    return f"alpha{a}"


def _list_embeddings_pkls(dir_path: str) -> list[str]:
    """List candidate embeddings .pkl files in a directory.

    Heuristic: prefer files containing 'emb' in the filename; if none match,
    fall back to all .pkl files.
    """
    try:
        entries = [
            os.path.join(dir_path, f)
            for f in os.listdir(dir_path)
            if f.endswith(".pkl") and not f.startswith(".")
        ]
    except FileNotFoundError:
        return []

    entries = sorted(entries)
    preferred = [p for p in entries if "emb" in os.path.basename(p).lower()]
    return preferred if len(preferred) else entries


def run_prob1_test_set_batch(
    *,
    fr_features_root: str = DEFAULT_FR_FEATURES_ROOT,
    quality_dir: str = DEFAULT_QUALITY_DIR,
    ca_fiqa_data_root: str = DEFAULT_CA_FIQA_DATA_ROOT,
    out_root: str = DEFAULT_OUT_ROOT,
    run_all: bool = False,
    fmr_target: float = 1e-3,
    threshold_method: str = "percentile",
    weight_alphas: list[float] | None = None,
):
    if weight_alphas is None:
        weight_alphas = list(DEFAULT_WEIGHT_ALPHAS)

    fr_features_root = str(fr_features_root)
    quality_dir = str(quality_dir)
    ca_fiqa_data_root = str(ca_fiqa_data_root)
    out_root = str(out_root)

    if not os.path.isdir(fr_features_root):
        raise FileNotFoundError(f"fr_features_root not found: {fr_features_root}")
    if not os.path.isdir(quality_dir):
        raise FileNotFoundError(f"quality_dir not found: {quality_dir}")

    datasets = [
        os.path.join(fr_features_root, d)
        for d in sorted(os.listdir(fr_features_root))
        if os.path.isdir(os.path.join(fr_features_root, d)) and not d.startswith(".")
    ]

    if not run_all:
        dataset_set = set(DEFAULT_DATASET_ALLOWLIST)
        datasets = [d for d in datasets if os.path.basename(d) in dataset_set]

    if len(datasets) == 0:
        if run_all:
            raise FileNotFoundError(f"No dataset directories under: {fr_features_root}")
        raise FileNotFoundError(
            f"No datasets matched DEFAULT_DATASET_ALLOWLIST under: {fr_features_root} (allowlist={DEFAULT_DATASET_ALLOWLIST})"
        )

    if run_all:
        print(f"Mode: ALL datasets / ALL FR models")
    else:
        print(f"Mode: allowlist datasets={DEFAULT_DATASET_ALLOWLIST} fr_models={DEFAULT_FR_MODEL_ALLOWLIST}")

    print(f"Found {len(datasets)} datasets under: {fr_features_root}")

    for dataset_dir in datasets:
        dataset_name = os.path.basename(dataset_dir)
        pairs_csv = os.path.join(ca_fiqa_data_root, dataset_name, "pairs", "pairs.csv")
        if not os.path.exists(pairs_csv):
            print(f"[skip] Missing pairs: {pairs_csv}")
            continue

        # FR models are expected to be subdirectories under each dataset.
        model_dirs = [
            os.path.join(dataset_dir, d)
            for d in sorted(os.listdir(dataset_dir))
            if os.path.isdir(os.path.join(dataset_dir, d)) and not d.startswith(".")
        ]

        if not run_all:
            model_set = set(DEFAULT_FR_MODEL_ALLOWLIST)
            model_dirs = [md for md in model_dirs if os.path.basename(md) in model_set]

        # Fallback: dataset dir itself contains pkls (no per-model subfolders)
        dataset_pkls = [
            os.path.join(dataset_dir, f)
            for f in sorted(os.listdir(dataset_dir))
            if f.endswith(".pkl") and not f.startswith(".")
        ]

        jobs: list[tuple[str, str, str]] = []  # (fr_model_name, emb_tag, embeddings_pkl)

        if len(model_dirs) > 0:
            for md in model_dirs:
                fr_model = os.path.basename(md)
                emb_pkls = _list_embeddings_pkls(md)
                if len(emb_pkls) == 0:
                    print(f"[skip] No .pkl in {md}")
                    continue
                for emb_pkl in emb_pkls:
                    emb_tag = _derive_embedding_name(emb_pkl)
                    jobs.append((fr_model, emb_tag, emb_pkl))
        elif len(dataset_pkls) > 0:
            for emb_pkl in dataset_pkls:
                fr_model = _derive_embedding_name(emb_pkl)
                if (not run_all) and fr_model not in set(DEFAULT_FR_MODEL_ALLOWLIST):
                    continue
                emb_tag = fr_model
                jobs.append((fr_model, emb_tag, emb_pkl))
        else:
            print(f"[skip] No models or .pkl files in: {dataset_dir}")
            continue

        if len(jobs) == 0:
            print(f"[skip] No matching FR models/embeddings for dataset: {dataset_name}")
            continue

        print(f"\n=== Dataset: {dataset_name} ({len(jobs)} FR models) ===")
        for fr_model, emb_tag, emb_pkl in jobs:
            out_dir = os.path.join(out_root, dataset_name, fr_model)
            os.makedirs(out_dir, exist_ok=True)
            print(f"[run] {dataset_name}/{fr_model}/{emb_tag} -> {out_dir}")
            try:
                embeddings, id_to_idx = load_embeddings_pkl(emb_pkl)
                pairs_df = load_pairs_csv(pairs_csv)

                harm_pack = compute_harm_scores(
                    embeddings=embeddings,
                    pairs_df=pairs_df,
                    id_to_idx=id_to_idx,
                    fmr_target=fmr_target,
                    threshold_method=threshold_method,
                )
                if harm_pack is None:
                    print(f"[skip] No threshold for {dataset_name}/{fr_model}/{emb_tag}")
                    continue

                img_ids, harm, c_fn, c_fa, c_gen_above, c_imp_below, min_gen_sim, max_imp_sim, threshold, base_fmr = harm_pack

                labels_arr = pairs_df["label"].astype(int).to_numpy()
                total_genuine_pairs = int(np.sum(labels_arr == 1))
                total_impostor_pairs = int(np.sum(labels_arr == 0))
                genuine_above_thr = int(np.sum(c_gen_above) // 2)
                genuine_below_thr = int(np.sum(c_fn) // 2)
                impostor_above_thr = int(np.sum(c_fa) // 2)
                impostor_below_thr = int(np.sum(c_imp_below) // 2)

                # Save dataset-level ranking by combined harm (ascending: rank 1 = lowest harm).
                margin = min_gen_sim - max_imp_sim
                margin_for_sort = np.where(np.isfinite(margin), margin, -np.inf)
                order = np.lexsort((-margin_for_sort, harm))
                rank = np.empty_like(order)
                rank[order] = np.arange(1, len(order) + 1)
                ranking_df = pd.DataFrame({
                    "img": np.array(img_ids)[order],
                    "rank": rank[order],
                    "gen_harm": c_fn[order],
                    "imp_harm": c_fa[order],
                    "combine": harm[order],
                    "min_gen_sim": min_gen_sim[order],
                    "max_imp_sim": max_imp_sim[order],
                    "margin": margin[order],
                })
                ranking_out = os.path.join(out_dir, f"harm_ranking__{emb_tag}.csv")
                ranking_df.to_csv(ranking_out, index=False)

                # Evaluate each FIQA method available for this dataset.
                for d in sorted(os.listdir(quality_dir)):
                    method_dir = os.path.join(quality_dir, d)
                    if not os.path.isdir(method_dir) or d.startswith("."):
                        continue
                    qpath = _quality_pkl_for_dataset(method_dir, dataset_name)
                    if not os.path.exists(qpath):
                        continue
                    qdf = load_quality_scores_pkl(qpath)
                    for alpha in weight_alphas:
                        _run_harm_evaluation_for_method(
                            method_name=d,
                            out_dir=out_dir,
                            img_ids=img_ids,
                            harm=harm,
                            c_fn=c_fn,
                            c_fa=c_fa,
                            min_gen_sim=min_gen_sim,
                            max_imp_sim=max_imp_sim,
                            quality_df=qdf,
                            threshold=threshold,
                            fmr_target=fmr_target,
                            base_fmr=base_fmr,
                            total_genuine_pairs=total_genuine_pairs,
                            total_impostor_pairs=total_impostor_pairs,
                            genuine_above_thr=genuine_above_thr,
                            genuine_below_thr=genuine_below_thr,
                            impostor_above_thr=impostor_above_thr,
                            impostor_below_thr=impostor_below_thr,
                            weight_alpha=float(alpha),
                            alpha_tag=_alpha_tag(float(alpha)),
                        )

                # Aggregate leaderboard across all methods for this dataset/FR model.
                for alpha in weight_alphas:
                    save_spearman_leaderboard_for_alpha(out_dir, float(alpha))
            except Exception as e:
                print(f"[fail] {dataset_name}/{fr_model}/{emb_tag}: {e}")

    # Aggregate a single comprehensive CSV for all datasets/models/methods.
    save_comprehensive_results(out_root, weight_alphas)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Batch-run FIQA harm-based evaluation")
    p.add_argument(
        "--fr-features-root",
        default=DEFAULT_FR_FEATURES_ROOT,
        help="Root folder containing dataset subfolders (default: fiq_baselines/fr_features)",
    )
    p.add_argument(
        "--quality-dir",
        default=DEFAULT_QUALITY_DIR,
        help="Directory containing FIQA method subfolders",
    )
    p.add_argument(
        "--ca-fiqa-data-root",
        default=DEFAULT_CA_FIQA_DATA_ROOT,
        help="Root folder containing {dataset}/pairs/pairs.csv",
    )
    p.add_argument(
        "--out-root",
        default=DEFAULT_OUT_ROOT,
        help="Output root directory (default: output/fiqa_harm)",
    )
    p.add_argument(
        "--run-all",
        action="store_true",
        help="Ignore allowlists and run all datasets / all FR models.",
    )
    p.add_argument("--fmr", type=float, default=1e-3, help="Target FMR")
    p.add_argument(
        "--threshold-method",
        choices=["percentile", "roc"],
        default="percentile",
        help="FMR threshold method",
    )
    p.add_argument(
        "--weight-alphas",
        default=",".join(format(a, "g") for a in DEFAULT_WEIGHT_ALPHAS),
        help="Comma-separated alphas for w(s)=(R_H(s)/R)^alpha, e.g. 1,2,3",
    )
    args = p.parse_args()

    alpha_values = [
        float(x.strip())
        for x in str(args.weight_alphas).split(",")
        if str(x).strip() != ""
    ]
    if len(alpha_values) == 0:
        alpha_values = list(DEFAULT_WEIGHT_ALPHAS)

    run_prob1_test_set_batch(
        fr_features_root=args.fr_features_root,
        quality_dir=args.quality_dir,
        ca_fiqa_data_root=args.ca_fiqa_data_root,
        out_root=args.out_root,
        run_all=bool(args.run_all),
        fmr_target=args.fmr,
        threshold_method=args.threshold_method,
        weight_alphas=alpha_values,
    )
