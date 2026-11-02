#!/usr/bin/env python3
"""
Fixed-Threshold Proportional Pair Discard EDC (FT-PPD-EDC)

Solution 3 Same number of pair of gen and imp rejection at Fix Threshold.
Solution 3 Proportional rejection of genuine and impostor pairs at fixed threshold.

Run:

    python sol3_same_gen_imp_fix_thr.py \
        --fr-features-root /home/bw/FIQA/fiq_baselines/fr_features \
        --quality-dir /home/bw/FIQA/fiq_baselines/quality_scores \
        --ca-fiqa-data-root /home/bw/FIQA/ca-fiqa/data \
        --out-root output/sol3_same_gen_imp_min_fix_thr \
        --threshold-method roc \
        --pair-quality min

To run everything (all datasets / all FR models found under `--fr-features-root`):

    python sol3_same_gen_imp_fix_thr.py --run-all

This script scans all datasets and FR models under:

    /home/bw/FIQA/fiq_baselines/fr_features/*

For each dataset it loads the verification pairs from:

    /home/bw/FIQA/ca-fiqa/data/{dataset_name}/pairs/pairs.csv

For each FR model inside that dataset it loads all embeddings PKLs and generates an
EVR comparison plot across all FIQA methods found under:

    /home/bw/FIQA/fiq_baselines/quality_scores/<method>/{dataset_name}-quality.pkl

Outputs are written to:

    output/sol3_same_gen_imp_fix_thr/{dataset_name}/{fr_model}/
"""

import os
import pickle

import numpy as np  # type: ignore[reportMissingImports]
import pandas as pd  # type: ignore[reportMissingImports]


DEFAULT_FR_FEATURES_ROOT = "/home/bw/FIQA/fiq_baselines/fr_features"
DEFAULT_QUALITY_DIR = "/home/bw/FIQA/fiq_baselines/quality_scores"
DEFAULT_CA_FIQA_DATA_ROOT = "/home/bw/FIQA/ca-fiqa/data"
DEFAULT_OUT_ROOT = os.path.join("output", "sol3_same_gen_imp_fix_thr")

# Default allowlists (used unless --run-all is passed)
DEFAULT_DATASET_ALLOWLIST = ["adience", "lfw","cfp-fp" ,"agedb", "calfw", "cplfw", "xqlfw"]
DEFAULT_FR_MODEL_ALLOWLIST = ["adaface", "arcface_o", "magface","swinface"]

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

METHOD_PLOT_ORDER = [
    # Supervised regression
    "faceqan",
    "faceqnet",
    "faceqgen",
    "lightqnet",

    # Unsupervised
    "ser-fiq",
    "sdd-fiqa",

    # Recognition model–based
    "cr-fiqa(L)",
    "clib-fiqa",
    "grafiqs",
    "froqAda",
    "pfe",
    "pcnet",
    "vit-fiqa",

    # Diffusion-based
    "diffiqa(R)",
    "diffiqa",
    "ediffiqa(L)",
    "ediffiqa(M)",
    "ediffiqa(S)",
]

METHOD_COLOR_PALETTE = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
    "#aec7e8",
    "#ffbb78",
    "#98df8a",
    "#ff9896",
    "#c5b0d5",
    "#c49c94",
    "#f7b6d2",
    "#c7c7c7",
    "#dbdb8d",
    "#9edae5",
]

DEFAULT_METHOD_ALLOWLIST = [
    "ser-fiq",
    "ediffiqa(L)",
    "diffiqa(R)",
    "cr-fiqa(L)",
    "clib-fiqa",
    "grafiqs",
    "froqAda",
    "faceqan",
    "faceqnet",
    "sdd-fiqa",
]

PLOT_FONT_SIZES = {
    "title": 30,
    "labels": 32,
    "ticks": 30,
    "legend": 19,
}

PLOT_LINE_WIDTHS = {
    "mean": 4.2,
    "summary": 2.8,
    "reference": 1.6,
    "fmr": 2.8,
}

PLOT_ALPHAS = {
    "per_method": 0.8,
    "fmr": 0.8,
}

PLOT_REJECT_TICKS = np.arange(0, 101, 10, dtype=float)

def _norm_image_id(x):
    s = str(x)
    return os.path.basename(s)


def _display_dataset_name(dataset_name: str) -> str:
    return DATASET_DISPLAY_NAMES.get(str(dataset_name), str(dataset_name))


def _display_method_name(method_name: str) -> str:
    return METHOD_DISPLAY_NAMES.get(str(method_name), str(method_name))


def _display_fr_model_name(fr_model_name: str) -> str:
    return FR_MODEL_DISPLAY_NAMES.get(str(fr_model_name), str(fr_model_name))


def _method_sort_key(method_name: str) -> tuple[int, str]:
    method_name = str(method_name)
    try:
        return (METHOD_PLOT_ORDER.index(method_name), _display_method_name(method_name))
    except ValueError:
        return (len(METHOD_PLOT_ORDER), _display_method_name(method_name))


def _method_color(method_name: str) -> str:
    method_name = str(method_name)
    try:
        idx = METHOD_PLOT_ORDER.index(method_name)
        return METHOD_COLOR_PALETTE[idx % len(METHOD_COLOR_PALETTE)]
    except ValueError:
        fallback_idx = sum(ord(ch) for ch in method_name) % len(METHOD_COLOR_PALETTE)
        return METHOD_COLOR_PALETTE[fallback_idx]


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
    
    return sims


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
    Find verification threshold at the closest achievable FMR
    using ROC-style operating point selection (standard FIQA practice).

    Args:
        similarities : ndarray (N,)
            Similarity scores (higher = more similar)
        labels : ndarray (N,)
            1 = genuine, 0 = impostor
        target_fmr : float
            Desired FMR (e.g., 1e-3)

    Returns:
        threshold : float
            Similarity threshold
        actual_fmr : float
            Achieved FMR at this threshold
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


def compute_fnmr(similarities, labels, threshold):
    """
    FNMR = (# of genuine pairs with sim < threshold) / (# of genuine pairs)
    np.mean(genuine_sims < threshold)  → computes this fraction.
    """
    genuine_sims = similarities[labels == 1]
    if len(genuine_sims) == 0:
        return np.nan
    return np.mean(genuine_sims < threshold)


def error_vs_reject(embeddings, pairs_df, quality_df, id_to_idx,
                    reject_steps=20, fmr_target=1e-3, threshold_method="percentile",
                    pair_quality_method="mean", cer_lambda=0.5):
    """
    Iteratively reject low-quality samples and compute FNMR, FMR, and CER.
    
    Args:
        embeddings: Face embeddings array
        pairs_df: DataFrame with columns [img1_id, img2_id, label]
        quality_df: DataFrame with columns [image_id, score]
        id_to_idx: Dict mapping image_id to embedding index
        reject_steps: int or array-like of rejection fractions
        fmr_target: Target FMR (e.g., 1e-3)
        threshold_method: "percentile" or "roc" - method for FMR threshold calculation
                         "percentile": percentile-based (default)
                         "roc": ROC-style closest achievable FMR
        pair_quality_method: "mean" or "min" - method to aggregate pair quality
    
    Returns:
        reject_rates: Array of rejection rates
        fnmrs: Array of FNMR values
        fmrs: Array of FMR values
        cers: Array of CER values
        detailed_results: List of dicts with comprehensive metrics
        quality_df: Quality scores dataframe
        rejection_info: List of lists - rejected image info per step
    """
    # Convert reject_steps to array if it's an integer
    if isinstance(reject_steps, int):
        reject_steps = np.linspace(0, 1, reject_steps)
    else:
        reject_steps = np.array(reject_steps)
    
    valid_mask = pairs_df["img1_id"].isin(id_to_idx) & pairs_df["img2_id"].isin(id_to_idx)
    pairs_df = pairs_df[valid_mask].reset_index(drop=True)

    pairs_idx = np.array([
        [id_to_idx[i1], id_to_idx[i2], lbl]
        for i1, i2, lbl in zip(pairs_df["img1_id"], pairs_df["img2_id"], pairs_df["label"])
    ], dtype=int)

    # Map image quality for pair-wise mean quality
    quality_map = dict(zip(quality_df["image_id"], quality_df["score"]))
    img_ids = list(id_to_idx.keys())  # Deterministic order

    sims = compute_similarity(embeddings, pairs_idx)
    labels = pairs_idx[:, 2].astype(int)

    if threshold_method == "roc":
        fixed_thr, base_fmr = find_threshold_closest_fmr(sims, labels, fmr_target)
    else:
        fixed_thr, base_fmr = find_threshold_for_fmr(sims, labels, fmr_target)

    if fixed_thr is None:
        return np.array([]), np.array([]), np.array([]), np.array([]), [], quality_df, []

    reject_rates, fnmrs, fmrs, cers = [], [], [], []
    detailed_results = []
    rejection_info = []  # Store per-step rejection info

    # Precompute pair quality
    if pair_quality_method == "min":
        pair_q = np.array([
            min(quality_map.get(img_ids[i1], 0.0), quality_map.get(img_ids[i2], 0.0))
            for i1, i2 in pairs_idx[:, :2]
        ])
    else:
        pair_q = np.array([
            0.5 * (
                quality_map.get(img_ids[i1], 0.0) + quality_map.get(img_ids[i2], 0.0)
            )
            for i1, i2 in pairs_idx[:, :2]
        ])

    for reject_fraction in reject_steps:
        # Reject low-quality pairs proportionally to class sizes.
        gen_idx = np.where(labels == 1)[0]
        imp_idx = np.where(labels == 0)[0]

        total_pairs = len(labels)
        total_genuine = len(gen_idx)
        total_impostor = len(imp_idx)

        target_reject_total = int(reject_fraction * total_pairs)

        if target_reject_total <= 0:
            reject_gen = np.array([], dtype=int)
            reject_imp = np.array([], dtype=int)
        else:
            # Proportional allocation by class size.
            target_reject_gen = int(round(target_reject_total * (total_genuine / total_pairs)))
            target_reject_gen = min(target_reject_gen, total_genuine)

            target_reject_imp = target_reject_total - target_reject_gen
            target_reject_imp = min(target_reject_imp, total_impostor)

            # If one class is full, reassign remaining budget to the other class.
            assigned_total = target_reject_gen + target_reject_imp
            remaining_budget = target_reject_total - assigned_total

            if remaining_budget > 0:
                extra_gen_capacity = total_genuine - target_reject_gen
                add_gen = min(remaining_budget, extra_gen_capacity)
                target_reject_gen += add_gen
                remaining_budget -= add_gen

            if remaining_budget > 0:
                extra_imp_capacity = total_impostor - target_reject_imp
                add_imp = min(remaining_budget, extra_imp_capacity)
                target_reject_imp += add_imp
                remaining_budget -= add_imp

            # Sort each class by pair quality (low to high).
            gen_order = gen_idx[np.argsort(pair_q[gen_idx])]
            imp_order = imp_idx[np.argsort(pair_q[imp_idx])]

            reject_gen = gen_order[:target_reject_gen]
            reject_imp = imp_order[:target_reject_imp]

        reject_pairs = np.concatenate([reject_gen, reject_imp])

        mask_keep = np.ones(total_pairs, dtype=bool)
        mask_keep[reject_pairs] = False

        sims_keep = sims[mask_keep]
        labels_keep = labels[mask_keep]

        # Count genuine and impostor pairs after rejection
        genuine_remaining = np.sum(labels_keep == 1)
        impostor_remaining = np.sum(labels_keep == 0)
        # Stop if insufficient impostors to reliably compute target FMR (standard practice)
        min_impostors_needed = int(np.ceil(1.0 / fmr_target))  # e.g., 1000 for 1e-3
        if impostor_remaining < min_impostors_needed:
            break  # stop the EVR curve

        thr = fixed_thr
        
        fnmr = compute_fnmr(sims_keep, labels_keep, thr)
        impostor_sims = sims_keep[labels_keep == 0]
        actual_fmr = np.mean(impostor_sims >= thr) if len(impostor_sims) > 0 else np.nan
        cer = cer_lambda * actual_fmr + (1.0 - cer_lambda) * fnmr
        
        # Count pairs above/below threshold
        genuine_sims = sims_keep[labels_keep == 1]
        
        gen_below = np.sum(genuine_sims < thr) if len(genuine_sims) > 0 else 0
        gen_above = np.sum(genuine_sims >= thr) if len(genuine_sims) > 0 else 0
        imp_below = np.sum(impostor_sims < thr) if len(impostor_sims) > 0 else 0
        imp_above = np.sum(impostor_sims >= thr) if len(impostor_sims) > 0 else 0

        actual_reject_count = len(reject_pairs)
        actual_reject_rate = actual_reject_count / total_pairs if total_pairs else 0.0
        reject_rates.append(actual_reject_rate)
        fnmrs.append(fnmr)
        fmrs.append(actual_fmr)
        cers.append(cer)
        
        # Comprehensive results with sample counts
        detailed_results.append({
            "reject_rate": actual_reject_rate,
            "threshold": thr,
            "fmr": actual_fmr,
            "fnmr": fnmr,
            "cer": cer,
            "cer_lambda": cer_lambda,
            "total_pairs": total_pairs,
            "pairs_rejected": int(actual_reject_count),
            "pairs_remaining": int(total_pairs - actual_reject_count),
            "total_pairs_after_rejection": genuine_remaining + impostor_remaining,
            "genuine_pairs_after_rejection": genuine_remaining,
            "impostor_pairs_after_rejection": impostor_remaining,
            "rejected_genuine_pairs": int(len(reject_gen)),
            "rejected_impostor_pairs": int(len(reject_imp)),
            "genuine_below_threshold": gen_below,
            "genuine_above_threshold": gen_above,
            "impostor_below_threshold": imp_below,
            "impostor_above_threshold": imp_above,
            "total_genuine_pairs_before_rejection": total_genuine,
            "total_impostor_pairs_before_rejection": total_impostor
        })
        
        # Store per-step rejection info at pair level
        rejected_pairs_this_step = []
        for pair_idx in reject_pairs:
            i1, i2, lbl = pairs_idx[pair_idx]
            rejected_pairs_this_step.append({
                "img1_id": img_ids[i1],
                "img2_id": img_ids[i2],
                "label": int(lbl),
                "pair_quality": float(pair_q[pair_idx]),
                "reject_fraction": reject_fraction,
            })
        rejection_info.append(rejected_pairs_this_step)

    # Ensure reject_rates and fnmrs are numpy arrays of the same length
    reject_rates = np.array(reject_rates)
    fnmrs = np.array(fnmrs)
    fmrs = np.array(fmrs)
    cers = np.array(cers)

    return reject_rates, fnmrs, fmrs, cers, detailed_results, quality_df, rejection_info


def compute_pauc(reject_rates, fnmrs, max_reject=0.3):
    """Compute partial AUC up to max_reject."""
    if len(reject_rates) > 1:
        # Filter for rejection rates up to max_reject
        mask = np.round(reject_rates, 5) <= max_reject
        reject_rates_filtered = reject_rates[mask]
        fnmrs_filtered = fnmrs[mask]
        
        if len(reject_rates_filtered) > 1:
            # NumPy compatibility: np.trapezoid is not available in older versions.
            trap = getattr(np, 'trapezoid', None)
            if trap is None:
                return np.trapz(fnmrs_filtered, reject_rates_filtered)
            return trap(fnmrs_filtered, reject_rates_filtered)
    return np.nan


def _quality_pkl_for_dataset(method_dir: str, dataset_name: str) -> str:
    return os.path.join(method_dir, f"{dataset_name}-quality.pkl")


def plot_evr_compare_dataset(
    *,
    dataset_name: str,
    embeddings_pkl: str,
    quality_methods_dir: str,
    pairs_csv: str,
    out_path: str,
    fmr_target: float = 1e-3,
    reject_steps=None,
    threshold_method: str = "roc",
    pair_quality_method: str = "mean",
    run_all: bool = False,
    save_all_methods_values: bool = True,
    cer_lambda: float = 0.5,
):
    try:
        import matplotlib.pyplot as plt  # type: ignore[reportMissingImports]
    except Exception as e:
        raise SystemExit(f"matplotlib is required for plotting: {e}")

    if reject_steps is None:
        reject_steps = np.arange(0, 0.98, 0.05)

    embeddings, id_to_idx = load_embeddings_pkl(embeddings_pkl)
    pairs_df = load_pairs_csv(pairs_csv)

    methods_for_plot = []
    curves = {}
    fmr_curves = {}
    cer_curves = {}
    method_to_pauc: dict[str, float] = {}
    detailed_by_method = {}
    allowed_methods = set(DEFAULT_METHOD_ALLOWLIST)

    def _run(method_name, quality_pkl_path):
        qdf = load_quality_scores_pkl(quality_pkl_path)
        rr, fn, fm, cer, detailed_results, _, _ = error_vs_reject(
            embeddings,
            pairs_df,
            qdf,
            id_to_idx,
            reject_steps=reject_steps,
            fmr_target=fmr_target,
            threshold_method=threshold_method,
            pair_quality_method=pair_quality_method,
            cer_lambda=cer_lambda,
        )
        p = compute_pauc(rr, fn, max_reject=0.3)
        curves[method_name] = (rr, fn)
        fmr_curves[method_name] = (rr, fm)
        cer_curves[method_name] = (rr, cer)
        method_to_pauc[method_name] = p
        detailed_by_method[method_name] = detailed_results

    available_methods = []
    method_qpath: dict[str, str] = {}
    for d in sorted(os.listdir(quality_methods_dir)):
        p = os.path.join(quality_methods_dir, d)
        if not os.path.isdir(p) or d.startswith("."):
            continue
        qpath = _quality_pkl_for_dataset(p, dataset_name)
        if os.path.exists(qpath):
            available_methods.append(d)
            method_qpath[d] = qpath

    if run_all:
        methods_for_plot = list(available_methods)
    else:
        methods_for_plot = [m for m in available_methods if m in allowed_methods]

    methods_for_csv = list(available_methods) if (run_all or save_all_methods_values) else list(methods_for_plot)
    methods_to_compute = sorted(set(methods_for_plot) | set(methods_for_csv), key=_method_sort_key)
    for method_name in methods_to_compute:
        _run(method_name, method_qpath[method_name])

    if len(methods_for_plot) == 0:
        raise FileNotFoundError(
            f"No quality PKLs found for plotted methods for dataset '{dataset_name}' under: {quality_methods_dir}"
        )

    methods_for_plot.sort(key=_method_sort_key)

    pauc_rows = [
        {
            "method": _display_method_name(method_name),
            "m_n": method_name,
            "pAUC@0.3": method_to_pauc.get(method_name, np.nan),
        }
        for method_name in methods_for_csv
    ]
    pauc_df = pd.DataFrame(pauc_rows)
    print(pauc_df["pAUC@0.3"].isna().sum())
    if not pauc_df.empty:
        pauc_df["rank"] = pauc_df["pAUC@0.3"].rank(method="min", ascending=True).astype("Int64")
        pauc_df = pauc_df.sort_values(["rank", "pAUC@0.3", "method"], ascending=[True, True, True])

    plt.figure(figsize=(12, 8))
    for m in methods_for_plot:
        rr, fn = curves[m]
        plt.plot(
            rr,
            fn,
            label=_display_method_name(m),
            color=_method_color(m),
            linewidth=PLOT_LINE_WIDTHS["summary"],
            alpha=PLOT_ALPHAS["per_method"],
        )

    plt.xlabel("Discard rate (%)", fontsize=PLOT_FONT_SIZES["labels"])
    plt.ylabel("FNMR", fontsize=PLOT_FONT_SIZES["labels"])
    plt.xlim(0.0, 1.0)
    plt.xticks(PLOT_REJECT_TICKS / 100.0, [str(int(v)) for v in PLOT_REJECT_TICKS], fontsize=PLOT_FONT_SIZES["ticks"])
    plt.yticks(fontsize=PLOT_FONT_SIZES["ticks"])
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right", fontsize=PLOT_FONT_SIZES["legend"])

    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = str(out_path)
    if not os.path.isabs(out_path):
        out_path = os.path.join(base_dir, out_path)
    os.makedirs(os.path.dirname(out_path) or base_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=600)
    plt.close()

    # Save FMR vs discard rate plot (fixed threshold) with explicit filename.
    fnmr_stem = os.path.splitext(out_path)[0]
    fmr_stem = fnmr_stem.replace("fnmr_vs_reject__", "fmr_vs_reject__")
    if fmr_stem == fnmr_stem:
        fmr_stem = fnmr_stem.replace("fnmr_vs_reject__", "fmr_vs_reject__")
    fmr_out_path = fmr_stem + ".pdf"
    plt.figure(figsize=(12, 8))
    for m in methods_for_plot:
        rr, fm = fmr_curves[m]
        plt.plot(
            rr,
            fm,
            label=_display_method_name(m),
            color=_method_color(m),
            linewidth=PLOT_LINE_WIDTHS["fmr"],
            alpha=PLOT_ALPHAS["fmr"],
        )
    plt.xlabel("Discard rate (%)", fontsize=PLOT_FONT_SIZES["labels"])
    plt.ylabel("FMR", fontsize=PLOT_FONT_SIZES["labels"])
    plt.xlim(0.0, 1.0)
    plt.xticks(PLOT_REJECT_TICKS / 100.0, [str(int(v)) for v in PLOT_REJECT_TICKS], fontsize=PLOT_FONT_SIZES["ticks"])
    plt.yticks(fontsize=PLOT_FONT_SIZES["ticks"])
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right", fontsize=PLOT_FONT_SIZES["legend"])
    plt.tight_layout()
    plt.savefig(fmr_out_path, dpi=600)
    plt.close()

    # Save CER vs reject rate plot
    cer_stem = fnmr_stem.replace("fnmr_vs_reject__", "cer_vs_reject__")
    if cer_stem == fnmr_stem:
        cer_stem = fnmr_stem.replace("fnmr_vs_reject__", "cer_vs_reject__")
    cer_out_path = cer_stem + ".pdf"
    plt.figure(figsize=(12, 8))
    for m in methods_for_plot:
        rr, cer_vals = cer_curves[m]
        plt.plot(
            rr,
            cer_vals,
            label=_display_method_name(m),
            color=_method_color(m),
            linewidth=PLOT_LINE_WIDTHS["fmr"],
            alpha=PLOT_ALPHAS["fmr"],
        )
    plt.xlabel("Discard rate (%)", fontsize=PLOT_FONT_SIZES["labels"])
    plt.ylabel(f"CER", fontsize=PLOT_FONT_SIZES["labels"])
    plt.xlim(0.0, 1.0)
    plt.xticks(PLOT_REJECT_TICKS / 100.0, [str(int(v)) for v in PLOT_REJECT_TICKS], fontsize=PLOT_FONT_SIZES["ticks"])
    plt.yticks(fontsize=PLOT_FONT_SIZES["ticks"])
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right", fontsize=PLOT_FONT_SIZES["legend"])
    plt.tight_layout()
    plt.savefig(cer_out_path, dpi=600)
    plt.close()

    pauc_csv = os.path.splitext(out_path)[0] + "_pauc.csv"
    pauc_df.to_csv(pauc_csv, index=False)

    # Save CER-vs-reject summary scores per method.
    cer_rows = []
    for method_name in methods_for_csv:
        rr, cer_vals = cer_curves.get(method_name, (np.array([]), np.array([])))
        cer_score = compute_pauc(rr, cer_vals, max_reject=0.3)
        cer_rows.append(
            {
                "method": _display_method_name(method_name),
                "m_n": method_name,
                "pAUC@0.3": cer_score,
            }
        )
    cer_df = pd.DataFrame(cer_rows)
    print(cer_df["pAUC@0.3"].isna().sum())
    if not cer_df.empty:
        cer_df["rank"] = cer_df["pAUC@0.3"].rank(method="min", ascending=True).astype("Int64")
        cer_df = cer_df.sort_values(["rank", "pAUC@0.3", "method"], ascending=[True, True, True])
    cer_csv = cer_stem + "_scores.csv"
    cer_df.to_csv(cer_csv, index=False)

    # Save per-method detailed EVR tables
    out_prefix = os.path.splitext(out_path)[0]
    detailed_dir = out_prefix + "_detailed"
    os.makedirs(detailed_dir, exist_ok=True)
    for method_name in methods_for_csv:
        detailed = detailed_by_method.get(method_name, [])
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(method_name))
        pd.DataFrame(detailed).to_csv(os.path.join(detailed_dir, f"{safe}.csv"), index=False)

    print(f"Saved plot: {out_path}")
    print(f"Saved pAUC table: {pauc_csv}")
    print(f"Saved FMR plot: {fmr_out_path}")
    print(f"Saved CER plot: {cer_out_path}")
    print(f"Saved CER scores: {cer_csv}")
    print(f"Saved detailed CSVs: {detailed_dir}/")
    return cer_df


def _save_combined_grid_image(png_paths, titles, out_path, ncols=3):
    """Create a single grid image from a list of PNG plots."""
    try:
        import matplotlib.pyplot as plt  # type: ignore[reportMissingImports]
        import matplotlib.image as mpimg  # type: ignore[reportMissingImports]
    except Exception as e:
        raise SystemExit(f"matplotlib is required for combining plots: {e}")

    if len(png_paths) == 0:
        raise ValueError("No PNGs provided for combined grid")

    n = len(png_paths)
    ncols = max(1, int(ncols))
    nrows = int(np.ceil(n / ncols))

    # Size tuned for readability; each tile ~5x4 inches
    fig_w = 5.0 * ncols
    fig_h = 4.2 * nrows
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_w, fig_h))

    # Normalize axes to a flat list
    if isinstance(axes, np.ndarray):
        axes_list = axes.ravel().tolist()
    else:
        axes_list = [axes]

    for ax in axes_list:
        ax.axis("off")

    for i, (png_path, title) in enumerate(zip(png_paths, titles)):
        ax = axes_list[i]
        img = mpimg.imread(png_path)
        ax.imshow(img)
        ax.set_title(str(title), fontsize=11)
        ax.axis("off")

    # Hide any remaining empty tiles
    for j in range(len(png_paths), len(axes_list)):
        axes_list[j].axis("off")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def _derive_embedding_name(path: str) -> str:
    """Derive a stable short name from an embeddings file path."""
    base = os.path.basename(str(path))
    name = os.path.splitext(base)[0]
    for suffix in ("-embeddings", "_embeddings", "_embeddings_dict", "_embedding", "-embedding"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return name


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
    plot_both_scopes: bool = False,
    fmr_target: float = 1e-3,
    reject_steps=None,
    threshold_method: str = "roc",
    pair_quality_method: str = "mean",
    save_all_methods_values: bool = True,
    cer_lambda: float = 0.5,
):
    if reject_steps is None:
        reject_steps = np.arange(0, 1, 0.05)

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

    if plot_both_scopes:
        print("Mode: plotting BOTH scopes (default_methods and all_methods)")
    elif run_all:
        print(f"Mode: ALL datasets / ALL FR models")
    else:
        print(f"Mode: allowlist datasets={DEFAULT_DATASET_ALLOWLIST} fr_models={DEFAULT_FR_MODEL_ALLOWLIST}")

    print(f"Found {len(datasets)} datasets under: {fr_features_root}")
    comprehensive_rows_by_scope: dict[str, list[dict]] = {}

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

        print(f"\n=== Dataset: {_display_dataset_name(dataset_name)} ({len(jobs)} FR models) ===")
        for fr_model, emb_tag, emb_pkl in jobs:
            if plot_both_scopes:
                scopes = [("default_methods", False), ("all_methods", True)]
            else:
                scopes = [("all_methods", True)] if run_all else [("default_methods", False)]

            for method_scope_dir, scope_run_all in scopes:
                out_dir = os.path.join(out_root, method_scope_dir, fr_model, dataset_name)
                out_path = os.path.join(out_dir, f"fnmr_vs_reject__{emb_tag}.pdf")
                os.makedirs(out_dir, exist_ok=True)
                print(
                    f"[run] {_display_dataset_name(dataset_name)}/{_display_fr_model_name(fr_model)}/{emb_tag} ({method_scope_dir}) -> {out_path}"
                )
                try:
                    cer_df = plot_evr_compare_dataset(
                        dataset_name=dataset_name,
                        embeddings_pkl=emb_pkl,
                        quality_methods_dir=quality_dir,
                        pairs_csv=pairs_csv,
                        out_path=out_path,
                        fmr_target=fmr_target,
                        reject_steps=reject_steps,
                        threshold_method=threshold_method,
                        pair_quality_method=pair_quality_method,
                        run_all=scope_run_all,
                        save_all_methods_values=save_all_methods_values,
                        cer_lambda=cer_lambda,
                    )
                    if isinstance(cer_df, pd.DataFrame) and (not cer_df.empty):
                        for _, row in cer_df.iterrows():
                            comprehensive_rows_by_scope.setdefault(method_scope_dir, []).append(
                                {
                                    "dataset": _display_dataset_name(dataset_name),
                                    "fr_model": _display_fr_model_name(fr_model),
                                    "fiqa": _display_method_name(row.get("m_n", row.get("method", ""))),
                                    "pAUC@0.3(cer)": row.get("pAUC@0.3", np.nan),
                                    "rank": row.get("rank", np.nan),
                                }
                            )
                except Exception as e:
                    print(f"[fail] {dataset_name}/{fr_model}/{emb_tag} ({method_scope_dir}): {e}")

    for scope_name, rows in comprehensive_rows_by_scope.items():
        if len(rows) == 0:
            continue
        comprehensive_df = pd.DataFrame(rows)
        comprehensive_df = comprehensive_df[
            ["dataset", "fr_model", "fiqa", "pAUC@0.3(cer)", "rank"]
        ]
        comprehensive_df = comprehensive_df.sort_values(
            ["dataset", "fr_model", "rank", "fiqa"], ascending=[True, True, True, True]
        )
        scope_dir = os.path.join(out_root, scope_name)
        os.makedirs(scope_dir, exist_ok=True)
        comprehensive_results_csv = os.path.join(scope_dir, "comprehensive_results.csv")
        comprehensive_df.to_csv(comprehensive_results_csv, index=False)
        print(f"Saved comprehensive results: {comprehensive_results_csv}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Batch-run RCE for IJCB EDC Problem 1")
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
        help="Output root directory (default: output/sol3_same_gen_imp_fix_thr)",
    )
    p.add_argument(
        "--run-all",
        action="store_true",
        help="Ignore allowlists and run all datasets, FR models, and FIQA methods.",
    )
    p.add_argument(
        "--plot-both-scopes",
        action="store_true",
        help="Generate both default_methods and all_methods plots in one run.",
    )
    p.add_argument("--fmr", type=float, default=1e-3, help="Target FMR")
    p.add_argument(
        "--threshold-method",
        choices=["percentile", "roc"],
        default="roc",
        help="FMR threshold method",
    )
    p.add_argument(
        "--pair-quality",
        choices=["mean", "min"],
        default="mean",
        help="Pair quality aggregation method",
    )
    p.add_argument(
        "--save-all-methods-values",
        dest="save_all_methods_values",
        action="store_true",
        default=True,
        help="When not using --run-all, keep default plotting methods but save CSV values for all available FIQA methods (default: enabled).",
    )
    p.add_argument(
        "--defaults-csv-only",
        dest="save_all_methods_values",
        action="store_false",
        help="When not using --run-all, save CSV values only for the plotted default methods.",
    )
    p.add_argument(
        "--cer-lambda",
        type=float,
        default=0.5,
        help="Lambda for CER = lambda * FMR + (1-lambda) * FNMR",
    )
    args = p.parse_args()

    run_prob1_test_set_batch(
        fr_features_root=args.fr_features_root,
        quality_dir=args.quality_dir,
        ca_fiqa_data_root=args.ca_fiqa_data_root,
        out_root=args.out_root,
        run_all=bool(args.run_all),
        plot_both_scopes=bool(args.plot_both_scopes),
        fmr_target=args.fmr,
        threshold_method=args.threshold_method,
        pair_quality_method=args.pair_quality,
        save_all_methods_values=bool(args.save_all_methods_values),
        cer_lambda=args.cer_lambda,
    )
