#!/usr/bin/env python3
"""Error vs Discard Characteristic (EDC) - Problem 1: Test-Set Divergence.

Run:

    python prob1_test_set.py \
        --fr-features-root /home/bw/FIQA/fiq_baselines/fr_features \
        --quality-dir /home/bw/FIQA/fiq_baselines/quality_scores \
        --ca-fiqa-data-root /home/bw/FIQA/ca-fiqa/data \
        --out-root output_submission/prob1_test_set \
        --threshold-method roc \
        --skip-evr-plot \
        --skip-additional-plots

To run everything (all datasets / all FR models found under `--fr-features-root`):

    python prob1_test_set.py --run-all --skip-evr-plot --skip-additional-plots

This script scans all datasets and FR models under:

    /home/bw/FIQA/fiq_baselines/fr_features/*

For each dataset it loads the verification pairs from:

    /home/bw/FIQA/ca-fiqa/data/{dataset_name}/pairs/pairs.csv

For each FR model inside that dataset it loads all embeddings PKLs and generates an
EVR comparison plot across all FIQA methods found under:

    /home/bw/FIQA/fiq_baselines/quality_scores/<method>/{dataset_name}-quality.pkl

Outputs are written to:

    output_submission/prob1_test_set/{dataset_name}/{fr_model}/
"""

import os
import pickle
from itertools import combinations

import numpy as np  # type: ignore[reportMissingImports]
import pandas as pd  # type: ignore[reportMissingImports]


DEFAULT_FR_FEATURES_ROOT = "/home/bw/FIQA/fiq_baselines/fr_features"
DEFAULT_QUALITY_DIR = "/home/bw/FIQA/fiq_baselines/quality_scores"
DEFAULT_CA_FIQA_DATA_ROOT = "/home/bw/FIQA/ca-fiqa/data"
DEFAULT_OUT_ROOT = os.path.join("output_submission", "prob1_test_set")

DATASET_DISPLAY_NAMES = {
    "adience": "Adience",
    "lfw": "LFW",
    "cplfw": "CPLFW",
    "xqlfw": "XQLFW",
    "calfw": "CALFW",
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
}

METHOD_ALIASES = {
    "ser-fiqa": "ser-fiq",
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
]

# Default allowlists (used unless --run-all is passed)
DEFAULT_DATASET_ALLOWLIST = ["adience", "lfw", "calfw", "cplfw", "xqlfw"]
DEFAULT_FR_MODEL_ALLOWLIST = ["adaface", "arcface_o", "magface","swinface"]


# Problem 1 (different test set at each discard level)
DEFAULT_PROB1_DISCARD_RATE = 0.30
DEFAULT_PROB1_FIQA_METHOD_A = "ser-fiq"

# DEFAULT_PROB1_FIQA_METHOD_BS = ["magface", "faceqnet", "sdd-fiqa", "ediffiqa(L)",
#     "diffiqa(R)",
#     "cr-fiqa(L)",
#     "clib-fiqa",
#     "grafiqs",
#     "froqAda",
#     "magface",
#     "ser-fiq",
#     "faceqan",]


DEFAULT_PROB1_FIQA_METHOD_BS = [
    "ediffiqa(L)",
    "diffiqa(R)",
    "cr-fiqa(L)",
    "clib-fiqa",
    "grafiqs",
    "froqAda",
    "faceqan",
    "faceqnet",
    "sdd-fiqa"
]

GROUP_1_METHOD = [
    "ediffiqa(L)",
    "cr-fiqa(L)",
    "faceqnet",
    "ser-fiqa",
]

PLOT_FONT_SIZES = {
    "title": 30,
    "labels": 34,
    "ticks": 32,
    "legend": 20,
}

PLOT_LINE_WIDTHS = {
    "standard": 3.0,
    "mean": 4.2,
    "summary": 3.4,
    "evr_default": 2.8,
    "evr_highlight": 3.6,
    "reference": 2.8,
    "guide": 2.8,
}

PLOT_ALPHAS = {
    "per_method": 0.8,
}


def _display_dataset_name(dataset_name: str) -> str:
    return DATASET_DISPLAY_NAMES.get(str(dataset_name), str(dataset_name))


def _canonical_method_name(method_name: str) -> str:
    return METHOD_ALIASES.get(str(method_name), str(method_name))


def _display_method_name(method_name: str) -> str:
    canonical_name = _canonical_method_name(method_name)
    return METHOD_DISPLAY_NAMES.get(canonical_name, canonical_name)


def _method_sort_key(method_name: str) -> tuple[int, str]:
    method_name = _canonical_method_name(method_name)
    try:
        return (METHOD_PLOT_ORDER.index(method_name), _display_method_name(method_name))
    except ValueError:
        return (len(METHOD_PLOT_ORDER), _display_method_name(method_name))


def _method_color(method_name: str) -> str:
    method_name = _canonical_method_name(method_name)
    try:
        idx = METHOD_PLOT_ORDER.index(method_name)
        return METHOD_COLOR_PALETTE[idx % len(METHOD_COLOR_PALETTE)]
    except ValueError:
        fallback_idx = sum(ord(ch) for ch in method_name) % len(METHOD_COLOR_PALETTE)
        return METHOD_COLOR_PALETTE[fallback_idx]


def _resolve_requested_methods(
    requested_method_names: list[str],
    available_method_quality_dfs: dict[str, pd.DataFrame],
) -> list[str]:
    resolved_method_names = []
    for method_name in requested_method_names:
        canonical_name = _canonical_method_name(method_name)
        if canonical_name in available_method_quality_dfs and canonical_name not in resolved_method_names:
            resolved_method_names.append(canonical_name)
            continue
        if method_name in available_method_quality_dfs and method_name not in resolved_method_names:
            resolved_method_names.append(method_name)
    return resolved_method_names


def _pairwise_method_label(method_a: str, method_b: str) -> str:
    return f"{_display_method_name(method_a)} vs {_display_method_name(method_b)}"


def _all_vs_all_method_pairs(method_names: list[str]) -> list[tuple[str, str]]:
    return list(combinations(sorted(method_names, key=_method_sort_key), 2))

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


def _retained_image_set_for_discard_rate(
    *,
    img_ids: list[str],
    quality_df: pd.DataFrame,
    discard_rate: float,
) -> set[str]:
    """Return retained image-id set after discarding bottom discard_rate by FIQA score."""
    if not (0.0 <= float(discard_rate) <= 1.0):
        raise ValueError(f"discard_rate must be in [0,1]: {discard_rate}")

    quality_map = dict(zip(quality_df["image_id"], quality_df["score"]))
    quality_scores = np.array([float(quality_map.get(img_id, 0.0)) for img_id in img_ids], dtype=float)

    order = np.argsort(quality_scores)  # low -> high
    n = len(order)
    n_discard = min(int(float(discard_rate) * n), n)
    keep_idx = order[n_discard:]
    return {img_ids[int(i)] for i in keep_idx}


def _pair_counts_for_retained_set(
    *,
    pairs_df: pd.DataFrame,
    retained_images: set[str],
) -> dict:
    kept = pairs_df["img1_id"].isin(retained_images) & pairs_df["img2_id"].isin(retained_images)
    kept_df = pairs_df[kept]
    labels = kept_df["label"].astype(int).to_numpy()
    genuine = int(np.sum(labels == 1))
    impostor = int(np.sum(labels == 0))
    total = genuine + impostor
    genuine_ratio = (genuine / total) if total > 0 else np.nan
    gen_imp_ratio = (genuine / impostor) if impostor > 0 else np.nan
    return {
        "pairs_total": total,
        "pairs_genuine": genuine,
        "pairs_impostor": impostor,
        "class_ratio_genuine": genuine_ratio,
        "genuine_to_impostor_ratio": gen_imp_ratio,
    }


def _save_prob1_bar_chart(
    *,
    method_a: str,
    method_b: str,
    counts_original: dict,
    counts_a: dict,
    counts_b: dict,
    out_path: str,
):
    try:
        import matplotlib.pyplot as plt  # type: ignore[reportMissingImports]
    except Exception as e:
        raise SystemExit(f"matplotlib is required for plotting: {e}")

    method_a_display = _display_method_name(method_a)
    method_b_display = _display_method_name(method_b)

    labels = ["Original (0%)", f"{method_a_display} (20%)", f"{method_b_display} (20%)"]
    genuine_vals = [
        counts_original["pairs_genuine"],
        counts_a["pairs_genuine"],
        counts_b["pairs_genuine"],
    ]
    impostor_vals = [
        counts_original["pairs_impostor"],
        counts_a["pairs_impostor"],
        counts_b["pairs_impostor"],
    ]

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9, 5.2))
    bars_gen = ax.bar(x, genuine_vals, label="Genuine pairs", color="#4C72B0")
    bars_imp = ax.bar(
        x,
        impostor_vals,
        bottom=genuine_vals,
        label="Impostor pairs",
        color="#DD8452",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("# pairs")
    ax.grid(True, axis="y", alpha=0.2)
    ax.legend()

    totals = [g + i for g, i in zip(genuine_vals, impostor_vals)]
    base_genuine = counts_original["pairs_genuine"] if counts_original["pairs_genuine"] > 0 else 0
    base_impostor = counts_original["pairs_impostor"] if counts_original["pairs_impostor"] > 0 else 0
    label_dx = 0.22
    for xpos, g_val, i_val in zip(x, genuine_vals, impostor_vals):
        if base_genuine > 0:
            g_pct = (g_val / base_genuine) * 100.0
            g_label = f"G {g_val}\n({g_pct:.1f}%)"
        else:
            g_label = f"G {g_val}"
        if base_impostor > 0:
            i_pct = (i_val / base_impostor) * 100.0
            i_label = f"I {i_val}\n({i_pct:.1f}%)"
        else:
            i_label = f"I {i_val}"

        ax.text(
            xpos - label_dx,
            g_val / 2.0,
            g_label,
            ha="center",
            va="center",
            fontsize=7,
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white", alpha=0.85, edgecolor="none"),
        )
        ax.text(
            xpos + label_dx,
            g_val + (i_val / 2.0),
            i_label,
            ha="center",
            va="center",
            fontsize=7,
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white", alpha=0.85, edgecolor="none"),
        )
    base_total = totals[0] if len(totals) else 0
    for xpos, total in zip(x, totals):
        if base_total > 0:
            pct = (total / base_total) * 100.0
            total_label = f"{total}\n({pct:.1f}%)"
        else:
            total_label = str(total)
        ax.text(
            xpos,
            total / 2.0,
            total_label,
            ha="center",
            va="center",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85, edgecolor="none"),
        )

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _save_prob1_confusion_heatmap(
    *,
    method_a: str,
    method_b: str,
    retained_a: set[str],
    retained_b: set[str],
    total_images: int,
    discard_rate: float,
    out_path: str,
):
    try:
        import matplotlib.pyplot as plt  # type: ignore[reportMissingImports]
    except Exception as e:
        raise SystemExit(f"matplotlib is required for plotting: {e}")

    inter = len(retained_a & retained_b)
    only_a = len(retained_a) - inter
    only_b = len(retained_b) - inter
    both_removed = int(max(0, total_images - (inter + only_a + only_b)))

    jaccard = (inter / (inter + only_a + only_b)) if (inter + only_a + only_b) > 0 else np.nan
    overlap_pct = jaccard * 100.0 if np.isfinite(jaccard) else np.nan
    method_a_display = _display_method_name(method_a)
    method_b_display = _display_method_name(method_b)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    mat = np.array([[inter, only_a], [only_b, both_removed]], dtype=float)
    im = ax.imshow(mat, cmap="Blues")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Kept", "Removed"])
    ax.set_yticklabels(["Kept", "Removed"])
    ax.set_xlabel(method_b_display)
    ax.set_ylabel(method_a_display)

    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                str(int(mat[i, j])),
                ha="center",
                va="center",
                fontsize=11,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.75, edgecolor="none"),
            )

    note = "Overlap (Jaccard): " + (f"{overlap_pct:.1f}%" if np.isfinite(overlap_pct) else "nan")
    ax.text(0.5, -0.16, note, ha="center", va="center", transform=ax.transAxes, fontsize=10)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _save_prob1_pair_retention_ratios(
    *,
    method_a: str,
    method_b: str,
    counts_original: dict,
    counts_a: dict,
    counts_b: dict,
    discard_rate: float,
    out_path: str,
):
    """Plot normalized pair-set retention ratios as a clean bar chart."""
    try:
        import matplotlib.pyplot as plt  # type: ignore[reportMissingImports]
    except Exception as e:
        raise SystemExit(f"matplotlib is required for plotting: {e}")

    method_a_display = _display_method_name(method_a)
    method_b_display = _display_method_name(method_b)

    base_g = float(counts_original.get("pairs_genuine", 0))
    base_i = float(counts_original.get("pairs_impostor", 0))

    def _ratio(counts, key, base):
        if base <= 0:
            return np.nan
        return float(counts.get(key, 0)) / base

    ratios = {
        method_a_display: (
            _ratio(counts_a, "pairs_genuine", base_g),
            _ratio(counts_a, "pairs_impostor", base_i),
        ),
        method_b_display: (
            _ratio(counts_b, "pairs_genuine", base_g),
            _ratio(counts_b, "pairs_impostor", base_i),
        ),
    }

    labels = list(ratios.keys())
    genuine_vals = [ratios[m][0] for m in labels]
    impostor_vals = [ratios[m][1] for m in labels]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    ax.bar(x - width / 2, genuine_vals, width, label="|G_r|/|G_0|", color="#4C72B0")
    ax.bar(x + width / 2, impostor_vals, width, label="|I_r|/|I_0|", color="#DD8452")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Retention ratio")
    ax.grid(True, axis="y", alpha=0.2)
    ax.legend(fontsize=9, frameon=False)

    for xpos, g_val, i_val in zip(x, genuine_vals, impostor_vals):
        if np.isfinite(g_val):
            ax.text(xpos - width / 2, g_val + 0.02, f"{g_val:.3f}", ha="center", va="bottom", fontsize=9)
        if np.isfinite(i_val):
            ax.text(xpos + width / 2, i_val + 0.02, f"{i_val:.3f}", ha="center", va="bottom", fontsize=9)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _save_prob1_overlap_curve(
    *,
    method_a: str,
    method_b: str,
    img_ids: list[str],
    qa: pd.DataFrame,
    qb: pd.DataFrame,
    discard_rates: list[float],
    out_path: str,
):
    try:
        import matplotlib.pyplot as plt  # type: ignore[reportMissingImports]
    except Exception as e:
        raise SystemExit(f"matplotlib is required for plotting: {e}")

    method_a_display = _display_method_name(method_a)
    method_b_display = _display_method_name(method_b)

    jaccard_pct = []
    for dr in discard_rates:
        retained_a = _retained_image_set_for_discard_rate(img_ids=img_ids, quality_df=qa, discard_rate=dr)
        retained_b = _retained_image_set_for_discard_rate(img_ids=img_ids, quality_df=qb, discard_rate=dr)
        inter = retained_a & retained_b
        union = retained_a | retained_b
        j = (len(inter) / len(union)) if len(union) > 0 else np.nan
        jaccard_pct.append(j * 100.0 if np.isfinite(j) else np.nan)

    x = np.array(discard_rates, dtype=float) * 100.0
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.plot(x, jaccard_pct, marker="o", linewidth=PLOT_LINE_WIDTHS["standard"], color="#333333")
    ax.set_xlabel("Discard rate (%)")
    ax.set_ylabel("Retained-set overlap (Jaccard, %)")
    ax.set_ylim(0, 100)
    ax.grid(True, axis="y", alpha=0.25)

    for xi, yi in zip(x, jaccard_pct):
        if np.isfinite(yi):
            ax.text(xi, yi + 2.0, f"{yi:.1f}%", ha="center", va="bottom", fontsize=9)

    drop_idx = None
    for i, yi in enumerate(jaccard_pct):
        if np.isfinite(yi) and yi < 100.0:
            drop_idx = i
            break
    if drop_idx is not None:
        drop_x = x[drop_idx]
        ax.axvline(drop_x, color="#999999", linestyle="--", linewidth=PLOT_LINE_WIDTHS["guide"])
        ax.text(
            drop_x,
            5,
            f"first drop at {drop_x:.0f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            rotation=90,
            color="#666666",
        )

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _pair_set_for_retained(
    *,
    pairs_df: pd.DataFrame,
    retained_images: set[str],
    label_filter: int | None = None,
) -> set[tuple[str, str]]:
    kept = pairs_df["img1_id"].isin(retained_images) & pairs_df["img2_id"].isin(retained_images)
    if label_filter is not None:
        kept = kept & (pairs_df["label"].astype(int) == int(label_filter))
    kept_df = pairs_df[kept]
    pairs = set()
    for i1, i2 in zip(kept_df["img1_id"], kept_df["img2_id"]):
        if i1 <= i2:
            pairs.add((i1, i2))
        else:
            pairs.add((i2, i1))
    return pairs


def _jaccard_overlap(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if len(union) == 0:
        return 1.0
    return len(a & b) / len(union)


def _nanmean_std_by_column(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.full(values.shape[1], np.nan, dtype=float)
    std = np.full(values.shape[1], np.nan, dtype=float)
    for idx in range(values.shape[1]):
        col = values[:, idx]
        finite = col[np.isfinite(col)]
        if finite.size == 0:
            continue
        mean[idx] = float(np.mean(finite))
        std[idx] = float(np.std(finite))
    return mean, std


def _save_prob1_group_sample_overlap_plot(
    *,
    group_method_names: list[str],
    method_quality_dfs: dict[str, pd.DataFrame],
    img_ids: list[str],
    discard_rates: list[float],
    out_path: str,
):
    try:
        import matplotlib.pyplot as plt  # type: ignore[reportMissingImports]
        from matplotlib.lines import Line2D  # type: ignore[reportMissingImports]
    except Exception as e:
        raise SystemExit(f"matplotlib is required for plotting: {e}")

    selected_method_names = _resolve_requested_methods(group_method_names, method_quality_dfs)
    method_pairs = _all_vs_all_method_pairs(selected_method_names)
    if len(method_pairs) == 0:
        return

    retained_sets_by_method = {
        method_name: [
            _retained_image_set_for_discard_rate(
                img_ids=img_ids,
                quality_df=method_quality_dfs[method_name],
                discard_rate=discard_rate,
            )
            for discard_rate in discard_rates
        ]
        for method_name in selected_method_names
    }

    overlaps_by_pair: dict[tuple[str, str], list[float]] = {}
    for method_a, method_b in method_pairs:
        overlaps_by_pair[(method_a, method_b)] = [
            (overlap * 100.0) if np.isfinite(overlap) else np.nan
            for retained_a, retained_b in zip(retained_sets_by_method[method_a], retained_sets_by_method[method_b])
            for overlap in [_jaccard_overlap(retained_a, retained_b)]
        ]

    overlap_matrix = np.array(list(overlaps_by_pair.values()), dtype=float)
    mean_overlap, _ = _nanmean_std_by_column(overlap_matrix)
    x = np.array(discard_rates, dtype=float) * 100.0

    fig, ax = plt.subplots(figsize=(13.2, 7.2))

    for pair_idx, ((method_a, method_b), overlap_values) in enumerate(overlaps_by_pair.items()):
        ax.plot(
            x,
            overlap_values,
            linewidth=PLOT_LINE_WIDTHS["reference"],
            color=METHOD_COLOR_PALETTE[pair_idx % len(METHOD_COLOR_PALETTE)],
            alpha=PLOT_ALPHAS["per_method"],
        )

    ax.plot(x, mean_overlap, color="#111111", linewidth=PLOT_LINE_WIDTHS["mean"])

    ax.set_xlabel("Discard rate (%)", fontsize=PLOT_FONT_SIZES["labels"])
    ax.set_ylabel("Sample overlap (%)", fontsize=PLOT_FONT_SIZES["labels"])
    ax.set_xlim(0, 90)
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(v)) for v in x], fontsize=PLOT_FONT_SIZES["ticks"])
    ax.set_ylim(0, 100)
    ax.tick_params(axis="y", labelsize=PLOT_FONT_SIZES["ticks"])
    ax.grid(True, axis="y", alpha=0.25)

    style_legend = ax.legend(
        handles=[
            Line2D([0], [0], color="#9A9A9A", linewidth=PLOT_LINE_WIDTHS["reference"], label="Methods overlap"),
            Line2D([0], [0], color="#111111", linewidth=PLOT_LINE_WIDTHS["mean"], label="Mean overlap"),
        ],
        loc="upper right",
        bbox_to_anchor=(0.98, 0.98),
        frameon=True,
        framealpha=0.92,
        fontsize=PLOT_FONT_SIZES["legend"],
    )
    ax.add_artist(style_legend)

    pair_handles = [
        Line2D(
            [0],
            [0],
            color=METHOD_COLOR_PALETTE[pair_idx % len(METHOD_COLOR_PALETTE)],
            linewidth=PLOT_LINE_WIDTHS["summary"],
            label=_pairwise_method_label(method_a, method_b),
        )
        for pair_idx, (method_a, method_b) in enumerate(overlaps_by_pair.keys())
    ]
    ax.legend(
        handles=pair_handles,
        loc="lower right",
        bbox_to_anchor=(0.98, 0.02),
        frameon=True,
        framealpha=0.92,
        fontsize=PLOT_FONT_SIZES["legend"],
        handlelength=2.2,
    )

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=600)
    plt.close(fig)


def _save_prob1_group_pair_overlap_plot(
    *,
    group_method_names: list[str],
    method_quality_dfs: dict[str, pd.DataFrame],
    img_ids: list[str],
    pairs_df: pd.DataFrame,
    discard_rates: list[float],
    out_path: str,
):
    try:
        import matplotlib.pyplot as plt  # type: ignore[reportMissingImports]
        from matplotlib.lines import Line2D  # type: ignore[reportMissingImports]
    except Exception as e:
        raise SystemExit(f"matplotlib is required for plotting: {e}")

    selected_method_names = _resolve_requested_methods(group_method_names, method_quality_dfs)
    method_pairs = _all_vs_all_method_pairs(selected_method_names)
    if len(method_pairs) == 0:
        return

    retained_sets_by_method = {
        method_name: [
            _retained_image_set_for_discard_rate(
                img_ids=img_ids,
                quality_df=method_quality_dfs[method_name],
                discard_rate=discard_rate,
            )
            for discard_rate in discard_rates
        ]
        for method_name in selected_method_names
    }

    pair_sets_by_method_and_label: dict[tuple[str, int], list[set[tuple[str, str]]]] = {}
    for method_name in selected_method_names:
        for label_filter in (1, 0):
            pair_sets_by_method_and_label[(method_name, label_filter)] = [
                _pair_set_for_retained(
                    pairs_df=pairs_df,
                    retained_images=retained_images,
                    label_filter=label_filter,
                )
                for retained_images in retained_sets_by_method[method_name]
            ]

    genuine_by_pair: dict[tuple[str, str], list[float]] = {}
    impostor_by_pair: dict[tuple[str, str], list[float]] = {}
    for method_a, method_b in method_pairs:
        genuine_by_pair[(method_a, method_b)] = [
            (overlap * 100.0) if np.isfinite(overlap) else np.nan
            for pairs_a, pairs_b in zip(
                pair_sets_by_method_and_label[(method_a, 1)],
                pair_sets_by_method_and_label[(method_b, 1)],
            )
            for overlap in [_jaccard_overlap(pairs_a, pairs_b)]
        ]
        impostor_by_pair[(method_a, method_b)] = [
            (overlap * 100.0) if np.isfinite(overlap) else np.nan
            for pairs_a, pairs_b in zip(
                pair_sets_by_method_and_label[(method_a, 0)],
                pair_sets_by_method_and_label[(method_b, 0)],
            )
            for overlap in [_jaccard_overlap(pairs_a, pairs_b)]
        ]

    genuine_matrix = np.array(list(genuine_by_pair.values()), dtype=float)
    impostor_matrix = np.array(list(impostor_by_pair.values()), dtype=float)
    genuine_mean, _ = _nanmean_std_by_column(genuine_matrix)
    impostor_mean, _ = _nanmean_std_by_column(impostor_matrix)
    x = np.array(discard_rates, dtype=float) * 100.0

    fig, ax = plt.subplots(figsize=(13.6, 7.4))

    for pair_idx, (method_a, method_b) in enumerate(genuine_by_pair.keys()):
        pair_color = METHOD_COLOR_PALETTE[pair_idx % len(METHOD_COLOR_PALETTE)]
        ax.plot(
            x,
            genuine_by_pair[(method_a, method_b)],
            linewidth=PLOT_LINE_WIDTHS["reference"],
            linestyle="-",
            color=pair_color,
            alpha=PLOT_ALPHAS["per_method"],
        )
        ax.plot(
            x,
            impostor_by_pair[(method_a, method_b)],
            linewidth=PLOT_LINE_WIDTHS["reference"],
            linestyle="--",
            color=pair_color,
            alpha=PLOT_ALPHAS["per_method"],
        )

    ax.plot(x, genuine_mean, color="#0B3D0B", linewidth=PLOT_LINE_WIDTHS["mean"], linestyle="-")
    ax.plot(x, impostor_mean, color="#8B0000", linewidth=PLOT_LINE_WIDTHS["mean"], linestyle="--")

    ax.set_xlabel("Discard rate (%)", fontsize=PLOT_FONT_SIZES["labels"])
    ax.set_ylabel("Pair overlap (%)", fontsize=PLOT_FONT_SIZES["labels"])
    ax.set_xlim(0, 90)
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(v)) for v in x], fontsize=PLOT_FONT_SIZES["ticks"])
    ax.set_ylim(0, 100)
    ax.tick_params(axis="y", labelsize=PLOT_FONT_SIZES["ticks"])
    ax.grid(True, axis="y", alpha=0.25)

    style_legend = ax.legend(
        handles=[
            Line2D([0], [0], color="#9A9A9A", linewidth=PLOT_LINE_WIDTHS["reference"], linestyle="-", label="Methods (Genuine)"),
            Line2D([0], [0], color="#9A9A9A", linewidth=PLOT_LINE_WIDTHS["reference"], linestyle="--", label="Methods (Impostor)"),
            Line2D([0], [0], color="#0B3D0B", linewidth=PLOT_LINE_WIDTHS["mean"], linestyle="-", label="Mean Genuine"),
            Line2D([0], [0], color="#8B0000", linewidth=PLOT_LINE_WIDTHS["mean"], linestyle="--", label="Mean Impostor"),
        ],
        loc="upper right",
        bbox_to_anchor=(0.98, 0.98),
        frameon=True,
        framealpha=0.92,
        fontsize=PLOT_FONT_SIZES["legend"],
    )
    ax.add_artist(style_legend)

    pair_handles = [
        Line2D(
            [0],
            [0],
            color=METHOD_COLOR_PALETTE[pair_idx % len(METHOD_COLOR_PALETTE)],
            linewidth=PLOT_LINE_WIDTHS["summary"],
            linestyle="-",
            label=_pairwise_method_label(method_a, method_b),
        )
        for pair_idx, (method_a, method_b) in enumerate(genuine_by_pair.keys())
    ]
    ax.legend(
        handles=pair_handles,
        loc="lower right",
        bbox_to_anchor=(0.98, 0.02),
        frameon=True,
        framealpha=0.92,
        fontsize=PLOT_FONT_SIZES["legend"],
        handlelength=2.2,
    )

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=600)
    plt.close(fig)


def _save_prob1_reference_sample_overlap_plot(
    *,
    reference_method: str,
    method_quality_dfs: dict[str, pd.DataFrame],
    img_ids: list[str],
    discard_rates: list[float],
    out_path: str,
):
    try:
        import matplotlib.pyplot as plt  # type: ignore[reportMissingImports]
        from matplotlib.lines import Line2D  # type: ignore[reportMissingImports]
        from matplotlib.patches import Patch  # type: ignore[reportMissingImports]
    except Exception as e:
        raise SystemExit(f"matplotlib is required for plotting: {e}")

    if reference_method not in method_quality_dfs:
        raise ValueError(f"Reference FIQA method not found: {reference_method}")

    reference_display = _display_method_name(reference_method)

    method_names = sorted(
        [m for m in method_quality_dfs if m != reference_method],
        key=_method_sort_key,
    )
    if len(method_names) == 0:
        return

    reference_sets = [
        _retained_image_set_for_discard_rate(
            img_ids=img_ids,
            quality_df=method_quality_dfs[reference_method],
            discard_rate=dr,
        )
        for dr in discard_rates
    ]

    overlaps_by_method: dict[str, list[float]] = {}
    for method_name in method_names:
        method_sets = [
            _retained_image_set_for_discard_rate(
                img_ids=img_ids,
                quality_df=method_quality_dfs[method_name],
                discard_rate=dr,
            )
            for dr in discard_rates
        ]
        overlaps_by_method[method_name] = [
            (overlap * 100.0) if np.isfinite(overlap) else np.nan
            for reference_set, method_set in zip(reference_sets, method_sets)
            for overlap in [_jaccard_overlap(reference_set, method_set)]
        ]

    overlap_matrix = np.array(list(overlaps_by_method.values()), dtype=float)
    mean_overlap, _ = _nanmean_std_by_column(overlap_matrix)
    x = np.array(discard_rates, dtype=float) * 100.0

    fig, ax = plt.subplots(figsize=(13.2, 7.2))

    for method_name in method_names:
        ax.plot(
            x,
            overlaps_by_method[method_name],
            linewidth=PLOT_LINE_WIDTHS["reference"],
            color=_method_color(method_name),
            alpha=PLOT_ALPHAS["per_method"],
        )

    ax.plot(x, mean_overlap, color="#111111", linewidth=PLOT_LINE_WIDTHS["mean"])

    ax.set_xlabel("Discard rate (%)", fontsize=PLOT_FONT_SIZES["labels"])
    ax.set_ylabel(f"Sample overlap (%)", fontsize=PLOT_FONT_SIZES["labels"])
    ax.set_xlim(0, 90)
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(v)) for v in x], fontsize=PLOT_FONT_SIZES["ticks"])
    ax.set_ylim(0, 100)
    ax.tick_params(axis="y", labelsize=PLOT_FONT_SIZES["ticks"])
    ax.grid(True, axis="y", alpha=0.25)

    legend_handles = [
        Line2D([0], [0], color="#9A9A9A", linewidth=PLOT_LINE_WIDTHS["reference"], label=f"Per-method overlap vs {reference_display}"),
        Line2D([0], [0], color="#111111", linewidth=PLOT_LINE_WIDTHS["mean"], label="Mean overlap"),
    ]
    style_legend = ax.legend(
        handles=legend_handles,
        loc="upper right",
        bbox_to_anchor=(0.98, 0.98),
        frameon=True,
        framealpha=0.92,
        fontsize=PLOT_FONT_SIZES["legend"],
    )
    ax.add_artist(style_legend)

    method_handles = [
        Line2D([0], [0], color=_method_color(method_name), linewidth=PLOT_LINE_WIDTHS["summary"], label=_display_method_name(method_name))
        for method_name in method_names
    ]
    ax.legend(
        handles=method_handles,
        loc="lower right",
        bbox_to_anchor=(0.98, 0.02),
        frameon=True,
        framealpha=0.92,
        fontsize=PLOT_FONT_SIZES["legend"],
        handlelength=2.2,
    )

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=600)
    plt.close(fig)


def _save_prob1_reference_pair_overlap_plot(
    *,
    reference_method: str,
    method_quality_dfs: dict[str, pd.DataFrame],
    img_ids: list[str],
    pairs_df: pd.DataFrame,
    discard_rates: list[float],
    out_path: str,
):
    try:
        import matplotlib.pyplot as plt  # type: ignore[reportMissingImports]
        from matplotlib.lines import Line2D  # type: ignore[reportMissingImports]
        from matplotlib.patches import Patch  # type: ignore[reportMissingImports]
    except Exception as e:
        raise SystemExit(f"matplotlib is required for plotting: {e}")

    if reference_method not in method_quality_dfs:
        raise ValueError(f"Reference FIQA method not found: {reference_method}")

    reference_display = _display_method_name(reference_method)

    method_names = sorted(
        [m for m in method_quality_dfs if m != reference_method],
        key=_method_sort_key,
    )
    if len(method_names) == 0:
        return

    def _pair_sets_for_method(quality_df: pd.DataFrame, label_filter: int) -> list[set[tuple[str, str]]]:
        pair_sets = []
        for dr in discard_rates:
            retained = _retained_image_set_for_discard_rate(
                img_ids=img_ids,
                quality_df=quality_df,
                discard_rate=dr,
            )
            pair_sets.append(
                _pair_set_for_retained(
                    pairs_df=pairs_df,
                    retained_images=retained,
                    label_filter=label_filter,
                )
            )
        return pair_sets

    reference_genuine_sets = _pair_sets_for_method(method_quality_dfs[reference_method], label_filter=1)
    reference_impostor_sets = _pair_sets_for_method(method_quality_dfs[reference_method], label_filter=0)

    genuine_by_method: dict[str, list[float]] = {}
    impostor_by_method: dict[str, list[float]] = {}
    for method_name in method_names:
        method_genuine_sets = _pair_sets_for_method(method_quality_dfs[method_name], label_filter=1)
        method_impostor_sets = _pair_sets_for_method(method_quality_dfs[method_name], label_filter=0)
        genuine_by_method[method_name] = [
            (overlap * 100.0) if np.isfinite(overlap) else np.nan
            for reference_set, method_set in zip(reference_genuine_sets, method_genuine_sets)
            for overlap in [_jaccard_overlap(reference_set, method_set)]
        ]
        impostor_by_method[method_name] = [
            (overlap * 100.0) if np.isfinite(overlap) else np.nan
            for reference_set, method_set in zip(reference_impostor_sets, method_impostor_sets)
            for overlap in [_jaccard_overlap(reference_set, method_set)]
        ]

    genuine_matrix = np.array(list(genuine_by_method.values()), dtype=float)
    impostor_matrix = np.array(list(impostor_by_method.values()), dtype=float)
    genuine_mean, _ = _nanmean_std_by_column(genuine_matrix)
    impostor_mean, _ = _nanmean_std_by_column(impostor_matrix)
    x = np.array(discard_rates, dtype=float) * 100.0

    fig, ax = plt.subplots(figsize=(13.6, 7.4))

    for method_name in method_names:
        ax.plot(
            x,
            genuine_by_method[method_name],
            linewidth=PLOT_LINE_WIDTHS["reference"],
            linestyle="-",
            color=_method_color(method_name),
            alpha=PLOT_ALPHAS["per_method"],
        )
        ax.plot(
            x,
            impostor_by_method[method_name],
            linewidth=PLOT_LINE_WIDTHS["reference"],
            linestyle="--",
            color=_method_color(method_name),
            alpha=PLOT_ALPHAS["per_method"],
        )

    ax.plot(x, genuine_mean, color="#0B3D0B", linewidth=PLOT_LINE_WIDTHS["mean"], linestyle="-", label="Mean Genuine")
    ax.plot(x, impostor_mean, color="#8B0000", linewidth=PLOT_LINE_WIDTHS["mean"], linestyle="--", label="Mean Impostor")

    ax.set_xlabel("Discard rate (%)", fontsize=PLOT_FONT_SIZES["labels"])
    ax.set_ylabel("Pair overlap (%)", fontsize=PLOT_FONT_SIZES["labels"])
    ax.set_xlim(0, 90)
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(v)) for v in x], fontsize=PLOT_FONT_SIZES["ticks"])
    ax.set_ylim(0, 100)
    ax.tick_params(axis="y", labelsize=PLOT_FONT_SIZES["ticks"])
    ax.grid(True, axis="y", alpha=0.25)

    legend_handles = [
        Line2D([0], [0], color="#9A9A9A", linewidth=PLOT_LINE_WIDTHS["reference"], linestyle="-", label=f"Per-method Genuine vs {reference_display}"),
        Line2D([0], [0], color="#9A9A9A", linewidth=PLOT_LINE_WIDTHS["reference"], linestyle="--", label=f"Per-method Impostor vs {reference_display}"),
        Line2D([0], [0], color="#0B3D0B", linewidth=PLOT_LINE_WIDTHS["mean"], linestyle="-", label="Mean Genuine"),
        Line2D([0], [0], color="#8B0000", linewidth=PLOT_LINE_WIDTHS["mean"], linestyle="--", label="Mean Impostor"),
    ]
    style_legend = ax.legend(
        handles=legend_handles,
        loc="upper right",
        bbox_to_anchor=(0.98, 0.98),
        frameon=True,
        framealpha=0.92,
        fontsize=PLOT_FONT_SIZES["legend"],
    )
    ax.add_artist(style_legend)

    method_handles = [
        Line2D([0], [0], color=_method_color(method_name), linewidth=PLOT_LINE_WIDTHS["summary"], linestyle="-", label=_display_method_name(method_name))
        for method_name in method_names
    ]
    ax.legend(
        handles=method_handles,
        loc="lower right",
        bbox_to_anchor=(0.98, 0.02),
        frameon=True,
        framealpha=0.92,
        fontsize=PLOT_FONT_SIZES["legend"],
        handlelength=2.2,
    )

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=600)
    plt.close(fig)


def _save_prob1_pair_stability_curves(
    *,
    method_a: str,
    method_b: str,
    img_ids: list[str],
    qa: pd.DataFrame,
    qb: pd.DataFrame,
    pairs_df: pd.DataFrame,
    discard_rates: list[float],
    out_path_global: str,
    out_path_adjacent: str,
):
    try:
        import matplotlib.pyplot as plt  # type: ignore[reportMissingImports]
    except Exception as e:
        raise SystemExit(f"matplotlib is required for plotting: {e}")

    def _jacc(a: set[tuple[str, str]], b: set[tuple[str, str]]) -> float:
        if not a and not b:
            return np.nan
        union = a | b
        if len(union) == 0:
            return np.nan
        return len(a & b) / len(union)

    def _pair_sets_for_method(quality_df: pd.DataFrame):
        sets_g = []
        sets_i = []
        for dr in discard_rates:
            retained = _retained_image_set_for_discard_rate(
                img_ids=img_ids,
                quality_df=quality_df,
                discard_rate=dr,
            )
            sets_g.append(
                _pair_set_for_retained(
                    pairs_df=pairs_df,
                    retained_images=retained,
                    label_filter=1,
                )
            )
            sets_i.append(
                _pair_set_for_retained(
                    pairs_df=pairs_df,
                    retained_images=retained,
                    label_filter=0,
                )
            )
        return sets_g, sets_i

    def _plot_curves(x, g_vals, i_vals, title, out_path):
        fig, ax = plt.subplots(figsize=(7.6, 4.8))
        ax.plot(x, g_vals, marker="o", linewidth=PLOT_LINE_WIDTHS["standard"], label="Genuine", color="#4C72B0")
        ax.plot(x, i_vals, marker="o", linewidth=PLOT_LINE_WIDTHS["standard"], label="Impostor", color="#DD8452")
        ax.set_xlabel("Discard rate (%)")
        ax.set_ylabel("Jaccard (%)")
        ax.set_ylim(0, 100)
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(fontsize=9, frameon=False)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        fig.tight_layout()
        fig.savefig(out_path, dpi=200)
        plt.close(fig)

    x = np.array(discard_rates, dtype=float) * 100.0

    for method_name, quality_df in ((method_a, qa), (method_b, qb)):
        method_display = _display_method_name(method_name)
        g_sets, i_sets = _pair_sets_for_method(quality_df)

        g_global = [
            (_jacc(s, g_sets[0]) * 100.0 if np.isfinite(_jacc(s, g_sets[0])) else np.nan)
            for s in g_sets
        ]
        i_global = [
            (_jacc(s, i_sets[0]) * 100.0 if np.isfinite(_jacc(s, i_sets[0])) else np.nan)
            for s in i_sets
        ]

        g_adjacent = [np.nan]
        i_adjacent = [np.nan]
        for idx in range(1, len(g_sets)):
            g_adjacent.append(
                _jacc(g_sets[idx], g_sets[idx - 1]) * 100.0
                if np.isfinite(_jacc(g_sets[idx], g_sets[idx - 1]))
                else np.nan
            )
            i_adjacent.append(
                _jacc(i_sets[idx], i_sets[idx - 1]) * 100.0
                if np.isfinite(_jacc(i_sets[idx], i_sets[idx - 1]))
                else np.nan
            )

        _plot_curves(
            x,
            g_global,
            i_global,
            f"Global stability vs original: {method_display}",
            out_path_global.replace("{method}", str(method_name)),
        )
        _plot_curves(
            x,
            g_adjacent,
            i_adjacent,
            f"Adjacent-step stability (churn): {method_display}",
            out_path_adjacent.replace("{method}", str(method_name)),
        )


def _save_prob1_pair_overlap_curve(
    *,
    method_a: str,
    method_b: str,
    img_ids: list[str],
    qa: pd.DataFrame,
    qb: pd.DataFrame,
    pairs_df: pd.DataFrame,
    discard_rates: list[float],
    label_filter: int | None,
    out_path: str,
):
    try:
        import matplotlib.pyplot as plt  # type: ignore[reportMissingImports]
    except Exception as e:
        raise SystemExit(f"matplotlib is required for plotting: {e}")

    method_a_display = _display_method_name(method_a)
    method_b_display = _display_method_name(method_b)

    jaccard_pct = []
    for dr in discard_rates:
        retained_a = _retained_image_set_for_discard_rate(img_ids=img_ids, quality_df=qa, discard_rate=dr)
        retained_b = _retained_image_set_for_discard_rate(img_ids=img_ids, quality_df=qb, discard_rate=dr)
        pairs_a = _pair_set_for_retained(
            pairs_df=pairs_df,
            retained_images=retained_a,
            label_filter=label_filter,
        )
        pairs_b = _pair_set_for_retained(
            pairs_df=pairs_df,
            retained_images=retained_b,
            label_filter=label_filter,
        )
        inter = pairs_a & pairs_b
        union = pairs_a | pairs_b
        j = (len(inter) / len(union)) if len(union) > 0 else np.nan
        jaccard_pct.append(j * 100.0 if np.isfinite(j) else np.nan)

    x = np.array(discard_rates, dtype=float) * 100.0
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.plot(x, jaccard_pct, marker="o", linewidth=PLOT_LINE_WIDTHS["standard"], color="#333333")
    ax.set_xlabel("Discard rate (%)")
    ax.set_ylabel("Pair overlap (Jaccard, %)")
    if label_filter is None:
        label_txt = "all pairs"
    else:
        label_txt = "Genuine" if int(label_filter) == 1 else "Impostor"
    ax.set_ylim(0, 100)
    ax.grid(True, axis="y", alpha=0.25)

    for xi, yi in zip(x, jaccard_pct):
        if np.isfinite(yi):
            ax.text(xi, yi + 2.0, f"{yi:.1f}%", ha="center", va="bottom", fontsize=9)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _save_prob1_pair_instability(
    *,
    method_a: str,
    method_b: str,
    img_ids: list[str],
    qa: pd.DataFrame,
    qb: pd.DataFrame,
    pairs_df: pd.DataFrame,
    discard_rates: list[float],
    out_path: str,
):
    try:
        import matplotlib.pyplot as plt  # type: ignore[reportMissingImports]
    except Exception as e:
        raise SystemExit(f"matplotlib is required for plotting: {e}")

    method_a_display = _display_method_name(method_a)
    method_b_display = _display_method_name(method_b)

    discard_pct = [dr * 100.0 for dr in discard_rates]
    a_genuine = []
    a_impostor = []
    b_genuine = []
    b_impostor = []
    for dr in discard_rates:
        retained_a = _retained_image_set_for_discard_rate(
            img_ids=img_ids,
            quality_df=qa,
            discard_rate=dr,
        )
        retained_b = _retained_image_set_for_discard_rate(
            img_ids=img_ids,
            quality_df=qb,
            discard_rate=dr,
        )
        counts_a = _pair_counts_for_retained_set(pairs_df=pairs_df, retained_images=retained_a)
        counts_b = _pair_counts_for_retained_set(pairs_df=pairs_df, retained_images=retained_b)
        a_genuine.append(counts_a["pairs_genuine"])
        a_impostor.append(counts_a["pairs_impostor"])
        b_genuine.append(counts_b["pairs_genuine"])
        b_impostor.append(counts_b["pairs_impostor"])

    fig, ax = plt.subplots(figsize=(7.8, 4.9))
    ax.plot(discard_pct, a_genuine, marker="o", linewidth=PLOT_LINE_WIDTHS["standard"], label=f"{method_a_display} Genuine")
    ax.plot(discard_pct, a_impostor, marker="o", linewidth=PLOT_LINE_WIDTHS["standard"], label=f"{method_a_display} Impostor")
    ax.plot(discard_pct, b_genuine, marker="s", linewidth=PLOT_LINE_WIDTHS["standard"], label=f"{method_b_display} Genuine")
    ax.plot(discard_pct, b_impostor, marker="s", linewidth=PLOT_LINE_WIDTHS["standard"], label=f"{method_b_display} Impostor")
    ax.set_xlabel("Discard rate (%)")
    ax.set_ylabel("# pairs")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=8, ncol=2)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _save_prob1_concrete_example(
    *,
    method_a: str,
    method_b: str,
    retained_a: set[str],
    retained_b: set[str],
    discard_rate: float,
    out_path: str,
):
    try:
        import matplotlib.pyplot as plt  # type: ignore[reportMissingImports]
    except Exception as e:
        raise SystemExit(f"matplotlib is required for plotting: {e}")

    method_a_display = _display_method_name(method_a)
    method_b_display = _display_method_name(method_b)

    inter = len(retained_a & retained_b)
    only_a = len(retained_a) - inter
    only_b = len(retained_b) - inter
    union = len(retained_a | retained_b)
    jaccard = (inter / union) if union > 0 else np.nan
    jaccard_pct = jaccard * 100.0 if np.isfinite(jaccard) else np.nan

    totals = np.array([inter, only_a, only_b], dtype=float)
    if totals.sum() > 0:
        fracs = totals / totals.sum() * 100.0
    else:
        fracs = np.array([np.nan, np.nan, np.nan])

    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    left = 0.0
    colors = ["#4C72B0", "#55A868", "#C44E52"]
    labels = [f"Overlap ({method_a_display} & {method_b_display})", f"Only {method_a_display}", f"Only {method_b_display}"]
    for val, frac, color, label in zip(totals, fracs, colors, labels):
        ax.barh(0, frac, left=left, color=color, label=label)
        if np.isfinite(frac):
            ax.text(left + frac / 2.0, 0, str(int(val)), ha="center", va="center", fontsize=10, color="white")
        left += 0 if not np.isfinite(frac) else float(frac)

    ax.set_xlim(0, 100)
    ax.set_yticks([])
    ax.set_xlabel("Share of retained union (%)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=3, frameon=False, fontsize=9)

    note = "Jaccard: " + (f"{jaccard_pct:.1f}%" if np.isfinite(jaccard_pct) else "nan")
    ax.text(0.5, 1.15, note, ha="center", va="bottom", transform=ax.transAxes, fontsize=10)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def run_problem2_divergence(
    *,
    dataset_name: str,
    fr_model: str,
    emb_tag: str,
    embeddings_pkl: str,
    quality_dir: str,
    pairs_csv: str,
    out_dir: str,
    discard_rate: float = DEFAULT_PROB1_DISCARD_RATE,
    method_a: str = DEFAULT_PROB1_FIQA_METHOD_A,
    method_b: str | list[str] = DEFAULT_PROB1_FIQA_METHOD_BS,
    skip_additional_plots: bool = False,
):
    """Problem 2: show that FIQA methods induce different retained sets at fixed discard rate."""
    embeddings, id_to_idx = load_embeddings_pkl(embeddings_pkl)
    _ = embeddings  # embeddings not used; id_to_idx defines available image universe

    pairs_df = load_pairs_csv(pairs_csv)
    valid_mask = pairs_df["img1_id"].isin(id_to_idx) & pairs_df["img2_id"].isin(id_to_idx)
    pairs_df = pairs_df[valid_mask].reset_index(drop=True)

    img_ids = list(id_to_idx.keys())

    method_bs = [method_b] if isinstance(method_b, str) else list(method_b)
    qa_path = _quality_pkl_for_dataset(os.path.join(quality_dir, method_a), dataset_name)
    if not os.path.exists(qa_path):
        print(f"[skip] Problem2 missing quality pkl for method_a: {qa_path}")
        return
    qa = load_quality_scores_pkl(qa_path)
    method_quality_dfs = {str(method_a): qa}
    discard_rates = [i / 10.0 for i in range(10)]
    additional_plots_dir = os.path.join(out_dir, "additional_plots")

    counts_original = _pair_counts_for_retained_set(
        pairs_df=pairs_df,
        retained_images=set(img_ids),
    )
    retained_a = _retained_image_set_for_discard_rate(img_ids=img_ids, quality_df=qa, discard_rate=discard_rate)
    counts_a = _pair_counts_for_retained_set(pairs_df=pairs_df, retained_images=retained_a)

    for mb in method_bs:
        qb_path = _quality_pkl_for_dataset(os.path.join(quality_dir, mb), dataset_name)
        if not os.path.exists(qb_path):
            print(f"[skip] Problem2 missing quality pkl: {qb_path}")
            continue

        qb = load_quality_scores_pkl(qb_path)
        method_quality_dfs[str(mb)] = qb
        if skip_additional_plots:
            continue

        retained_b = _retained_image_set_for_discard_rate(img_ids=img_ids, quality_df=qb, discard_rate=discard_rate)
        counts_b = _pair_counts_for_retained_set(pairs_df=pairs_df, retained_images=retained_b)

        inter = retained_a & retained_b
        union = retained_a | retained_b

        jaccard = (len(inter) / len(union)) if len(union) > 0 else np.nan
        overlap_a_in_b = (len(inter) / len(retained_a)) if len(retained_a) > 0 else np.nan
        overlap_b_in_a = (len(inter) / len(retained_b)) if len(retained_b) > 0 else np.nan

        row = {
            "dataset": dataset_name,
            "dataset_display": _display_dataset_name(dataset_name),
            "fr_model": fr_model,
            "embeddings_tag": emb_tag,
            "discard_rate": float(discard_rate),
            "method_a": str(method_a),
            "method_b": str(mb),
            "method_a_display": _display_method_name(method_a),
            "method_b_display": _display_method_name(mb),
            "retained_images_a": int(len(retained_a)),
            "retained_images_b": int(len(retained_b)),
            "retained_images_intersection": int(len(inter)),
            "retained_images_union": int(len(union)),
            "retained_images_jaccard": jaccard,
            "retained_images_jaccard_pct": jaccard * 100.0 if np.isfinite(jaccard) else np.nan,
            "overlap_frac_a_in_b": overlap_a_in_b,
            "overlap_frac_b_in_a": overlap_b_in_a,
            "overlap_pct_a_in_b": overlap_a_in_b * 100.0 if np.isfinite(overlap_a_in_b) else np.nan,
            "overlap_pct_b_in_a": overlap_b_in_a * 100.0 if np.isfinite(overlap_b_in_a) else np.nan,
            **{f"a_{k}": v for k, v in counts_a.items()},
            **{f"b_{k}": v for k, v in counts_b.items()},
        }

        out_dir_ab = os.path.join(additional_plots_dir, f"{method_a}_vs_{mb}")
        os.makedirs(out_dir_ab, exist_ok=True)
        prefix = os.path.join(
            out_dir_ab,
            f"prob1_r{int(round(discard_rate * 100)):02d}_{method_a}_vs_{mb}",
        )

        # Option A: stacked bar chart
        _save_prob1_bar_chart(
            method_a=method_a,
            method_b=mb,
            counts_original=counts_original,
            counts_a=counts_a,
            counts_b=counts_b,
            out_path=prefix + "_bar.pdf",
        )

        # Option B: confusion heatmap (retained/removed)
        _save_prob1_confusion_heatmap(
            method_a=method_a,
            method_b=mb,
            retained_a=retained_a,
            retained_b=retained_b,
            total_images=len(img_ids),
            discard_rate=discard_rate,
            out_path=prefix + "_confusion_heatmap.pdf",
        )

        # Option B2: normalized pair-set retention ratios (clean bar chart)
        _save_prob1_pair_retention_ratios(
            method_a=method_a,
            method_b=mb,
            counts_original=counts_original,
            counts_a=counts_a,
            counts_b=counts_b,
            discard_rate=discard_rate,
            out_path=prefix + "_pair_retention_ratios.pdf",
        )

        # Option C: overlap vs discard rate (image sets)
        _save_prob1_overlap_curve(
            method_a=method_a,
            method_b=mb,
            img_ids=img_ids,
            qa=qa,
            qb=qb,
            discard_rates=discard_rates,
            out_path=prefix + "_overlap_curve.pdf",
        )

        # Option D: pair overlap vs discard rate (all/genuine/impostor)
        _save_prob1_pair_overlap_curve(
            method_a=method_a,
            method_b=mb,
            img_ids=img_ids,
            qa=qa,
            qb=qb,
            pairs_df=pairs_df,
            discard_rates=discard_rates,
            label_filter=None,
            out_path=prefix + "_pair_overlap_all.pdf",
        )
        _save_prob1_pair_overlap_curve(
            method_a=method_a,
            method_b=mb,
            img_ids=img_ids,
            qa=qa,
            qb=qb,
            pairs_df=pairs_df,
            discard_rates=discard_rates,
            label_filter=1,
            out_path=prefix + "_pair_overlap_genuine.pdf",
        )
        _save_prob1_pair_overlap_curve(
            method_a=method_a,
            method_b=mb,
            img_ids=img_ids,
            qa=qa,
            qb=qb,
            pairs_df=pairs_df,
            discard_rates=discard_rates,
            label_filter=0,
            out_path=prefix + "_pair_overlap_impostor.pdf",
        )

        # Option D2: pair stability curves (global vs original, adjacent-step churn)
        _save_prob1_pair_stability_curves(
            method_a=method_a,
            method_b=mb,
            img_ids=img_ids,
            qa=qa,
            qb=qb,
            pairs_df=pairs_df,
            discard_rates=discard_rates,
            out_path_global=prefix + "_{method}_pair_stability_global.pdf",
            out_path_adjacent=prefix + "_{method}_pair_stability_adjacent.pdf",
        )

        # Option E: pair set instability (combined)
        _save_prob1_pair_instability(
            method_a=method_a,
            method_b=mb,
            img_ids=img_ids,
            qa=qa,
            qb=qb,
            pairs_df=pairs_df,
            discard_rates=discard_rates,
            out_path=prefix + "_pair_instability.pdf",
        )

        # Option F: concrete example at current discard
        _save_prob1_concrete_example(
            method_a=method_a,
            method_b=mb,
            retained_a=retained_a,
            retained_b=retained_b,
            discard_rate=discard_rate,
            out_path=prefix + "_overlap_example.pdf",
        )

        # Option G: table
        pd.DataFrame([row]).to_csv(prefix + "_table.csv", index=False)
        print(
			f"[prob1] Saved: {prefix}_bar.pdf, {prefix}_confusion_heatmap.pdf, {prefix}_pair_retention_ratios.pdf, {prefix}_overlap_curve.pdf, {prefix}_pair_overlap_all.pdf, {prefix}_pair_overlap_genuine.pdf, {prefix}_pair_overlap_impostor.pdf, {prefix}_{method_a}_pair_stability_global.pdf, {prefix}_{method_a}_pair_stability_adjacent.pdf, {prefix}_{mb}_pair_stability_global.pdf, {prefix}_{mb}_pair_stability_adjacent.pdf, {prefix}_pair_instability.pdf, {prefix}_overlap_example.pdf, {prefix}_table.csv"
        )

    if len(method_quality_dfs) > 1:
        compact_prefix = os.path.join(out_dir, f"prob1_reference_{str(method_a).replace('-', '_')}")
        _save_prob1_reference_pair_overlap_plot(
            reference_method=str(method_a),
            method_quality_dfs=method_quality_dfs,
            img_ids=img_ids,
            pairs_df=pairs_df,
            discard_rates=discard_rates,
			out_path=compact_prefix + "_pair_overlap_summary.pdf",
        )
        _save_prob1_reference_sample_overlap_plot(
            reference_method=str(method_a),
            method_quality_dfs=method_quality_dfs,
            img_ids=img_ids,
            discard_rates=discard_rates,
            out_path=compact_prefix + "_sample_overlap_summary.pdf",
        )
        print(
            f"[prob1] Saved: {compact_prefix}_pair_overlap_summary.pdf, {compact_prefix}_sample_overlap_summary.pdf"
        )

    group1_methods = _resolve_requested_methods(GROUP_1_METHOD, method_quality_dfs)
    if len(group1_methods) > 1:
        group1_pair_out_path = os.path.join(out_dir, "prob1_group1_pair_overlap_summary.pdf")
        _save_prob1_group_pair_overlap_plot(
            group_method_names=group1_methods,
            method_quality_dfs=method_quality_dfs,
            img_ids=img_ids,
            pairs_df=pairs_df,
            discard_rates=discard_rates,
            out_path=group1_pair_out_path,
        )
        group1_sample_out_path = os.path.join(out_dir, "prob1_group_1_sample_overlap_summary.pdf")
        _save_prob1_group_sample_overlap_plot(
            group_method_names=group1_methods,
            method_quality_dfs=method_quality_dfs,
            img_ids=img_ids,
            discard_rates=discard_rates,
            out_path=group1_sample_out_path,
        )
        print(
            f"[prob1] Saved: {group1_pair_out_path}, {group1_sample_out_path}"
        )


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
                    reject_steps=20, fmr_target=1e-3, threshold_method="percentile"):
    """
    Iteratively reject low-quality samples and compute FNMR.
    
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
    
    Returns:
        reject_rates: Array of rejection rates
        fnmrs: Array of FNMR values
        detailed_results: List of dicts with comprehensive metrics (13 columns)
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

    # Ensure consistent ordering between embeddings and quality scores
    quality_map = dict(zip(quality_df["image_id"], quality_df["score"]))
    img_ids = list(id_to_idx.keys())  # Deterministic order
    quality_scores = np.array([quality_map.get(img_id, 0.0) for img_id in img_ids])

    sims = compute_similarity(embeddings, pairs_idx)
    labels = pairs_idx[:, 2].astype(int)

    reject_rates, fnmrs = [], []
    detailed_results = []
    rejection_info = []  # Store per-step rejection info

    for reject_fraction in reject_steps:
        # Sort by quality (low to high) and reject bottom fraction
        sorted_idx = np.argsort(quality_scores)  # low → high
        num_reject = int(reject_fraction * len(sorted_idx))
        reject_idx = sorted_idx[:num_reject]
        keep_idx = sorted_idx[num_reject:]

        # Keep only pairs where both images are in keep_idx
        mask_keep = np.isin(pairs_idx[:, 0], keep_idx) & np.isin(pairs_idx[:, 1], keep_idx)

        sims_keep = sims[mask_keep]
        labels_keep = labels[mask_keep]

        # Count genuine and impostor pairs after rejection
        genuine_remaining = np.sum(labels_keep == 1)
        impostor_remaining = np.sum(labels_keep == 0)
        total_genuine = np.sum(labels == 1)
        total_impostor = np.sum(labels == 0)

        # Stop if insufficient impostors to reliably compute target FMR (standard practice)
        min_impostors_needed = int(np.ceil(1.0 / fmr_target))  # e.g., 1000 for 1e-3
        if impostor_remaining < min_impostors_needed:
            break  # stop the EVR curve

        # Find threshold for target FMR using selected method
        if threshold_method == "roc":
            thr, actual_fmr = find_threshold_closest_fmr(sims_keep, labels_keep, fmr_target)
        else:  # "percentile"
            thr, actual_fmr = find_threshold_for_fmr(sims_keep, labels_keep, fmr_target)
        
        if thr is None:
            # No impostors remaining - skip this step
            continue
        
        fnmr = compute_fnmr(sims_keep, labels_keep, thr)
        
        # Count pairs above/below threshold
        genuine_sims = sims_keep[labels_keep == 1]
        impostor_sims = sims_keep[labels_keep == 0]
        
        gen_below = np.sum(genuine_sims < thr) if len(genuine_sims) > 0 else 0
        gen_above = np.sum(genuine_sims >= thr) if len(genuine_sims) > 0 else 0
        imp_below = np.sum(impostor_sims < thr) if len(impostor_sims) > 0 else 0
        imp_above = np.sum(impostor_sims >= thr) if len(impostor_sims) > 0 else 0

        reject_rates.append(reject_fraction)
        fnmrs.append(fnmr)
        
        # Comprehensive results with sample counts
        detailed_results.append({
            "reject_rate": reject_fraction,
            "threshold": thr,
            "fmr": actual_fmr,  # Use actual FMR from threshold calculation
            "fnmr": fnmr,
            "total_samples": len(quality_scores),
            "samples_rejected": num_reject,
            "samples_remaining": len(keep_idx),
            "total_pairs_after_rejection": genuine_remaining + impostor_remaining,
            "genuine_pairs_after_rejection": genuine_remaining,
            "impostor_pairs_after_rejection": impostor_remaining,
            "genuine_below_threshold": gen_below,
            "genuine_above_threshold": gen_above,
            "impostor_below_threshold": imp_below,
            "impostor_above_threshold": imp_above,
            "total_genuine_pairs_before_rejection": total_genuine,
            "total_impostor_pairs_before_rejection": total_impostor
        })
        
        # Store rejection information for this step (non-duplicated)
        rejected_images_this_step = [
            {
                "image_id": img_ids[idx],
                "quality_score": quality_scores[idx],
                "reject_fraction": reject_fraction
            }
            for idx in reject_idx
        ]
        rejection_info.append(rejected_images_this_step)

    # Ensure reject_rates and fnmrs are numpy arrays of the same length
    reject_rates = np.array(reject_rates)
    fnmrs = np.array(fnmrs)

    return reject_rates, fnmrs, detailed_results, quality_df, rejection_info


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
    threshold_method: str = "percentile",
):
    try:
        import matplotlib.pyplot as plt  # type: ignore[reportMissingImports]
    except Exception as e:
        raise SystemExit(f"matplotlib is required for plotting: {e}")

    if reject_steps is None:
        reject_steps = np.arange(0, 0.98, 0.05)

    embeddings, id_to_idx = load_embeddings_pkl(embeddings_pkl)
    pairs_df = load_pairs_csv(pairs_csv)

    methods = []
    curves = {}
    pauc_rows = []
    detailed_by_method = {}
    dataset_display = _display_dataset_name(dataset_name)

    def _run(method_name, quality_pkl_path):
        qdf = load_quality_scores_pkl(quality_pkl_path)
        rr, fn, detailed_results, _, _ = error_vs_reject(
            embeddings,
            pairs_df,
            qdf,
            id_to_idx,
            reject_steps=reject_steps,
            fmr_target=fmr_target,
            threshold_method=threshold_method,
        )
        p = compute_pauc(rr, fn, max_reject=0.3)
        curves[method_name] = (rr, fn)
        pauc_rows.append({
            "Method": method_name,
            "Method Display": _display_method_name(method_name),
            "pAUC@0.3": p,
        })
        methods.append(method_name)
        detailed_by_method[method_name] = detailed_results

    for d in sorted(os.listdir(quality_methods_dir)):
        p = os.path.join(quality_methods_dir, d)
        if not os.path.isdir(p) or d.startswith("."):
            continue
        qpath = _quality_pkl_for_dataset(p, dataset_name)
        if os.path.exists(qpath):
            _run(d, qpath)

    if len(methods) == 0:
        raise FileNotFoundError(
            f"No quality PKLs found for dataset '{dataset_name}' under: {quality_methods_dir}"
        )

    pauc_df = pd.DataFrame(pauc_rows).sort_values("pAUC@0.3", ascending=True)

    plot_methods = sorted(methods, key=_method_sort_key)

    plt.figure(figsize=(12, 8))
    for m in plot_methods:
        rr, fn = curves[m]
        p = pauc_df[pauc_df["Method"] == m]["pAUC@0.3"].values
        p_txt = "nan" if len(p) == 0 or np.isnan(p[0]) else f"{p[0]:.4f}"
        is_prop = m == "proposed"
        lw = PLOT_LINE_WIDTHS["evr_highlight"] if is_prop else PLOT_LINE_WIDTHS["evr_default"]
        color = "#111111" if is_prop else _method_color(m)
        z = 5 if is_prop else 2
        plt.plot(
            rr,
            fn,
            label=f"{_display_method_name(m)} (pAUC={p_txt})",
            color=color,
            linewidth=lw,
            zorder=z,
        )

    plt.xlabel("Discard rate (%)")
    plt.ylabel("FNMR")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=9)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = str(out_path)
    if not os.path.isabs(out_path):
        out_path = os.path.join(base_dir, out_path)
    os.makedirs(os.path.dirname(out_path) or base_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    pauc_csv = os.path.splitext(out_path)[0] + "_pauc.csv"
    pauc_df.to_csv(pauc_csv, index=False)

    # Save per-method detailed EVR tables
    out_prefix = os.path.splitext(out_path)[0]
    detailed_dir = out_prefix + "_detailed"
    os.makedirs(detailed_dir, exist_ok=True)
    for method_name, detailed in detailed_by_method.items():
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(method_name))
        pd.DataFrame(detailed).to_csv(os.path.join(detailed_dir, f"{safe}.csv"), index=False)

    print(f"Saved plot: {out_path}")
    print(f"Saved pAUC table: {pauc_csv}")
    print(f"Saved detailed CSVs: {detailed_dir}/")


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
    fmr_target: float = 1e-3,
    reject_steps=None,
    threshold_method: str = "percentile",
    skip_evr_plot: bool = False,
    skip_additional_plots: bool = False,
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
            additional_plots_dir = os.path.join(out_dir, "additional_plots")
            out_path = os.path.join(additional_plots_dir, f"evr_compare__{emb_tag}.pdf")
            print(f"[run] {dataset_name}/{fr_model}/{emb_tag} -> {out_path}")
            try:
                if (not skip_additional_plots) and (not skip_evr_plot):
                    os.makedirs(additional_plots_dir, exist_ok=True)
                    plot_evr_compare_dataset(
                        dataset_name=dataset_name,
                        embeddings_pkl=emb_pkl,
                        quality_methods_dir=quality_dir,
                        pairs_csv=pairs_csv,
                        out_path=out_path,
                        fmr_target=fmr_target,
                        reject_steps=reject_steps,
                        threshold_method=threshold_method,
                    )

                run_problem2_divergence(
                    dataset_name=dataset_name,
                    fr_model=fr_model,
                    emb_tag=emb_tag,
                    embeddings_pkl=emb_pkl,
                    quality_dir=quality_dir,
                    pairs_csv=pairs_csv,
                    out_dir=out_dir,
                    discard_rate=DEFAULT_PROB1_DISCARD_RATE,
                    method_a=DEFAULT_PROB1_FIQA_METHOD_A,
                    method_b=DEFAULT_PROB1_FIQA_METHOD_BS,
                    skip_additional_plots=skip_additional_plots,
                )
            except Exception as e:
                print(f"[fail] {dataset_name}/{fr_model}/{emb_tag}: {e}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Batch-run EVR for IJCB EVD Problem 1")
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
        help="Output root directory (default: output/prob1_test_set)",
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
        "--skip-evr-plot",
        action="store_true",
        help="Skip EVR curve plotting and only run Problem 2 divergence outputs.",
    )
    p.add_argument(
        "--skip-additional-plots",
        action="store_true",
        help="Skip all outputs under additional_plots and only save the compact summary plots.",
    )
    args = p.parse_args()

    run_prob1_test_set_batch(
        fr_features_root=args.fr_features_root,
        quality_dir=args.quality_dir,
        ca_fiqa_data_root=args.ca_fiqa_data_root,
        out_root=args.out_root,
        run_all=bool(args.run_all),
        fmr_target=args.fmr,
        threshold_method=args.threshold_method,
        skip_evr_plot=bool(args.skip_evr_plot),
        skip_additional_plots=bool(args.skip_additional_plots),
    )
