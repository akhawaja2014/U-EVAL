#!/usr/bin/env python3
"""
Problem 2: Threshold Drift

Plot threshold vs discard rate summaries for EDC detailed CSV folders.

Run:
	python prob2_thr.py --root /home/bw/FIQA/evd_ijcb/output/evd/default_methods \
	  --out-dir /home/bw/FIQA/evd_ijcb/output_submission/prob2_thr_plots

Searches under a root directory (default: output/evd) for EDC detailed folders
(`*_detailed`) and creates one threshold-vs-discard summary plot per folder.
Each plot overlays all FIQA methods for the same dataset / FR model and highlights
the mean threshold curve.

By default, only the configured default FIQA methods are plotted. Use `--run-all`
to include every method found in each detailed EDC folder.
"""

from __future__ import annotations

import argparse
from functools import lru_cache
from pathlib import Path

import numpy as np  # type: ignore[reportMissingImports]
import pandas as pd  # type: ignore[reportMissingImports]


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
    "ser-fiq": "SER-FIQA",
    "faceqan": "FaceQAN",
    "faceqnet": "FaceQnet",
    "sdd-fiqa": "SDD-FIQA",
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
	"summary": 3.4,
	"reference": 2.8,
	"fmr": 2.8,
}

PLOT_ALPHAS = {
	"per_method": 0.8,
	"fmr": 0.8,
}

FMR_PLOT_COLOR = "#ff0000"

PLOT_REJECT_TICKS = np.arange(0, 101, 10, dtype=float)

ALIGNED_FMR_HIGHLIGHT_COLOR = "#b22222"


def _style_fmr_axis(ax_fmr) -> None:
	ax_fmr.set_ylabel("FMR", fontsize=PLOT_FONT_SIZES["labels"], color=FMR_PLOT_COLOR)
	ax_fmr.tick_params(axis="y", labelsize=PLOT_FONT_SIZES["ticks"], colors=FMR_PLOT_COLOR)
	ax_fmr.spines["right"].set_color(FMR_PLOT_COLOR)


def _format_fmr_value(value: float) -> str:
	if not np.isfinite(value):
		return "n/a"
	return f"{value:.4f}"


def _format_fmr_factor(value: float, baseline: float) -> str:
	if not np.isfinite(value) or not np.isfinite(baseline) or baseline <= 0:
		return "n/a"
	ratio = value / baseline
	if ratio >= 1:
		return f"{ratio:.1f}x"
	return f"{ratio:.2f}x"


def _format_fmr_change_annotation(value: float, baseline: float) -> str:
	if not np.isfinite(value) or not np.isfinite(baseline) or baseline <= 0:
		return f"FMR {_format_fmr_value(value)}"
	ratio = value / baseline
	pct_change = (ratio - 1.0) * 100.0
	return f"FMR {_format_fmr_value(value)}\n{ratio:.2f}x ({pct_change:+.0f}%)"


def _format_pct_change(value: float) -> str:
	if not np.isfinite(value):
		return "n/a"
	return f"{value:+.0f}%"


def _format_threshold_value(value: float) -> str:
	if not np.isfinite(value):
		return "n/a"
	return f"{value:.2f}"


def _threshold_to_exact_fmr(impostor_sims: np.ndarray, thresholds) -> np.ndarray:
	threshold_arr = np.asarray(thresholds, dtype=float)
	if threshold_arr.ndim == 0:
		threshold_arr = threshold_arr.reshape(1)
	if len(impostor_sims) == 0:
		return np.full_like(threshold_arr, np.nan, dtype=float)
	return np.array(
		[
			float(np.mean(impostor_sims >= threshold)) if np.isfinite(threshold) else np.nan
			for threshold in threshold_arr
		],
		dtype=float,
	)


def _exact_fmr_to_threshold(impostor_sims: np.ndarray, fmrs) -> np.ndarray:
	fmr_arr = np.asarray(fmrs, dtype=float)
	if fmr_arr.ndim == 0:
		fmr_arr = fmr_arr.reshape(1)
	if len(impostor_sims) == 0:
		return np.full_like(fmr_arr, np.nan, dtype=float)
	sorted_sims = np.sort(np.asarray(impostor_sims, dtype=float))
	quantiles = np.clip(1.0 - fmr_arr, 0.0, 1.0)
	return np.quantile(sorted_sims, quantiles, method="linear")


def _add_aligned_fmr_axis(
	ax,
	ax_fmr,
	impostor_sims: np.ndarray,
	baseline_fmr: float,
	threshold_ylim=None,
	max_ticks: int = 6,
) -> None:
	if not np.isfinite(baseline_fmr) or baseline_fmr <= 0:
		return

	if threshold_ylim is None:
		y_limits = np.asarray(ax.get_ylim(), dtype=float)
	else:
		y_limits = np.asarray(threshold_ylim, dtype=float)

	ax.set_ylim(y_limits[0], y_limits[1])

	# Use a small fixed number of evenly spaced ticks to avoid overlap
	threshold_ticks = np.linspace(y_limits[0], y_limits[1], max_ticks, dtype=float)
	fmr_ticks = _threshold_to_exact_fmr(impostor_sims, threshold_ticks)

	finite_mask = np.isfinite(threshold_ticks) & np.isfinite(fmr_ticks)
	threshold_ticks = threshold_ticks[finite_mask]
	fmr_ticks = fmr_ticks[finite_mask]

	if len(threshold_ticks) == 0:
		return

	# Left axis: FMR(t) and threshold t on the same aligned positions
	ax.set_yticks(threshold_ticks)
	ax.set_yticklabels(
		[
			f"{_format_fmr_value(fmr)}  {_format_threshold_value(threshold)}"
			for threshold, fmr in zip(threshold_ticks, fmr_ticks)
		],
		fontsize=PLOT_FONT_SIZES["ticks"],
	)

	# Right axis: percentage change only, aligned to the same threshold positions
	fmr_pct_ticks = ((fmr_ticks / baseline_fmr) - 1.0) * 100.0

	# IMPORTANT: use the same data coordinates as the left axis
	ax_fmr.set_ylim(y_limits[0], y_limits[1])
	ax_fmr.set_yticks(threshold_ticks)
	ax_fmr.set_yticklabels(
		[_format_pct_change(pct) for pct in fmr_pct_ticks],
		fontsize=PLOT_FONT_SIZES["ticks"],
		color=FMR_PLOT_COLOR,
	)
	ax_fmr.set_ylabel(
		"Change in FMR (%)",
		fontsize=PLOT_FONT_SIZES["labels"],
		color=FMR_PLOT_COLOR,
		rotation=270,
		labelpad=22,
	)
	ax_fmr.tick_params(axis="y", colors=FMR_PLOT_COLOR)
	ax_fmr.spines["right"].set_color(FMR_PLOT_COLOR)


def _annotate_fmr_change(ax, reject_rate: float, threshold: float, fmr: float, baseline_fmr: float, label: str) -> None:
	ax.scatter(
		[reject_rate],
		[threshold],
		s=120,
		color=ALIGNED_FMR_HIGHLIGHT_COLOR,
		edgecolor="white",
		linewidth=1.2,
		zorder=6,
	)
	ax.annotate(
		f"{label}\n{_format_fmr_change_annotation(fmr, baseline_fmr)}",
		xy=(reject_rate, threshold),
		xytext=(12, 12),
		textcoords="offset points",
		fontsize=max(PLOT_FONT_SIZES["legend"] - 2, 10),
		color=ALIGNED_FMR_HIGHLIGHT_COLOR,
		bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": ALIGNED_FMR_HIGHLIGHT_COLOR, "alpha": 0.95},
		arrowprops={"arrowstyle": "->", "color": ALIGNED_FMR_HIGHLIGHT_COLOR, "lw": 1.4},
	)


def _build_fmr_change_summary(plot_df: pd.DataFrame) -> str | None:
	if plot_df.empty:
		return None
	baseline_fmr = float(plot_df["fmr"].iloc[0])
	if not np.isfinite(baseline_fmr) or baseline_fmr <= 0 or len(plot_df) <= 1:
		return f"FMR: {_format_fmr_value(baseline_fmr)}"
	ratio_change = np.abs((plot_df["fmr"] / baseline_fmr) - 1.0)
	ratio_change.iloc[0] = -np.inf
	peak_reject = float(ratio_change.idxmax())
	peak_fmr = float(plot_df.loc[peak_reject, "fmr"])
	pct_change = ((peak_fmr / baseline_fmr) - 1.0) * 100.0
	return (
		f"Baseline FMR(0): {_format_fmr_value(baseline_fmr)}\n"
		f"Max delta FMR(t): {pct_change:+.0f}% at {peak_reject:.0f}% discard"
	)


def _build_endpoint_fmr_summary(plot_df: pd.DataFrame, preferred_reject_rate: float = 100.0) -> str | None:
	if plot_df.empty:
		return None

	baseline_reject = float(plot_df.index[0])
	baseline_fmr = float(plot_df["fmr"].iloc[0])
	if not np.isfinite(baseline_fmr):
		return None

	if preferred_reject_rate in plot_df.index:
		end_reject = float(preferred_reject_rate)
	else:
		end_reject = float(plot_df.index[-1])
	end_fmr = float(plot_df.loc[end_reject, "fmr"])

	if not np.isfinite(end_fmr) or baseline_fmr <= 0:
		return (
			f"FMR {_format_fmr_value(baseline_fmr)} at {baseline_reject:.0f}%\n"
			f"FMR {_format_fmr_value(end_fmr)} at {end_reject:.0f}%"
		)

	ratio = end_fmr / baseline_fmr
	pct_change = (ratio - 1.0) * 100.0
	change_word = "hike" if pct_change >= 0 else "drop"
	return (
		f"FMR {_format_fmr_value(baseline_fmr)} at {baseline_reject:.0f}%\n"
		f"FMR {_format_fmr_value(end_fmr)} at {end_reject:.0f}%\n"
		f"{ratio:.2f}x ({pct_change:+.0f}%) {change_word}"
	)


def _display_dataset_name(dataset_name: str) -> str:
	return DATASET_DISPLAY_NAMES.get(str(dataset_name), str(dataset_name))


def _display_method_name(method_name: str) -> str:
	return METHOD_DISPLAY_NAMES.get(str(method_name), str(method_name))


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


def _safe_method_name(method_name: str) -> str:
	return "".join(c if c.isalnum() or c in "-_" else "_" for c in str(method_name))


def _load_pauc_method_map(detailed_dir: Path) -> tuple[dict[str, str], dict[str, float]]:
	base_name = detailed_dir.name
	if not base_name.endswith("_detailed"):
		return {}, {}

	prefix = base_name[: -len("_detailed")]
	pauc_path = detailed_dir.parent / f"{prefix}_pauc.csv"
	if not pauc_path.exists():
		return {}, {}

	pauc_df = pd.read_csv(pauc_path)
	col_lookup = {str(col).strip().lower(): str(col) for col in pauc_df.columns}
	method_col = col_lookup.get("method")
	canonical_col = col_lookup.get("m_n") or method_col
	pauc_col = col_lookup.get("pauc@0.3")
	if canonical_col is None:
		return {}, {}

	safe_to_method: dict[str, str] = {}
	for _, row in pauc_df.iterrows():
		canonical_name = str(row[canonical_col])
		if method_col is not None:
			display_name = str(row[method_col])
			display_safe = _safe_method_name(display_name)
			safe_to_method[display_safe] = canonical_name
			safe_to_method[display_safe.lower()] = canonical_name
		canonical_safe = _safe_method_name(canonical_name)
		safe_to_method[canonical_safe] = canonical_name
		safe_to_method[canonical_safe.lower()] = canonical_name

	method_to_pauc = {}
	if pauc_col is not None:
		for _, row in pauc_df.iterrows():
			method_to_pauc[str(row[canonical_col])] = float(row[pauc_col])
	return safe_to_method, method_to_pauc


def _mean_threshold_by_reject_rate(series_by_method: dict[str, pd.Series]) -> pd.Series:
	threshold_df = pd.concat(series_by_method.values(), axis=1).sort_index()
	return threshold_df.mean(axis=1, skipna=True)


def _derive_embedding_name(path: Path) -> str:
	name = path.stem
	for suffix in ("-embeddings", "_embeddings", "_embeddings_dict", "_embedding", "-embedding"):
		if name.endswith(suffix):
			return name[: -len(suffix)]
	return name


def _resolve_embeddings_pkl(fr_features_root: Path, dataset_name: str, fr_model: str, emb_tag: str) -> Path:
	dataset_dir = fr_features_root / dataset_name
	model_dir = dataset_dir / fr_model
	candidate_dirs = [model_dir] if model_dir.is_dir() else [dataset_dir]

	for candidate_dir in candidate_dirs:
		pkls = sorted(path for path in candidate_dir.glob("*.pkl") if path.is_file())
		preferred = [path for path in pkls if "emb" in path.name.lower()]
		pkls = preferred if preferred else pkls
		matching = [path for path in pkls if _derive_embedding_name(path) == emb_tag]
		if matching:
			return matching[0]
		if len(pkls) == 1:
			return pkls[0]

	raise FileNotFoundError(
		f"Could not resolve embeddings PKL for dataset={dataset_name}, fr_model={fr_model}, emb_tag={emb_tag} under {fr_features_root}"
	)


@lru_cache(maxsize=None)
def _load_original_impostor_similarities(
	fr_features_root_str: str,
	ca_fiqa_data_root_str: str,
	dataset_name: str,
	fr_model: str,
	emb_tag: str,
) -> np.ndarray:
	from prob1_test_set import compute_similarity, load_embeddings_pkl, load_pairs_csv  # type: ignore[reportMissingImports]

	fr_features_root = Path(fr_features_root_str)
	ca_fiqa_data_root = Path(ca_fiqa_data_root_str)
	embeddings_pkl = _resolve_embeddings_pkl(fr_features_root, dataset_name, fr_model, emb_tag)
	pairs_csv = ca_fiqa_data_root / dataset_name / "pairs" / "pairs.csv"
	if not pairs_csv.exists():
		raise FileNotFoundError(f"Pairs CSV not found: {pairs_csv}")

	embeddings, id_to_idx = load_embeddings_pkl(str(embeddings_pkl))
	pairs_df = load_pairs_csv(str(pairs_csv))
	valid_mask = pairs_df["img1_id"].isin(id_to_idx) & pairs_df["img2_id"].isin(id_to_idx)
	pairs_df = pairs_df[valid_mask].reset_index(drop=True)
	if pairs_df.empty:
		raise ValueError(f"No valid pairs found for dataset={dataset_name}, fr_model={fr_model}")

	pairs_idx = np.array(
		[
			[id_to_idx[i1], id_to_idx[i2], lbl]
			for i1, i2, lbl in zip(pairs_df["img1_id"], pairs_df["img2_id"], pairs_df["label"])
		],
		dtype=int,
	)
	similarities = compute_similarity(embeddings, pairs_idx)
	labels = pairs_idx[:, 2]
	return np.asarray(similarities[labels == 0], dtype=float)


def _compute_original_fmr_from_thresholds(impostor_sims: np.ndarray, threshold_series: pd.Series) -> pd.Series:
	threshold_series = threshold_series.sort_index()
	fmr_values = np.array(
		[
			float(np.mean(impostor_sims >= threshold)) if np.isfinite(threshold) and len(impostor_sims) > 0 else np.nan
			for threshold in threshold_series.to_numpy(dtype=float)
		],
		dtype=float,
	)
	return pd.Series(fmr_values, index=threshold_series.index.to_numpy(dtype=float), name=threshold_series.name)


def _load_threshold_series_by_method(
	detailed_dir: Path,
	*,
	run_all: bool = False,
) -> tuple[dict[str, pd.Series], dict[str, float]]:
	safe_to_method, method_to_pauc = _load_pauc_method_map(detailed_dir)
	csv_paths = sorted(detailed_dir.glob("*.csv"))
	if not csv_paths:
		raise ValueError(f"No CSV files found in {detailed_dir}")

	allowed_methods = set(DEFAULT_METHOD_ALLOWLIST)
	series_by_method: dict[str, pd.Series] = {}
	for csv_path in csv_paths:
		df = pd.read_csv(csv_path)
		if "reject_rate" not in df.columns or "threshold" not in df.columns:
			continue

		stem = csv_path.stem
		method_name = safe_to_method.get(stem, safe_to_method.get(stem.lower(), stem))
		if (not run_all) and method_name not in allowed_methods:
			continue
		reject_rate_pct = df["reject_rate"].to_numpy(dtype=float) * 100.0
		thresholds = df["threshold"].to_numpy(dtype=float)
		series_by_method[method_name] = pd.Series(thresholds, index=reject_rate_pct, name=method_name)

	if not series_by_method:
		if run_all:
			raise ValueError(f"No detailed EVR CSVs with reject_rate/threshold columns in {detailed_dir}")
		raise ValueError(
			f"No detailed EVR CSVs matched DEFAULT_METHOD_ALLOWLIST in {detailed_dir} "
			f"(allowlist={DEFAULT_METHOD_ALLOWLIST})"
		)

	return series_by_method, method_to_pauc


def _plot_individual_threshold(
	detailed_dir: Path,
	method_name: str,
	series: pd.Series,
	original_fmr_series: pd.Series,
	out_path: Path,
) -> None:
	try:
		import matplotlib.pyplot as plt  # type: ignore[reportMissingImports]
		from matplotlib.lines import Line2D  # type: ignore[reportMissingImports]
	except Exception as e:
		raise SystemExit(f"matplotlib is required for plotting: {e}")

	fr_model = detailed_dir.parent.parent.name
	dataset_name = detailed_dir.parent.name
	series = series.sort_index()
	original_fmr_series = original_fmr_series.sort_index()
	x_values = series.index.to_numpy(dtype=float)
	y_values = series.to_numpy(dtype=float)

	fig, ax = plt.subplots(figsize=(13.2, 7.2))
	ax_fmr = ax.twinx()
	ax.plot(
		x_values,
		y_values,
		linewidth=PLOT_LINE_WIDTHS["summary"],
		color=_method_color(method_name),
		marker="o",
	)
	ax_fmr.plot(
		original_fmr_series.index.to_numpy(dtype=float),
		original_fmr_series.to_numpy(dtype=float),
		linewidth=PLOT_LINE_WIDTHS["fmr"],
		color=FMR_PLOT_COLOR,
		linestyle="--",
		marker="s",
		alpha=PLOT_ALPHAS["fmr"],
	)
	ax.set_xlabel("Discard rate (%)", fontsize=PLOT_FONT_SIZES["labels"])
	ax.set_ylabel("Threshold", fontsize=PLOT_FONT_SIZES["labels"])
	_style_fmr_axis(ax_fmr)
	ax.set_xlim(0.0, 100.0)
	ax.set_xticks(PLOT_REJECT_TICKS)
	ax.set_xticklabels([str(int(v)) for v in PLOT_REJECT_TICKS], fontsize=PLOT_FONT_SIZES["ticks"])
	ax.tick_params(axis="y", labelsize=PLOT_FONT_SIZES["ticks"])
	ax.grid(True, axis="y", alpha=0.25)
	ax.legend(
		[
			Line2D([0], [0], color=_method_color(method_name), linewidth=PLOT_LINE_WIDTHS["summary"], marker="o"),
			Line2D([0], [0], color=FMR_PLOT_COLOR, linewidth=PLOT_LINE_WIDTHS["fmr"], linestyle="--", marker="s"),
		],
		["Threshold", "FMR at threshold"],
		loc="upper right",
		frameon=True,
		framealpha=0.92,
		fontsize=PLOT_FONT_SIZES["legend"],
	)

	out_path.parent.mkdir(parents=True, exist_ok=True)
	fig.tight_layout()
	fig.savefig(out_path, dpi=600)
	plt.close(fig)


def _plot_threshold_summary(
	detailed_dir: Path,
	out_path: Path,
	*,
	fr_features_root: Path,
	ca_fiqa_data_root: Path,
	run_all: bool = False,
) -> None:
	try:
		import matplotlib.pyplot as plt  # type: ignore[reportMissingImports]
		import matplotlib.patheffects as pe  # type: ignore[reportMissingImports]
		from matplotlib.lines import Line2D  # type: ignore[reportMissingImports]
	except Exception as e:
		raise SystemExit(f"matplotlib is required for plotting: {e}")

	series_by_method, method_to_pauc = _load_threshold_series_by_method(detailed_dir, run_all=run_all)

	method_names = list(series_by_method.keys())
	method_names.sort(key=_method_sort_key)
	mean_series = _mean_threshold_by_reject_rate({name: series_by_method[name] for name in method_names})

	fr_model = detailed_dir.parent.parent.name
	dataset_name = detailed_dir.parent.name
	prefix = detailed_dir.name[: -len("_detailed")] if detailed_dir.name.endswith("_detailed") else detailed_dir.name
	emb_tag = prefix.replace("evr_compare__", "", 1)
	original_impostor_sims = _load_original_impostor_similarities(
		str(fr_features_root),
		str(ca_fiqa_data_root),
		dataset_name,
		fr_model,
		emb_tag,
	)
	summary_fmr_series = _compute_original_fmr_from_thresholds(original_impostor_sims, mean_series)

	fig, ax = plt.subplots(figsize=(13.2, 7.2))
	ax_fmr = ax.twinx()

	for method_name in method_names:
		series = series_by_method[method_name].sort_index()
		ax.plot(
			series.index.to_numpy(dtype=float),
			series.to_numpy(dtype=float),
			linewidth=PLOT_LINE_WIDTHS["reference"],
			color=_method_color(method_name),
			alpha=PLOT_ALPHAS["per_method"],
		)

	ax.plot(
		mean_series.index.to_numpy(dtype=float),
		mean_series.to_numpy(dtype=float),
		color="#111111",
		linewidth=PLOT_LINE_WIDTHS["mean"],
	)
	ax_fmr.plot(
		summary_fmr_series.index.to_numpy(dtype=float),
		summary_fmr_series.to_numpy(dtype=float),
		color=FMR_PLOT_COLOR,
		linewidth=PLOT_LINE_WIDTHS["fmr"],
		linestyle="--",
	)

	ax.set_xlabel("Discard rate (%)", fontsize=PLOT_FONT_SIZES["labels"])
	ax.set_ylabel("Threshold", fontsize=PLOT_FONT_SIZES["labels"])
	_style_fmr_axis(ax_fmr)
	ax.set_xlim(0.0, 100.0)
	ax.set_xticks(PLOT_REJECT_TICKS)
	ax.set_xticklabels([str(int(v)) for v in PLOT_REJECT_TICKS], fontsize=PLOT_FONT_SIZES["ticks"])
	ax.tick_params(axis="y", labelsize=PLOT_FONT_SIZES["ticks"])
	ax.grid(True, axis="y", alpha=0.25)

	style_legend = ax.legend(
		handles=[
			Line2D([0], [0], color="#9A9A9A", linewidth=PLOT_LINE_WIDTHS["reference"], label="Per-method threshold (t)"),
			Line2D([0], [0], color="#111111", linewidth=PLOT_LINE_WIDTHS["mean"], label="Mean threshold and FMR(t)"),
			Line2D([0], [0], color=FMR_PLOT_COLOR, linewidth=PLOT_LINE_WIDTHS["fmr"], linestyle="--", label="FMR at mean threshold"),
		],
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

	out_path.parent.mkdir(parents=True, exist_ok=True)
	fig.tight_layout()
	fig.savefig(out_path, dpi=600)
	plt.close(fig)


def _plot_threshold_summary_with_fmr_info(
	detailed_dir: Path,
	out_path: Path,
	*,
	fr_features_root: Path,
	ca_fiqa_data_root: Path,
	run_all: bool = False,
) -> None:
	try:
		import matplotlib.pyplot as plt  # type: ignore[reportMissingImports]
		from matplotlib.lines import Line2D  # type: ignore[reportMissingImports]
	except Exception as e:
		raise SystemExit(f"matplotlib is required for plotting: {e}")

	series_by_method, _ = _load_threshold_series_by_method(detailed_dir, run_all=run_all)
	method_names = list(series_by_method.keys())
	method_names.sort(key=_method_sort_key)
	mean_series = _mean_threshold_by_reject_rate({name: series_by_method[name] for name in method_names})

	fr_model = detailed_dir.parent.parent.name
	dataset_name = detailed_dir.parent.name
	prefix = detailed_dir.name[: -len("_detailed")] if detailed_dir.name.endswith("_detailed") else detailed_dir.name
	emb_tag = prefix.replace("evr_compare__", "", 1)
	original_impostor_sims = _load_original_impostor_similarities(
		str(fr_features_root),
		str(ca_fiqa_data_root),
		dataset_name,
		fr_model,
		emb_tag,
	)
	summary_fmr_series = _compute_original_fmr_from_thresholds(original_impostor_sims, mean_series)
	plot_df = pd.DataFrame({"threshold": mean_series, "fmr": summary_fmr_series}).dropna().sort_index()

	fig, ax = plt.subplots(figsize=(13.2, 7.2))
	ax_fmr = ax.twinx()

	for method_name in method_names:
		series = series_by_method[method_name].sort_index()
		ax.plot(
			series.index.to_numpy(dtype=float),
			series.to_numpy(dtype=float),
			linewidth=PLOT_LINE_WIDTHS["reference"],
			color=_method_color(method_name),
			alpha=PLOT_ALPHAS["per_method"],
		)

	ax.plot(
		mean_series.index.to_numpy(dtype=float),
		mean_series.to_numpy(dtype=float),
		color="#111111",
		linewidth=PLOT_LINE_WIDTHS["mean"],
	)
	ax_fmr.plot(
		summary_fmr_series.index.to_numpy(dtype=float),
		summary_fmr_series.to_numpy(dtype=float),
		color=FMR_PLOT_COLOR,
		linewidth=PLOT_LINE_WIDTHS["fmr"],
		linestyle="--",
	)

	ax.set_xlabel("Discard rate (%)", fontsize=PLOT_FONT_SIZES["labels"])
	ax.set_ylabel("Threshold", fontsize=PLOT_FONT_SIZES["labels"])
	_style_fmr_axis(ax_fmr)
	ax.set_xlim(0.0, 100.0)
	ax.set_xticks(PLOT_REJECT_TICKS)
	ax.set_xticklabels([str(int(v)) for v in PLOT_REJECT_TICKS], fontsize=PLOT_FONT_SIZES["ticks"])
	ax.tick_params(axis="y", labelsize=PLOT_FONT_SIZES["ticks"])
	ax.grid(True, axis="y", alpha=0.25)

	style_legend = ax.legend(
		handles=[
			Line2D([0], [0], color="#9A9A9A", linewidth=PLOT_LINE_WIDTHS["reference"], label="Per-method threshold (t)"),
			Line2D([0], [0], color="#111111", linewidth=PLOT_LINE_WIDTHS["mean"], label="Mean threshold and FMR(t)"),
			Line2D([0], [0], color=FMR_PLOT_COLOR, linewidth=PLOT_LINE_WIDTHS["fmr"], linestyle="--", label="FMR at mean threshold"),
		],
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

	endpoint_summary = _build_endpoint_fmr_summary(plot_df)
	if endpoint_summary:
		ax.text(
			0.5,
			0.02,
			endpoint_summary,
			transform=ax.transAxes,
			va="bottom",
			ha="center",
			fontsize=max(PLOT_FONT_SIZES["legend"] - 1, 10),
			color=ALIGNED_FMR_HIGHLIGHT_COLOR,
			bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": ALIGNED_FMR_HIGHLIGHT_COLOR, "alpha": 0.95},
		)

	out_path.parent.mkdir(parents=True, exist_ok=True)
	fig.tight_layout()
	fig.savefig(out_path, dpi=600)
	plt.close(fig)


def _plot_threshold_fmr_aligned_summary(
	detailed_dir: Path,
	out_path: Path,
	*,
	fr_features_root: Path,
	ca_fiqa_data_root: Path,
	run_all: bool = False,
) -> None:
	try:
		import matplotlib.pyplot as plt  # type: ignore[reportMissingImports]
		from matplotlib.lines import Line2D  # type: ignore[reportMissingImports]
	except Exception as e:
		raise SystemExit(f"matplotlib is required for plotting: {e}")

	series_by_method, _ = _load_threshold_series_by_method(detailed_dir, run_all=run_all)
	method_names = list(series_by_method.keys())
	method_names.sort(key=_method_sort_key)
	mean_series = _mean_threshold_by_reject_rate(
		{name: series_by_method[name] for name in method_names}
	).sort_index()

	fr_model = detailed_dir.parent.parent.name
	dataset_name = detailed_dir.parent.name
	prefix = detailed_dir.name[: -len("_detailed")] if detailed_dir.name.endswith("_detailed") else detailed_dir.name
	emb_tag = prefix.replace("evr_compare__", "", 1)

	original_impostor_sims = _load_original_impostor_similarities(
		str(fr_features_root),
		str(ca_fiqa_data_root),
		dataset_name,
		fr_model,
		emb_tag,
	)
	summary_fmr_series = _compute_original_fmr_from_thresholds(
		original_impostor_sims,
		mean_series,
	).sort_index()

	plot_df = pd.DataFrame({"threshold": mean_series, "fmr": summary_fmr_series}).dropna().sort_index()
	if plot_df.empty:
		raise ValueError(f"No valid threshold/FMR pairs available for aligned plot in {detailed_dir}")

	if 0.0 in plot_df.index:
		baseline_fmr = float(plot_df.loc[0.0, "fmr"])
	else:
		baseline_fmr = float(plot_df["fmr"].iloc[0])

	if not np.isfinite(baseline_fmr) or baseline_fmr <= 0:
		raise ValueError(f"Invalid baseline FMR for aligned plot in {detailed_dir}")

	fig, ax = plt.subplots(figsize=(13.2, 7.2))
	ax_fmr = ax.twinx()

	# Plot only threshold curves
	for method_name in method_names:
		series = series_by_method[method_name].sort_index()
		ax.plot(
			series.index.to_numpy(dtype=float),
			series.to_numpy(dtype=float),
			linewidth=PLOT_LINE_WIDTHS["reference"],
			color=_method_color(method_name),
			alpha=PLOT_ALPHAS["per_method"],
		)

	ax.plot(
		mean_series.index.to_numpy(dtype=float),
		mean_series.to_numpy(dtype=float),
		color="#111111",
		linewidth=PLOT_LINE_WIDTHS["mean"],
		alpha=0.85,
		zorder=4,
	)

	ax.set_xlabel("Discard rate (%)", fontsize=PLOT_FONT_SIZES["labels"])
	ax.set_ylabel("")
	ax.set_xlim(0.0, 100.0)
	ax.set_xticks(PLOT_REJECT_TICKS)
	ax.set_xticklabels([str(int(v)) for v in PLOT_REJECT_TICKS], fontsize=PLOT_FONT_SIZES["ticks"])
	ax.tick_params(axis="y", labelsize=PLOT_FONT_SIZES["ticks"])
	ax.grid(True, axis="y", alpha=0.25)

	threshold_ylim = ax.get_ylim()
	fig.canvas.draw()

	# Aligned axis relabeling only; no FMR line plotted here
	_add_aligned_fmr_axis(
		ax,
		ax_fmr,
		original_impostor_sims,
		baseline_fmr,
		threshold_ylim=threshold_ylim,
		max_ticks=6,
	)

	# Left-side combined header
	ax.text(
		-0.18,
		1.02,
		"FMR(t)",
		transform=ax.transAxes,
		va="bottom",
		ha="center",
		fontsize=PLOT_FONT_SIZES["ticks"],
		color="black",
	)
	ax.text(
		-0.03,
		1.02,
		"t",
		transform=ax.transAxes,
		va="bottom",
		ha="center",
		fontsize=PLOT_FONT_SIZES["ticks"],
	)

	style_legend = ax.legend(
		handles=[
			Line2D([0], [0], color="#9A9A9A", linewidth=PLOT_LINE_WIDTHS["reference"], label="Per-method threshold (t)"),
			Line2D([0], [0], color="#111111", linewidth=PLOT_LINE_WIDTHS["mean"], label="Mean threshold and FMR(t)"),
		],
		loc="upper right",
		bbox_to_anchor=(0.98, 0.98),
		frameon=True,
		framealpha=0.92,
		fontsize=PLOT_FONT_SIZES["legend"],
	)
	ax.add_artist(style_legend)

	method_handles = [
		Line2D(
			[0],
			[0],
			color=_method_color(method_name),
			linewidth=PLOT_LINE_WIDTHS["summary"],
			label=_display_method_name(method_name),
		)
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

	out_path.parent.mkdir(parents=True, exist_ok=True)
	fig.subplots_adjust(bottom=0.14)
	fig.savefig(out_path, dpi=600, bbox_inches="tight", pad_inches=0.02)
	plt.close(fig)


def _get_original_impostor_fmr_series_for_method(
	detailed_dir: Path,
	method_series: pd.Series,
	fr_features_root: Path,
	ca_fiqa_data_root: Path,
) -> pd.Series:
	fr_model = detailed_dir.parent.parent.name
	dataset_name = detailed_dir.parent.name
	prefix = detailed_dir.name[: -len("_detailed")] if detailed_dir.name.endswith("_detailed") else detailed_dir.name
	emb_tag = prefix.replace("evr_compare__", "", 1)
	original_impostor_sims = _load_original_impostor_similarities(
		str(fr_features_root),
		str(ca_fiqa_data_root),
		dataset_name,
		fr_model,
		emb_tag,
	)
	return _compute_original_fmr_from_thresholds(original_impostor_sims, method_series)


def _default_output_path(root: Path, detailed_dir: Path, out_dir: Path | None) -> Path:
	fr_model = detailed_dir.parent.parent.name
	dataset_name = detailed_dir.parent.name
	filename = f"prob_thr_{fr_model}_{dataset_name}.pdf"

	if out_dir is None:
		return detailed_dir.parent / "additional_plots" / filename

	rel_parent = detailed_dir.parent.relative_to(root)
	return out_dir / rel_parent / "additional_plots" / filename


def _default_individual_output_dir(root: Path, detailed_dir: Path, out_dir: Path | None) -> Path:
	if out_dir is None:
		return detailed_dir.parent / "additional_plots" / "individual_thresholds"

	rel_parent = detailed_dir.parent.relative_to(root)
	return out_dir / rel_parent / "additional_plots" / "individual_thresholds"


def _default_aligned_output_path(root: Path, detailed_dir: Path, out_dir: Path | None) -> Path:
	fr_model = detailed_dir.parent.parent.name
	dataset_name = detailed_dir.parent.name
	filename = f"prob_thr_aligned_fmr_{fr_model}_{dataset_name}.pdf"

	if out_dir is None:
		return detailed_dir.parent / "additional_plots" / filename

	rel_parent = detailed_dir.parent.relative_to(root)
	return out_dir / rel_parent / "additional_plots" / filename


def _default_info_output_path(root: Path, detailed_dir: Path, out_dir: Path | None) -> Path:
	fr_model = detailed_dir.parent.parent.name
	dataset_name = detailed_dir.parent.name
	filename = f"prob_thr_info_{fr_model}_{dataset_name}.pdf"

	if out_dir is None:
		return detailed_dir.parent / "additional_plots" / filename

	rel_parent = detailed_dir.parent.relative_to(root)
	return out_dir / rel_parent / "additional_plots" / filename


def _save_aligned_threshold_fmr_plot(
	root: Path,
	detailed_dir: Path,
	out_dir: Path | None,
	*,
	fr_features_root: Path,
	ca_fiqa_data_root: Path,
	run_all: bool,
) -> None:
	aligned_out_path = _default_aligned_output_path(root, detailed_dir, out_dir)
	_plot_threshold_fmr_aligned_summary(
		detailed_dir,
		aligned_out_path,
		fr_features_root=fr_features_root,
		ca_fiqa_data_root=ca_fiqa_data_root,
		run_all=run_all,
	)
	print(f"Saved: {aligned_out_path}")


def main() -> int:
	parser = argparse.ArgumentParser(
		description="Plot threshold vs discard rate summaries for EVR detailed CSV folders"
	)
	parser.add_argument(
		"--root",
		default="/home/bw/FIQA/evd_ijcb/output/evd",
		help="Root directory to search for EVR detailed folders",
	)
	parser.add_argument(
		"--out-dir",
		default=None,
		help="Optional output directory for plots (defaults to alongside each detailed folder)",
	)
	parser.add_argument(
		"--run-all",
		action="store_true",
		help="Ignore the default FIQA-method allowlist and plot all methods found.",
	)
	parser.add_argument(
		"--fr-features-root",
		default="/home/bw/FIQA/fiq_baselines/fr_features",
		help="Root directory containing FR embeddings for original-data FMR computation.",
	)
	parser.add_argument(
		"--ca-fiqa-data-root",
		default="/home/bw/FIQA/ca-fiqa/data",
		help="Root directory containing pairs.csv files for original-data FMR computation.",
	)
	args = parser.parse_args()

	root = Path(args.root)
	fr_features_root = Path(args.fr_features_root)
	ca_fiqa_data_root = Path(args.ca_fiqa_data_root)
	if not root.exists():
		raise SystemExit(f"Root not found: {root}")
	if not fr_features_root.exists():
		raise SystemExit(f"FR features root not found: {fr_features_root}")
	if not ca_fiqa_data_root.exists():
		raise SystemExit(f"CA-FIQA data root not found: {ca_fiqa_data_root}")

	detailed_dirs = sorted(path for path in root.rglob("*_detailed") if path.is_dir())
	if not detailed_dirs:
		print(f"No *_detailed folders found under: {root}")
		return 0

	out_dir = Path(args.out_dir) if args.out_dir else None
	if out_dir:
		out_dir.mkdir(parents=True, exist_ok=True)

	for detailed_dir in detailed_dirs:
		try:
			series_by_method, method_to_pauc = _load_threshold_series_by_method(
				detailed_dir,
				run_all=bool(args.run_all),
			)
			method_names = list(series_by_method.keys())
			method_names.sort(key=_method_sort_key)

			out_path = _default_output_path(root, detailed_dir, out_dir)
			_plot_threshold_summary(
				detailed_dir,
				out_path,
				fr_features_root=fr_features_root,
				ca_fiqa_data_root=ca_fiqa_data_root,
				run_all=bool(args.run_all),
			)
			print(f"Saved: {out_path}")

			info_out_path = _default_info_output_path(root, detailed_dir, out_dir)
			_plot_threshold_summary_with_fmr_info(
				detailed_dir,
				info_out_path,
				fr_features_root=fr_features_root,
				ca_fiqa_data_root=ca_fiqa_data_root,
				run_all=bool(args.run_all),
			)
			print(f"Saved: {info_out_path}")

			try:
				_save_aligned_threshold_fmr_plot(
					root,
					detailed_dir,
					out_dir,
					fr_features_root=fr_features_root,
					ca_fiqa_data_root=ca_fiqa_data_root,
					run_all=bool(args.run_all),
				)
			except Exception as aligned_error:
				print(f"[aligned-skip] {detailed_dir}: {aligned_error}")

			individual_out_dir = _default_individual_output_dir(root, detailed_dir, out_dir)
			for method_name in method_names:
				individual_out_path = individual_out_dir / f"{_safe_method_name(method_name)}_threshold.pdf"
				original_fmr_series = _get_original_impostor_fmr_series_for_method(
					detailed_dir,
					series_by_method[method_name],
					fr_features_root,
					ca_fiqa_data_root,
				)
				_plot_individual_threshold(
					detailed_dir,
					method_name,
					series_by_method[method_name],
					original_fmr_series,
					individual_out_path,
				)
				print(f"Saved: {individual_out_path}")
		except Exception as e:
			print(f"[skip] {detailed_dir}: {e}")

	return 0


if __name__ == "__main__":
	raise SystemExit(main())
