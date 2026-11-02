#!/usr/bin/env python3
"""Plot weighting functions used in RCE.

This visualizes the weighting rule used in `rce.py`:

	w(s) = (R_H(s) / R) ** alpha

where:
	- R_H(s) is the 1-based harm rank of sample s
	- R is the total number of ranked samples

Run examples:

	python sol4_rce_function.py

	python sol4_rce_function.py \
		--max-rank 100 \
		--alphas 0,10 \
		--out output/sol4_rce_function/sol4_rce_function.pdf
"""

from __future__ import annotations

import argparse
import os

import numpy as np  # type: ignore[reportMissingImports]


DEFAULT_ALPHAS = [0.0, 1.0, 2.0, 3.0]
DEFAULT_MAX_RANK = 1000
DEFAULT_OUT = os.path.join("output", "sol4_rce_function.pdf")

PLOT_FONT_SIZES = {
	"labels": 32,
	"ticks": 30,
	"legend": 19,
}


def weights_for_ranks(ranks: np.ndarray, alpha: float) -> np.ndarray:
	"""Compute weights from ranks using sol4 weighting rule."""
	if ranks.size == 0:
		return np.array([], dtype=float)
	r_total = float(np.max(ranks))
	return np.power(ranks.astype(float) / r_total, float(alpha))


def plot_weighting_function(*, max_rank: int, alphas: list[float], out_path: str) -> str:
	"""Plot w(s) across ranks for all requested alpha values."""
	try:
		import matplotlib.pyplot as plt  # type: ignore[reportMissingImports]
	except Exception as exc:
		raise SystemExit(f"matplotlib is required for plotting: {exc}")

	if max_rank < 1:
		raise ValueError("max_rank must be >= 1")
	if len(alphas) == 0:
		raise ValueError("At least one alpha value is required")

	ranks = np.arange(1, int(max_rank) + 1, dtype=float)
	if max_rank == 1:
		rank_scale_1_to_100 = np.array([1.0], dtype=float)
	else:
		# Map rank index to a common interpretation scale where 1 is best and 100 is worst.
		rank_scale_1_to_100 = 1.0 + 99.0 * (ranks - 1.0) / float(max_rank - 1)

	plt.figure(figsize=(10, 6))
	for alpha in alphas:
		weights = weights_for_ranks(ranks, alpha)
		plt.plot(rank_scale_1_to_100, weights, linewidth=2.5, label=rf"$\gamma={float(alpha):g}$")

	plt.xlabel("Quality rank (100 worst, 1 best)", fontsize=PLOT_FONT_SIZES["labels"])
	plt.ylabel("Weight w(s)", fontsize=PLOT_FONT_SIZES["labels"])
	plt.xlim(100, 1)
	plt.ylim(0, 1.05)
	plt.xticks([100, 90, 80, 70, 60, 50, 40, 30, 20, 10, 1], fontsize=PLOT_FONT_SIZES["ticks"])
	plt.yticks(fontsize=PLOT_FONT_SIZES["ticks"])
	plt.grid(True, alpha=0.3)
	plt.legend(loc="best", fontsize=PLOT_FONT_SIZES["legend"])

	out_path = str(out_path)
	out_dir = os.path.dirname(out_path)
	if out_dir:
		os.makedirs(out_dir, exist_ok=True)

	plt.tight_layout()
	plt.savefig(out_path, dpi=600)
	plt.close()
	return out_path


def _parse_alphas(value: str) -> list[float]:
	vals = [x.strip() for x in str(value).split(",") if x.strip() != ""]
	if len(vals) == 0:
		return list(DEFAULT_ALPHAS)
	return [float(v) for v in vals]


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Plot weighting function used in RCE"
	)
	parser.add_argument(
		"--max-rank",
		type=int,
		default=DEFAULT_MAX_RANK,
		help="Maximum rank R used on x-axis (default: 1000)",
	)
	parser.add_argument(
		"--alphas",
		default=",".join(format(a, "g") for a in DEFAULT_ALPHAS),
		help="Comma-separated gamma values, e.g. 0,1,2,3",
	)
	parser.add_argument(
		"--out",
		default=DEFAULT_OUT,
		help="Output image path (default: output/sol4_rce_function.pdf)",
	)
	args = parser.parse_args()

	alphas = _parse_alphas(args.alphas)
	out_path = plot_weighting_function(
		max_rank=int(args.max_rank),
		alphas=alphas,
		out_path=str(args.out),
	)
	print(f"Saved weighting plot: {out_path}")


if __name__ == "__main__":
	main()

