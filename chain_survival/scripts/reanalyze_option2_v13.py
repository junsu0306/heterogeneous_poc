"""Reanalyze legacy Option 2 activations with v13-normalized metrics.

The stored arrays contain spatially averaged activations from one build and one
calibration condition. They are useful for candidate generation only; they
cannot pass the v13 multi-build mechanism/model gates.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


BOUNDARIES = ("layer1.2_shallow", "layer4.2_deep")


def svd_summary(matrix: np.ndarray) -> dict:
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False)
    energy = np.square(singular)
    total = energy.sum() + 1e-12
    return {
        "top1_fraction": float(energy[0] / total),
        "top3_fraction": float(energy[:3].sum() / total),
        "effective_rank": float(singular.sum() ** 2 / total),
    }


def correlations(gpu: np.ndarray, residual: np.ndarray) -> np.ndarray:
    values = []
    for channel in range(gpu.shape[1]):
        x = np.abs(gpu[:, channel])
        y = np.abs(residual[:, channel])
        if np.std(x) == 0 or np.std(y) == 0:
            values.append(np.nan)
        else:
            values.append(float(spearmanr(x, y).statistic))
    return np.asarray(values, dtype=np.float64)


def summarize(path: Path) -> dict:
    with np.load(path) as archive:
        gpu_all = np.asarray(archive["A_gpu"], dtype=np.float64)
        dla_all = np.asarray(archive["A_dla"], dtype=np.float64)
        normal = np.asarray(archive["normal"], dtype=np.int64)
    gpu = gpu_all[:, normal]
    residual = dla_all[:, normal] - gpu
    mean_abs_per_channel = np.mean(np.abs(residual), axis=0)
    corr = correlations(gpu, residual)
    top = np.argsort(-mean_abs_per_channel)[: min(10, len(normal))]

    mean_abs_residual = float(np.mean(np.abs(residual)))
    gpu_mean_abs = float(np.mean(np.abs(gpu)))
    gpu_rms = float(np.sqrt(np.mean(np.square(gpu))))
    gpu_std = float(np.std(gpu))
    return {
        "path": str(path),
        "n_images": int(gpu.shape[0]),
        "n_normal_channels": int(gpu.shape[1]),
        "mean_abs_residual": mean_abs_residual,
        "gpu_mean_abs": gpu_mean_abs,
        "gpu_rms": gpu_rms,
        "gpu_std_about_global_mean": gpu_std,
        "residual_over_gpu_mean_abs": mean_abs_residual / (gpu_mean_abs + 1e-12),
        "residual_over_gpu_rms": mean_abs_residual / (gpu_rms + 1e-12),
        "residual_over_gpu_std": mean_abs_residual / (gpu_std + 1e-12),
        "activation_residual_spearman_all_normal": {
            "median": float(np.nanmedian(corr)),
            "q25": float(np.nanquantile(corr, 0.25)),
            "q75": float(np.nanquantile(corr, 0.75)),
        },
        "activation_residual_spearman_top10_by_divergence": {
            "median": float(np.nanmedian(corr[top])),
            "values": [float(value) for value in corr[top]],
            "channels": [int(normal[index]) for index in top],
        },
        "centered_residual_svd": svd_summary(residual),
        "limitations": [
            "single build",
            "single calibration subset",
            "GPU per-channel versus DLA total difference is not decomposed",
            "stored activations are spatial means",
            "quantization step and true endpoint occupancy are unavailable",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir", type=Path, default=Path("chain_survival/results")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("chain_survival/results/v13/option2_v13_reanalysis.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = {
        boundary: summarize(args.results_dir / f"option2_{boundary}.npz")
        for boundary in BOUNDARIES
    }
    shallow = rows["layer1.2_shallow"]
    deep = rows["layer4.2_deep"]
    comparison = {
        "deep_over_shallow_raw_mean_abs_residual": (
            deep["mean_abs_residual"] / shallow["mean_abs_residual"]
        ),
        "deep_over_shallow_residual_over_gpu_mean_abs": (
            deep["residual_over_gpu_mean_abs"]
            / shallow["residual_over_gpu_mean_abs"]
        ),
        "deep_over_shallow_residual_over_gpu_rms": (
            deep["residual_over_gpu_rms"] / shallow["residual_over_gpu_rms"]
        ),
        "interpretation": (
            "The raw depth ratio is not a causal accumulation estimate because "
            "activation scales differ and granularity/backend effects are not decomposed."
        ),
    }
    result = {
        "schema_version": 1,
        "boundaries": rows,
        "comparison": comparison,
        "gate": {
            "decision": "PILOT_ONLY",
            "reason": "v13 B-model requires multi-build/multi-calibration consensus after B1/B2.",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps({"output": str(args.output), "comparison": comparison, "gate": result["gate"]}, indent=2))


if __name__ == "__main__":
    main()
