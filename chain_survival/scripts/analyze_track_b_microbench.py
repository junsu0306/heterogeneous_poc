"""Analyze paired GPU/DLA outputs produced by run_track_b_microbench.py."""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


def residual_metrics(gpu: np.ndarray, dla: np.ndarray) -> dict:
    gpu = np.asarray(gpu, dtype=np.float64)
    dla = np.asarray(dla, dtype=np.float64)
    residual = dla - gpu
    flat = residual.reshape(residual.shape[0], -1)
    centered = flat - flat.mean(axis=0, keepdims=True)
    gram = centered @ centered.T
    eigenvalues = np.maximum(np.linalg.eigvalsh(gram), 0.0)[::-1]
    total = eigenvalues.sum() + 1e-12
    effective_rank = float(
        eigenvalues.sum() ** 2 / (np.square(eigenvalues).sum() + 1e-12)
    )
    return {
        "mean_abs": float(np.mean(np.abs(residual))),
        "rms": float(np.sqrt(np.mean(np.square(residual)))),
        "relative_mean_abs": float(
            np.mean(np.abs(residual)) / (np.mean(np.abs(gpu)) + 1e-12)
        ),
        "normalized_by_gpu_rms": float(
            np.mean(np.abs(residual))
            / (np.sqrt(np.mean(np.square(gpu))) + 1e-12)
        ),
        "zero_occupancy_gpu": float(np.mean(gpu == 0)),
        "zero_occupancy_dla": float(np.mean(dla == 0)),
        "centered_top1_fraction": float(eigenvalues[0] / total),
        "centered_effective_rank": effective_rank,
        "mean_direction": flat.mean(axis=0).astype(np.float32),
        "per_image_l2": np.linalg.norm(flat, axis=1).astype(np.float32),
    }


def vector_metrics(value: np.ndarray) -> dict:
    value = np.asarray(value, dtype=np.float64)
    return {
        "mean_abs": float(np.mean(np.abs(value))),
        "rms": float(np.sqrt(np.mean(np.square(value)))),
        "mean_direction": value.reshape(value.shape[0], -1).mean(axis=0).astype(np.float32),
    }


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    if denominator == 0:
        return 0.0
    return float(np.dot(a, b) / denominator)


def serializable_metrics(metrics: dict) -> dict:
    return {
        key: value
        for key, value in metrics.items()
        if key not in {"mean_direction", "per_image_l2"}
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-index",
        type=Path,
        default=Path(
            "chain_survival/results/v13/microbench_hardware/run_index.json"
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("chain_survival/results/v13/microbench_manifest.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("chain_survival/results/v13/microbench_hardware_analysis.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.run_index.exists():
        blocked = {
            "schema_version": 1,
            "status": "BLOCKED",
            "reason": f"hardware run index does not exist: {args.run_index}",
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(blocked, indent=2))
        print(json.dumps(blocked, indent=2))
        return

    run_index = json.loads(args.run_index.read_text())
    manifest = json.loads(args.manifest.read_text())
    specs = {spec["model_id"]: spec for spec in manifest["models"]}
    records = [record for record in run_index["records"] if record["status"] == "OK"]
    mapping = {
        (
            record["model_id"],
            int(record["calibration"]),
            int(record["build"]),
            record["backend"],
        ): record
        for record in records
    }

    paired = []
    internal = {}
    for model_id, spec in specs.items():
        for calibration in range(run_index["settings"]["calibrations"]):
            for build in range(run_index["settings"]["builds"]):
                gpu_record = mapping.get((model_id, calibration, build, "gpu"))
                dla_record = mapping.get((model_id, calibration, build, "dla"))
                if gpu_record is None or dla_record is None:
                    continue
                with np.load(gpu_record["activation_path"]) as gpu_archive, np.load(
                    dla_record["activation_path"]
                ) as dla_archive:
                    for amplitude_key in sorted(
                        set(gpu_archive.files) & set(dla_archive.files)
                    ):
                        metrics = residual_metrics(
                            gpu_archive[amplitude_key], dla_archive[amplitude_key]
                        )
                        key = (model_id, calibration, build, amplitude_key)
                        internal[key] = metrics
                        paired.append(
                            {
                                "model_id": model_id,
                                "family": spec["family"],
                                "calibration": calibration,
                                "build": build,
                                "amplitude": amplitude_key,
                                **serializable_metrics(metrics),
                            }
                        )

    stability = []
    grouped_directions = {}
    for key, metrics in internal.items():
        model_id, _, _, amplitude = key
        grouped_directions.setdefault((model_id, amplitude), []).append(
            metrics["mean_direction"]
        )
    for (model_id, amplitude), directions in grouped_directions.items():
        pairwise = [cosine(a, b) for a, b in combinations(directions, 2)]
        stability.append(
            {
                "model_id": model_id,
                "family": specs[model_id]["family"],
                "amplitude": amplitude,
                "n_conditions": len(directions),
                "median_pairwise_direction_cosine": (
                    float(np.median(pairwise)) if pairwise else None
                ),
                "min_pairwise_direction_cosine": (
                    float(np.min(pairwise)) if pairwise else None
                ),
            }
        )

    causal = {}
    family_axis = {
        "graph_break": "requested_materialized_boundaries",
        "repeated_block": "n_blocks",
        "reduction": "reduction_length",
        "dataflow": "groups",
    }
    for family, axis in family_axis.items():
        points = []
        for spec in specs.values():
            if spec["family"] != family:
                continue
            values = [
                row["normalized_by_gpu_rms"]
                for row in paired
                if row["model_id"] == spec["model_id"] and row["amplitude"] == "amp_1"
            ]
            if values:
                points.append(
                    {
                        "model_id": spec["model_id"],
                        "x": float(spec["variable"][axis]),
                        "mean_normalized_residual": float(np.mean(values)),
                        "n_conditions": len(values),
                    }
                )
        points.sort(key=lambda row: row["x"])
        if len(points) >= 3:
            rho = float(
                spearmanr(
                    [point["x"] for point in points],
                    [point["mean_normalized_residual"] for point in points],
                ).statistic
            )
        else:
            rho = None
        causal[family] = {"axis": axis, "points": points, "spearman": rho}

    granularity = []
    for calibration in range(run_index["settings"]["calibrations"]):
        for build in range(run_index["settings"]["builds"]):
            condition = {}
            for model_id in (
                "granularity_per_channel",
                "granularity_per_tensor",
            ):
                for backend in ("gpu", "dla"):
                    record = mapping.get((model_id, calibration, build, backend))
                    if record is not None:
                        condition[(model_id, backend)] = np.load(
                            record["activation_path"]
                        )
            required = {
                ("granularity_per_channel", "gpu"),
                ("granularity_per_tensor", "gpu"),
                ("granularity_per_tensor", "dla"),
            }
            if not required <= set(condition):
                for archive in condition.values():
                    archive.close()
                continue
            for amplitude_key in condition[
                ("granularity_per_tensor", "gpu")
            ].files:
                g_pc = condition[("granularity_per_channel", "gpu")][amplitude_key]
                g_pt = condition[("granularity_per_tensor", "gpu")][amplitude_key]
                d_pt = condition[("granularity_per_tensor", "dla")][amplitude_key]
                granularity.append(
                    {
                        "calibration": calibration,
                        "build": build,
                        "amplitude": amplitude_key,
                        "granularity_grid_effect": serializable_metrics(
                            vector_metrics(g_pt - g_pc)
                        ),
                        "backend_after_coarse_grid": serializable_metrics(
                            vector_metrics(d_pt - g_pt)
                        ),
                        "total_proxy": serializable_metrics(vector_metrics(d_pt - g_pc)),
                    }
                )
            for archive in condition.values():
                archive.close()

    amp1_stability = [
        row
        for row in stability
        if row["amplitude"] == "amp_1" and row["n_conditions"] >= 3
    ]
    direction_candidates = [
        row
        for row in amp1_stability
        if row["median_pairwise_direction_cosine"] is not None
        and row["median_pairwise_direction_cosine"] >= 0.8
    ]
    result = {
        "schema_version": 1,
        "status": "ANALYZED",
        "n_successful_engine_records": len(records),
        "n_paired_conditions": len(paired),
        "paired_residuals": paired,
        "direction_stability": stability,
        "causal_trends": causal,
        "granularity_proxy_decomposition": granularity,
        "gate": {
            "direction_threshold": 0.8,
            "n_amp1_direction_candidates": len(direction_candidates),
            "decision": (
                "INCOMPLETE"
                if not direction_candidates
                else "REQUIRES_SCALE_AND_ENDPOINT_REVIEW"
            ),
            "reason": (
                "True quantization scales, endpoint occupancy, inspector-confirmed "
                "fusion/materialization, and input-vs-build variance must be reviewed "
                "before B-micro GO."
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(
        json.dumps(
            {
                "output": str(args.output),
                "n_paired_conditions": len(paired),
                "gate": result["gate"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
