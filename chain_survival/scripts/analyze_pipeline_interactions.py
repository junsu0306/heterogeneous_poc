"""Analyze non-additive and environment-dependent v15 pipeline effects."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    left = left.reshape(-1).astype(np.float64)
    right = right.reshape(-1).astype(np.float64)
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return float(np.dot(left, right) / max(denominator, 1e-12))


def load_environment(index_path: Path) -> dict[str, Any]:
    index = json.loads(index_path.read_text())
    captures = {}
    paths = None
    for record in index["records"]:
        if (
            record["status"] != "OK"
            or not record.get("strict_artifact_gate", True)
        ):
            continue
        data = np.load(record["capture_path"])
        current_paths = data["paths"]
        if paths is None:
            paths = current_paths
        elif not np.array_equal(paths, current_paths):
            raise ValueError(f"unpaired capture in {index_path}")
        captures[record["state_id"]] = data["logits"].astype(np.float64)
    settings = index["settings"]
    return {
        "index": str(index_path),
        "calibration": settings.get("calibration"),
        "build_id": settings.get("build_id"),
        "dla_core": settings.get("dla_core"),
        "paths": paths,
        "captures": captures,
    }


def residual_summary(value: np.ndarray) -> dict[str, float]:
    per_image_rms = np.sqrt(np.mean(value**2, axis=1))
    return {
        "mean_abs": float(np.mean(np.abs(value))),
        "rms": float(np.sqrt(np.mean(value**2))),
        "max_abs": float(np.max(np.abs(value))),
        "mean_image_rms": float(np.mean(per_image_rms)),
        "median_image_rms": float(np.median(per_image_rms)),
        "p95_image_rms": float(np.quantile(per_image_rms, 0.95)),
    }


def pair_kind(left: dict[str, Any], right: dict[str, Any]) -> str:
    same_calibration = left["calibration"] == right["calibration"]
    same_build = left["build_id"] == right["build_id"]
    if same_calibration and not same_build:
        return "build_only"
    if not same_calibration and same_build:
        return "calibration_only"
    if same_calibration and same_build:
        return "replicate"
    return "calibration_and_build"


def environment_effects(environment: dict[str, Any]) -> dict[str, Any]:
    states = environment["captures"]
    effects: dict[str, Any] = {}

    def add(name: str, required: list[str], value: np.ndarray) -> None:
        if all(state in states for state in required):
            effects[name] = {
                "required_states": required,
                "summary": residual_summary(value),
                "residual": value,
            }

    if all(state in states for state in ("S5", "S4", "S2", "S1")):
        add(
            "quantization_x_compilation",
            ["S1", "S2", "S4", "S5"],
            states["S5"] - states["S4"] - states["S2"] + states["S1"],
        )
    if all(state in states for state in ("S6", "S5")):
        add(
            "explicit_backend",
            ["S5", "S6"],
            states["S6"] - states["S5"],
        )
    if all(state in states for state in ("S8", "S7")):
        add(
            "implicit_backend",
            ["S7", "S8"],
            states["S8"] - states["S7"],
        )
    if all(state in states for state in ("S7", "S5")):
        add(
            "implicit_vs_explicit_gpu",
            ["S5", "S7"],
            states["S7"] - states["S5"],
        )
    return effects


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--capture-indexes",
        nargs="+",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "chain_survival/results/v15/ablations/pipeline_interactions.json"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    environments = [load_environment(path) for path in args.capture_indexes]
    effect_records = []
    residuals_by_name: dict[str, list[dict[str, Any]]] = {}
    for environment in environments:
        identity = (
            f"{environment['calibration']}__build{environment['build_id']}"
            f"__dla{environment['dla_core']}"
        )
        effects = environment_effects(environment)
        serializable = {}
        for name, effect in effects.items():
            residuals_by_name.setdefault(name, []).append(
                {
                    "identity": identity,
                    "calibration": environment["calibration"],
                    "build_id": environment["build_id"],
                    "dla_core": environment["dla_core"],
                    "residual": effect["residual"],
                    "effect_rms": effect["summary"]["rms"],
                }
            )
            serializable[name] = {
                "required_states": effect["required_states"],
                "summary": effect["summary"],
            }
        effect_records.append(
            {
                "environment": identity,
                "capture_index": environment["index"],
                "effects": serializable,
            }
        )

    stability = {}
    for name, records in residuals_by_name.items():
        comparisons = []
        for left_index in range(len(records)):
            for right_index in range(left_index + 1, len(records)):
                left_record = records[left_index]
                right_record = records[right_index]
                left = left_record["residual"]
                right = right_record["residual"]
                comparisons.append(
                    {
                        "left": left_record["identity"],
                        "right": right_record["identity"],
                        "pair_kind": pair_kind(left_record, right_record),
                        "cosine": cosine(left, right),
                        "difference": residual_summary(right - left),
                    }
                )
        grouped = {}
        for kind in (
            "build_only",
            "calibration_only",
            "calibration_and_build",
            "replicate",
        ):
            selected = [item for item in comparisons if item["pair_kind"] == kind]
            grouped[kind] = {
                "pair_count": len(selected),
                "worst_cosine": (
                    min(item["cosine"] for item in selected)
                    if selected
                    else None
                ),
                "max_difference_rms": (
                    max(item["difference"]["rms"] for item in selected)
                    if selected
                    else None
                ),
            }

        calibration_gates = {}
        calibration_names = sorted(
            {str(record["calibration"]) for record in records}
        )
        for calibration in calibration_names:
            selected_records = [
                record
                for record in records
                if str(record["calibration"]) == calibration
            ]
            selected_pairs = [
                item
                for item in comparisons
                if item["pair_kind"] == "build_only"
                and item["left"].startswith(f"{calibration}__")
                and item["right"].startswith(f"{calibration}__")
            ]
            effect_floor = min(
                record["effect_rms"] for record in selected_records
            )
            build_noise_ceiling = max(
                (
                    item["difference"]["rms"]
                    for item in selected_pairs
                ),
                default=None,
            )
            if build_noise_ceiling is None:
                ratio = None
                ratio_interpretation = "unavailable_without_build_pair"
                gate = None
            elif build_noise_ceiling == 0.0:
                ratio = None
                ratio_interpretation = "unbounded_no_observed_build_noise"
                gate = effect_floor > 0.0
            else:
                ratio = effect_floor / build_noise_ceiling
                ratio_interpretation = "finite"
                gate = ratio >= 3.0
            calibration_gates[calibration] = {
                "build_count": len(selected_records),
                "effect_rms_floor": effect_floor,
                "build_noise_rms_ceiling": build_noise_ceiling,
                "effect_to_build_noise_ratio": ratio,
                "ratio_interpretation": ratio_interpretation,
                "gate_effect_gt_3x_build_noise": gate,
            }
        passed_calibrations = sum(
            gate["gate_effect_gt_3x_build_noise"] is True
            for gate in calibration_gates.values()
        )
        stability[name] = {
            "environment_count": len(records),
            "pairwise": comparisons,
            "pair_groups": grouped,
            "worst_cosine": (
                min(item["cosine"] for item in comparisons)
                if comparisons
                else None
            ),
            "calibration_gates": calibration_gates,
            "gate_effect_gt_3x_build_noise_in_at_least_2_calibrations": (
                passed_calibrations >= 2
            ),
        }
    payload = {
        "schema_version": 1,
        "paired": True,
        "environments": effect_records,
        "stability": stability,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(json.dumps({"output": str(args.output), **payload}, indent=2))


if __name__ == "__main__":
    main()
