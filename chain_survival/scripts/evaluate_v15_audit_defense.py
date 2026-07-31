"""Evaluate v15 multi-state audit and randomized-build defenses."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np


AUDITS = {
    "source_export": [("S0", "S1")],
    "export_fp16": [("S1", "S3")],
    "export_qdq": [("S1", "S4")],
    "gpu_hybrid_dla": [("S7", "S8")],
    "minimal_strict_3state": [("S0", "S1"), ("S1", "S4")],
    "strict_pairwise_max": list(
        itertools.combinations(["S0", "S1", "S2", "S3", "S4"], 2)
    ),
    "available_pairwise_max": list(
        itertools.combinations(
            ["S0", "S1", "S2", "S3", "S4", "S5", "S7", "S8"],
            2,
        )
    ),
}


def load_environment(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    captures = {}
    strict = {}
    metrics = {}
    for record in payload["records"]:
        if record.get("status") != "OK":
            continue
        state = record["state_id"]
        captures[state] = np.load(record["capture_path"])
        strict[state] = bool(record.get("strict_artifact_gate", True))
        metrics[state] = record["metrics"]
    return {
        "calibration": payload["settings"]["calibration"],
        "build_id": int(payload["settings"]["build_id"]),
        "capture_index": str(path),
        "captures": captures,
        "strict": strict,
        "metrics": metrics,
    }


def centered_logit_distance(
    left: np.ndarray, right: np.ndarray
) -> np.ndarray:
    left = left.astype(np.float64)
    right = right.astype(np.float64)
    left = left - left.mean(axis=1, keepdims=True)
    right = right - right.mean(axis=1, keepdims=True)
    return np.sqrt(np.mean((right - left) ** 2, axis=1))


def pair_scores(
    environment: dict[str, Any],
    pair: tuple[str, str],
    group: str,
) -> np.ndarray:
    left, right = pair
    key = f"{group}_logits"
    return centered_logit_distance(
        environment["captures"][left][key],
        environment["captures"][right][key],
    )


def auc(negative: np.ndarray, positive: np.ndarray) -> float:
    values = np.concatenate([negative, positive])
    labels = np.concatenate(
        [
            np.zeros(len(negative), dtype=np.int8),
            np.ones(len(positive), dtype=np.int8),
        ]
    )
    order = np.argsort(values, kind="mergesort")
    values = values[order]
    labels = labels[order]
    concordant = 0.0
    negatives_before = 0
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[end] == values[start]:
            end += 1
        block = labels[start:end]
        block_positive = int(block.sum())
        block_negative = len(block) - block_positive
        concordant += (
            block_positive * negatives_before
            + 0.5 * block_positive * block_negative
        )
        negatives_before += block_negative
        start = end
    denominator = len(negative) * len(positive)
    return float(concordant / denominator)


def calibrated_audit_scores(
    environments: list[dict[str, Any]],
    pairs: list[tuple[str, str]],
    reference_calibrations: set[str],
    build_id: int,
) -> tuple[dict[tuple[str, str], float], dict[str, np.ndarray]]:
    selected = [
        environment
        for environment in environments
        if environment["build_id"] == build_id
    ]
    scales = {}
    for pair in pairs:
        clean = np.concatenate(
            [
                pair_scores(environment, pair, "clean")
                for environment in selected
                if environment["calibration"]
                in reference_calibrations
            ]
        )
        scales[pair] = max(
            float(np.quantile(clean, 0.99, method="higher")),
            1e-12,
        )
    scores = {}
    for environment in selected:
        identity = environment["calibration"]
        for group in ("clean", "trigger"):
            normalized = [
                pair_scores(environment, pair, group) / scales[pair]
                for pair in pairs
            ]
            scores[f"{identity}:{group}"] = np.max(
                np.stack(normalized), axis=0
            )
    return scales, scores


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(array.min()),
        "max": float(array.max()),
        "mean": float(array.mean()),
    }


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
            "chain_survival/results/v15/defense/"
            "multi_state_audit.json"
        ),
    )
    parser.add_argument("--target-fpr", type=float, default=0.01)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    environments = [
        load_environment(path) for path in args.capture_indexes
    ]
    reference_calibrations = {
        "calib_shadow_1",
        "calib_shadow_2",
    }
    blind_calibration = "calib_blind_1"
    audit_results = {}
    for name, pairs in AUDITS.items():
        per_build = []
        for build_id in sorted(
            {environment["build_id"] for environment in environments}
        ):
            scales, scores = calibrated_audit_scores(
                environments,
                pairs,
                reference_calibrations,
                build_id,
            )
            shadow_clean = np.concatenate(
                [
                    scores[f"{calibration}:clean"]
                    for calibration in reference_calibrations
                ]
            )
            threshold = float(
                np.quantile(
                    shadow_clean,
                    1.0 - args.target_fpr,
                    method="higher",
                )
            )
            blind_clean = scores[f"{blind_calibration}:clean"]
            blind_trigger = scores[f"{blind_calibration}:trigger"]
            per_build.append(
                {
                    "build_id": build_id,
                    "state_count": len(
                        {state for pair in pairs for state in pair}
                    ),
                    "pairs": [list(pair) for pair in pairs],
                    "pair_clean_p99_scales": {
                        f"{left}:{right}": value
                        for (left, right), value in scales.items()
                    },
                    "threshold_from_shadow_clean": threshold,
                    "shadow_empirical_fpr": float(
                        np.mean(shadow_clean > threshold)
                    ),
                    "blind_fpr": float(
                        np.mean(blind_clean > threshold)
                    ),
                    "blind_tpr": float(
                        np.mean(blind_trigger > threshold)
                    ),
                    "blind_auc": auc(blind_clean, blind_trigger),
                }
            )
        audit_results[name] = {
            "per_build": per_build,
            "blind_fpr": summarize(
                [item["blind_fpr"] for item in per_build]
            ),
            "blind_tpr": summarize(
                [item["blind_tpr"] for item in per_build]
            ),
            "blind_auc": summarize(
                [item["blind_auc"] for item in per_build]
            ),
            "inference_state_count": per_build[0]["state_count"],
        }

    randomized_build = {}
    for state in ("S3", "S7", "S8"):
        per_calibration = {}
        for calibration in sorted(
            {environment["calibration"] for environment in environments}
        ):
            asr = [
                float(environment["metrics"][state]["trigger_asr"])
                for environment in environments
                if environment["calibration"] == calibration
            ]
            accuracy = [
                float(
                    environment["metrics"][state]["clean_accuracy"]
                )
                for environment in environments
                if environment["calibration"] == calibration
            ]
            per_calibration[calibration] = {
                "uniform_build_expected_asr": float(np.mean(asr)),
                "worst_build_asr": float(np.max(asr)),
                "best_build_asr": float(np.min(asr)),
                "uniform_build_expected_clean_accuracy": float(
                    np.mean(accuracy)
                ),
                "gate_expected_asr_le_0_10": float(np.mean(asr))
                <= 0.10,
            }
        randomized_build[state] = per_calibration

    eligible = {
        name: value
        for name, value in audit_results.items()
        if value["blind_fpr"]["max"] <= args.target_fpr
    }
    selected_name = (
        max(
            eligible,
            key=lambda name: (
                eligible[name]["blind_tpr"]["min"],
                -eligible[name]["inference_state_count"],
            ),
        )
        if eligible
        else None
    )
    payload = {
        "schema_version": 1,
        "paired": True,
        "environment_count": len(environments),
        "target_fpr": args.target_fpr,
        "threshold_training_calibrations": sorted(
            reference_calibrations
        ),
        "blind_calibration": blind_calibration,
        "score": (
            "maximum clean-p99-normalized centered-logit RMS "
            "across configured state pairs"
        ),
        "audits": audit_results,
        "randomized_build": randomized_build,
        "selection": {
            "selected_audit": selected_name,
            "criterion": (
                "maximize worst-build blind TPR subject to "
                "worst-build blind FPR <= target; break ties by state count"
            ),
            "selected_result": (
                audit_results[selected_name]
                if selected_name is not None
                else None
            ),
        },
        "gates": {
            "blind_audit_available": selected_name is not None,
            "blind_fpr_le_target_all_builds": selected_name is not None,
            "randomized_build_alone_suppresses_s7_asr_le_0_10": all(
                value["gate_expected_asr_le_0_10"]
                for value in randomized_build["S7"].values()
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(
        json.dumps(
            {
                "output": str(args.output),
                "selected_audit": selected_name,
                "selected_result": payload["selection"][
                    "selected_result"
                ],
                "randomized_s7": randomized_build["S7"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
