"""Analyze paired logit transitions across v15 deployment states."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


PRIMARY_CONTRASTS = {
    "export_S0_to_S1": ("S0", "S1"),
    "fp_compile_S1_to_S2": ("S1", "S2"),
    "fp16_policy_S2_to_S3": ("S2", "S3"),
    "quantization_S1_to_S4": ("S1", "S4"),
    "explicit_target_compile_S4_to_S5": ("S4", "S5"),
    "explicit_backend_S5_to_S6": ("S5", "S6"),
    "implicit_qc_gpu_S1_to_S7": ("S1", "S7"),
    "implicit_backend_S7_to_S8": ("S7", "S8"),
}


def load_capture(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path)
    return {key: data[key] for key in data.files}


def bootstrap_mean_ci(
    values: np.ndarray,
    seed: int,
    samples: int,
) -> list[float]:
    generator = np.random.default_rng(seed)
    means = []
    for _ in range(samples):
        indices = generator.integers(0, len(values), size=len(values))
        means.append(float(np.mean(values[indices])))
    return [
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    ]


def contrast(
    left: dict[str, np.ndarray],
    right: dict[str, np.ndarray],
    seed: int,
    bootstrap_samples: int,
) -> dict[str, Any]:
    if not np.array_equal(left["paths"], right["paths"]):
        raise ValueError("capture path order mismatch")
    left_logits = left["logits"].astype(np.float64)
    right_logits = right["logits"].astype(np.float64)
    difference = right_logits - left_logits
    per_image_mae = np.mean(np.abs(difference), axis=1)
    per_image_rms = np.sqrt(np.mean(difference**2, axis=1))
    norms = np.linalg.norm(left_logits, axis=1) * np.linalg.norm(
        right_logits, axis=1
    )
    cosine = np.sum(left_logits * right_logits, axis=1) / np.maximum(
        norms, 1e-12
    )
    left_predictions = left_logits.argmax(1)
    right_predictions = right_logits.argmax(1)
    labels = left["labels"]
    return {
        "n_images": len(labels),
        "mean_abs": float(np.mean(per_image_mae)),
        "mean_abs_bootstrap_95ci": bootstrap_mean_ci(
            per_image_mae, seed, bootstrap_samples
        ),
        "rms": float(np.mean(per_image_rms)),
        "rms_bootstrap_95ci": bootstrap_mean_ci(
            per_image_rms, seed + 1, bootstrap_samples
        ),
        "max_abs": float(np.max(np.abs(difference))),
        "mean_cosine": float(np.mean(cosine)),
        "prediction_flip_rate": float(
            np.mean(left_predictions != right_predictions)
        ),
        "left_accuracy": float(np.mean(left_predictions == labels)),
        "right_accuracy": float(np.mean(right_predictions == labels)),
        "accuracy_delta": float(
            np.mean(right_predictions == labels)
            - np.mean(left_predictions == labels)
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--capture-index",
        type=Path,
        default=Path(
            "chain_survival/results/v15/captures/run_index.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "chain_survival/results/v15/ablations/state_transitions.json"
        ),
    )
    parser.add_argument("--seed", type=int, default=1501)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    index = json.loads(args.capture_index.read_text())
    records = {
        record["state_id"]: record
        for record in index["records"]
        if record["status"] == "OK"
        and record.get("strict_artifact_gate", True)
    }
    captures = {
        state: load_capture(Path(record["capture_path"]))
        for state, record in records.items()
    }
    analyses = {}
    unavailable = {}
    for name, (left, right) in PRIMARY_CONTRASTS.items():
        missing = [state for state in (left, right) if state not in captures]
        if missing:
            unavailable[name] = missing
            continue
        analyses[name] = {
            "left": left,
            "right": right,
            **contrast(
                captures[left],
                captures[right],
                args.seed,
                args.bootstrap_samples,
            ),
        }
    payload = {
        "schema_version": 1,
        "capture_index": str(args.capture_index),
        "paired": True,
        "bootstrap_samples": args.bootstrap_samples,
        "contrasts": analyses,
        "unavailable": unavailable,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(json.dumps({"output": str(args.output), **payload}, indent=2))


if __name__ == "__main__":
    main()
