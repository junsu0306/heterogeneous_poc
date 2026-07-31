"""Benchmark per-state and dual-execution audit latency for v15 P9."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import onnxruntime as ort
import torch

sys.path.insert(0, str(Path("chain_survival/scripts").resolve()))
import run_dclbd_survival as survival  # noqa: E402
from reproduce_dclbd_baseline import (  # noqa: E402
    CifarParquetDataset,
    preprocess_batch,
)


def benchmark(
    function: Callable[[], Any],
    warmup: int,
    repeats: int,
) -> dict[str, float]:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    values = []
    for _ in range(repeats):
        started = time.perf_counter()
        function()
        torch.cuda.synchronize()
        values.append((time.perf_counter() - started) * 1000.0)
    array = np.asarray(values, dtype=np.float64)
    return {
        "repeats": repeats,
        "mean_ms_per_batch": float(array.mean()),
        "median_ms_per_batch": float(np.median(array)),
        "p95_ms_per_batch": float(np.quantile(array, 0.95)),
        "min_ms_per_batch": float(array.min()),
        "max_ms_per_batch": float(array.max()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--state-indexes",
        nargs="+",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("common/datasets/cifar10_hf"),
    )
    parser.add_argument(
        "--attacked-checkpoint",
        type=Path,
        default=Path(
            "chain_survival/results/v15/dclbd_baseline/"
            "attacked_model.pth"
        ),
    )
    parser.add_argument(
        "--trigger-checkpoint",
        type=Path,
        default=Path(
            "chain_survival/results/v15/dclbd_baseline/trigger.pth"
        ),
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "chain_survival/results/v15/defense/"
            "audit_latency.json"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda")
    dataset = CifarParquetDataset(
        args.data_dir / "test.parquet",
        args.data_dir / "test_tensor.pt",
        train=False,
    )
    inputs = preprocess_batch(
        dataset.images[:100], device, train=False
    )
    input_array = inputs.detach().cpu().numpy()
    model, _, _ = survival.load_attack(
        args.attacked_checkpoint,
        args.trigger_checkpoint,
        device,
    )
    source_result = benchmark(
        lambda: model(inputs),
        args.warmup,
        args.repeats,
    )
    source_result["ms_per_image"] = (
        source_result["mean_ms_per_batch"] / len(inputs)
    )

    environments = []
    reference_cache: dict[str, dict[str, float]] = {}
    for index_path in args.state_indexes:
        payload = json.loads(index_path.read_text())
        calibration = payload["settings"]["calibration"]
        records = [
            record
            for record in payload["records"]
            if record.get("status") in {"OK", "REFERENCE"}
        ]
        for build_id in sorted(
            {int(record["build_id"]) for record in records}
        ):
            selected = {
                record["state_id"]: record
                for record in records
                if int(record["build_id"]) == build_id
            }
            state_results = {"S0": source_result}
            for state in ("S1", "S4"):
                record = selected.get(state)
                if record is None:
                    continue
                artifact = record["artifact_path"]
                cache_key = f"{state}:{artifact}"
                if cache_key not in reference_cache:
                    session = ort.InferenceSession(
                        artifact,
                        providers=["CPUExecutionProvider"],
                    )
                    result = benchmark(
                        lambda current=session: current.run(
                            ["logits", "embedding"],
                            {"input": input_array},
                        ),
                        args.warmup,
                        args.repeats,
                    )
                    result["ms_per_image"] = (
                        result["mean_ms_per_batch"] / len(inputs)
                    )
                    reference_cache[cache_key] = result
                state_results[state] = reference_cache[cache_key]
            for state in ("S2", "S3", "S5", "S7", "S8"):
                record = selected.get(state)
                if record is None or not record.get("engine_path"):
                    continue
                runner = survival.MultiOutputEngineRunner(
                    survival.trt_helpers.load_engine(
                        record["engine_path"]
                    )
                )
                result = benchmark(
                    lambda current=runner: current.run(input_array),
                    args.warmup,
                    args.repeats,
                )
                result["ms_per_image"] = (
                    result["mean_ms_per_batch"] / len(inputs)
                )
                state_results[state] = result
            audit_costs = {}
            for name, states in {
                "source_export": ("S0", "S1"),
                "export_qdq": ("S1", "S4"),
                "gpu_hybrid_dla": ("S7", "S8"),
                "minimal_strict_3state": ("S0", "S1", "S4"),
            }.items():
                if not all(state in state_results for state in states):
                    continue
                mean = sum(
                    state_results[state]["mean_ms_per_batch"]
                    for state in states
                )
                audit_costs[name] = {
                    "states": list(states),
                    "sequential_mean_ms_per_batch": mean,
                    "sequential_ms_per_image": mean / len(inputs),
                    "overhead_vs_source_only": (
                        mean / source_result["mean_ms_per_batch"]
                    ),
                }
            environments.append(
                {
                    "calibration": calibration,
                    "build_id": build_id,
                    "state_index": str(index_path),
                    "states": state_results,
                    "audit_costs": audit_costs,
                }
            )

    cost_summary = {}
    for name in (
        "source_export",
        "export_qdq",
        "gpu_hybrid_dla",
        "minimal_strict_3state",
    ):
        values = [
            environment["audit_costs"][name][
                "sequential_ms_per_image"
            ]
            for environment in environments
            if name in environment["audit_costs"]
        ]
        if not values:
            continue
        cost_summary[name] = {
            "environment_count": len(values),
            "min_ms_per_image": float(min(values)),
            "max_ms_per_image": float(max(values)),
            "mean_ms_per_image": float(np.mean(values)),
        }
    payload = {
        "schema_version": 1,
        "batch_size": len(inputs),
        "warmup": args.warmup,
        "repeats": args.repeats,
        "measurement": (
            "sequential wall-clock latency with CUDA synchronization; "
            "preprocessing excluded; both ONNX outputs retained"
        ),
        "energy_measurement": "unavailable_without_power-rail sampling",
        "environments": environments,
        "audit_cost_summary": cost_summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(
        json.dumps(
            {
                "output": str(args.output),
                "source": source_result,
                "audit_cost_summary": cost_summary,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
