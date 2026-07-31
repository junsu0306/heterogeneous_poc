"""Measure partial Jetson rail power for the selected S0-S1 audit."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import threading
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


RAILS = ("VDD_GPU_SOC", "VDD_CPU_CV")


def parse_rails(line: str) -> dict[str, float] | None:
    values = {}
    for rail in RAILS:
        match = re.search(rf"{rail} ([0-9]+)mW", line)
        if match is None:
            return None
        values[rail] = float(match.group(1))
    values["partial_rail_sum"] = sum(values.values())
    return values


def run_for_duration(
    function: Callable[[], Any],
    duration: float,
) -> tuple[int, float]:
    count = 0
    started = time.monotonic()
    while time.monotonic() - started < duration:
        function()
        torch.cuda.synchronize()
        count += 1
    return count, time.monotonic() - started


def summarize_phase(
    samples: list[dict[str, Any]],
    start: float,
    end: float,
    batches: int | None = None,
    batch_size: int | None = None,
) -> dict[str, Any]:
    selected = [
        sample for sample in samples if start <= sample["time"] <= end
    ]
    result: dict[str, Any] = {
        "sample_count": len(selected),
        "duration_seconds": end - start,
    }
    for rail in (*RAILS, "partial_rail_sum"):
        values = [sample[rail] for sample in selected]
        result[f"{rail}_mean_mw"] = (
            float(np.mean(values)) if values else None
        )
        result[f"{rail}_p95_mw"] = (
            float(np.quantile(values, 0.95)) if values else None
        )
    if batches is not None and batch_size is not None:
        result["batches"] = batches
        result["images"] = batches * batch_size
        result["gross_partial_rail_mj_per_image"] = (
            result["partial_rail_sum_mean_mw"]
            * result["duration_seconds"]
            / result["images"]
        )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-onnx",
        type=Path,
        default=Path(
            "chain_survival/results/v15/dclbd_baseline/ort_final.onnx"
        ),
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
    parser.add_argument("--idle-seconds", type=float, default=3.0)
    parser.add_argument("--work-seconds", type=float, default=8.0)
    parser.add_argument("--interval-ms", type=int, default=100)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "chain_survival/results/v15/defense/audit_power.json"
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
    session = ort.InferenceSession(
        str(args.source_onnx),
        providers=["CPUExecutionProvider"],
    )

    def source() -> None:
        model(inputs)

    def audit() -> None:
        model(inputs)
        session.run(
            ["logits", "embedding"], {"input": input_array}
        )

    for _ in range(5):
        audit()
    torch.cuda.synchronize()

    process = subprocess.Popen(
        ["tegrastats", "--interval", str(args.interval_ms)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    samples: list[dict[str, Any]] = []

    def collect() -> None:
        if process.stdout is None:
            return
        for line in process.stdout:
            rails = parse_rails(line)
            if rails is not None:
                samples.append({"time": time.monotonic(), **rails})

    thread = threading.Thread(target=collect, daemon=True)
    thread.start()
    phases = {}
    try:
        idle_start = time.monotonic()
        time.sleep(args.idle_seconds)
        idle_end = time.monotonic()

        source_start = time.monotonic()
        source_batches, _ = run_for_duration(
            source, args.work_seconds
        )
        source_end = time.monotonic()

        idle2_start = time.monotonic()
        time.sleep(args.idle_seconds)
        idle2_end = time.monotonic()

        audit_start = time.monotonic()
        audit_batches, _ = run_for_duration(
            audit, args.work_seconds
        )
        audit_end = time.monotonic()

        time.sleep(0.5)
        phases["idle_before_source"] = summarize_phase(
            samples, idle_start, idle_end
        )
        phases["source_only"] = summarize_phase(
            samples,
            source_start,
            source_end,
            source_batches,
            len(inputs),
        )
        phases["idle_before_audit"] = summarize_phase(
            samples, idle2_start, idle2_end
        )
        phases["source_export_audit"] = summarize_phase(
            samples,
            audit_start,
            audit_end,
            audit_batches,
            len(inputs),
        )
    finally:
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        thread.join(timeout=2)

    for work_name, idle_name in (
        ("source_only", "idle_before_source"),
        ("source_export_audit", "idle_before_audit"),
    ):
        work = phases[work_name]
        idle = phases[idle_name]
        incremental = (
            work["partial_rail_sum_mean_mw"]
            - idle["partial_rail_sum_mean_mw"]
        )
        work["idle_subtracted_partial_rail_mean_mw"] = incremental
        work["idle_subtracted_partial_rail_mj_per_image"] = (
            incremental
            * work["duration_seconds"]
            / work["images"]
        )
    payload = {
        "schema_version": 1,
        "tegrastats_interval_ms": args.interval_ms,
        "rails": list(RAILS),
        "rail_scope": (
            "partial platform power only; GPU/SOC plus CPU/CV, "
            "not total board energy"
        ),
        "batch_size": len(inputs),
        "phases": phases,
        "audit_incremental_energy_over_source_mj_per_image": (
            phases["source_export_audit"][
                "idle_subtracted_partial_rail_mj_per_image"
            ]
            - phases["source_only"][
                "idle_subtracted_partial_rail_mj_per_image"
            ]
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(json.dumps({"output": str(args.output), **payload}, indent=2))


if __name__ == "__main__":
    main()
