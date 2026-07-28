"""Validate mathematical controls in the generated Track B microbench suite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort


def run_output(spec: dict, input_array: np.ndarray) -> np.ndarray:
    session = ort.InferenceSession(
        spec["artifact"]["path"], providers=["CPUExecutionProvider"]
    )
    outputs = session.run(None, {"input": input_array})
    names = [output.name for output in session.get_outputs()]
    return np.asarray(outputs[names.index(spec["comparison_output"])], dtype=np.float64)


def diff(a: np.ndarray, b: np.ndarray) -> dict:
    delta = a - b
    return {
        "max_abs": float(np.max(np.abs(delta))),
        "mean_abs": float(np.mean(np.abs(delta))),
        "relative_mean_abs": float(
            np.mean(np.abs(delta)) / (np.mean(np.abs(a)) + 1e-12)
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("chain_survival/results/v13/microbench_manifest.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("chain_survival/results/v13/microbench_cpu_controls.json"),
    )
    parser.add_argument("--seed", type=int, default=2301)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.manifest.open() as handle:
        manifest = json.load(handle)
    specs = {spec["model_id"]: spec for spec in manifest["models"]}
    rng = np.random.default_rng(args.seed)
    shared_input = rng.standard_normal((1, 16, 16, 16), dtype=np.float32)

    fusion_a = run_output(specs["fusion_fused_candidate"], shared_input)
    fusion_b = run_output(specs["fusion_materialized_candidate"], shared_input)
    fusion = diff(fusion_a, fusion_b)

    graph_outputs = {
        model_id: run_output(spec, shared_input)
        for model_id, spec in specs.items()
        if spec["family"] == "graph_break"
    }
    graph_reference = graph_outputs["graph_break_0"]
    graph_break = {
        model_id: diff(graph_reference, output)
        for model_id, output in graph_outputs.items()
        if model_id != "graph_break_0"
    }

    granularity_outputs = {
        model_id: run_output(spec, shared_input)
        for model_id, spec in specs.items()
        if spec["family"] == "granularity_proxy"
    }
    granularity = {
        "per_channel_vs_fp32": diff(
            granularity_outputs["granularity_per_channel"],
            granularity_outputs["granularity_fp32"],
        ),
        "per_tensor_vs_fp32": diff(
            granularity_outputs["granularity_per_tensor"],
            granularity_outputs["granularity_fp32"],
        ),
        "per_tensor_vs_per_channel": diff(
            granularity_outputs["granularity_per_tensor"],
            granularity_outputs["granularity_per_channel"],
        ),
    }

    exact_control_max = max(
        [fusion["max_abs"]]
        + [summary["max_abs"] for summary in graph_break.values()]
    )
    result = {
        "schema_version": 1,
        "manifest": str(args.manifest),
        "seed": args.seed,
        "fusion_exact_control": fusion,
        "graph_break_exact_controls": graph_break,
        "granularity_grid_effect": granularity,
        "gate": {
            "required_exact_control_max_abs": 1e-6,
            "observed_exact_control_max_abs": exact_control_max,
            "decision": "GO" if exact_control_max <= 1e-6 else "NO-GO",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps({"output": str(args.output), **result["gate"], "granularity": granularity}, indent=2))


if __name__ == "__main__":
    main()
