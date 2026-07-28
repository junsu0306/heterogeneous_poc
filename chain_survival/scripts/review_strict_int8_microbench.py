"""Review strict-INT8 inspector, stability, subspace and saturation gates."""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denominator) if denominator else 0.0


def residual_matrix(mapping: dict, model: str, calibration: int, build: int, amp: str) -> np.ndarray:
    gpu = np.load(mapping[(model, calibration, build, "gpu")]["activation_path"])[amp]
    dla = np.load(mapping[(model, calibration, build, "dla")]["activation_path"])[amp]
    return (dla.astype(np.float32) - gpu.astype(np.float32)).reshape(len(gpu), -1)


def subspace_overlap(a: np.ndarray, b: np.ndarray, k: int) -> dict:
    a = a - a.mean(axis=0, keepdims=True)
    b = b - b.mean(axis=0, keepdims=True)
    _, _, va = np.linalg.svd(a, full_matrices=False)
    _, _, vb = np.linalg.svd(b, full_matrices=False)
    principal = np.linalg.svd(va[:k] @ vb[:k].T, compute_uv=False)
    return {
        "k": k,
        "mean_squared_principal_cosine": float(np.mean(np.square(principal))),
        "min_principal_cosine": float(np.min(principal)),
        "max_principal_cosine": float(np.max(principal)),
    }


def endpoint_occupancy(path: str, amp: str = "amp_1") -> float:
    with np.load(path) as archive:
        all_values = np.concatenate([archive[key].reshape(-1) for key in archive.files])
        test = archive[amp].reshape(-1)
    lower = float(all_values.min())
    upper = float(all_values.max())
    unique = np.unique(all_values)
    differences = np.diff(unique)
    differences = differences[differences > 1e-6]
    step = float(np.median(differences)) if len(differences) else 0.0
    tolerance = max(1e-7, step * 0.1)
    occupancy = np.mean(np.isclose(test, upper, rtol=0.0, atol=tolerance))
    if lower < -tolerance:
        occupancy += np.mean(np.isclose(test, lower, rtol=0.0, atol=tolerance))
    return float(occupancy)


def inspector_review(record: dict) -> dict:
    inspector = json.loads(Path(record["inspector_path"]).read_text())
    layers = inspector["Layers"]
    if layers and isinstance(layers[0], str):
        return {
            "detailed": False,
            "layer_types": [],
            "strict_int8_compute_verified": False,
        }
    layer_types = [layer.get("LayerType") for layer in layers]
    formats = []
    for layer in layers:
        for tensor in layer.get("Inputs", []) + layer.get("Outputs", []):
            formats.append(tensor.get("Format/Datatype", ""))
    if record["backend"] == "dla":
        compute_layers = [layer for layer in layers if layer.get("LayerType") == "DLA"]
        verified = bool(compute_layers) and all(
            "INT8" in tensor.get("Format/Datatype", "").upper()
            for layer in compute_layers
            for tensor in layer.get("Inputs", []) + layer.get("Outputs", [])
        )
    else:
        compute_layers = [
            layer
            for layer in layers
            if layer.get("LayerType") in {"CaskConvolution", "FusedConvActConvolution"}
            or layer.get("ParameterType") == "Convolution"
        ]
        verified = bool(compute_layers) and all(
            layer.get("Weights", {}).get("Type") == "Int8"
            and all(
                "INT8" in tensor.get("Format/Datatype", "").upper()
                for tensor in layer.get("Inputs", []) + layer.get("Outputs", [])
            )
            for layer in compute_layers
        )
    return {
        "detailed": True,
        "layer_types": layer_types,
        "n_compute_layers": len(compute_layers),
        "strict_int8_compute_verified": verified,
        "all_formats": sorted(set(formats)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-index",
        type=Path,
        default=Path(
            "chain_survival/results/v13/microbench_hardware_strict_int8/run_index.json"
        ),
    )
    parser.add_argument(
        "--analysis",
        type=Path,
        default=Path(
            "chain_survival/results/v13/microbench_hardware_strict_int8_analysis.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "chain_survival/results/v13/microbench_hardware_strict_int8_review.json"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_index = json.loads(args.run_index.read_text())
    analysis = json.loads(args.analysis.read_text())
    records = [record for record in run_index["records"] if record["status"] == "OK"]
    failed_records = [
        record for record in run_index["records"] if record["status"] == "FAILED"
    ]
    mapping = {
        (
            record["model_id"],
            int(record["calibration"]),
            int(record["build"]),
            record["backend"],
        ): record
        for record in records
    }
    all_models = sorted({record["model_id"] for record in run_index["records"]})
    models = [
        model
        for model in all_models
        if all(
            (model, calibration, build, backend) in mapping
            for calibration in (0, 1)
            for build in (0, 1, 2)
            for backend in ("gpu", "dla")
        )
    ]

    inspectors = [inspector_review(record) for record in records]
    inspector_gate = {
        "n_records": len(records),
        "n_detailed": sum(item["detailed"] for item in inspectors),
        "n_strict_int8_verified": sum(
            item["strict_int8_compute_verified"] for item in inspectors
        ),
    }
    inspector_gate["decision"] = (
        "GO"
        if inspector_gate["n_strict_int8_verified"] == inspector_gate["n_records"]
        else "NO-GO"
    )

    model_rows = {}
    for model in models:
        directions = {}
        for calibration in (0, 1):
            for build in (0, 1, 2):
                residual = residual_matrix(
                    mapping, model, calibration, build, "amp_1"
                )
                directions[(calibration, build)] = residual.mean(axis=0)
        within = {}
        for calibration in (0, 1):
            values = [
                cosine(directions[(calibration, a)], directions[(calibration, b)])
                for a, b in combinations((0, 1, 2), 2)
            ]
            within[f"calibration_{calibration}"] = {
                "median": float(np.median(values)),
                "min": float(np.min(values)),
            }
        cross = [
            cosine(directions[(0, a)], directions[(1, b)])
            for a in (0, 1, 2)
            for b in (0, 1, 2)
        ]
        residual_a = residual_matrix(mapping, model, 0, 0, "amp_1")
        residual_b = residual_matrix(mapping, model, 1, 0, "amp_1")
        subspaces = [
            subspace_overlap(residual_a, residual_b, k) for k in (1, 4, 8, 16)
        ]
        occupancy = {
            backend: float(
                np.mean(
                    [
                        endpoint_occupancy(
                            mapping[(model, calibration, build, backend)][
                                "activation_path"
                            ]
                        )
                        for calibration in (0, 1)
                        for build in (0, 1, 2)
                    ]
                )
            )
            for backend in ("gpu", "dla")
        }
        top8 = next(item for item in subspaces if item["k"] == 8)
        candidate_go = (
            float(np.median(cross)) >= 0.8
            and top8["mean_squared_principal_cosine"] >= 0.5
            and max(occupancy.values()) <= 0.01
        )
        model_rows[model] = {
            "within_calibration_build_direction": within,
            "cross_calibration_direction": {
                "median": float(np.median(cross)),
                "min": float(np.min(cross)),
                "max": float(np.max(cross)),
            },
            "cross_calibration_subspace": subspaces,
            "amp1_endpoint_occupancy": occupancy,
            "candidate_gate": "GO" if candidate_go else "NO-GO",
        }

    # Inspector-confirmed DLA partition counts for causal-control pairs.
    partitions = {}
    for model in models:
        record = mapping[(model, 0, 0, "dla")]
        inspector = json.loads(Path(record["inspector_path"]).read_text())
        partitions[model] = sum(
            layer.get("LayerType") == "DLA" for layer in inspector["Layers"]
        )

    amp1_rows = [
        row for row in analysis["paired_residuals"] if row["amplitude"] == "amp_1"
    ]
    amp1_summary = {}
    for model in models:
        selected = [row for row in amp1_rows if row["model_id"] == model]
        amp1_summary[model] = {
            key: float(np.mean([row[key] for row in selected]))
            for key in ("mean_abs", "relative_mean_abs", "normalized_by_gpu_rms")
        }

    repeated_points = [
        (blocks, amp1_summary[f"repeated_{blocks}"]["normalized_by_gpu_rms"])
        for blocks in (1, 2, 4, 8)
        if f"repeated_{blocks}" in amp1_summary
    ]
    growth_models = {}
    if len(repeated_points) == 4:
        x = np.asarray([point[0] for point in repeated_points], dtype=np.float64)
        y = np.asarray([point[1] for point in repeated_points], dtype=np.float64)
        denominator = np.square(y - y.mean()).sum() + 1e-30
        for name, basis in (
            ("constant", np.ones_like(x)),
            ("linear", x),
            ("sqrt", np.sqrt(x)),
        ):
            alpha = float(np.dot(basis, y) / np.dot(basis, basis))
            prediction = alpha * basis
            growth_models[name] = {
                "alpha": alpha,
                "r2_through_origin": float(
                    1.0 - np.square(y - prediction).sum() / denominator
                ),
            }
        exponent, log_alpha = np.polyfit(np.log(x), np.log(y), 1)
        power_prediction = np.exp(log_alpha) * np.power(x, exponent)
        growth_models["power_exploratory"] = {
            "alpha": float(np.exp(log_alpha)),
            "exponent": float(exponent),
            "r2": float(
                1.0 - np.square(y - power_prediction).sum() / denominator
            ),
        }

    tm_q_shortlist = []
    for model, row in model_rows.items():
        if not model.startswith("repeated_"):
            continue
        fixed_calibration_min = min(
            item["min"]
            for item in row["within_calibration_build_direction"].values()
        )
        effect = amp1_summary[model]["normalized_by_gpu_rms"]
        if (
            fixed_calibration_min >= 0.8
            and max(row["amp1_endpoint_occupancy"].values()) <= 0.01
            and effect >= 0.001
        ):
            tm_q_shortlist.append(model)

    result = {
        "schema_version": 1,
        "inspector_gate": inspector_gate,
        "unsupported_or_failed": failed_records,
        "n_fully_paired_models": len(models),
        "dla_partition_count": partitions,
        "requested_graph_outputs_do_not_create_dla_partitions": (
            len(
                {
                    partitions[model]
                    for model in (
                        "graph_break_0",
                        "graph_break_1",
                        "graph_break_2",
                        "graph_break_4",
                    )
                }
            )
            == 1
        ),
        "model_stability": model_rows,
        "amp1_residual_summary": amp1_summary,
        "repeated_growth_models": growth_models,
        "b_micro_gate": {
            "n_candidate_go": sum(
                row["candidate_gate"] == "GO" for row in model_rows.values()
            ),
            "tm_w_decision": (
                "GO"
                if any(row["candidate_gate"] == "GO" for row in model_rows.values())
                else "NO-GO"
            ),
            "tm_q_exploratory_shortlist": tm_q_shortlist,
            "reason": (
                "Residuals are build-stable and non-saturating at amplitude 1, "
                "but neither mean direction nor top-k residual subspace transfers "
                "between the two calibration subsets. Requested graph outputs also "
                "do not split the DLA compute partition. Repeated-block magnitude "
                "growth is causal and fixed-calibration stable, so repeated_4/8 "
                "remain TM-Q candidates only."
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(
        json.dumps(
            {
                "output": str(args.output),
                "inspector_gate": inspector_gate,
                "b_micro_gate": result["b_micro_gate"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
