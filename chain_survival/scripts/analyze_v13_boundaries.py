"""Analyze multi-build/calibration strict-INT8 boundary summaries."""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denominator) if denominator else 0.0


def load_feature(path: str) -> np.ndarray:
    with np.load(path) as archive:
        value = archive["pooled4"].astype(np.float32)
    return value.reshape(value.shape[0], -1)


def endpoint_occupancy(path: str) -> float:
    with np.load(path) as archive:
        values = archive["hist_values"]
        counts = archive["hist_counts"]
    total = counts.sum()
    occupancy = counts[-1]
    if values[0] < 0:
        occupancy += counts[0]
    return float(occupancy / total)


def subspace_overlap(a: np.ndarray, b: np.ndarray, k: int = 8) -> float:
    a = a - a.mean(axis=0, keepdims=True)
    b = b - b.mean(axis=0, keepdims=True)
    _, _, va = np.linalg.svd(a, full_matrices=False)
    _, _, vb = np.linalg.svd(b, full_matrices=False)
    principal = np.linalg.svd(va[:k] @ vb[:k].T, compute_uv=False)
    return float(np.mean(np.square(principal)))


def top_subspace(value: np.ndarray, k: int = 8) -> np.ndarray:
    centered = value - value.mean(axis=0, keepdims=True)
    _, _, vectors = np.linalg.svd(centered, full_matrices=False)
    return vectors[:k]


def inspector_review(record: dict) -> dict:
    inspector = json.loads(Path(record["inspector_path"]).read_text())
    layers = inspector["Layers"]
    if not layers or isinstance(layers[0], str):
        return {
            "detailed": False,
            "strict_int8_compute_verified": False,
            "n_compute_layers": 0,
        }
    def is_output_reformat(layer: dict) -> bool:
        if layer.get("LayerType") == "Reformat":
            return True
        inputs = layer.get("Inputs", [])
        outputs = layer.get("Outputs", [])
        return (
            layer.get("LayerType") == "kgen"
            and bool(inputs)
            and bool(outputs)
            and all(
                "INT8" in tensor.get("Format/Datatype", "").upper()
                for tensor in inputs
            )
            and all(
                "FLOAT" in tensor.get("Format/Datatype", "").upper()
                for tensor in outputs
            )
        )

    compute_layers = [layer for layer in layers if not is_output_reformat(layer)]
    formats_verified = bool(compute_layers) and all(
        "INT8" in tensor.get("Format/Datatype", "").upper()
        for layer in compute_layers
        for tensor in layer.get("Inputs", []) + layer.get("Outputs", [])
    )
    if record["backend"] == "dla":
        dla_layers = [
            layer for layer in compute_layers if layer.get("LayerType") == "DLA"
        ]
        verified = bool(dla_layers) and formats_verified
    else:
        convolution_layers = [
            layer
            for layer in compute_layers
            if (
                "Conv" in layer.get("LayerType", "")
                or layer.get("ParameterType") == "Convolution"
                or layer.get("ConvParameterType") == "Convolution"
            )
        ]
        weights_verified = bool(convolution_layers) and all(
            (
                layer.get("Weights", layer.get("ConvWeights", {})).get("Type")
                == "Int8"
            )
            for layer in convolution_layers
        )
        verified = formats_verified and weights_verified
    return {
        "detailed": True,
        "strict_int8_compute_verified": verified,
        "n_compute_layers": len(compute_layers),
        "n_dla_partitions": sum(
            layer.get("LayerType") == "DLA" for layer in compute_layers
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-index",
        type=Path,
        default=Path("chain_survival/results/v13/boundary_strict_int8/run_index.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("chain_survival/results/v13/boundary_strict_int8_analysis.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run = json.loads(args.run_index.read_text())
    records = [record for record in run["records"] if record["status"] == "OK"]
    inspector_rows = [inspector_review(record) for record in records]
    inspector_gate = {
        "n_records": len(records),
        "n_detailed": sum(row["detailed"] for row in inspector_rows),
        "n_strict_int8_compute_verified": sum(
            row["strict_int8_compute_verified"] for row in inspector_rows
        ),
        "gpu_compute_layer_range": [
            min(
                row["n_compute_layers"]
                for row, record in zip(inspector_rows, records)
                if record["backend"] == "gpu"
            ),
            max(
                row["n_compute_layers"]
                for row, record in zip(inspector_rows, records)
                if record["backend"] == "gpu"
            ),
        ],
        "dla_partition_range": [
            min(
                row["n_dla_partitions"]
                for row, record in zip(inspector_rows, records)
                if record["backend"] == "dla"
            ),
            max(
                row["n_dla_partitions"]
                for row, record in zip(inspector_rows, records)
                if record["backend"] == "dla"
            ),
        ],
    }
    inspector_gate["decision"] = (
        "GO"
        if (
            inspector_gate["n_records"]
            == inspector_gate["n_detailed"]
            == inspector_gate["n_strict_int8_compute_verified"]
        )
        else "NO-GO"
    )
    mapping = {
        (
            record["boundary"],
            int(record["calibration"]),
            int(record["build"]),
            record["backend"],
        ): record
        for record in records
    }
    boundaries = sorted({record["boundary"] for record in records})
    rows = {}
    for boundary in boundaries:
        if not all(
            (boundary, calibration, build, backend) in mapping
            for calibration in range(run["settings"]["calibrations"])
            for build in range(run["settings"]["builds"])
            for backend in ("gpu", "dla")
        ):
            rows[boundary] = {"decision": "INCOMPLETE"}
            continue
        residuals = {}
        normalized = []
        occupancy = []
        for calibration in range(run["settings"]["calibrations"]):
            for build in range(run["settings"]["builds"]):
                gpu_record = mapping[(boundary, calibration, build, "gpu")]
                dla_record = mapping[(boundary, calibration, build, "dla")]
                gpu = load_feature(gpu_record["activation_path"])
                dla = load_feature(dla_record["activation_path"])
                residual = dla - gpu
                residuals[(calibration, build)] = residual
                normalized.append(
                    float(
                        np.mean(np.abs(residual))
                        / (np.sqrt(np.mean(np.square(gpu))) + 1e-12)
                    )
                )
                occupancy.extend(
                    [
                        endpoint_occupancy(gpu_record["activation_path"]),
                        endpoint_occupancy(dla_record["activation_path"]),
                    ]
                )
        directions = {key: value.mean(axis=0) for key, value in residuals.items()}
        within = []
        for calibration in range(run["settings"]["calibrations"]):
            within.extend(
                cosine(directions[(calibration, a)], directions[(calibration, b)])
                for a, b in combinations(range(run["settings"]["builds"]), 2)
            )
        cross = [
            cosine(directions[(0, a)], directions[(1, b)])
            for a in range(run["settings"]["builds"])
            for b in range(run["settings"]["builds"])
        ]
        subspaces = {
            key: top_subspace(value, 8) for key, value in residuals.items()
        }
        within_subspace = {}
        for calibration in range(run["settings"]["calibrations"]):
            values = []
            for a, b in combinations(range(run["settings"]["builds"]), 2):
                singular = np.linalg.svd(
                    subspaces[(calibration, a)]
                    @ subspaces[(calibration, b)].T,
                    compute_uv=False,
                )
                values.append(float(np.mean(np.square(singular))))
            within_subspace[f"calibration_{calibration}"] = values
        cross_subspace = []
        for a in range(run["settings"]["builds"]):
            for b in range(run["settings"]["builds"]):
                singular = np.linalg.svd(
                    subspaces[(0, a)] @ subspaces[(1, b)].T,
                    compute_uv=False,
                )
                cross_subspace.append(float(np.mean(np.square(singular))))

        stacked = np.concatenate(list(subspaces.values()), axis=0)
        _, consensus_singular, consensus_vectors = np.linalg.svd(
            stacked, full_matrices=False
        )
        consensus = consensus_vectors[:8]
        consensus_overlap = {}
        consensus_energy = {}
        for key, subspace in subspaces.items():
            principal = np.linalg.svd(
                subspace @ consensus.T, compute_uv=False
            )
            label = f"cal{key[0]}_build{key[1]}"
            consensus_overlap[label] = float(np.mean(np.square(principal)))
            centered = residuals[key] - residuals[key].mean(
                axis=0, keepdims=True
            )
            consensus_energy[label] = float(
                np.sum(np.square(centered @ consensus.T))
                / (np.sum(np.square(centered)) + 1e-30)
            )
        consensus_path = (
            args.output.parent / f"{boundary}_consensus_subspace.npz"
        )
        np.savez(
            consensus_path,
            directions=consensus.astype(np.float32),
            singular_values=consensus_singular.astype(np.float32),
            feature="pooled4_flattened",
            pooled_shape=np.asarray([4, 4], dtype=np.int64),
        )

        tm_w_go = (
            min(consensus_overlap.values()) >= 0.5
            and min(consensus_energy.values()) >= 0.5
            and max(occupancy) <= 0.01
        )
        tm_q_go = min(within) >= 0.8 and max(occupancy) <= 0.01
        rows[boundary] = {
            "normalized_residual_mean": float(np.mean(normalized)),
            "fixed_calibration_build_cosine_min": float(np.min(within)),
            "cross_calibration_direction_median": float(np.median(cross)),
            "within_calibration_top8_overlap": within_subspace,
            "cross_calibration_top8_overlap": {
                "median": float(np.median(cross_subspace)),
                "min": float(np.min(cross_subspace)),
                "max": float(np.max(cross_subspace)),
            },
            "consensus_top8_overlap": {
                "values": consensus_overlap,
                "median": float(np.median(list(consensus_overlap.values()))),
                "min": float(min(consensus_overlap.values())),
            },
            "consensus_residual_energy": {
                "values": consensus_energy,
                "median": float(np.median(list(consensus_energy.values()))),
                "min": float(min(consensus_energy.values())),
            },
            "consensus_subspace_path": str(consensus_path),
            "empirical_endpoint_occupancy_max": float(max(occupancy)),
            "tm_w_mean_direction_gate": (
                "GO"
                if np.median(cross) >= 0.8 and max(occupancy) <= 0.01
                else "NO-GO"
            ),
            "tm_w_consensus_subspace_gate": "GO" if tm_w_go else "NO-GO",
            "tm_q_mean_direction_gate": "GO" if tm_q_go else "NO-GO",
        }
    result = {
        "schema_version": 1,
        "run_index": str(args.run_index),
        "inspector_gate": inspector_gate,
        "boundaries": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({"output": str(args.output), "boundaries": rows}, indent=2))


if __name__ == "__main__":
    main()
