"""B5 low-cost real-hardware perturbation controllability probe."""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional

sys.path.insert(0, str(Path("common/scripts").resolve()))
import trt_runtime as runtime  # noqa: E402

sys.path.insert(0, str(Path("chain_survival/scripts").resolve()))
import models_cfg as model_config  # noqa: E402
from run_paths import load_split  # noqa: E402


BENIGN_NAMES = {
    "benign_noise_0.02",
    "benign_brightness_0.05",
}


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denominator) if denominator else 0.0


def patch_pattern(kind: str, size: int, seed: int = 4301) -> np.ndarray:
    if kind == "gray_hi":
        return np.full((3, size, size), 2.0, dtype=np.float32)
    if kind == "gray_lo":
        return np.full((3, size, size), -2.0, dtype=np.float32)
    if kind in {"red", "green", "blue"}:
        value = np.full((3, size, size), -2.0, dtype=np.float32)
        value[{"red": 0, "green": 1, "blue": 2}[kind]] = 2.0
        return value
    if kind == "checker":
        grid = np.indices((size, size)).sum(axis=0) % 2
        value = np.where(grid == 0, -2.0, 2.0).astype(np.float32)
        return np.repeat(value[None, :, :], 3, axis=0)
    if kind == "stripes":
        grid = np.arange(size)[None, :] % 4 < 2
        value = np.where(grid, -2.0, 2.0).astype(np.float32)
        value = np.repeat(value, size, axis=0)
        return np.repeat(value[None, :, :], 3, axis=0)
    if kind == "random":
        rng = np.random.default_rng(seed)
        return rng.uniform(-2.0, 2.0, size=(3, size, size)).astype(np.float32)
    raise ValueError(kind)


def perturbations(inputs: np.ndarray) -> dict[str, np.ndarray]:
    output = {}
    rng = np.random.default_rng(4302)
    output["benign_noise_0.02"] = inputs + rng.normal(
        0.0, 0.02, size=inputs.shape
    ).astype(np.float32)
    output["benign_brightness_0.05"] = inputs + np.float32(0.05)
    specs = [
        ("gray_hi", 16, "bottom_right"),
        ("gray_lo", 16, "bottom_right"),
        ("red", 16, "bottom_right"),
        ("green", 16, "bottom_right"),
        ("blue", 16, "bottom_right"),
        ("checker", 16, "bottom_right"),
        ("stripes", 16, "bottom_right"),
        ("random", 16, "bottom_right"),
        ("checker", 32, "bottom_right"),
        ("random", 32, "top_left"),
    ]
    height, width = inputs.shape[-2:]
    for kind, size, location in specs:
        value = inputs.copy()
        if location == "bottom_right":
            row, column = height - size, width - size
        else:
            row, column = 0, 0
        value[:, :, row : row + size, column : column + size] = patch_pattern(
            kind, size
        )
        output[f"{kind}_{size}_{location}"] = value
    return output


def run_pooled(engine, inputs: np.ndarray, endpoints: tuple[float, float]) -> tuple[np.ndarray, float]:
    runner = runtime.EngineRunner(engine)
    pooled = []
    endpoint_count = 0
    total = 0
    lower, upper = endpoints
    for index in range(len(inputs)):
        activation = runner.run(inputs[index : index + 1])[0].astype(np.float32)
        tensor = torch.from_numpy(activation).unsqueeze(0)
        pooled.append(
            functional.adaptive_avg_pool2d(tensor, (4, 4))[0].numpy()
        )
        endpoint_count += int(np.count_nonzero(activation == upper))
        if lower < 0:
            endpoint_count += int(np.count_nonzero(activation == lower))
        total += activation.size
    return np.stack(pooled).reshape(len(inputs), -1), endpoint_count / total


def clean_data(path: str, n_images: int) -> tuple[np.ndarray, tuple[float, float], float]:
    with np.load(path) as archive:
        pooled = archive["pooled4"][:n_images].astype(np.float32).reshape(n_images, -1)
        values = archive["hist_values"]
        counts = archive["hist_counts"]
    occupancy = int(counts[-1])
    if values[0] < 0:
        occupancy += int(counts[0])
    return pooled, (float(values[0]), float(values[-1])), occupancy / int(counts.sum())


def summarize_candidates(
    condition_results: dict[str, dict],
    candidate_names: list[str],
) -> dict[str, dict]:
    summary = {}
    for name in candidate_names:
        means = [
            np.asarray(result[name]["mean_projected_vector"], dtype=np.float64)
            for result in condition_results.values()
        ]
        global_direction = np.mean(means, axis=0)
        global_direction /= np.linalg.norm(global_direction) + 1e-30
        condition_cosines = [
            cosine(a, b) for a, b in combinations(means, 2)
        ]
        worst_positive = 1.0
        condition_score_means = []
        norm_means = []
        endpoint_deltas = []
        endpoint_occupancies = []
        for result in condition_results.values():
            projected = np.asarray(result[name]["projected_vectors"], dtype=np.float64)
            scores = projected @ global_direction
            worst_positive = min(worst_positive, float(np.mean(scores > 0)))
            condition_score_means.append(float(np.mean(scores)))
            norm_means.append(result[name]["projected_norm_mean"])
            endpoint_deltas.extend(
                [
                    result[name]["gpu_endpoint_delta"],
                    result[name]["dla_endpoint_delta"],
                ]
            )
            endpoint_occupancies.extend(
                [
                    result[name]["gpu_endpoint_occupancy"],
                    result[name]["dla_endpoint_occupancy"],
                ]
            )
        summary[name] = {
            "is_benign_control": name in BENIGN_NAMES,
            "condition_direction_cosine_min": (
                float(min(condition_cosines)) if condition_cosines else 1.0
            ),
            "condition_direction_cosine_median": (
                float(np.median(condition_cosines)) if condition_cosines else 1.0
            ),
            "condition_score_mean_min": float(min(condition_score_means)),
            "image_positive_fraction_worst_condition": worst_positive,
            "projected_norm_mean": float(np.mean(norm_means)),
            "endpoint_delta_max": float(max(endpoint_deltas)),
            "endpoint_occupancy_max": float(max(endpoint_occupancies)),
        }

    benign_norms = [
        summary[name]["projected_norm_mean"]
        for name in BENIGN_NAMES
        if name in summary
    ]
    benign_norm = max(benign_norms) if benign_norms else None
    for row in summary.values():
        row["norm_over_max_benign_control"] = (
            row["projected_norm_mean"] / (benign_norm + 1e-30)
            if benign_norm is not None
            else None
        )
        row["screen_gate"] = (
            "GO"
            if (
                not row["is_benign_control"]
                and row["condition_direction_cosine_min"] >= 0.8
                and row["image_positive_fraction_worst_condition"] >= 0.7
                and row["norm_over_max_benign_control"] is not None
                and row["norm_over_max_benign_control"] >= 3.0
                and row["endpoint_occupancy_max"] <= 0.001
            )
            else "NO-GO"
        )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-index",
        type=Path,
        default=Path("chain_survival/results/v13/boundary_strict_int8/run_index.json"),
    )
    parser.add_argument(
        "--splits",
        type=Path,
        default=Path("chain_survival/results/v13/splits_v13.json"),
    )
    parser.add_argument(
        "--subspace",
        type=Path,
        default=Path("chain_survival/results/v13/layer4.2_consensus_subspace.npz"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("chain_survival/results/v13/controllability_screen.json"),
    )
    parser.add_argument("--boundary", default="layer4.2")
    parser.add_argument("--model", default="resnet50")
    parser.add_argument("--n-images", type=int, default=32)
    parser.add_argument("--builds", type=int, nargs="+", default=[0])
    parser.add_argument("--calibrations", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--patterns", nargs="*")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run = json.loads(args.run_index.read_text())
    split_data = json.loads(args.splits.read_text())
    records = {
        (
            record["boundary"],
            int(record["calibration"]),
            int(record["build"]),
            record["backend"],
        ): record
        for record in run["records"]
        if record["status"] == "OK"
    }
    transform = model_config.get_transform(args.model)
    entries = split_data["splits"]["mechanism_discovery"][: args.n_images]
    inputs = load_split(split_data["imagenet_root"], entries, transform)
    candidates = perturbations(inputs)
    if args.patterns:
        requested = set(args.patterns)
        candidates = {key: value for key, value in candidates.items() if key in requested}
    with np.load(args.subspace) as archive:
        subspace = archive["directions"].astype(np.float32)

    condition_results = {}
    for calibration in args.calibrations:
        for build in args.builds:
            gpu_record = records[(args.boundary, calibration, build, "gpu")]
            dla_record = records[(args.boundary, calibration, build, "dla")]
            gpu_clean, gpu_endpoints, gpu_clean_occupancy = clean_data(
                gpu_record["activation_path"], args.n_images
            )
            dla_clean, dla_endpoints, dla_clean_occupancy = clean_data(
                dla_record["activation_path"], args.n_images
            )
            gpu_engine = runtime.load_engine(gpu_record["engine_path"])
            dla_engine = runtime.load_engine(dla_record["engine_path"])
            condition_key = f"cal{calibration}_build{build}"
            condition_results[condition_key] = {}
            for name, perturbed in candidates.items():
                gpu_triggered, gpu_occupancy = run_pooled(
                    gpu_engine, perturbed, gpu_endpoints
                )
                dla_triggered, dla_occupancy = run_pooled(
                    dla_engine, perturbed, dla_endpoints
                )
                interaction = (
                    (dla_triggered - dla_clean)
                    - (gpu_triggered - gpu_clean)
                )
                projected = interaction @ subspace.T
                condition_results[condition_key][name] = {
                    "mean_projected_vector": projected.mean(axis=0).tolist(),
                    "projected_norm_mean": float(
                        np.linalg.norm(projected, axis=1).mean()
                    ),
                    "projected_vectors": projected.tolist(),
                    "gpu_endpoint_occupancy": gpu_occupancy,
                    "dla_endpoint_occupancy": dla_occupancy,
                    "gpu_endpoint_delta": gpu_occupancy - gpu_clean_occupancy,
                    "dla_endpoint_delta": dla_occupancy - dla_clean_occupancy,
                }

    summary = summarize_candidates(condition_results, list(candidates))
    tm_q_by_calibration = {}
    for calibration in args.calibrations:
        prefix = f"cal{calibration}_"
        scoped = {
            key: value
            for key, value in condition_results.items()
            if key.startswith(prefix)
        }
        scoped_summary = summarize_candidates(scoped, list(candidates))
        tm_q_by_calibration[str(calibration)] = {
            "summary": scoped_summary,
            "candidates": [
                name
                for name, row in scoped_summary.items()
                if row["screen_gate"] == "GO"
            ],
        }
    result = {
        "schema_version": 1,
        "boundary": args.boundary,
        "n_images": args.n_images,
        "builds": args.builds,
        "calibrations": args.calibrations,
        "subspace": str(args.subspace),
        "summary": summary,
        "tm_q_by_calibration": tm_q_by_calibration,
        "condition_results": condition_results,
        "gate": {
            "tm_w_candidates": [
                name for name, row in summary.items() if row["screen_gate"] == "GO"
            ],
            "tm_q_candidates": {
                calibration: value["candidates"]
                for calibration, value in tm_q_by_calibration.items()
            },
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(
        json.dumps(
            {
                "output": str(args.output),
                "gate": result["gate"],
                "summary": summary,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
