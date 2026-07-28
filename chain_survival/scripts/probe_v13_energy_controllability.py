"""Rotation-invariant B5 screen for residual-energy trigger candidates.

The signed residual vector was unstable across TensorRT builds/calibrations,
while the ResNet-50 layer4.2 top-8 residual subspace was reproducible. This
screen therefore measures squared trigger-path interaction energy inside that
fixed subspace and compares every perturbation with Gaussian noise at matched
pixel-space RMS.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional

sys.path.insert(0, str(Path("common/scripts").resolve()))
import trt_runtime as runtime  # noqa: E402

sys.path.insert(0, str(Path("chain_survival/scripts").resolve()))
import models_cfg as model_config  # noqa: E402
from run_paths import load_split  # noqa: E402


PIXEL_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32).reshape(
    1, 3, 1, 1
)
PIXEL_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32).reshape(
    1, 3, 1, 1
)


def candidate_specs() -> list[dict]:
    specs = []
    for rms in (2, 4, 8):
        specs.append(
            {
                "name": f"gaussian_rms{rms}",
                "kind": "gaussian",
                "rms_255": rms,
                "is_control": True,
            }
        )
        for sign, label in ((1, "plus"), (-1, "minus")):
            specs.extend(
                [
                    {
                        "name": f"brightness_{label}_rms{rms}",
                        "kind": "brightness",
                        "sign": sign,
                        "rms_255": rms,
                    },
                    {
                        "name": f"contrast_{label}_rms{rms}",
                        "kind": "contrast",
                        "sign": sign,
                        "rms_255": rms,
                    },
                ]
            )
        for frequency in (1, 2, 4, 8):
            specs.append(
                {
                    "name": f"sine_gray_f{frequency}_rms{rms}",
                    "kind": "sine_gray",
                    "frequency": frequency,
                    "rms_255": rms,
                }
            )
    for rms in (4, 8):
        for frequency in (1, 2, 4):
            specs.append(
                {
                    "name": f"sine_chromatic_f{frequency}_rms{rms}",
                    "kind": "sine_chromatic",
                    "frequency": frequency,
                    "rms_255": rms,
                }
            )
        for grid in (4, 8):
            specs.append(
                {
                    "name": f"smooth_random_g{grid}_rms{rms}",
                    "kind": "smooth_random",
                    "grid": grid,
                    "rms_255": rms,
                }
            )
        for sign, label in ((1, "center_hi"), (-1, "center_lo")):
            specs.append(
                {
                    "name": f"vignette_{label}_rms{rms}",
                    "kind": "vignette",
                    "sign": sign,
                    "rms_255": rms,
                }
            )
        for color in ("red", "green", "blue"):
            specs.append(
                {
                    "name": f"color_{color}_rms{rms}",
                    "kind": "color",
                    "color": color,
                    "rms_255": rms,
                }
            )
        for size in (64, 96, 128):
            specs.append(
                {
                    "name": f"checker_br_s{size}_rms{rms}",
                    "kind": "checker_patch",
                    "size": size,
                    "rms_255": rms,
                }
            )
    for spec in specs:
        spec.setdefault("is_control", False)
        spec["reference_control"] = (
            None
            if spec["is_control"]
            else f"gaussian_rms{spec['rms_255']}"
        )
    return specs


def normalize_delta(raw: np.ndarray, target_rms: float) -> np.ndarray:
    flat = raw.reshape(len(raw), -1)
    rms = np.sqrt(np.mean(np.square(flat), axis=1))
    scale = target_rms / np.maximum(rms, 1e-30)
    delta = raw * scale.reshape(-1, 1, 1, 1)
    return np.clip(delta, -4.0 * target_rms, 4.0 * target_rms)


def raw_pattern(spec: dict, pixels: np.ndarray) -> np.ndarray:
    count, _, height, width = pixels.shape
    kind = spec["kind"]
    if kind == "gaussian":
        rng = np.random.default_rng(5100 + int(spec["rms_255"]))
        return rng.normal(size=pixels.shape).astype(np.float32)
    if kind == "brightness":
        return np.full_like(pixels, float(spec["sign"]))
    if kind == "contrast":
        spatial_mean = pixels.mean(axis=(2, 3), keepdims=True)
        return float(spec["sign"]) * (pixels - spatial_mean)

    rows = np.linspace(0.0, 1.0, height, endpoint=False, dtype=np.float32)
    columns = np.linspace(0.0, 1.0, width, endpoint=False, dtype=np.float32)
    yy, xx = np.meshgrid(rows, columns, indexing="ij")
    if kind in {"sine_gray", "sine_chromatic"}:
        frequency = int(spec["frequency"])
        pattern = np.sin(
            2.0 * np.pi * frequency * (0.73 * xx + 0.41 * yy) + 0.37
        ).astype(np.float32)
        if kind == "sine_gray":
            color = np.ones((3, 1, 1), dtype=np.float32)
        else:
            color = np.asarray([1.0, -0.8, 0.35], dtype=np.float32).reshape(
                3, 1, 1
            )
        return np.broadcast_to(
            color[None] * pattern[None, None], pixels.shape
        ).copy()
    if kind == "smooth_random":
        rng = np.random.default_rng(
            5200 + int(spec["grid"]) * 10 + int(spec["rms_255"])
        )
        coarse = rng.normal(
            size=(count, 3, int(spec["grid"]), int(spec["grid"]))
        ).astype(np.float32)
        smooth = functional.interpolate(
            torch.from_numpy(coarse),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        ).numpy()
        return smooth - smooth.mean(axis=(2, 3), keepdims=True)
    if kind == "vignette":
        radial = np.sqrt(np.square(xx - 0.5) + np.square(yy - 0.5))
        pattern = (0.5 - radial).astype(np.float32) * float(spec["sign"])
        return np.broadcast_to(
            pattern[None, None], pixels.shape
        ).copy()
    if kind == "color":
        color = np.full((3,), -0.5, dtype=np.float32)
        color[{"red": 0, "green": 1, "blue": 2}[spec["color"]]] = 1.0
        return np.broadcast_to(
            color.reshape(1, 3, 1, 1), pixels.shape
        ).copy()
    if kind == "checker_patch":
        size = int(spec["size"])
        raw = np.zeros_like(pixels)
        grid_y, grid_x = np.indices((size, size))
        checker = np.where(
            ((grid_x // 8) + (grid_y // 8)) % 2 == 0, -1.0, 1.0
        ).astype(np.float32)
        row, column = height - size, width - size
        raw[:, :, row:, column:] = checker.reshape(1, 1, size, size)
        return raw
    raise KeyError(kind)


def perturb(
    normalized_inputs: np.ndarray, spec: dict
) -> tuple[np.ndarray, dict]:
    pixels = np.clip(
        normalized_inputs * PIXEL_STD + PIXEL_MEAN, 0.0, 1.0
    )
    target_rms = float(spec["rms_255"]) / 255.0
    delta = normalize_delta(raw_pattern(spec, pixels), target_rms)
    perturbed_pixels = np.clip(pixels + delta, 0.0, 1.0)
    actual_delta = perturbed_pixels - pixels
    per_image_rms = np.sqrt(
        np.mean(np.square(actual_delta), axis=(1, 2, 3))
    )
    per_image_linf = np.max(np.abs(actual_delta), axis=(1, 2, 3))
    normalized = (perturbed_pixels - PIXEL_MEAN) / PIXEL_STD
    return normalized.astype(np.float32), {
        "target_rms": target_rms,
        "actual_rms_mean": float(np.mean(per_image_rms)),
        "actual_rms_min": float(np.min(per_image_rms)),
        "actual_rms_max": float(np.max(per_image_rms)),
        "actual_linf_mean": float(np.mean(per_image_linf)),
        "actual_linf_max": float(np.max(per_image_linf)),
    }


def endpoints(path: str) -> tuple[float, float]:
    with np.load(path) as archive:
        values = archive["hist_values"]
    return float(values[0]), float(values[-1])


def run_pooled(
    runner: runtime.EngineRunner,
    inputs: np.ndarray,
    quantized_endpoints: tuple[float, float],
) -> tuple[np.ndarray, float]:
    pooled = []
    endpoint_count = 0
    total = 0
    lower, upper = quantized_endpoints
    for index in range(len(inputs)):
        activation = runner.run(inputs[index : index + 1])[0].astype(
            np.float32
        )
        tensor = torch.from_numpy(activation).unsqueeze(0)
        pooled.append(
            functional.adaptive_avg_pool2d(tensor, (4, 4))[0].numpy()
        )
        endpoint_count += int(np.count_nonzero(activation == upper))
        if lower < 0:
            endpoint_count += int(np.count_nonzero(activation == lower))
        total += activation.size
    return (
        np.stack(pooled).reshape(len(inputs), -1),
        endpoint_count / total,
    )


def summarize_scope(
    condition_results: dict[str, dict],
    specs: list[dict],
    perturbation_metrics: dict[str, dict],
) -> dict[str, dict]:
    summary = {}
    for spec in specs:
        name = spec["name"]
        if spec["is_control"]:
            summary[name] = {
                "is_control": True,
                "screen_gate": "CONTROL",
            }
            continue
        control = spec["reference_control"]
        energy_ratios = []
        paired_fractions = []
        interaction_gpu_ratios = []
        endpoint_occupancies = []
        energy_means = []
        for condition in condition_results.values():
            candidate_row = condition[name]
            control_row = condition[control]
            candidate_energy = np.asarray(
                candidate_row["interaction_energy"], dtype=np.float64
            )
            control_energy = np.asarray(
                control_row["interaction_energy"], dtype=np.float64
            )
            energy_ratios.append(
                float(
                    np.mean(candidate_energy)
                    / (np.mean(control_energy) + 1e-30)
                )
            )
            paired_fractions.append(
                float(np.mean(candidate_energy > control_energy))
            )
            interaction_gpu_ratios.append(
                candidate_row["interaction_norm_mean"]
                / (candidate_row["gpu_effect_norm_mean"] + 1e-30)
            )
            endpoint_occupancies.extend(
                [
                    candidate_row["gpu_endpoint_occupancy"],
                    candidate_row["dla_endpoint_occupancy"],
                ]
            )
            energy_means.append(float(np.mean(candidate_energy)))
        actual_rms = perturbation_metrics[name]["actual_rms_mean"]
        control_rms = perturbation_metrics[control]["actual_rms_mean"]
        rms_ratio = actual_rms / (control_rms + 1e-30)
        row = {
            "is_control": False,
            "reference_control": control,
            "energy_ratio_worst_condition": float(min(energy_ratios)),
            "energy_ratio_median_condition": float(
                np.median(energy_ratios)
            ),
            "energy_ratio_best_condition": float(max(energy_ratios)),
            "paired_fraction_worst_condition": float(
                min(paired_fractions)
            ),
            "paired_fraction_median_condition": float(
                np.median(paired_fractions)
            ),
            "interaction_to_gpu_effect_ratio_worst": float(
                min(interaction_gpu_ratios)
            ),
            "condition_energy_coefficient_of_variation": float(
                np.std(energy_means) / (np.mean(energy_means) + 1e-30)
            ),
            "actual_rms_over_control": float(rms_ratio),
            "endpoint_occupancy_max": float(max(endpoint_occupancies)),
        }
        row["screen_gate"] = (
            "GO"
            if (
                row["energy_ratio_worst_condition"] >= 1.25
                and row["paired_fraction_worst_condition"] >= 0.65
                and 0.8 <= row["actual_rms_over_control"] <= 1.2
                and row["endpoint_occupancy_max"] <= 0.001
            )
            else "NO-GO"
        )
        summary[name] = row
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-index",
        type=Path,
        default=Path(
            "chain_survival/results/v13/"
            "boundary_strict_int8/run_index.json"
        ),
    )
    parser.add_argument(
        "--splits",
        type=Path,
        default=Path("chain_survival/results/v13/splits_v13.json"),
    )
    parser.add_argument(
        "--subspace",
        type=Path,
        default=Path(
            "chain_survival/results/v13/layer4.2_consensus_subspace.npz"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "chain_survival/results/v13/"
            "residual_energy_controllability_screen.json"
        ),
    )
    parser.add_argument("--boundary", default="layer4.2")
    parser.add_argument("--model", default="resnet50")
    parser.add_argument("--split-role", default="mechanism_discovery")
    parser.add_argument("--offset", type=int, default=64)
    parser.add_argument("--n-images", type=int, default=64)
    parser.add_argument("--builds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument(
        "--calibrations", type=int, nargs="+", default=[0, 1]
    )
    parser.add_argument("--patterns", nargs="*")
    parser.add_argument("--overwrite", action="store_true")
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
    entries = split_data["splits"][args.split_role][
        args.offset : args.offset + args.n_images
    ]
    if len(entries) != args.n_images:
        raise ValueError("requested image range exceeds the split")
    inputs = load_split(split_data["imagenet_root"], entries, transform)
    with np.load(args.subspace) as archive:
        subspace = archive["directions"].astype(np.float32)

    specs = candidate_specs()
    if args.patterns:
        requested = set(args.patterns)
        selected = [spec for spec in specs if spec["name"] in requested]
        missing = requested - {spec["name"] for spec in selected}
        if missing:
            raise ValueError(f"unknown patterns: {sorted(missing)}")
        needed_controls = {
            spec["reference_control"]
            for spec in selected
            if spec["reference_control"] is not None
        }
        specs = [
            spec
            for spec in specs
            if spec["name"] in requested or spec["name"] in needed_controls
        ]

    perturbation_metrics = {}
    for spec in specs:
        _, metrics = perturb(inputs, spec)
        perturbation_metrics[spec["name"]] = metrics

    settings = {
        "run_index": str(args.run_index),
        "splits": str(args.splits),
        "subspace": str(args.subspace),
        "boundary": args.boundary,
        "model": args.model,
        "split_role": args.split_role,
        "offset": args.offset,
        "n_images": args.n_images,
        "builds": args.builds,
        "calibrations": args.calibrations,
        "patterns": [spec["name"] for spec in specs],
        "pixel_rms_values_255": sorted(
            {int(spec["rms_255"]) for spec in specs}
        ),
    }
    condition_results = {}
    if args.output.exists() and not args.overwrite:
        previous = json.loads(args.output.read_text())
        if previous.get("settings") != settings:
            raise ValueError(
                "existing output settings differ; pass --overwrite"
            )
        condition_results = previous.get("condition_results", {})

    def write_checkpoint(status: str) -> None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "status": status,
                    "settings": settings,
                    "perturbation_specs": specs,
                    "perturbation_metrics": perturbation_metrics,
                    "condition_results": condition_results,
                },
                indent=2,
            )
        )

    for calibration in args.calibrations:
        for build in args.builds:
            condition_key = f"cal{calibration}_build{build}"
            if condition_key in condition_results:
                print(f"[skip] {condition_key}", flush=True)
                continue
            gpu_record = records[
                (args.boundary, calibration, build, "gpu")
            ]
            dla_record = records[
                (args.boundary, calibration, build, "dla")
            ]
            gpu_engine = runtime.load_engine(gpu_record["engine_path"])
            dla_engine = runtime.load_engine(dla_record["engine_path"])
            gpu_runner = runtime.EngineRunner(gpu_engine)
            dla_runner = runtime.EngineRunner(dla_engine)
            gpu_endpoints = endpoints(gpu_record["activation_path"])
            dla_endpoints = endpoints(dla_record["activation_path"])
            gpu_clean, _ = run_pooled(
                gpu_runner, inputs, gpu_endpoints
            )
            dla_clean, _ = run_pooled(
                dla_runner, inputs, dla_endpoints
            )
            condition_results[condition_key] = {}
            for spec_index, spec in enumerate(specs, start=1):
                triggered, _ = perturb(inputs, spec)
                gpu_triggered, gpu_occupancy = run_pooled(
                    gpu_runner, triggered, gpu_endpoints
                )
                dla_triggered, dla_occupancy = run_pooled(
                    dla_runner, triggered, dla_endpoints
                )
                gpu_effect = gpu_triggered - gpu_clean
                dla_effect = dla_triggered - dla_clean
                interaction = dla_effect - gpu_effect
                projected_interaction = interaction @ subspace.T
                projected_gpu = gpu_effect @ subspace.T
                projected_dla = dla_effect @ subspace.T
                interaction_energy = np.sum(
                    np.square(projected_interaction), axis=1
                )
                condition_results[condition_key][spec["name"]] = {
                    "interaction_energy": interaction_energy.tolist(),
                    "interaction_norm_mean": float(
                        np.mean(np.sqrt(interaction_energy))
                    ),
                    "gpu_effect_norm_mean": float(
                        np.mean(
                            np.linalg.norm(projected_gpu, axis=1)
                        )
                    ),
                    "dla_effect_norm_mean": float(
                        np.mean(
                            np.linalg.norm(projected_dla, axis=1)
                        )
                    ),
                    "gpu_endpoint_occupancy": gpu_occupancy,
                    "dla_endpoint_occupancy": dla_occupancy,
                }
                if spec_index % 10 == 0 or spec_index == len(specs):
                    print(
                        f"[{condition_key}] "
                        f"{spec_index}/{len(specs)} patterns",
                        flush=True,
                    )
            write_checkpoint("RUNNING")
            del gpu_runner, dla_runner, gpu_engine, dla_engine
            gc.collect()

    summary = summarize_scope(
        condition_results, specs, perturbation_metrics
    )
    tm_q_by_calibration = {}
    for calibration in args.calibrations:
        scoped = {
            key: value
            for key, value in condition_results.items()
            if key.startswith(f"cal{calibration}_")
        }
        scoped_summary = summarize_scope(
            scoped, specs, perturbation_metrics
        )
        tm_q_by_calibration[str(calibration)] = {
            "summary": scoped_summary,
            "candidates": [
                name
                for name, row in scoped_summary.items()
                if row["screen_gate"] == "GO"
            ],
        }
    candidates = [
        name for name, row in summary.items() if row["screen_gate"] == "GO"
    ]
    ranked = sorted(
        (
            (name, row)
            for name, row in summary.items()
            if not row["is_control"]
        ),
        key=lambda item: (
            item[1]["energy_ratio_worst_condition"],
            item[1]["paired_fraction_worst_condition"],
        ),
        reverse=True,
    )
    result = {
        "schema_version": 1,
        "status": "COMPLETED",
        "settings": settings,
        "perturbation_specs": specs,
        "perturbation_metrics": perturbation_metrics,
        "summary": summary,
        "tm_q_by_calibration": tm_q_by_calibration,
        "condition_results": condition_results,
        "gate": {
            "tm_w_candidates": candidates,
            "tm_q_candidates": {
                calibration: value["candidates"]
                for calibration, value in tm_q_by_calibration.items()
            },
            "top_five": [
                {
                    "name": name,
                    "energy_ratio_worst_condition": row[
                        "energy_ratio_worst_condition"
                    ],
                    "paired_fraction_worst_condition": row[
                        "paired_fraction_worst_condition"
                    ],
                    "screen_gate": row["screen_gate"],
                }
                for name, row in ranked[:5]
            ],
        },
    }
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result["gate"], indent=2))


if __name__ == "__main__":
    main()
