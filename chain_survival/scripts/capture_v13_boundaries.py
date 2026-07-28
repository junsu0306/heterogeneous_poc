"""Capture strict-INT8 GPU/DLA ResNet-50 boundary summaries for Track B B4.

The runner stores spatially aware 4x4 pooled activations, channel mean/max and
an exact value histogram. Full 56x56 tensors are not retained, avoiding
multi-gigabyte artifacts while preserving coarse spatial structure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import onnx
import torch
import torch.nn.functional as functional
from onnx import shape_inference
from onnx import utils as onnx_utils

sys.path.insert(0, str(Path("common/scripts").resolve()))
import trt_runtime as runtime  # noqa: E402

sys.path.insert(0, str(Path("chain_survival/scripts").resolve()))
import models_cfg as model_config  # noqa: E402
from run_paths import load_split  # noqa: E402


BOUNDARIES = {
    "layer1.2": "/layer1/layer1.2/Add_output_0",
    "layer4.2": "/layer4/layer4.2/Add_output_0",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize_engine(engine, inputs: np.ndarray) -> dict[str, np.ndarray]:
    runner = runtime.EngineRunner(engine)
    pooled = []
    channel_mean = []
    channel_max = []
    histogram = {}
    output_shape = None
    for index in range(len(inputs)):
        activation = runner.run(inputs[index : index + 1])[0].astype(np.float32)
        output_shape = list(activation.shape)
        tensor = torch.from_numpy(activation).unsqueeze(0)
        pooled.append(
            functional.adaptive_avg_pool2d(tensor, (4, 4))[0].numpy().astype(np.float16)
        )
        flat = activation.reshape(activation.shape[0], -1)
        channel_mean.append(flat.mean(axis=1).astype(np.float32))
        channel_max.append(flat.max(axis=1).astype(np.float32))
        values, counts = np.unique(activation, return_counts=True)
        for value, count in zip(values, counts):
            key = float(value)
            histogram[key] = histogram.get(key, 0) + int(count)
    hist_values = np.asarray(sorted(histogram), dtype=np.float32)
    hist_counts = np.asarray([histogram[float(value)] for value in hist_values], dtype=np.int64)
    return {
        "pooled4": np.stack(pooled),
        "channel_mean": np.stack(channel_mean),
        "channel_max": np.stack(channel_max),
        "hist_values": hist_values,
        "hist_counts": hist_counts,
        "output_shape": np.asarray(output_shape, dtype=np.int64),
    }


def extract_boundaries(source: Path, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    inferred_path = output_dir / "resnet50_inferred.onnx"
    if not inferred_path.exists():
        onnx.save(shape_inference.infer_shapes(onnx.load(source)), inferred_path)
    paths = {}
    for name, tensor in BOUNDARIES.items():
        path = output_dir / f"resnet50_{name}.onnx"
        if not path.exists():
            onnx_utils.extract_model(
                str(inferred_path), str(path), ["input"], [tensor]
            )
        paths[name] = path
    return paths


def inspect(engine) -> str:
    inspector = engine.create_engine_inspector()
    return inspector.get_engine_information(
        runtime.trt.LayerInformationFormat.JSON
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-onnx",
        type=Path,
        default=Path("chain_survival/onnx/resnet50.onnx"),
    )
    parser.add_argument(
        "--splits",
        type=Path,
        default=Path("chain_survival/results/v13/splits_v13.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("chain_survival/results/v13/boundary_strict_int8"),
    )
    parser.add_argument("--boundaries", nargs="+", choices=sorted(BOUNDARIES), default=sorted(BOUNDARIES))
    parser.add_argument("--builds", type=int, default=3)
    parser.add_argument("--calibrations", type=int, default=2)
    parser.add_argument("--n-calib", type=int, default=200)
    parser.add_argument("--n-discovery", type=int, default=128)
    parser.add_argument("--allow-gpu-fallback", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    onnx_dir = args.output_dir / "onnx"
    engine_dir = args.output_dir / "engines"
    cache_dir = args.output_dir / "calibration_cache"
    engine_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    boundary_paths = extract_boundaries(args.source_onnx, onnx_dir)

    split_data = json.loads(args.splits.read_text())
    root = split_data["imagenet_root"]
    transform = model_config.get_transform("resnet50")
    discovery_entries = split_data["splits"]["mechanism_discovery"][: args.n_discovery]
    discovery = load_split(root, discovery_entries, transform)
    calibration_roles = ["calib_shadow_1", "calib_shadow_2"][: args.calibrations]

    index_path = args.output_dir / "run_index.json"
    records = {}
    if index_path.exists() and not args.overwrite:
        previous = json.loads(index_path.read_text())
        for record in previous.get("records", []):
            key = (
                record["boundary"],
                int(record["calibration"]),
                int(record["build"]),
                record["backend"],
            )
            records[key] = record

    def write_index() -> None:
        index_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "captured_at": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
                    "settings": {
                        "source_onnx": str(args.source_onnx),
                        "source_onnx_sha256": sha256(args.source_onnx),
                        "splits": str(args.splits),
                        "splits_sha256": sha256(args.splits),
                        "boundaries": args.boundaries,
                        "builds": args.builds,
                        "calibrations": args.calibrations,
                        "n_calib": args.n_calib,
                        "n_discovery": args.n_discovery,
                        "allow_gpu_fallback": args.allow_gpu_fallback,
                        "strict_layer_int8": True,
                        "discovery_paths_sha256": hashlib.sha256(
                            "\n".join(entry["path"] for entry in discovery_entries).encode()
                        ).hexdigest(),
                    },
                    "records": list(records.values()),
                },
                indent=2,
            )
        )

    for boundary in args.boundaries:
        onnx_path = boundary_paths[boundary]
        for calibration_index, calibration_role in enumerate(calibration_roles):
            entries = split_data["splits"][calibration_role][: args.n_calib]
            calibration = load_split(root, entries, transform)
            calibration_samples = [
                calibration[index : index + 1] for index in range(len(calibration))
            ]
            for build_index in range(args.builds):
                for backend in ("gpu", "dla"):
                    key = (boundary, calibration_index, build_index, backend)
                    stem = f"{boundary}__cal{calibration_index}__build{build_index}__{backend}"
                    activation_path = args.output_dir / f"{stem}.npz"
                    serialized_path = engine_dir / f"{stem}.engine"
                    inspector_path = engine_dir / f"{stem}.inspector.json"
                    cache_path = cache_dir / f"{boundary}__cal{calibration_index}__{backend}.cache"
                    if activation_path.exists() and not args.overwrite:
                        continue
                    try:
                        calibrator = runtime.EntropyListCalibrator(
                            [] if cache_path.exists() else calibration_samples,
                            str(cache_path),
                        )
                        serialized = runtime.build_int8_engine(
                            str(onnx_path),
                            backend,
                            calibrator,
                            allow_gpu_fallback=(
                                args.allow_gpu_fallback if backend == "dla" else True
                            ),
                            force_layer_int8=True,
                            detailed_inspector=True,
                        )
                        if serialized is None:
                            raise RuntimeError("TensorRT returned no serialized engine")
                        serialized_path.write_bytes(bytes(serialized))
                        engine = runtime.load_engine(serialized)
                        inspector_path.write_text(inspect(engine))
                        summary = summarize_engine(engine, discovery)
                        np.savez(activation_path, **summary)
                        records[key] = {
                            "boundary": boundary,
                            "calibration": calibration_index,
                            "calibration_role": calibration_role,
                            "build": build_index,
                            "backend": backend,
                            "status": "OK",
                            "activation_path": str(activation_path),
                            "activation_sha256": sha256(activation_path),
                            "engine_path": str(serialized_path),
                            "engine_sha256": sha256(serialized_path),
                            "inspector_path": str(inspector_path),
                            "calibration_cache_path": str(cache_path),
                            "calibration_cache_sha256": (
                                sha256(cache_path) if cache_path.exists() else None
                            ),
                        }
                    except Exception as error:
                        records[key] = {
                            "boundary": boundary,
                            "calibration": calibration_index,
                            "calibration_role": calibration_role,
                            "build": build_index,
                            "backend": backend,
                            "status": "FAILED",
                            "error": repr(error),
                        }
                    write_index()
    status = [record["status"] for record in records.values()]
    print(
        json.dumps(
            {
                "output": str(index_path),
                "n_ok": status.count("OK"),
                "n_failed": status.count("FAILED"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
