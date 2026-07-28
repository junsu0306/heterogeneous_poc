"""Capture v13 strict-INT8 boundaries for VGG-16 and GoogLeNet.

The exported heads end at one architecture-specific deep boundary:

- VGG-16: the final convolutional ReLU before the last max-pool
- GoogLeNet: the inception5b branch-merge output

Only mechanism-discovery images are evaluated. Calibration comes from the two
shadow calibration roles created by ``prepare_v13_splits.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as functional

sys.path.insert(0, str(Path("common/scripts").resolve()))
import trt_runtime as runtime  # noqa: E402

sys.path.insert(0, str(Path("chain_survival/scripts").resolve()))
import models_cfg as model_config  # noqa: E402
from run_paths import load_split  # noqa: E402


BOUNDARIES = {
    "vgg16": "vgg16.features29",
    "vgg19": "vgg19.features35",
    "googlenet": "googlenet.inception5b",
}


class VGG16Boundary(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        model = model_config.get_model("vgg16", pretrained=True)
        self.features = model.features[:30]

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.features(inputs)


class VGG19Boundary(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        model = model_config.get_model("vgg19", pretrained=True)
        self.features = model.features[:36]

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.features(inputs)


class GoogLeNetBoundary(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        model = model_config.get_model("googlenet", pretrained=True)
        self.transform_input = model.transform_input
        self.register_buffer(
            "input_scale",
            torch.tensor(
                [0.229 / 0.5, 0.224 / 0.5, 0.225 / 0.5]
            ).reshape(1, 3, 1, 1),
        )
        self.register_buffer(
            "input_bias",
            torch.tensor(
                [
                    (0.485 - 0.5) / 0.5,
                    (0.456 - 0.5) / 0.5,
                    (0.406 - 0.5) / 0.5,
                ]
            ).reshape(1, 3, 1, 1),
        )
        self.conv1 = model.conv1
        self.maxpool1 = model.maxpool1
        self.conv2 = model.conv2
        self.conv3 = model.conv3
        self.maxpool2 = model.maxpool2
        self.inception3a = model.inception3a
        self.inception3b = model.inception3b
        self.maxpool3 = model.maxpool3
        self.inception4a = model.inception4a
        self.inception4b = model.inception4b
        self.inception4c = model.inception4c
        self.inception4d = model.inception4d
        self.inception4e = model.inception4e
        self.maxpool4 = model.maxpool4
        self.inception5a = model.inception5a
        self.inception5b = model.inception5b

    def _transform_input(self, inputs: torch.Tensor) -> torch.Tensor:
        if not self.transform_input:
            return inputs
        return inputs * self.input_scale + self.input_bias

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = self._transform_input(inputs)
        inputs = self.conv1(inputs)
        inputs = self.maxpool1(inputs)
        inputs = self.conv2(inputs)
        inputs = self.conv3(inputs)
        inputs = self.maxpool2(inputs)
        inputs = self.inception3a(inputs)
        inputs = self.inception3b(inputs)
        inputs = self.maxpool3(inputs)
        inputs = self.inception4a(inputs)
        inputs = self.inception4b(inputs)
        inputs = self.inception4c(inputs)
        inputs = self.inception4d(inputs)
        inputs = self.inception4e(inputs)
        inputs = self.maxpool4(inputs)
        inputs = self.inception5a(inputs)
        return self.inception5b(inputs)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def boundary_model(model_name: str) -> torch.nn.Module:
    if model_name == "vgg16":
        return VGG16Boundary()
    if model_name == "vgg19":
        return VGG19Boundary()
    if model_name == "googlenet":
        return GoogLeNetBoundary()
    raise KeyError(model_name)


def export_boundary(model_name: str, path: Path, overwrite: bool = False) -> None:
    if path.exists() and not overwrite:
        return
    model = boundary_model(model_name).eval()
    dummy = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy,
            path,
            opset_version=17,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["boundary"],
            dynamo=False,
        )


def validate_onnx(model_name: str, path: Path) -> dict:
    torch.manual_seed(1304)
    model = boundary_model(model_name).eval()
    inputs = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        expected = model(inputs).numpy()
    session = ort.InferenceSession(
        str(path), providers=["CPUExecutionProvider"]
    )
    observed = session.run(None, {"input": inputs.numpy()})[0]
    difference = observed.astype(np.float64) - expected.astype(np.float64)
    expected_flat = expected.astype(np.float64).reshape(-1)
    observed_flat = observed.astype(np.float64).reshape(-1)
    error_rms = float(np.sqrt(np.mean(np.square(difference))))
    reference_rms = float(np.sqrt(np.mean(np.square(expected_flat))))
    return {
        "shape": list(observed.shape),
        "max_abs_difference": float(np.max(np.abs(difference))),
        "mean_abs_difference": float(np.mean(np.abs(difference))),
        "error_rms": error_rms,
        "reference_rms": reference_rms,
        "normalized_error_rms": error_rms / (reference_rms + 1e-30),
        "cosine_similarity": float(
            np.dot(expected_flat, observed_flat)
            / (
                np.linalg.norm(expected_flat)
                * np.linalg.norm(observed_flat)
                + 1e-30
            )
        ),
        "allclose_rtol_1e-4_atol_1e-5": bool(
            np.allclose(observed, expected, rtol=1e-4, atol=1e-5)
        ),
        "allclose_rtol_1e-4_atol_2e-5": bool(
            np.allclose(observed, expected, rtol=1e-4, atol=2e-5)
        ),
    }


def inspect(engine) -> str:
    inspector = engine.create_engine_inspector()
    return inspector.get_engine_information(
        runtime.trt.LayerInformationFormat.JSON
    )


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
            functional.adaptive_avg_pool2d(tensor, (4, 4))[0]
            .numpy()
            .astype(np.float16)
        )
        flat = activation.reshape(activation.shape[0], -1)
        channel_mean.append(flat.mean(axis=1).astype(np.float32))
        channel_max.append(flat.max(axis=1).astype(np.float32))
        values, counts = np.unique(activation, return_counts=True)
        for value, count in zip(values, counts):
            key = float(value)
            histogram[key] = histogram.get(key, 0) + int(count)
    hist_values = np.asarray(sorted(histogram), dtype=np.float32)
    hist_counts = np.asarray(
        [histogram[float(value)] for value in hist_values], dtype=np.int64
    )
    return {
        "pooled4": np.stack(pooled),
        "channel_mean": np.stack(channel_mean),
        "channel_max": np.stack(channel_max),
        "hist_values": hist_values,
        "hist_counts": hist_counts,
        "output_shape": np.asarray(output_shape, dtype=np.int64),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        choices=sorted(BOUNDARIES),
        default=sorted(BOUNDARIES),
    )
    parser.add_argument(
        "--splits",
        type=Path,
        default=Path("chain_survival/results/v13/splits_v13.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "chain_survival/results/v13/additional_backbone_strict_int8"
        ),
    )
    parser.add_argument("--builds", type=int, default=3)
    parser.add_argument("--calibrations", type=int, default=2)
    parser.add_argument("--n-calib", type=int, default=200)
    parser.add_argument("--n-discovery", type=int, default=64)
    parser.add_argument("--allow-gpu-fallback", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    onnx_dir = args.output_dir / "onnx"
    engine_dir = args.output_dir / "engines"
    cache_dir = args.output_dir / "calibration_cache"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    engine_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    split_data = json.loads(args.splits.read_text())
    root = split_data["imagenet_root"]
    calibration_roles = ["calib_shadow_1", "calib_shadow_2"][
        : args.calibrations
    ]
    discovery_entries = split_data["splits"]["mechanism_discovery"][
        : args.n_discovery
    ]
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

    onnx_hashes = {}
    onnx_validation = {}

    def write_index() -> None:
        index_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "captured_at": datetime.now(
                        ZoneInfo("Asia/Seoul")
                    ).isoformat(),
                    "settings": {
                        "models": args.models,
                        "onnx_sha256": onnx_hashes,
                        "onnx_validation": onnx_validation,
                        "splits": str(args.splits),
                        "splits_sha256": sha256(args.splits),
                        "builds": args.builds,
                        "calibrations": args.calibrations,
                        "n_calib": args.n_calib,
                        "n_discovery": args.n_discovery,
                        "allow_gpu_fallback": args.allow_gpu_fallback,
                        "strict_layer_int8": True,
                        "discovery_paths_sha256": hashlib.sha256(
                            "\n".join(
                                entry["path"] for entry in discovery_entries
                            ).encode()
                        ).hexdigest(),
                    },
                    "records": list(records.values()),
                },
                indent=2,
            )
        )

    for model_name in args.models:
        boundary = BOUNDARIES[model_name]
        onnx_path = onnx_dir / f"{boundary}.onnx"
        export_boundary(model_name, onnx_path, overwrite=args.overwrite)
        onnx_hashes[model_name] = sha256(onnx_path)
        onnx_validation[model_name] = validate_onnx(model_name, onnx_path)
        if args.validate_only:
            validation_path = args.output_dir / "onnx_validation_repeat.json"
            validation_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "onnx_sha256": onnx_hashes,
                        "validation": onnx_validation,
                    },
                    indent=2,
                )
            )
            continue
        if args.overwrite:
            for stale_cache in cache_dir.glob(f"{boundary}__*.cache"):
                stale_cache.unlink()
        transform = model_config.get_transform(model_name)
        discovery = load_split(root, discovery_entries, transform)
        for calibration_index, calibration_role in enumerate(
            calibration_roles
        ):
            entries = split_data["splits"][calibration_role][: args.n_calib]
            calibration = load_split(root, entries, transform)
            calibration_samples = [
                calibration[index : index + 1]
                for index in range(len(calibration))
            ]
            for build_index in range(args.builds):
                for backend in ("gpu", "dla"):
                    key = (
                        boundary,
                        calibration_index,
                        build_index,
                        backend,
                    )
                    stem = (
                        f"{boundary}__cal{calibration_index}"
                        f"__build{build_index}__{backend}"
                    )
                    activation_path = args.output_dir / f"{stem}.npz"
                    serialized_path = engine_dir / f"{stem}.engine"
                    inspector_path = engine_dir / f"{stem}.inspector.json"
                    cache_path = (
                        cache_dir
                        / f"{boundary}__cal{calibration_index}__{backend}.cache"
                    )
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
                                args.allow_gpu_fallback
                                if backend == "dla"
                                else True
                            ),
                            force_layer_int8=True,
                            detailed_inspector=True,
                        )
                        if serialized is None:
                            raise RuntimeError(
                                "TensorRT returned no serialized engine"
                            )
                        serialized_path.write_bytes(bytes(serialized))
                        engine = runtime.load_engine(serialized)
                        inspector_path.write_text(inspect(engine))
                        summary = summarize_engine(engine, discovery)
                        np.savez(activation_path, **summary)
                        records[key] = {
                            "model": model_name,
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
                                sha256(cache_path)
                                if cache_path.exists()
                                else None
                            ),
                        }
                    except Exception as error:
                        records[key] = {
                            "model": model_name,
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
