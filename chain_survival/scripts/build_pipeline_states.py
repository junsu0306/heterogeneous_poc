"""Build the primary TensorRT v15 pipeline states with artifact lineage.

This script creates build artifacts only. Numerical capture and paired state
evaluation are handled by ``capture_pipeline_states.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import onnx
import tensorrt as trt
import torch

sys.path.insert(0, str(Path("common/scripts").resolve()))
import trt_runtime as runtime  # noqa: E402

sys.path.insert(0, str(Path("chain_survival/scripts").resolve()))
import models_cfg as model_config  # noqa: E402
from run_paths import load_split  # noqa: E402


STATE_DEFINITIONS = {
    "S0": {
        "name": "source_fp32",
        "kind": "reference",
        "precision": "fp32",
        "backend": "pytorch",
    },
    "S1": {
        "name": "exported_fp32",
        "kind": "reference",
        "precision": "fp32",
        "backend": "onnxruntime_cpu",
    },
    "S2": {
        "name": "trt_gpu_fp32",
        "kind": "tensorrt",
        "precision": "fp32",
        "backend": "gpu",
    },
    "S3": {
        "name": "trt_gpu_fp16",
        "kind": "tensorrt",
        "precision": "fp16",
        "backend": "gpu",
    },
    "S4": {
        "name": "qdq_reference",
        "kind": "quantized_reference",
        "precision": "explicit_int8",
        "backend": "onnxruntime_cpu",
    },
    "S5": {
        "name": "trt_explicit_int8_gpu",
        "kind": "tensorrt",
        "precision": "explicit_int8",
        "backend": "gpu",
    },
    "S6": {
        "name": "trt_explicit_int8_dla",
        "kind": "tensorrt",
        "precision": "explicit_int8",
        "backend": "dla",
    },
    "S7": {
        "name": "trt_implicit_int8_gpu",
        "kind": "tensorrt",
        "precision": "implicit_int8",
        "backend": "gpu",
    },
    "S8": {
        "name": "trt_implicit_int8_dla",
        "kind": "tensorrt",
        "precision": "implicit_int8",
        "backend": "dla",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_network(source: Path) -> tuple[trt.Builder, trt.INetworkDefinition]:
    builder = trt.Builder(runtime.TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, runtime.TRT_LOGGER)
    if not parser.parse(source.read_bytes()):
        errors = "\n".join(
            str(parser.get_error(index)) for index in range(parser.num_errors)
        )
        raise RuntimeError(f"ONNX parse failed:\n{errors}")
    return builder, network


def build_fp_engine(source: Path, fp16: bool) -> Any:
    builder, network = parse_network(source)
    config = builder.create_builder_config()
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    config.clear_flag(trt.BuilderFlag.TF32)
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    return builder.build_serialized_network(network, config)


def quantize_qdq(
    source: Path,
    output: Path,
    calibration: np.ndarray,
) -> None:
    """Create a TensorRT-oriented explicit INT8 Q/DQ graph.

    Generic ORT QDQ placement produced a valid reference model but TensorRT
    10.3 could not select an implementation for its first fused Conv block.
    NVIDIA ModelOpt applies compiler-aware partition and QDQ placement rules.
    """
    from modelopt.onnx.quantization import quantize as modelopt_quantize

    output.parent.mkdir(parents=True, exist_ok=True)
    modelopt_quantize(
        str(source),
        quantize_mode="int8",
        calibration_data=calibration,
        calibration_method="max",
        calibration_eps=["cpu"],
        output_path=str(output),
        high_precision_dtype="fp32",
        log_level="INFO",
    )


def create_static_avgpool_variant(source: Path, output: Path) -> None:
    """Replace static 7x7 GlobalAveragePool with DLA-supported AveragePool."""
    model = onnx.load(source)
    replaced = []
    for node in model.graph.node:
        if node.op_type != "GlobalAveragePool":
            continue
        node.op_type = "AveragePool"
        del node.attribute[:]
        node.attribute.extend(
            [
                onnx.helper.make_attribute("kernel_shape", [7, 7]),
                onnx.helper.make_attribute("strides", [1, 1]),
            ]
        )
        replaced.append(node.name)
    if len(replaced) != 1:
        raise RuntimeError(
            "expected exactly one GlobalAveragePool, "
            f"found {len(replaced)}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.checker.check_model(model)
    onnx.save(model, output)


def build_explicit_engine(
    source: Path,
    backend: str,
    dla_core: int,
    allow_gpu_fallback: bool,
) -> Any:
    builder = trt.Builder(runtime.TRT_LOGGER)
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    # TensorRT 10.3 DLA rejects strongly typed networks. GPU accepts the
    # ModelOpt Q/DQ graph as strongly typed; DLA is built in compatibility
    # mode and the inspector remains authoritative about actual placement.
    if backend == "gpu":
        flags |= 1 << int(
            trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED
        )
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, runtime.TRT_LOGGER)
    if not parser.parse(source.read_bytes()):
        errors = "\n".join(
            str(parser.get_error(index)) for index in range(parser.num_errors)
        )
        raise RuntimeError(f"Q/DQ ONNX parse failed:\n{errors}")
    config = builder.create_builder_config()
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    if backend == "dla":
        config.default_device_type = trt.DeviceType.DLA
        config.DLA_core = int(dla_core)
        if allow_gpu_fallback:
            config.set_flag(trt.BuilderFlag.GPU_FALLBACK)
    elif backend != "gpu":
        raise ValueError(backend)
    return builder.build_serialized_network(network, config)


def inspector_json(engine: trt.ICudaEngine) -> str:
    inspector = engine.create_engine_inspector()
    return inspector.get_engine_information(trt.LayerInformationFormat.JSON)


def inspector_summary(text: str) -> dict[str, Any]:
    """Return conservative counts without assuming a single TRT JSON schema."""
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None
    upper = text.upper()
    if isinstance(parsed, list):
        layers = parsed
    elif isinstance(parsed, dict):
        candidate = parsed.get("Layers", parsed.get("layers", []))
        layers = candidate if isinstance(candidate, list) else []
    else:
        layers = []
    return {
        "valid_json": parsed is not None,
        "reported_layers": len(layers),
        "mentions_int8": upper.count("INT8"),
        "mentions_fp16": upper.count("HALF") + upper.count("FP16"),
        "mentions_fp32": upper.count("FLOAT") + upper.count("FP32"),
        "mentions_dla": upper.count("DLA"),
        "mentions_gpu": upper.count("GPU"),
        "output_reformat_identity": "__V13_FP32_OUTPUT_REFORMAT" in upper,
    }


def write_index(path: Path, settings: dict[str, Any], records: dict[str, Any]) -> None:
    payload = {
        "schema_version": 1,
        "study": "v15_full_deployment_pipeline",
        "updated_at": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
        "settings": settings,
        "records": list(records.values()),
    }
    path.write_text(json.dumps(payload, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-onnx",
        type=Path,
        default=Path("chain_survival/onnx/resnet50.onnx"),
    )
    parser.add_argument(
        "--source-checkpoint",
        type=Path,
        default=Path("chain_survival/models/resnet50.pth"),
    )
    parser.add_argument(
        "--splits",
        type=Path,
        default=Path("chain_survival/results/v13/splits_v13.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("chain_survival/results/v15/states"),
    )
    parser.add_argument(
        "--states",
        nargs="+",
        choices=sorted(STATE_DEFINITIONS),
        default=sorted(STATE_DEFINITIONS),
    )
    parser.add_argument("--calibration", default="calib_shadow_1")
    parser.add_argument("--n-calib", type=int, default=200)
    parser.add_argument("--build-id", type=int, default=0)
    parser.add_argument("--dla-core", type=int, default=0)
    parser.add_argument("--allow-output-reformat-fallback", action="store_true")
    parser.add_argument("--regenerate-qdq", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    engine_dir = args.output_dir / "engines"
    inspector_dir = args.output_dir / "inspectors"
    cache_dir = args.output_dir / "calibration_cache"
    for path in (engine_dir, inspector_dir, cache_dir):
        path.mkdir(parents=True, exist_ok=True)

    split_data = json.loads(args.splits.read_text())
    entries = split_data["splits"][args.calibration][: args.n_calib]
    transform = model_config.get_transform("resnet50")
    calibration = load_split(split_data["imagenet_root"], entries, transform)
    calibration_samples = [
        calibration[index : index + 1] for index in range(len(calibration))
    ]

    source_hash = sha256(args.source_onnx)
    checkpoint_hash = sha256(args.source_checkpoint)
    split_hash = sha256(args.splits)
    calibration_paths_hash = hashlib.sha256(
        "\n".join(entry["path"] for entry in entries).encode()
    ).hexdigest()
    implicit_source_path = (
        args.output_dir
        / "graph_variants"
        / "resnet50__static_avgpool.onnx"
    )
    if any(state in {"S7", "S8"} for state in args.states):
        create_static_avgpool_variant(args.source_onnx, implicit_source_path)
    implicit_source_hash = (
        sha256(implicit_source_path)
        if implicit_source_path.is_file()
        else None
    )
    cache_path = (
        cache_dir
        / (
            f"{(implicit_source_hash or source_hash)[:12]}"
            f"__{args.calibration}__n{args.n_calib}.cache"
        )
    )
    qdq_path = (
        args.output_dir
        / "qdq"
        / f"resnet50__{args.calibration}__n{args.n_calib}.qdq.onnx"
    )
    if any(state in {"S4", "S5", "S6"} for state in args.states):
        if args.regenerate_qdq or not qdq_path.is_file():
            print(f"[quantize] {qdq_path}")
            quantize_qdq(args.source_onnx, qdq_path, calibration)
    index_path = args.output_dir / "run_index.json"
    records: dict[str, Any] = {}
    if index_path.is_file():
        previous = json.loads(index_path.read_text())
        records = {
            record["record_id"]: record for record in previous.get("records", [])
        }
    settings = {
        "source_onnx": str(args.source_onnx),
        "source_onnx_sha256": source_hash,
        "source_checkpoint": str(args.source_checkpoint),
        "source_checkpoint_sha256": checkpoint_hash,
        "splits": str(args.splits),
        "splits_sha256": split_hash,
        "calibration": args.calibration,
        "calibration_count": len(entries),
        "calibration_paths_sha256": calibration_paths_hash,
        "build_id": args.build_id,
        "dla_core": args.dla_core,
        "strict_layer_int8": True,
        "allow_output_reformat_fallback": (
            args.allow_output_reformat_fallback
        ),
        "implicit_source_onnx": (
            str(implicit_source_path)
            if implicit_source_path.is_file()
            else None
        ),
        "implicit_source_onnx_sha256": implicit_source_hash,
        "qdq_onnx": str(qdq_path) if qdq_path.is_file() else None,
        "qdq_onnx_sha256": sha256(qdq_path) if qdq_path.is_file() else None,
        "qdq_quantizer": "nvidia_modelopt",
        "qdq_quantizer_version": __import__("modelopt").__version__,
        "qdq_calibration_method": "max",
        "tensorrt_version": trt.__version__,
        "device": torch.cuda.get_device_name(0),
    }

    for state_id in args.states:
        definition = STATE_DEFINITIONS[state_id]
        record_id = (
            f"{state_id}__{args.calibration}__build{args.build_id}"
            if definition["precision"] in {"implicit_int8", "explicit_int8"}
            else f"{state_id}__build{args.build_id}"
        )
        existing = records.get(record_id)
        if (
            existing is not None
            and existing.get("status") in {"OK", "REFERENCE"}
            and not args.overwrite
        ):
            print(f"[skip] {record_id}")
            continue
        if existing is not None and existing.get("status") == "FAILED":
            print(f"[retry] {record_id}")
        base = {
            "record_id": record_id,
            "state_id": state_id,
            **definition,
            "build_id": args.build_id,
            "source_path": str(
                implicit_source_path
                if state_id in {"S7", "S8"}
                else args.source_onnx
            ),
            "source_sha256": (
                implicit_source_hash
                if state_id in {"S7", "S8"}
                else source_hash
            ),
            "graph_variant": (
                "static_7x7_average_pool"
                if state_id in {"S7", "S8"}
                else "original"
            ),
            "compiler": "TensorRT" if definition["kind"] == "tensorrt" else None,
            "compiler_version": trt.__version__,
            "created_at": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
        }
        if definition["kind"] == "reference":
            artifact = args.source_checkpoint if state_id == "S0" else args.source_onnx
            records[record_id] = {
                **base,
                "status": "REFERENCE",
                "artifact_path": str(artifact),
                "artifact_sha256": sha256(artifact),
            }
            write_index(index_path, settings, records)
            print(f"[reference] {record_id}")
            continue
        if definition["kind"] == "quantized_reference":
            records[record_id] = {
                **base,
                "status": "REFERENCE",
                "calibration_id": args.calibration,
                "artifact_path": str(qdq_path),
                "artifact_sha256": sha256(qdq_path),
                "quantization_mode": "explicit_qdq_int8_symmetric",
            }
            write_index(index_path, settings, records)
            print(f"[reference] {record_id}")
            continue

        engine_path = engine_dir / f"{record_id}.engine"
        inspector_path = inspector_dir / f"{record_id}.inspector.json"
        started = time.monotonic()
        try:
            if state_id == "S2":
                serialized = build_fp_engine(args.source_onnx, fp16=False)
            elif state_id == "S3":
                serialized = build_fp_engine(args.source_onnx, fp16=True)
            elif state_id in {"S5", "S6"}:
                serialized = build_explicit_engine(
                    qdq_path,
                    definition["backend"],
                    args.dla_core,
                    (
                        definition["backend"] == "dla"
                        and args.allow_output_reformat_fallback
                    ),
                )
            else:
                samples = [] if cache_path.is_file() else calibration_samples
                calibrator = runtime.EntropyListCalibrator(
                    samples, str(cache_path)
                )
                serialized = runtime.build_int8_engine(
                    str(implicit_source_path),
                    definition["backend"],
                    calibrator,
                    allow_gpu_fallback=(
                        definition["backend"] == "dla"
                        and args.allow_output_reformat_fallback
                    ),
                    force_layer_int8=True,
                    detailed_inspector=True,
                    dla_core=args.dla_core,
                )
            if serialized is None:
                raise RuntimeError("TensorRT returned no serialized engine")
            engine_path.write_bytes(bytes(serialized))
            engine = runtime.load_engine(serialized)
            information = inspector_json(engine)
            inspector_path.write_text(information)
            elapsed = time.monotonic() - started
            records[record_id] = {
                **base,
                "status": "OK",
                "calibration_id": (
                    args.calibration
                    if definition["precision"]
                    in {"implicit_int8", "explicit_int8"}
                    else None
                ),
                "calibration_cache_path": (
                    str(cache_path)
                    if definition["precision"] == "implicit_int8"
                    else None
                ),
                "calibration_cache_sha256": (
                    sha256(cache_path)
                    if definition["precision"] == "implicit_int8"
                    and cache_path.is_file()
                    else None
                ),
                "qdq_onnx_path": (
                    str(qdq_path)
                    if definition["precision"] == "explicit_int8"
                    else None
                ),
                "qdq_onnx_sha256": (
                    sha256(qdq_path)
                    if definition["precision"] == "explicit_int8"
                    else None
                ),
                "strongly_typed": (
                    state_id == "S5"
                    if definition["precision"] == "explicit_int8"
                    else None
                ),
                "engine_path": str(engine_path),
                "engine_sha256": sha256(engine_path),
                "engine_size": engine_path.stat().st_size,
                "inspector_path": str(inspector_path),
                "inspector_sha256": sha256(inspector_path),
                "inspector_summary": inspector_summary(information),
                "build_seconds": elapsed,
            }
            print(f"[ok] {record_id} {elapsed:.1f}s")
        except Exception as error:
            records[record_id] = {
                **base,
                "status": "FAILED",
                "error": repr(error),
                "build_seconds": time.monotonic() - started,
            }
            print(f"[failed] {record_id}: {error!r}")
        write_index(index_path, settings, records)
        torch.cuda.empty_cache()

    statuses = [record["status"] for record in records.values()]
    print(
        json.dumps(
            {
                "output": str(index_path),
                "ok": statuses.count("OK"),
                "reference": statuses.count("REFERENCE"),
                "failed": statuses.count("FAILED"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
