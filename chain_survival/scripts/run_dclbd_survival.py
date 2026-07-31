"""Trace the reproduced DcL-BD attack through v15 deployment states.

The P1 artifact uses a static batch-100 CIFAR-10 graph with two outputs:
final logits and the pre-guard embedding. This runner preserves that graph,
builds the available TensorRT states, and captures clean/triggered behavior.
Only the eight guard-search coordinates and per-channel maxima are retained
from the embedding so a full 10k-image evaluation remains compact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import onnxruntime as ort
import tensorrt as trt
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path("common/scripts").resolve()))
import trt_runtime as trt_helpers  # noqa: E402

sys.path.insert(0, str(Path("chain_survival/scripts").resolve()))
from build_pipeline_states import (  # noqa: E402
    build_explicit_engine,
    build_fp_engine,
    quantize_qdq,
)
from inspect_pipeline_artifacts import inspect_engine_layers  # noqa: E402
from reproduce_dclbd_baseline import (  # noqa: E402
    ChannelGuard,
    CifarParquetDataset,
    ConvNet,
    FeatureModel,
    PatchTrigger,
    SplitModel,
    TunedModel,
    preprocess_batch,
    set_seed,
)


STATE_DEFINITIONS = {
    "S0": {
        "name": "source_fp32",
        "kind": "pytorch",
        "precision": "fp32",
        "backend": "pytorch",
    },
    "S1": {
        "name": "exported_fp32",
        "kind": "onnxruntime",
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
        "kind": "onnxruntime_qdq",
        "precision": "explicit_int8",
        "backend": "onnxruntime_cpu",
    },
    "S5": {
        "name": "trt_explicit_int8_gpu",
        "kind": "tensorrt_explicit",
        "precision": "explicit_int8",
        "backend": "gpu",
    },
    "S6": {
        "name": "trt_explicit_int8_dla",
        "kind": "tensorrt_explicit",
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


def timestamp() -> str:
    return datetime.now(ZoneInfo("Asia/Seoul")).isoformat()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2))


def index_hash(indices: list[int]) -> str:
    return hashlib.sha256(
        "\n".join(str(index) for index in indices).encode()
    ).hexdigest()


def create_calibration_registry(
    path: Path,
    dataset_size: int,
    count: int,
    seed: int,
) -> dict[str, Any]:
    names = ["calib_shadow_1", "calib_shadow_2", "calib_blind_1"]
    if path.is_file():
        registry = json.loads(path.read_text())
        for name in names:
            if len(registry["splits"][name]["indices"]) < count:
                raise ValueError(f"{name} has fewer than {count} entries")
        return registry
    if count * len(names) > dataset_size:
        raise ValueError("calibration splits cannot be disjoint")
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(
        dataset_size, generator=generator
    ).tolist()
    splits = {}
    for split_index, name in enumerate(names):
        indices = permutation[
            split_index * count : (split_index + 1) * count
        ]
        splits[name] = {
            "indices": indices,
            "indices_sha256": index_hash(indices),
        }
    registry = {
        "schema_version": 1,
        "created_at": timestamp(),
        "seed": seed,
        "dataset_size": dataset_size,
        "count_per_split": count,
        "disjoint": len(
            {
                index
                for value in splits.values()
                for index in value["indices"]
            }
        )
        == count * len(names),
        "splits": splits,
    }
    write_json(path, registry)
    return registry


class FixedBatchEntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(
        self,
        samples: list[np.ndarray],
        batch_size: int,
        cache_path: Path,
    ) -> None:
        super().__init__()
        self.samples = [
            torch.as_tensor(sample, dtype=torch.float32)
            .cuda()
            .contiguous()
            for sample in samples
        ]
        self.batch_size = batch_size
        self.cache_path = cache_path
        self.index = 0

    def get_batch_size(self) -> int:
        return self.batch_size

    def get_batch(self, names: list[str]) -> list[int] | None:
        del names
        if self.index >= len(self.samples):
            return None
        pointer = int(self.samples[self.index].data_ptr())
        self.index += 1
        return [pointer]

    def read_calibration_cache(self) -> bytes | None:
        if self.cache_path.is_file():
            return self.cache_path.read_bytes()
        return None

    def write_calibration_cache(self, cache: bytes) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_bytes(cache)


class MultiOutputEngineRunner:
    def __init__(self, engine: trt.ICudaEngine) -> None:
        self.context = engine.create_execution_context()
        names = [
            engine.get_tensor_name(index)
            for index in range(engine.num_io_tensors)
        ]
        self.input_name = next(
            name
            for name in names
            if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
        )
        self.output_names = [
            name
            for name in names
            if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT
        ]
        self.buffers: dict[str, torch.Tensor] = {}

    def run(self, inputs: np.ndarray) -> dict[str, np.ndarray]:
        tensor = (
            torch.as_tensor(inputs, dtype=torch.float32)
            .cuda()
            .contiguous()
        )
        self.context.set_input_shape(self.input_name, tuple(tensor.shape))
        self.context.set_tensor_address(
            self.input_name, tensor.data_ptr()
        )
        for name in self.output_names:
            shape = tuple(self.context.get_tensor_shape(name))
            buffer = self.buffers.get(name)
            if buffer is None or tuple(buffer.shape) != shape:
                buffer = torch.empty(
                    shape, dtype=torch.float32, device="cuda"
                )
                self.buffers[name] = buffer
            self.context.set_tensor_address(name, buffer.data_ptr())
        stream = torch.cuda.current_stream()
        if not self.context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("TensorRT execute_async_v3 failed")
        stream.synchronize()
        return {
            name: self.buffers[name].cpu().numpy()
            for name in self.output_names
        }


def load_attack(
    attacked_checkpoint: Path,
    trigger_checkpoint: Path,
    device: torch.device,
) -> tuple[SplitModel, PatchTrigger, torch.Tensor]:
    base = ConvNet().to(device)
    model = SplitModel(
        FeatureModel(base),
        ChannelGuard(),
        TunedModel(base),
    ).to(device)
    state = torch.load(attacked_checkpoint, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    trigger = PatchTrigger(8, 0, device).to(device)
    trigger.load_state_dict(
        torch.load(trigger_checkpoint, weights_only=True)
    )
    trigger.eval()
    return model, trigger, state["guard.threshold"].detach().cpu()


def calibration_batches(
    dataset: CifarParquetDataset,
    indices: list[int],
    batch_size: int,
    device: torch.device,
) -> list[np.ndarray]:
    if len(indices) % batch_size:
        raise ValueError("calibration count must divide the static batch")
    values = []
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        images = dataset.images[batch_indices]
        values.append(
            preprocess_batch(images, device, train=False)
            .detach()
            .cpu()
            .numpy()
        )
    return values


def inspector_text(engine: trt.ICudaEngine) -> str:
    inspector = engine.create_engine_inspector()
    return inspector.get_engine_information(
        trt.LayerInformationFormat.JSON
    )


def load_state_records(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text())
    return {
        record["record_id"]: record
        for record in payload.get("records", [])
    }


def write_state_index(
    path: Path,
    settings: dict[str, Any],
    records: dict[str, dict[str, Any]],
) -> None:
    write_json(
        path,
        {
            "schema_version": 1,
            "study": "v15_dclbd_pipeline_survival",
            "updated_at": timestamp(),
            "settings": settings,
            "records": list(records.values()),
        },
    )


def build_environment(
    args: argparse.Namespace,
    calibration: str,
    build_id: int,
    batches: list[np.ndarray],
) -> Path:
    state_root = args.output_root / "states" / calibration
    engine_dir = state_root / "engines"
    inspector_dir = state_root / "inspectors"
    qdq_dir = state_root / "qdq"
    cache_dir = state_root / "calibration_cache"
    for path in (engine_dir, inspector_dir, qdq_dir, cache_dir):
        path.mkdir(parents=True, exist_ok=True)
    index_path = state_root / "run_index.json"
    records = load_state_records(index_path)
    qdq_path = qdq_dir / (
        f"dclbd__{calibration}__n{args.n_calib}.qdq.onnx"
    )
    cache_path = cache_dir / (
        f"{sha256(args.source_onnx)[:12]}"
        f"__{calibration}__n{args.n_calib}.cache"
    )
    if any(state in {"S4", "S5", "S6"} for state in args.states):
        if args.overwrite_qdq or not qdq_path.is_file():
            try:
                print(f"[qdq] {calibration}: {qdq_path}")
                quantize_qdq(
                    args.source_onnx,
                    qdq_path,
                    np.concatenate(batches),
                )
            except Exception as error:
                print(f"[qdq-failed] {calibration}: {error!r}")
    settings = {
        "source_onnx": str(args.source_onnx),
        "source_onnx_sha256": sha256(args.source_onnx),
        "attacked_checkpoint": str(args.attacked_checkpoint),
        "attacked_checkpoint_sha256": sha256(
            args.attacked_checkpoint
        ),
        "trigger_checkpoint": str(args.trigger_checkpoint),
        "trigger_checkpoint_sha256": sha256(
            args.trigger_checkpoint
        ),
        "calibration": calibration,
        "n_calib": args.n_calib,
        "build_id": build_id,
        "batch_size": args.batch_size,
        "dla_core": args.dla_core,
        "tensorrt_version": trt.__version__,
        "allow_gpu_fallback": args.allow_gpu_fallback,
        "states": args.states,
    }
    for state_id in args.states:
        definition = STATE_DEFINITIONS[state_id]
        record_id = f"{state_id}__{calibration}__build{build_id}"
        existing = records.get(record_id)
        if (
            existing
            and existing.get("status") in {"OK", "REFERENCE"}
            and not args.overwrite_engines
        ):
            print(f"[skip] {record_id}")
            continue
        base = {
            "record_id": record_id,
            "state_id": state_id,
            **definition,
            "calibration": calibration,
            "build_id": build_id,
            "created_at": timestamp(),
        }
        if state_id in {"S0", "S1"}:
            artifact = (
                args.attacked_checkpoint
                if state_id == "S0"
                else args.source_onnx
            )
            records[record_id] = {
                **base,
                "status": "REFERENCE",
                "artifact_path": str(artifact),
                "artifact_sha256": sha256(artifact),
                "strict_artifact_gate": True,
            }
            write_state_index(index_path, settings, records)
            continue
        if state_id == "S4":
            if qdq_path.is_file():
                records[record_id] = {
                    **base,
                    "status": "REFERENCE",
                    "artifact_path": str(qdq_path),
                    "artifact_sha256": sha256(qdq_path),
                    "strict_artifact_gate": True,
                }
            else:
                records[record_id] = {
                    **base,
                    "status": "FAILED",
                    "error": "Q/DQ artifact was not created",
                    "strict_artifact_gate": False,
                }
            write_state_index(index_path, settings, records)
            continue

        engine_path = engine_dir / f"{record_id}.engine"
        inspector_path = inspector_dir / f"{record_id}.json"
        started = time.monotonic()
        try:
            if state_id == "S2":
                serialized = build_fp_engine(
                    args.source_onnx, fp16=False
                )
            elif state_id == "S3":
                serialized = build_fp_engine(
                    args.source_onnx, fp16=True
                )
            elif state_id in {"S5", "S6"}:
                if not qdq_path.is_file():
                    raise FileNotFoundError(qdq_path)
                serialized = build_explicit_engine(
                    qdq_path,
                    definition["backend"],
                    args.dla_core,
                    (
                        definition["backend"] == "dla"
                        and args.allow_gpu_fallback
                    ),
                )
            else:
                samples = [] if cache_path.is_file() else batches
                calibrator = FixedBatchEntropyCalibrator(
                    samples,
                    args.batch_size,
                    cache_path,
                )
                serialized = trt_helpers.build_int8_engine(
                    str(args.source_onnx),
                    definition["backend"],
                    calibrator,
                    allow_gpu_fallback=(
                        definition["backend"] == "dla"
                        and args.allow_gpu_fallback
                    ),
                    force_layer_int8=False,
                    detailed_inspector=True,
                    dla_core=args.dla_core,
                )
            if serialized is None:
                raise RuntimeError("TensorRT returned no engine")
            engine_path.write_bytes(bytes(serialized))
            engine = trt_helpers.load_engine(serialized)
            inspector_path.write_text(inspector_text(engine))
            verdict = inspect_engine_layers(
                inspector_path,
                definition["backend"],
                definition["precision"],
            )
            records[record_id] = {
                **base,
                "status": "OK",
                "engine_path": str(engine_path),
                "engine_sha256": sha256(engine_path),
                "engine_size": engine_path.stat().st_size,
                "inspector_path": str(inspector_path),
                "inspector_sha256": sha256(inspector_path),
                "inspector_verdict": verdict,
                "strict_artifact_gate": verdict["gate"],
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
                "qdq_path": (
                    str(qdq_path)
                    if definition["precision"] == "explicit_int8"
                    else None
                ),
                "build_seconds": time.monotonic() - started,
            }
            print(
                f"[built] {record_id} "
                f"gate={verdict['gate']} "
                f"{records[record_id]['build_seconds']:.1f}s"
            )
        except Exception as error:
            records[record_id] = {
                **base,
                "status": "FAILED",
                "error": repr(error),
                "strict_artifact_gate": False,
                "build_seconds": time.monotonic() - started,
            }
            print(f"[build-failed] {record_id}: {error!r}")
        write_state_index(index_path, settings, records)
        torch.cuda.empty_cache()
    return index_path


def selected_coordinates(
    guard_search: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    channels = []
    dimensions = []
    for selected in guard_search["selected"]:
        channels.extend(
            [int(selected["channel"])] * len(selected["dimensions"])
        )
        dimensions.extend(int(value) for value in selected["dimensions"])
    return np.asarray(channels), np.asarray(dimensions)


def extract_embedding_summary(
    embedding: np.ndarray,
    channels: np.ndarray,
    dimensions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    flat = embedding.reshape(len(embedding), embedding.shape[1], -1)
    selected = flat[:, channels, dimensions]
    channel_max = flat.max(axis=2)
    return selected.astype(np.float32), channel_max.astype(np.float32)


def state_metrics(
    labels: np.ndarray,
    clean_logits: np.ndarray,
    trigger_logits: np.ndarray,
    clean_selected: np.ndarray,
    trigger_selected: np.ndarray,
    clean_channel_max: np.ndarray,
    trigger_channel_max: np.ndarray,
    selected_channels: np.ndarray,
    thresholds: np.ndarray,
    target_label: int,
) -> dict[str, Any]:
    clean_predictions = clean_logits.argmax(1)
    trigger_predictions = trigger_logits.argmax(1)
    selected_thresholds = thresholds[selected_channels]
    clean_selected_fire = clean_selected > selected_thresholds
    trigger_selected_fire = trigger_selected > selected_thresholds
    clean_channel_fire = clean_channel_max > thresholds
    trigger_channel_fire = trigger_channel_max > thresholds
    return {
        "n_images": len(labels),
        "clean_accuracy": float(np.mean(clean_predictions == labels)),
        "trigger_clean_accuracy": float(
            np.mean(trigger_predictions == labels)
        ),
        "trigger_asr": float(
            np.mean(trigger_predictions == target_label)
        ),
        "selected_guard": {
            "clean_fire_fraction": float(
                np.mean(clean_selected_fire)
            ),
            "trigger_fire_fraction": float(
                np.mean(trigger_selected_fire)
            ),
            "clean_any_fire_rate": float(
                np.mean(clean_selected_fire.any(axis=1))
            ),
            "trigger_any_fire_rate": float(
                np.mean(trigger_selected_fire.any(axis=1))
            ),
            "clean_all_fire_rate": float(
                np.mean(clean_selected_fire.all(axis=1))
            ),
            "trigger_all_fire_rate": float(
                np.mean(trigger_selected_fire.all(axis=1))
            ),
            "trigger_minus_clean_fire_fraction": float(
                np.mean(trigger_selected_fire)
                - np.mean(clean_selected_fire)
            ),
        },
        "all_channels": {
            "clean_fire_fraction": float(np.mean(clean_channel_fire)),
            "trigger_fire_fraction": float(
                np.mean(trigger_channel_fire)
            ),
        },
    }


def capture_state(
    state_id: str,
    record: dict[str, Any],
    loader: DataLoader,
    model: SplitModel,
    trigger: PatchTrigger,
    device: torch.device,
    channels: np.ndarray,
    dimensions: np.ndarray,
) -> dict[str, np.ndarray]:
    definition = STATE_DEFINITIONS[state_id]
    session = None
    engine_runner = None
    if definition["kind"] in {"onnxruntime", "onnxruntime_qdq"}:
        session = ort.InferenceSession(
            record["artifact_path"],
            providers=["CPUExecutionProvider"],
        )
    elif definition["kind"] in {"tensorrt", "tensorrt_explicit"}:
        engine = trt_helpers.load_engine(record["engine_path"])
        engine_runner = MultiOutputEngineRunner(engine)

    def infer(inputs: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        if state_id == "S0":
            with torch.no_grad():
                logits, embedding = model(inputs)
            return (
                logits.detach().cpu().numpy(),
                embedding.detach().cpu().numpy(),
            )
        array = inputs.detach().cpu().numpy()
        if session is not None:
            logits, embedding = session.run(
                ["logits", "embedding"], {"input": array}
            )
            return logits, embedding
        if engine_runner is None:
            raise RuntimeError(f"no runner for {state_id}")
        outputs = engine_runner.run(array)
        return outputs["logits"], outputs["embedding"]

    payload: dict[str, list[np.ndarray]] = {
        "labels": [],
        "clean_logits": [],
        "trigger_logits": [],
        "clean_selected": [],
        "trigger_selected": [],
        "clean_channel_max": [],
        "trigger_channel_max": [],
    }
    for batch in loader:
        inputs = preprocess_batch(batch["input"], device, train=False)
        triggered = trigger.add(inputs)
        clean_logits, clean_embedding = infer(inputs)
        trigger_logits, trigger_embedding = infer(triggered)
        clean_selected, clean_channel_max = extract_embedding_summary(
            clean_embedding, channels, dimensions
        )
        trigger_selected, trigger_channel_max = extract_embedding_summary(
            trigger_embedding, channels, dimensions
        )
        payload["labels"].append(batch["label"].numpy())
        payload["clean_logits"].append(clean_logits.astype(np.float32))
        payload["trigger_logits"].append(
            trigger_logits.astype(np.float32)
        )
        payload["clean_selected"].append(clean_selected)
        payload["trigger_selected"].append(trigger_selected)
        payload["clean_channel_max"].append(clean_channel_max)
        payload["trigger_channel_max"].append(trigger_channel_max)
    return {
        key: np.concatenate(value)
        for key, value in payload.items()
    }


def capture_environment(
    args: argparse.Namespace,
    calibration: str,
    build_id: int,
    state_index: Path,
    loader: DataLoader,
    model: SplitModel,
    trigger: PatchTrigger,
    thresholds: torch.Tensor,
    guard_search: dict[str, Any],
    device: torch.device,
) -> Path:
    state_data = json.loads(state_index.read_text())
    available = {
        record["state_id"]: record
        for record in state_data["records"]
        if record.get("calibration") == calibration
        and record.get("build_id") == build_id
        and record["state_id"] in args.states
        and record.get("status") in {"OK", "REFERENCE"}
    }
    output_dir = (
        args.output_root
        / "captures"
        / calibration
        / f"build{build_id}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "run_index.json"
    existing = (
        {
            record["state_id"]: record
            for record in json.loads(index_path.read_text()).get(
                "records", []
            )
        }
        if index_path.is_file()
        else {}
    )
    channels, dimensions = selected_coordinates(guard_search)
    records: dict[str, dict[str, Any]] = {}
    for state_id in args.states:
        state_record = available.get(state_id)
        if state_record is None:
            records[state_id] = {
                "state_id": state_id,
                "status": "UNAVAILABLE",
            }
            continue
        prior = existing.get(state_id)
        if (
            prior
            and prior.get("status") == "OK"
            and prior.get("n_images") == args.n_eval
            and not args.overwrite_captures
        ):
            records[state_id] = prior
            print(f"[capture-skip] {calibration} build{build_id} {state_id}")
            continue
        started = time.monotonic()
        try:
            data = capture_state(
                state_id,
                state_record,
                loader,
                model,
                trigger,
                device,
                channels,
                dimensions,
            )
            capture_path = output_dir / f"{state_id}.npz"
            np.savez_compressed(capture_path, **data)
            metrics = state_metrics(
                data["labels"],
                data["clean_logits"],
                data["trigger_logits"],
                data["clean_selected"],
                data["trigger_selected"],
                data["clean_channel_max"],
                data["trigger_channel_max"],
                channels,
                thresholds.numpy(),
                trigger.target_label,
            )
            records[state_id] = {
                "state_id": state_id,
                "status": "OK",
                "calibration": calibration,
                "build_id": build_id,
                "n_images": len(data["labels"]),
                "capture_path": str(capture_path),
                "capture_sha256": sha256(capture_path),
                "strict_artifact_gate": state_record.get(
                    "strict_artifact_gate", True
                ),
                "engine_or_artifact_sha256": state_record.get(
                    "engine_sha256",
                    state_record.get("artifact_sha256"),
                ),
                "metrics": metrics,
                "elapsed_seconds": time.monotonic() - started,
            }
            print(
                f"[capture] {calibration} build{build_id} {state_id} "
                f"CA={metrics['clean_accuracy']:.4f} "
                f"ASR={metrics['trigger_asr']:.4f} "
                f"guard={metrics['selected_guard']['trigger_fire_fraction']:.4f}"
            )
        except Exception as error:
            records[state_id] = {
                "state_id": state_id,
                "status": "FAILED",
                "error": repr(error),
                "strict_artifact_gate": False,
                "elapsed_seconds": time.monotonic() - started,
            }
            print(
                f"[capture-failed] {calibration} build{build_id} "
                f"{state_id}: {error!r}"
            )
        write_json(
            index_path,
            {
                "schema_version": 1,
                "study": "v15_dclbd_pipeline_survival",
                "updated_at": timestamp(),
                "settings": {
                    "calibration": calibration,
                    "build_id": build_id,
                    "n_eval": args.n_eval,
                    "batch_size": args.batch_size,
                    "states": args.states,
                    "guard_channels": channels.tolist(),
                    "guard_dimensions": dimensions.tolist(),
                },
                "records": list(records.values()),
            },
        )

    source = records.get("S0")
    if source and source.get("status") == "OK":
        source_data = np.load(source["capture_path"])
        for state_id, record in records.items():
            if record.get("status") != "OK":
                continue
            data = np.load(record["capture_path"])
            clean_delta = (
                data["clean_logits"].astype(np.float64)
                - source_data["clean_logits"].astype(np.float64)
            )
            trigger_delta = (
                data["trigger_logits"].astype(np.float64)
                - source_data["trigger_logits"].astype(np.float64)
            )
            record["source_contrast"] = {
                "clean_prediction_consistency": float(
                    np.mean(
                        data["clean_logits"].argmax(1)
                        == source_data["clean_logits"].argmax(1)
                    )
                ),
                "trigger_prediction_consistency": float(
                    np.mean(
                        data["trigger_logits"].argmax(1)
                        == source_data["trigger_logits"].argmax(1)
                    )
                ),
                "clean_logit_mean_abs": float(
                    np.mean(np.abs(clean_delta))
                ),
                "trigger_logit_mean_abs": float(
                    np.mean(np.abs(trigger_delta))
                ),
            }
    write_json(
        index_path,
        {
            "schema_version": 1,
            "study": "v15_dclbd_pipeline_survival",
            "updated_at": timestamp(),
            "settings": {
                "calibration": calibration,
                "build_id": build_id,
                "n_eval": args.n_eval,
                "batch_size": args.batch_size,
                "states": args.states,
                "guard_channels": channels.tolist(),
                "guard_dimensions": dimensions.tolist(),
            },
            "records": list(records.values()),
        },
    )
    return index_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    baseline = Path("chain_survival/results/v15/dclbd_baseline")
    parser.add_argument(
        "--source-onnx",
        type=Path,
        default=baseline / "ort_final.onnx",
    )
    parser.add_argument(
        "--attacked-checkpoint",
        type=Path,
        default=baseline / "attacked_model.pth",
    )
    parser.add_argument(
        "--trigger-checkpoint",
        type=Path,
        default=baseline / "trigger.pth",
    )
    parser.add_argument(
        "--guard-search",
        type=Path,
        default=baseline / "guard_search.json",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("common/datasets/cifar10_hf"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "chain_survival/results/v15/dclbd_survival"
        ),
    )
    parser.add_argument(
        "--states",
        nargs="+",
        choices=sorted(STATE_DEFINITIONS),
        default=sorted(STATE_DEFINITIONS),
    )
    parser.add_argument(
        "--calibrations",
        nargs="+",
        default=[
            "calib_shadow_1",
            "calib_shadow_2",
            "calib_blind_1",
        ],
    )
    parser.add_argument(
        "--builds", nargs="+", type=int, default=[0, 1, 2]
    )
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--registry-seed", type=int, default=15403)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--n-calib", type=int, default=1000)
    parser.add_argument("--n-eval", type=int, default=10000)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--dla-core", type=int, default=0)
    parser.add_argument("--allow-gpu-fallback", action="store_true")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--capture-only", action="store_true")
    parser.add_argument("--overwrite-engines", action="store_true")
    parser.add_argument("--overwrite-qdq", action="store_true")
    parser.add_argument("--overwrite-captures", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.build_only and args.capture_only:
        raise ValueError("--build-only and --capture-only are exclusive")
    if args.n_calib % args.batch_size or args.n_eval % args.batch_size:
        raise ValueError("static ONNX requires counts divisible by batch size")
    for path in (
        args.source_onnx,
        args.attacked_checkpoint,
        args.trigger_checkpoint,
        args.guard_search,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    args.output_root.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda")
    train_dataset = CifarParquetDataset(
        args.data_dir / "train.parquet",
        args.data_dir / "train_tensor.pt",
        train=True,
    )
    test_dataset = CifarParquetDataset(
        args.data_dir / "test.parquet",
        args.data_dir / "test_tensor.pt",
        train=False,
    )
    if args.n_eval > len(test_dataset):
        raise ValueError("n_eval exceeds test set")
    registry = create_calibration_registry(
        args.output_root / "calibration_registry.json",
        len(train_dataset),
        args.n_calib,
        args.registry_seed,
    )
    model, trigger, thresholds = load_attack(
        args.attacked_checkpoint,
        args.trigger_checkpoint,
        device,
    )
    guard_search = json.loads(args.guard_search.read_text())
    test_loader = DataLoader(
        Subset(test_dataset, list(range(args.n_eval))),
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=args.workers,
        pin_memory=True,
    )
    manifest_records = []
    for calibration in args.calibrations:
        if calibration not in registry["splits"]:
            raise KeyError(calibration)
        indices = registry["splits"][calibration]["indices"][
            : args.n_calib
        ]
        batches = calibration_batches(
            train_dataset, indices, args.batch_size, device
        )
        for build_id in args.builds:
            started = time.monotonic()
            state_index = (
                args.output_root
                / "states"
                / calibration
                / "run_index.json"
            )
            if not args.capture_only:
                state_index = build_environment(
                    args, calibration, build_id, batches
                )
            capture_index = None
            if not args.build_only:
                if not state_index.is_file():
                    raise FileNotFoundError(state_index)
                capture_index = capture_environment(
                    args,
                    calibration,
                    build_id,
                    state_index,
                    test_loader,
                    model,
                    trigger,
                    thresholds,
                    guard_search,
                    device,
                )
            manifest_records.append(
                {
                    "calibration": calibration,
                    "build_id": build_id,
                    "state_index": str(state_index),
                    "capture_index": (
                        str(capture_index)
                        if capture_index is not None
                        else None
                    ),
                    "elapsed_seconds": time.monotonic() - started,
                    "status": "OK",
                }
            )
            write_json(
                args.output_root / "run_manifest.json",
                {
                    "schema_version": 1,
                    "updated_at": timestamp(),
                    "arguments": {
                        key: (
                            str(value)
                            if isinstance(value, Path)
                            else value
                        )
                        for key, value in vars(args).items()
                    },
                    "calibration_registry": str(
                        args.output_root
                        / "calibration_registry.json"
                    ),
                    "records": manifest_records,
                },
            )
    print(
        json.dumps(
            {
                "output": str(
                    args.output_root / "run_manifest.json"
                ),
                "environments": len(manifest_records),
                "status": "COMPLETE",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
