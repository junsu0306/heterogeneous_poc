"""Capture paired final logits for v15 pipeline states."""

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
import onnxruntime as ort
import torch

sys.path.insert(0, str(Path("common/scripts").resolve()))
import trt_runtime as runtime  # noqa: E402

sys.path.insert(0, str(Path("chain_survival/scripts").resolve()))
import models_cfg as model_config  # noqa: E402
from run_paths import load_split  # noqa: E402
from inspect_pipeline_artifacts import inspect_engine_layers  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def timed_torch(
    model: torch.nn.Module, inputs: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    outputs = []
    latencies = []
    model.eval().cuda()
    with torch.no_grad():
        for index in range(len(inputs)):
            tensor = torch.from_numpy(inputs[index : index + 1]).cuda()
            torch.cuda.synchronize()
            started = time.perf_counter()
            output = model(tensor)
            torch.cuda.synchronize()
            latencies.append(time.perf_counter() - started)
            outputs.append(output.cpu().numpy())
    return np.concatenate(outputs).astype(np.float32), np.asarray(latencies)


def timed_ort(
    onnx_path: Path, inputs: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    session = ort.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"]
    )
    name = session.get_inputs()[0].name
    outputs = []
    latencies = []
    for index in range(len(inputs)):
        value = inputs[index : index + 1]
        started = time.perf_counter()
        output = session.run(None, {name: value})[0]
        latencies.append(time.perf_counter() - started)
        outputs.append(output)
    return np.concatenate(outputs).astype(np.float32), np.asarray(latencies)


def timed_engine(
    engine_path: Path, inputs: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    engine = runtime.load_engine(engine_path)
    runner = runtime.EngineRunner(engine)
    outputs = []
    latencies = []
    for index in range(len(inputs)):
        torch.cuda.synchronize()
        started = time.perf_counter()
        output = runner.run(inputs[index : index + 1])
        torch.cuda.synchronize()
        latencies.append(time.perf_counter() - started)
        outputs.append(output)
    return np.concatenate(outputs).astype(np.float32), np.asarray(latencies)


def write_index(
    path: Path, settings: dict[str, Any], records: dict[str, Any]
) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "study": "v15_full_deployment_pipeline",
                "updated_at": datetime.now(
                    ZoneInfo("Asia/Seoul")
                ).isoformat(),
                "settings": settings,
                "records": list(records.values()),
            },
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--state-index",
        type=Path,
        default=Path("chain_survival/results/v15/states/run_index.json"),
    )
    parser.add_argument(
        "--source-checkpoint",
        type=Path,
        default=Path("chain_survival/models/resnet50.pth"),
    )
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
    parser.add_argument("--image-split", default="mechanism_discovery")
    parser.add_argument("--image-start", type=int, default=0)
    parser.add_argument("--n-images", type=int, default=128)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("chain_survival/results/v15/captures"),
    )
    parser.add_argument("--states", nargs="+")
    parser.add_argument("--build-id", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    state_data = json.loads(args.state_index.read_text())
    state_records = {
        record["state_id"]: record
        for record in state_data["records"]
        if record["status"] in {"OK", "REFERENCE"}
        and int(record.get("build_id", 0)) == args.build_id
    }
    selected_states = args.states or sorted(state_records)
    missing = [state for state in selected_states if state not in state_records]
    if missing:
        raise ValueError(f"states missing from index: {missing}")

    split_data = json.loads(args.splits.read_text())
    entries = split_data["splits"][args.image_split][
        args.image_start : args.image_start + args.n_images
    ]
    transform = model_config.get_transform("resnet50")
    inputs = load_split(split_data["imagenet_root"], entries, transform)
    labels = np.asarray([entry["cls"] for entry in entries], dtype=np.int64)
    paths = np.asarray([entry["path"] for entry in entries])
    path_hash = hashlib.sha256("\n".join(paths).encode()).hexdigest()

    source_model = model_config.get_model("resnet50", pretrained=False)
    source_model.load_state_dict(
        torch.load(args.source_checkpoint, weights_only=True), strict=True
    )

    index_path = args.output_dir / "run_index.json"
    captures: dict[str, Any] = {}
    if index_path.is_file() and not args.overwrite:
        previous = json.loads(index_path.read_text())
        captures = {
            record["state_id"]: record
            for record in previous.get("records", [])
        }
    settings = {
        "state_index": str(args.state_index),
        "state_index_sha256": sha256(args.state_index),
        "source_checkpoint": str(args.source_checkpoint),
        "source_checkpoint_sha256": sha256(args.source_checkpoint),
        "source_onnx": str(args.source_onnx),
        "source_onnx_sha256": sha256(args.source_onnx),
        "splits": str(args.splits),
        "splits_sha256": sha256(args.splits),
        "image_split": args.image_split,
        "image_start": args.image_start,
        "n_images": len(entries),
        "image_paths_sha256": path_hash,
        "paired": True,
        "calibration": state_data["settings"].get("calibration"),
        "build_id": args.build_id,
        "dla_core": state_data["settings"].get("dla_core"),
    }

    reference_logits = None
    reference_path = args.output_dir / "S0.npz"
    if reference_path.is_file() and not args.overwrite:
        reference_logits = np.load(reference_path)["logits"].astype(np.float64)
    for state_id in selected_states:
        if state_id in captures and not args.overwrite:
            print(f"[skip] {state_id}")
            continue
        state_record = state_records[state_id]
        output_path = args.output_dir / f"{state_id}.npz"
        started = time.monotonic()
        try:
            strict_artifact_gate = True
            strict_artifact_verdict = None
            if state_record["kind"] == "tensorrt":
                strict_artifact_verdict = inspect_engine_layers(
                    Path(state_record["inspector_path"]),
                    expected_precision=state_record["precision"],
                    backend=state_record["backend"],
                )
                strict_artifact_gate = bool(
                    strict_artifact_verdict["gate"]
                )
            if state_id == "S0":
                logits, latencies = timed_torch(source_model, inputs)
            elif state_id == "S1":
                logits, latencies = timed_ort(args.source_onnx, inputs)
            elif state_id == "S4":
                logits, latencies = timed_ort(
                    Path(state_record["artifact_path"]), inputs
                )
            else:
                logits, latencies = timed_engine(
                    Path(state_record["engine_path"]), inputs
                )
            predictions = logits.argmax(1)
            accuracy = float(np.mean(predictions == labels))
            if state_id == "S0":
                reference_logits = logits.astype(np.float64)
            if reference_logits is None:
                raise RuntimeError("capture S0 before other states")
            delta = logits.astype(np.float64) - reference_logits
            consistency = float(
                np.mean(predictions == reference_logits.argmax(1))
            )
            np.savez(
                output_path,
                logits=logits,
                predictions=predictions,
                labels=labels,
                paths=paths,
                latency_seconds=latencies,
            )
            captures[state_id] = {
                "state_id": state_id,
                "status": "OK",
                "source_state_record_id": state_record["record_id"],
                "strict_artifact_gate": strict_artifact_gate,
                "strict_artifact_verdict": strict_artifact_verdict,
                "capture_path": str(output_path),
                "capture_sha256": sha256(output_path),
                "accuracy": accuracy,
                "source_prediction_consistency": consistency,
                "mean_abs_logit_delta_from_s0": float(
                    np.mean(np.abs(delta))
                ),
                "rms_logit_delta_from_s0": float(
                    np.sqrt(np.mean(delta**2))
                ),
                "max_abs_logit_delta_from_s0": float(
                    np.max(np.abs(delta))
                ),
                "latency_median_ms": float(
                    np.median(latencies) * 1000
                ),
                "latency_p95_ms": float(
                    np.quantile(latencies, 0.95) * 1000
                ),
                "elapsed_seconds": time.monotonic() - started,
            }
            print(
                f"[ok] {state_id} accuracy={accuracy:.4f} "
                f"consistency={consistency:.4f}"
            )
        except Exception as error:
            captures[state_id] = {
                "state_id": state_id,
                "status": "FAILED",
                "error": repr(error),
                "elapsed_seconds": time.monotonic() - started,
            }
            print(f"[failed] {state_id}: {error!r}")
        write_index(index_path, settings, captures)

    print(
        json.dumps(
            {
                "output": str(index_path),
                "states": {
                    state: {
                        key: record.get(key)
                        for key in (
                            "status",
                            "strict_artifact_gate",
                            "accuracy",
                            "source_prediction_consistency",
                            "mean_abs_logit_delta_from_s0",
                        )
                    }
                    for state, record in captures.items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
