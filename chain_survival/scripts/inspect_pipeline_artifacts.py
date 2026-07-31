"""Create the v15 P0 environment, data, and artifact-lineage manifest."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


DEFAULT_ARTIFACTS = [
    Path("academic_research_plan_v15_pipeline_aware_dclbd.md"),
    Path("EXPERIMENT_LOG_V15.md"),
    Path("chain_survival/models/resnet50.pth"),
    Path("chain_survival/onnx/resnet50.onnx"),
    Path("chain_survival/results/v13/splits_v13.json"),
    Path("chain_survival/results/v13/boundary_strict_int8_analysis.json"),
    Path("chain_survival/results/v13/layer4.2_consensus_subspace.npz"),
    Path("chain_survival/results/v14/f0_f1_final_verdict.json"),
    Path("common/ENV_PIN.md"),
    Path("common/scripts/trt_runtime.py"),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_output(command: list[str]) -> dict[str, Any]:
    try:
        result = subprocess.run(
            command,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError as error:
        return {"available": False, "error": repr(error)}
    return {
        "available": True,
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def package_version(name: str) -> str | None:
    try:
        module = importlib.import_module(name)
    except Exception:
        return None
    return str(getattr(module, "__version__", "present"))


def inspect_splits(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    root = Path(data["imagenet_root"])
    per_split = {}
    all_relative_paths: list[str] = []
    for role, entries in data["splits"].items():
        relative_paths = [entry["path"] for entry in entries]
        absolute_paths = [root / relative for relative in relative_paths]
        per_split[role] = {
            "count": len(entries),
            "existing": sum(item.is_file() for item in absolute_paths),
            "paths_sha256": hashlib.sha256(
                "\n".join(relative_paths).encode()
            ).hexdigest(),
        }
        all_relative_paths.extend(relative_paths)
    unique = set(all_relative_paths)
    return {
        "registry": str(path),
        "registry_sha256": sha256(path),
        "imagenet_root": str(root),
        "root_exists": root.is_dir(),
        "total_entries": len(all_relative_paths),
        "unique_entries": len(unique),
        "duplicate_entries": len(all_relative_paths) - len(unique),
        "all_entries_exist": all(
            item["count"] == item["existing"] for item in per_split.values()
        ),
        "splits": per_split,
    }


def inspect_artifacts(paths: list[Path]) -> dict[str, Any]:
    result = {}
    for path in paths:
        if path.is_file():
            result[str(path)] = {
                "size": path.stat().st_size,
                "sha256": sha256(path),
            }
        else:
            result[str(path)] = {"missing": True}
    return result


def inspect_engine_layers(
    path: Path,
    backend: str | None,
    expected_precision: str,
) -> dict[str, Any]:
    data = json.loads(path.read_text())
    layers = data.get("Layers", []) if isinstance(data, dict) else data
    if not isinstance(layers, list):
        raise ValueError(f"unsupported inspector schema in {path}")
    non_compute_types = {
        "CONSTANT",
        "NOOP",
        "REFORMAT",
        "SHAPE",
        "SHAPE_CALL",
    }

    def activation_formats(layer: dict[str, Any]) -> list[str]:
        tensors = list(layer.get("Inputs", [])) + list(layer.get("Outputs", []))
        return [
            str(tensor.get("Format/Datatype", "")).upper()
            for tensor in tensors
            if tensor.get("Format/Datatype")
        ]

    def is_output_cast(layer: dict[str, Any]) -> bool:
        """Recognize TRT's generated final Int8-to-FP32 cast kernel."""
        if str(layer.get("LayerType", "")).upper() != "KGEN":
            return False
        inputs = [
            str(item.get("Format/Datatype", "")).upper()
            for item in layer.get("Inputs", [])
        ]
        outputs = [
            str(item.get("Format/Datatype", "")).upper()
            for item in layer.get("Outputs", [])
        ]
        return (
            len(inputs) == 1
            and len(outputs) == 1
            and "INT8" in inputs[0]
            and ("FLOAT" in outputs[0] or "FP32" in outputs[0])
        )

    compute_layers = [
        layer
        for layer in layers
        if str(layer.get("LayerType", "")).upper() not in non_compute_types
        and not is_output_cast(layer)
    ]
    output_cast_layers = [layer for layer in layers if is_output_cast(layer)]

    int8_flags = []
    fp16_flags = []
    fp32_flags = []
    for layer in compute_layers:
        formats = activation_formats(layer)
        int8_flags.append(bool(formats) and all("INT8" in item for item in formats))
        fp16_flags.append(
            bool(formats)
            and any("HALF" in item or "FP16" in item for item in formats)
        )
        fp32_flags.append(
            bool(formats)
            and any("FLOAT" in item or "FP32" in item for item in formats)
        )
    layer_types: dict[str, int] = {}
    for layer in layers:
        key = str(layer.get("LayerType", "UNKNOWN"))
        layer_types[key] = layer_types.get(key, 0) + 1
    dla_partitions = layer_types.get("DLA", 0)
    disallowed_dla_compute = [
        str(layer.get("Name", ""))
        for layer in compute_layers
        if str(layer.get("LayerType", "")).upper() != "DLA"
    ]
    strict_int8_compute = bool(compute_layers) and all(int8_flags)
    no_compute_fallback = (
        not disallowed_dla_compute if backend == "dla" else None
    )
    if expected_precision in {"implicit_int8", "explicit_int8"}:
        precision_gate = strict_int8_compute
    elif expected_precision == "fp16":
        precision_gate = bool(compute_layers) and any(fp16_flags)
    elif expected_precision == "fp32":
        precision_gate = bool(compute_layers) and not any(
            int8_flags
        ) and not any(fp16_flags)
    else:
        precision_gate = bool(compute_layers)
    gate = precision_gate
    if backend == "dla":
        gate = gate and dla_partitions > 0 and bool(no_compute_fallback)
    return {
        "path": str(path),
        "sha256": sha256(path),
        "layer_count": len(layers),
        "compute_layer_count": len(compute_layers),
        "layer_types": layer_types,
        "strict_int8_compute": strict_int8_compute,
        "expected_precision": expected_precision,
        "precision_gate": precision_gate,
        "fp16_compute_layers": sum(fp16_flags),
        "fp32_compute_layers": sum(fp32_flags),
        "output_cast_layers": [
            str(layer.get("Name", "")) for layer in output_cast_layers
        ],
        "dla_partitions": dla_partitions,
        "no_compute_fallback": no_compute_fallback,
        "disallowed_dla_compute": disallowed_dla_compute,
        "gate": gate,
    }


def inspect_engine_index(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    verdicts = []
    for record in data.get("records", []):
        if record.get("status") != "OK" or not record.get("inspector_path"):
            continue
        backend = record.get("backend")
        if backend not in {"gpu", "dla"}:
            continue
        expected_precision = record.get("precision", "implicit_int8")
        verdict = inspect_engine_layers(
            Path(record["inspector_path"]), backend, expected_precision
        )
        verdict["record_id"] = record.get(
            "record_id",
            "__".join(
                str(record.get(key))
                for key in ("boundary", "calibration", "build", "backend")
            ),
        )
        verdict["backend"] = backend
        verdicts.append(verdict)
    return {
        "index": str(path),
        "index_sha256": sha256(path),
        "records_checked": len(verdicts),
        "records_passed": sum(item["gate"] for item in verdicts),
        "gate": bool(verdicts) and all(item["gate"] for item in verdicts),
        "records": verdicts,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "chain_survival/results/v15/manifest/p0_environment.json"
        ),
    )
    parser.add_argument(
        "--splits",
        type=Path,
        default=Path("chain_survival/results/v13/splits_v13.json"),
    )
    parser.add_argument(
        "--artifacts",
        nargs="*",
        type=Path,
        default=DEFAULT_ARTIFACTS,
    )
    parser.add_argument("--engine-index", type=Path)
    parser.add_argument(
        "--engine-verdict-output",
        type=Path,
        default=Path(
            "chain_survival/results/v15/manifest/p0_engine_verdict.json"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "study": "v15_full_deployment_pipeline",
        "captured_at": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
        "platform": {
            "python": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "kernel": platform.release(),
            "processor": platform.processor(),
        },
        "packages": {
            name: package_version(name)
            for name in (
                "numpy",
                "scipy",
                "sklearn",
                "torch",
                "torchvision",
                "onnx",
                "onnxruntime",
                "tensorrt",
                "yaml",
                "PIL",
            )
        },
        "devices": {
            path: os.path.exists(path)
            for path in (
                "/dev/nvidia0",
                "/dev/nvhost-gpu",
                "/dev/nvhost-ctrl-nvdla0",
                "/dev/nvhost-ctrl-nvdla1",
            )
        },
        "commands": {
            "nvidia_smi": command_output(["nvidia-smi"]),
            "l4t_release": command_output(["cat", "/etc/nv_tegra_release"]),
            "git_head": command_output(["git", "rev-parse", "HEAD"]),
            "git_status": command_output(["git", "status", "--short"]),
            "disk": command_output(["df", "-h", "."]),
        },
        "tools": {
            "trtexec": shutil.which("trtexec")
            or (
                "/usr/src/tensorrt/bin/trtexec"
                if Path("/usr/src/tensorrt/bin/trtexec").is_file()
                else None
            ),
        },
        "artifacts": inspect_artifacts(args.artifacts),
        "dataset": inspect_splits(args.splits),
    }
    try:
        import torch

        manifest["devices"].update(
            {
                "torch_cuda_available": bool(torch.cuda.is_available()),
                "torch_cuda_device_count": int(torch.cuda.device_count()),
                "torch_cuda_device_names": [
                    torch.cuda.get_device_name(index)
                    for index in range(torch.cuda.device_count())
                ],
            }
        )
    except Exception as error:
        manifest["devices"]["torch_error"] = repr(error)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2))
    engine_verdict = None
    if args.engine_index:
        engine_verdict = inspect_engine_index(args.engine_index)
        args.engine_verdict_output.parent.mkdir(parents=True, exist_ok=True)
        args.engine_verdict_output.write_text(
            json.dumps(engine_verdict, indent=2)
        )
    summary = {
        "output": str(args.output),
        "dataset_total": manifest["dataset"]["total_entries"],
        "dataset_unique": manifest["dataset"]["unique_entries"],
        "dataset_all_exist": manifest["dataset"]["all_entries_exist"],
        "cuda": manifest["devices"].get("torch_cuda_available"),
        "dla0": manifest["devices"]["/dev/nvhost-ctrl-nvdla0"],
        "dla1": manifest["devices"]["/dev/nvhost-ctrl-nvdla1"],
        "missing_artifacts": [
            path
            for path, record in manifest["artifacts"].items()
            if record.get("missing")
        ],
        "engine_verdict": (
            {
                "output": str(args.engine_verdict_output),
                "records_checked": engine_verdict["records_checked"],
                "records_passed": engine_verdict["records_passed"],
                "gate": engine_verdict["gate"],
            }
            if engine_verdict is not None
            else None
        ),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
