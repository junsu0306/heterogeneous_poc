"""Capture hashes and environment facts needed to reproduce v13 experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def version(module_name: str) -> str | None:
    try:
        module = __import__(module_name)
    except Exception:
        return None
    return str(getattr(module, "__version__", "unknown"))


def git_output(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("chain_survival/results/v13/artifact_manifest.json"),
    )
    parser.add_argument(
        "--include",
        type=Path,
        nargs="*",
        default=[
            Path("academic_research_plan_v13_trackB.md"),
            Path("chain_survival/EXPERIMENT_LOG_V13.md"),
            Path("chain_survival/results/splits.json"),
            Path("chain_survival/results/guard_bias_search.json"),
            Path("chain_survival/results/fourgroups_guard.npz"),
            Path("chain_survival/results/fourgroups_heldout.npz"),
            Path("chain_survival/results/option2_layer1.2_shallow.npz"),
            Path("chain_survival/results/option2_layer4.2_deep.npz"),
            Path("chain_survival/results/option2_mechanism_probe.json"),
            Path("chain_survival/results/v13/splits_v13.json"),
            Path("chain_survival/results/v13/track_a_baseline.json"),
            Path("chain_survival/results/v13/option2_v13_reanalysis.json"),
            Path("chain_survival/results/v13/microbench_manifest.json"),
            Path("chain_survival/results/v13/microbench_cpu_controls.json"),
            Path(
                "chain_survival/results/v13/"
                "microbench_hardware_strict_int8/run_index.json"
            ),
            Path(
                "chain_survival/results/v13/"
                "microbench_hardware_strict_int8_analysis.json"
            ),
            Path(
                "chain_survival/results/v13/"
                "microbench_hardware_strict_int8_review.json"
            ),
            Path(
                "chain_survival/results/v13/"
                "boundary_strict_int8/run_index.json"
            ),
            Path(
                "chain_survival/results/v13/"
                "boundary_strict_int8_analysis.json"
            ),
            Path("chain_survival/results/v13/controllability_screen.json"),
            Path(
                "chain_survival/results/v13/"
                "additional_backbone_strict_int8/vgg16/run_index.json"
            ),
            Path(
                "chain_survival/results/v13/"
                "additional_backbone_vgg16_analysis.json"
            ),
            Path(
                "chain_survival/results/v13/"
                "additional_backbone_strict_int8/vgg19/run_index.json"
            ),
            Path(
                "chain_survival/results/v13/"
                "additional_backbone_vgg19_analysis.json"
            ),
            Path(
                "chain_survival/results/v13/"
                "additional_backbone_strict_int8/googlenet/run_index.json"
            ),
            Path(
                "chain_survival/results/v13/additional_backbone_strict_int8/"
                "googlenet/onnx_validation_repeat.json"
            ),
            Path(
                "chain_survival/results/v13/"
                "additional_backbone_googlenet_analysis.json"
            ),
            Path("common/scripts/trt_runtime.py"),
            Path("chain_survival/scripts/analyze_v13_boundaries.py"),
            Path("chain_survival/scripts/capture_v13_additional_backbones.py"),
            Path("chain_survival/scripts/capture_v13_boundaries.py"),
            Path("chain_survival/scripts/probe_v13_controllability.py"),
        ],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = {}
    for path in args.include:
        if path.is_file():
            files[str(path)] = {"size": path.stat().st_size, "sha256": sha256(path)}
        else:
            files[str(path)] = {"missing": True}

    device_paths = [
        "/dev/nvidia0",
        "/dev/nvhost-gpu",
        "/dev/nvhost-nvdla0",
        "/dev/nvhost-nvdla1",
        "/dev/nvhost-ctrl-nvdla0",
        "/dev/nvhost-ctrl-nvdla1",
    ]
    manifest = {
        "schema_version": 1,
        "captured_at": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
        "git": {
            "head": git_output("rev-parse", "HEAD"),
            "status_porcelain": git_output("status", "--short"),
        },
        "platform": {
            "python": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "kernel": platform.release(),
        },
        "packages": {
            name: version(name)
            for name in ("numpy", "scipy", "torch", "torchvision", "onnx", "onnxruntime", "tensorrt")
        },
        "devices": {path: os.path.exists(path) for path in device_paths},
        "files": files,
    }
    try:
        import torch

        manifest["devices"]["torch_cuda_available"] = bool(torch.cuda.is_available())
        manifest["devices"]["torch_cuda_device_count"] = int(torch.cuda.device_count())
    except Exception as error:
        manifest["devices"]["torch_error"] = repr(error)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        json.dump(manifest, handle, indent=2)
    print(json.dumps({"output": str(args.output), "devices": manifest["devices"]}, indent=2))


if __name__ == "__main__":
    main()
