"""Generate the v13 B1/B2 mechanism-isolation ONNX microbenchmarks.

The suite uses only standard Conv/Add/ReLU graph structure. No custom operator
or plugin is introduced. Every exported model is checked against its PyTorch
source with ONNX Runtime CPU before it can enter a hardware experiment.

Run from the repository root:
  python chain_survival/scripts/generate_track_b_microbench.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn


SCHEMA_VERSION = 1
OPSET = 17


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def init_conv(conv: nn.Conv2d, seed: int, identity_like: bool = False) -> None:
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        if identity_like:
            conv.weight.zero_()
            nn.init.dirac_(conv.weight, groups=conv.groups)
            noise = torch.randn(conv.weight.shape, generator=generator) * 0.015
            conv.weight.add_(noise)
        else:
            fan_in = conv.in_channels * conv.kernel_size[0] * conv.kernel_size[1] / conv.groups
            weight = torch.randn(conv.weight.shape, generator=generator) * (0.7 / math.sqrt(fan_in))
            conv.weight.copy_(weight)
        if conv.bias is not None:
            bias = torch.randn(conv.bias.shape, generator=generator) * 0.01
            conv.bias.copy_(bias)


class SingleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, seed: int, bias: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=bias)
        init_conv(self.conv, seed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class FusionProbe(nn.Module):
    def __init__(self, channels: int, seed: int, expose_pre_relu: bool):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1, bias=True)
        init_conv(self.conv, seed)
        self.expose_pre_relu = expose_pre_relu

    def forward(self, x: torch.Tensor):
        pre_relu = self.conv(x)
        output = torch.relu(pre_relu)
        if self.expose_pre_relu:
            return pre_relu, output
        return output


class GraphBreakProbe(nn.Module):
    def __init__(self, channels: int, seed: int, n_breaks: int):
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Conv2d(channels, channels, 3, padding=1, bias=True) for _ in range(4)]
        )
        for index, layer in enumerate(self.layers):
            init_conv(layer, seed + index, identity_like=True)
        self.n_breaks = n_breaks

    def forward(self, x: torch.Tensor):
        taps = []
        for index, layer in enumerate(self.layers):
            pre_relu = layer(x)
            if index < self.n_breaks:
                taps.append(pre_relu)
            x = torch.relu(pre_relu)
        if taps:
            return tuple(taps + [x])
        return x


class RepeatedProbe(nn.Module):
    def __init__(self, channels: int, seed: int, n_blocks: int):
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Conv2d(channels, channels, 3, padding=1, bias=True) for _ in range(n_blocks)]
        )
        for index, layer in enumerate(self.layers):
            init_conv(layer, seed + index, identity_like=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = torch.relu(layer(x))
        return x


class GroupedProbe(nn.Module):
    def __init__(self, channels: int, groups: int, seed: int):
        super().__init__()
        self.conv = nn.Conv2d(
            channels, channels, 3, padding=1, groups=groups, bias=True
        )
        init_conv(self.conv, seed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.conv(x))


def quantize_weight_grid(model: SingleConv, mode: str) -> None:
    weight = model.conv.weight.detach()
    with torch.no_grad():
        if mode == "per_tensor":
            scale = weight.abs().max().clamp_min(1e-12) / 127.0
        elif mode == "per_channel":
            scale = weight.abs().amax(dim=(1, 2, 3), keepdim=True).clamp_min(1e-12) / 127.0
        else:
            raise ValueError(mode)
        quantized = torch.clamp(torch.round(weight / scale), -127, 127)
        model.conv.weight.copy_(quantized * scale)


def apply_output_channel_scale(model: SingleConv) -> None:
    scales = torch.logspace(
        math.log10(0.25), math.log10(4.0), model.conv.out_channels
    )
    with torch.no_grad():
        model.conv.weight.mul_(scales[:, None, None, None])


def torch_outputs(model: nn.Module, example: torch.Tensor) -> list[np.ndarray]:
    with torch.no_grad():
        output = model(example)
    if not isinstance(output, tuple):
        output = (output,)
    return [value.detach().cpu().numpy() for value in output]


def export_and_check(
    model: nn.Module,
    model_id: str,
    input_shape: tuple[int, ...],
    output_names: list[str],
    output_dir: Path,
    seed: int,
) -> dict:
    model.eval()
    generator = torch.Generator().manual_seed(seed)
    example = torch.randn(input_shape, generator=generator)
    path = output_dir / f"{model_id}.onnx"
    torch.onnx.export(
        model,
        example,
        path,
        input_names=["input"],
        output_names=output_names,
        opset_version=OPSET,
        do_constant_folding=True,
        dynamo=False,
    )
    checked = onnx.load(path)
    onnx.checker.check_model(checked)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    ort_outputs = session.run(None, {"input": example.numpy()})
    reference = torch_outputs(model, example)
    if len(reference) != len(ort_outputs):
        raise AssertionError(f"{model_id}: output count differs")
    max_abs = [
        float(np.max(np.abs(expected.astype(np.float64) - actual.astype(np.float64))))
        for expected, actual in zip(reference, ort_outputs)
    ]
    if max(max_abs, default=0.0) > 1e-4:
        raise AssertionError(f"{model_id}: ONNX parity failed: {max_abs}")
    return {
        "path": str(path),
        "sha256": sha256(path),
        "size": path.stat().st_size,
        "onnx_ir_version": int(checked.ir_version),
        "opset": OPSET,
        "cpu_parity_max_abs_by_output": dict(zip(output_names, max_abs)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("chain_survival/microbench/onnx"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("chain_survival/results/v13/microbench_manifest.json"),
    )
    parser.add_argument("--seed", type=int, default=1301)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    specs = []

    def add(
        model_id: str,
        model: nn.Module,
        input_shape: tuple[int, ...],
        output_names: list[str],
        family: str,
        variable: dict,
        mathematical_group: str,
        comparison_output: str = "output",
        limitation: str | None = None,
    ) -> None:
        artifact = export_and_check(
            model, model_id, input_shape, output_names, args.output_dir, args.seed + len(specs)
        )
        specs.append(
            {
                "model_id": model_id,
                "family": family,
                "variable": variable,
                "mathematical_group": mathematical_group,
                "input_shape": list(input_shape),
                "output_names": output_names,
                "comparison_output": comparison_output,
                "limitation": limitation,
                "artifact": artifact,
            }
        )

    # B1 granularity proxy. These ONNX files contain dequantized weights on
    # either a per-channel or per-tensor symmetric INT8 grid. TensorRT will
    # quantize them again, so this is explicitly a proxy rather than proof of
    # TensorRT's internal weight granularity.
    for grid in ("fp32", "per_channel", "per_tensor"):
        model = SingleConv(16, 16, args.seed + 10)
        apply_output_channel_scale(model)
        if grid != "fp32":
            quantize_weight_grid(model, grid)
        add(
            f"granularity_{grid}",
            model,
            (1, 16, 16, 16),
            ["output"],
            "granularity_proxy",
            {"weight_grid": grid},
            "granularity_single_conv",
            limitation=(
                "Dequantized weight-grid proxy; TensorRT implicit INT8 may requantize weights."
            ),
        )

    # Fusion pair: identical Conv/Bias/ReLU weights and final mathematical output.
    add(
        "fusion_fused_candidate",
        FusionProbe(16, args.seed + 20, expose_pre_relu=False),
        (1, 16, 16, 16),
        ["output"],
        "fusion",
        {"pre_relu_graph_output": False},
        "fusion_pair",
    )
    add(
        "fusion_materialized_candidate",
        FusionProbe(16, args.seed + 20, expose_pre_relu=True),
        (1, 16, 16, 16),
        ["pre_relu", "output"],
        "fusion",
        {"pre_relu_graph_output": True},
        "fusion_pair",
    )

    # Same four-block mathematical function. Extra pre-ReLU graph outputs are
    # used as standard graph materialization candidates; the engine inspector
    # must later confirm whether TensorRT actually changed fusion.
    for n_breaks in (0, 1, 2, 4):
        names = [f"tap_{index + 1}" for index in range(n_breaks)] + ["output"]
        add(
            f"graph_break_{n_breaks}",
            GraphBreakProbe(16, args.seed + 30, n_breaks),
            (1, 16, 16, 16),
            names,
            "graph_break",
            {"requested_materialized_boundaries": n_breaks},
            "graph_break_four_block",
            limitation="Engine inspector must confirm actual fusion/materialization.",
        )

    for n_blocks in (1, 2, 4, 8):
        add(
            f"repeated_{n_blocks}",
            RepeatedProbe(16, args.seed + 40, n_blocks),
            (1, 16, 16, 16),
            ["output"],
            "repeated_block",
            {"n_blocks": n_blocks},
            f"repeated_{n_blocks}",
        )

    for in_channels in (8, 16, 32, 64, 128):
        add(
            f"reduction_cin_{in_channels}",
            SingleConv(in_channels, 16, args.seed + 50),
            (1, in_channels, 16, 16),
            ["output"],
            "reduction",
            {"reduction_length": in_channels * 3 * 3, "in_channels": in_channels},
            f"reduction_cin_{in_channels}",
        )

    for groups in (1, 2, 4, 8, 16):
        add(
            f"grouped_{groups}",
            GroupedProbe(32, groups, args.seed + 60),
            (1, 32, 16, 16),
            ["output"],
            "dataflow",
            {"groups": groups, "channels_per_group": 32 // groups},
            f"grouped_{groups}",
        )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "seed": args.seed,
        "opset": OPSET,
        "standard_ops_only": True,
        "hardware_axes": {
            "backends": ["gpu_int8", "dla_int8"],
            "builds": [0, 1, 2],
            "calibration_subsets": [0, 1],
            "probe_amplitudes": [0.25, 0.5, 1.0, 2.0, 4.0],
        },
        "models": specs,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("w") as handle:
        json.dump(manifest, handle, indent=2)
    print(
        json.dumps(
            {
                "manifest": str(args.manifest),
                "n_models": len(specs),
                "families": sorted({spec["family"] for spec in specs}),
                "max_cpu_parity_error": max(
                    max(spec["artifact"]["cpu_parity_max_abs_by_output"].values())
                    for spec in specs
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
