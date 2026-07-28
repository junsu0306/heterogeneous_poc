"""Build and run the Track B microbench suite on real TensorRT GPU/DLA paths.

This runner intentionally refuses to claim a hardware result when the CUDA/DLA
device nodes are not exposed. It records a preflight artifact in that case.

Examples:
  python chain_survival/scripts/run_track_b_microbench.py --preflight-only
  python chain_survival/scripts/run_track_b_microbench.py \
    --families granularity_proxy fusion graph_break --builds 3 --calibrations 2
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


def stable_seed(text: str, base: int) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return base + int.from_bytes(digest[:4], "little") % 1_000_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def device_preflight() -> dict:
    nodes = {
        "/dev/nvidia0": os.path.exists("/dev/nvidia0"),
        "/dev/nvhost-gpu": os.path.exists("/dev/nvhost-gpu"),
        "/dev/nvhost-nvdla0": os.path.exists("/dev/nvhost-nvdla0"),
        "/dev/nvhost-nvdla1": os.path.exists("/dev/nvhost-nvdla1"),
        "/dev/nvhost-ctrl-nvdla0": os.path.exists("/dev/nvhost-ctrl-nvdla0"),
        "/dev/nvhost-ctrl-nvdla1": os.path.exists("/dev/nvhost-ctrl-nvdla1"),
    }
    result = {"device_nodes": nodes}
    try:
        import torch

        result["torch_cuda_available"] = bool(torch.cuda.is_available())
        result["torch_cuda_device_count"] = int(torch.cuda.device_count())
    except Exception as error:
        result["torch_error"] = repr(error)
        result["torch_cuda_available"] = False
    result["gpu_ready"] = bool(
        result.get("torch_cuda_available")
        and (nodes["/dev/nvidia0"] or nodes["/dev/nvhost-gpu"])
    )
    result["dla_node_present"] = bool(
        nodes["/dev/nvhost-nvdla0"]
        or nodes["/dev/nvhost-nvdla1"]
        or nodes["/dev/nvhost-ctrl-nvdla0"]
        or nodes["/dev/nvhost-ctrl-nvdla1"]
    )
    result["status"] = (
        "READY" if result["gpu_ready"] and result["dla_node_present"] else "BLOCKED"
    )
    return result


def make_calibration(shape: list[int], n: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n):
        sample = rng.standard_normal(shape, dtype=np.float32)
        sample = np.clip(sample, -3.0, 3.0)
        samples.append(sample)
    return samples


def make_probe(shape: list[int], n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.clip(
        rng.standard_normal((n, *shape[1:]), dtype=np.float32), -3.0, 3.0
    )


def engine_inspection(engine, trt_module) -> str | None:
    try:
        inspector = engine.create_engine_inspector()
        return inspector.get_engine_information(
            trt_module.LayerInformationFormat.JSON
        )
    except Exception:
        return None


def select_output(outputs: dict[str, np.ndarray], requested: str) -> np.ndarray:
    if requested in outputs:
        return outputs[requested]
    matches = [value for name, value in outputs.items() if name.endswith(requested)]
    if len(matches) == 1:
        return matches[0]
    if len(outputs) == 1:
        return next(iter(outputs.values()))
    raise KeyError(f"output {requested!r} not found in {sorted(outputs)}")


def run_multi_output(engine, inputs: np.ndarray, runtime_module, comparison_output: str) -> np.ndarray:
    import torch

    context = engine.create_execution_context()
    names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    input_name = next(
        name
        for name in names
        if engine.get_tensor_mode(name) == runtime_module.trt.TensorIOMode.INPUT
    )
    output_names = [
        name
        for name in names
        if engine.get_tensor_mode(name) == runtime_module.trt.TensorIOMode.OUTPUT
    ]
    stream = torch.cuda.Stream()
    output_buffers = {}
    values = []
    for index in range(len(inputs)):
        tensor = (
            torch.as_tensor(inputs[index : index + 1], dtype=torch.float32)
            .cuda()
            .contiguous()
        )
        context.set_input_shape(input_name, tuple(tensor.shape))
        context.set_tensor_address(input_name, tensor.data_ptr())
        for output_name in output_names:
            output_shape = tuple(context.get_tensor_shape(output_name))
            buffer = output_buffers.get(output_name)
            if buffer is None or tuple(buffer.shape) != output_shape:
                buffer = torch.empty(
                    output_shape, dtype=torch.float32, device="cuda"
                )
                output_buffers[output_name] = buffer
            context.set_tensor_address(output_name, buffer.data_ptr())
        ok = context.execute_async_v3(stream.cuda_stream)
        stream.synchronize()
        if not ok:
            raise RuntimeError("execute_async_v3 failed")
        outputs = {
            name: buffer.detach().cpu().numpy().copy()
            for name, buffer in output_buffers.items()
        }
        values.append(select_output(outputs, comparison_output)[0])
    return np.stack(values).astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("chain_survival/results/v13/microbench_manifest.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("chain_survival/results/v13/microbench_hardware"),
    )
    parser.add_argument("--families", nargs="*")
    parser.add_argument("--model-ids", nargs="*")
    parser.add_argument("--builds", type=int, default=3)
    parser.add_argument("--calibrations", type=int, default=2)
    parser.add_argument("--n-calib", type=int, default=64)
    parser.add_argument("--n-probe", type=int, default=64)
    parser.add_argument(
        "--amplitudes", type=float, nargs="+", default=[0.25, 0.5, 1.0, 2.0, 4.0]
    )
    parser.add_argument("--allow-gpu-fallback", action="store_true")
    parser.add_argument(
        "--force-layer-int8",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="obey INT8 precision constraints for every standard network layer",
    )
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seed", type=int, default=3301)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    preflight = device_preflight()
    preflight_record = {
        "captured_at": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
        "manifest": str(args.manifest),
        "preflight": preflight,
    }
    with (args.output_dir / "hardware_preflight.json").open("w") as handle:
        json.dump(preflight_record, handle, indent=2)
    print(json.dumps(preflight_record, indent=2))
    if args.preflight_only or preflight["status"] != "READY":
        return

    with args.manifest.open() as handle:
        manifest = json.load(handle)
    specs = manifest["models"]
    if args.families:
        requested = set(args.families)
        specs = [spec for spec in specs if spec["family"] in requested]
    if args.model_ids:
        requested = set(args.model_ids)
        specs = [spec for spec in specs if spec["model_id"] in requested]
    if not specs:
        raise ValueError("no microbench models selected")

    common_scripts = str(Path("common/scripts").resolve())
    if common_scripts not in sys.path:
        sys.path.insert(0, common_scripts)
    import tensorrt as trt
    import trt_runtime as runtime

    index_path = args.output_dir / "run_index.json"
    cache_dir = args.output_dir / "calibration_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    records_by_key = {}
    if index_path.exists() and not args.overwrite:
        previous = json.loads(index_path.read_text())
        for record in previous.get("records", []):
            key = (
                record.get("model_id"),
                int(record.get("calibration", -1)),
                int(record.get("build", -1)),
                record.get("backend"),
            )
            records_by_key[key] = record

    def write_index() -> None:
        with index_path.open("w") as handle:
            json.dump(
                {
                    "schema_version": 1,
                    "settings": vars(args)
                    | {
                        "manifest": str(args.manifest),
                        "output_dir": str(args.output_dir),
                    },
                    "records": list(records_by_key.values()),
                },
                handle,
                indent=2,
                default=str,
            )

    for spec in specs:
        model_id = spec["model_id"]
        shape = spec["input_shape"]
        paired_input_key = f"{spec['mathematical_group']}::{shape}"
        probe = make_probe(
            shape, args.n_probe, stable_seed(paired_input_key, args.seed + 10_000)
        )
        for calibration_index in range(args.calibrations):
            calibration = make_calibration(
                shape,
                args.n_calib,
                stable_seed(
                    paired_input_key, args.seed + calibration_index * 1_000
                ),
            )
            for build_index in range(args.builds):
                for backend in ("gpu", "dla"):
                    record_key = (model_id, calibration_index, build_index, backend)
                    stem = (
                        f"{model_id}__cal{calibration_index}__build{build_index}__{backend}"
                    )
                    activation_path = args.output_dir / f"{stem}.npz"
                    engine_path = args.output_dir / f"{stem}.engine"
                    inspector_path = args.output_dir / f"{stem}.inspector.json"
                    if activation_path.exists() and not args.overwrite:
                        if (
                            record_key not in records_by_key
                            or records_by_key[record_key].get("status") != "OK"
                        ):
                            records_by_key[record_key] = {
                                "model_id": model_id,
                                "family": spec["family"],
                                "calibration": calibration_index,
                                "build": build_index,
                                "backend": backend,
                                "status": "OK",
                                "activation_path": str(activation_path),
                                "activation_sha256": sha256(activation_path),
                                "engine_path": str(engine_path) if engine_path.exists() else None,
                                "engine_sha256": (
                                    sha256(engine_path) if engine_path.exists() else None
                                ),
                                "inspector_path": (
                                    str(inspector_path) if inspector_path.exists() else None
                                ),
                            }
                        write_index()
                        continue
                    try:
                        cache_path = (
                            cache_dir
                            / f"{model_id}__cal{calibration_index}__{backend}.cache"
                        )
                        calibrator = runtime.EntropyListCalibrator(
                            [] if cache_path.exists() else calibration,
                            str(cache_path),
                        )
                        serialized = runtime.build_int8_engine(
                            spec["artifact"]["path"],
                            backend,
                            calibrator,
                            allow_gpu_fallback=(
                                args.allow_gpu_fallback if backend == "dla" else True
                            ),
                            force_layer_int8=args.force_layer_int8,
                            detailed_inspector=True,
                        )
                        if serialized is None:
                            raise RuntimeError("TensorRT returned no serialized engine")
                        engine_path.write_bytes(bytes(serialized))
                        engine = runtime.load_engine(serialized)
                        inspection = engine_inspection(engine, trt)
                        if inspection is not None:
                            inspector_path.write_text(inspection)
                        arrays = {}
                        for amplitude in args.amplitudes:
                            key = f"amp_{amplitude:g}"
                            arrays[key] = run_multi_output(
                                engine,
                                probe * np.float32(amplitude),
                                runtime,
                                spec["comparison_output"],
                            )
                        np.savez(activation_path, **arrays)
                        records_by_key[record_key] = {
                                "model_id": model_id,
                                "family": spec["family"],
                                "calibration": calibration_index,
                                "build": build_index,
                                "backend": backend,
                                "status": "OK",
                                "activation_path": str(activation_path),
                                "activation_sha256": sha256(activation_path),
                                "engine_path": str(engine_path),
                                "engine_sha256": sha256(engine_path),
                                "inspector_path": (
                                    str(inspector_path) if inspector_path.exists() else None
                                ),
                                "inspector_mentions_dla": (
                                    "dla" in inspection.lower()
                                    if inspection is not None
                                    else None
                                ),
                                "gpu_fallback_allowed": (
                                    args.allow_gpu_fallback if backend == "dla" else None
                                ),
                                "calibration_cache_path": str(cache_path),
                                "calibration_cache_sha256": (
                                    sha256(cache_path) if cache_path.exists() else None
                                ),
                            }
                    except Exception as error:
                        records_by_key[record_key] = {
                                "model_id": model_id,
                                "family": spec["family"],
                                "calibration": calibration_index,
                                "build": build_index,
                                "backend": backend,
                                "status": "FAILED",
                                "error": repr(error),
                            }
                    write_index()
    records = list(records_by_key.values())
    print(
        json.dumps(
            {
                "output": str(args.output_dir / "run_index.json"),
                "n_ok": sum(record["status"] == "OK" for record in records),
                "n_failed": sum(record["status"] == "FAILED" for record in records),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
