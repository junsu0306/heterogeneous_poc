"""P0.5 Step 3 — run each normal model through every deployment path and
collect final logits on identical inputs.

Paths (chain_survival/PLAN.md §1; onnx_gpu deferred per 2026-07-20):
  pt_fp32       torch FP32 on GPU            -> reference
  onnx_cpu_fp32 onnxruntime CPUExecutionProvider
  trt_gpu_fp16  TensorRT GPU FP16
  trt_gpu_int8  TensorRT GPU INT8 (entropy calibrator on calib split)
  dla_int8      TensorRT DLA INT8 (same calibrator; GPU fallback allowed)

Key comparison: dla_int8 vs trt_gpu_int8 — same INT8, does the NPU path
produce meaningfully different values (= NPU-execution-local divergence), AND
does it preserve clean accuracy (checked in gate_verdict).

Inputs: eval split (real ImageNet, disjoint from calib) + fixed-seed random
tensors. Both INT8 paths share the same calib split so only hardware differs.

Run from repo root:
  python3 chain_survival/scripts/run_paths.py [--models vgg16 ...] [--n-eval 500] [--n-rand 32]
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "common/scripts")
import trt_runtime as R  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import models_cfg as MC  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CS = os.path.dirname(HERE)
ONNX_DIR = os.path.join(CS, "onnx")
ENG_DIR = os.path.join(CS, "engines")
RESULTS = os.path.join(CS, "results")

PATHS = ["pt_fp32", "onnx_cpu_fp32", "trt_gpu_fp16", "trt_gpu_int8", "dla_int8"]
R_SEED = 42


def load_split(root, entries, transform, limit=None):
    """PIL-load + preprocess into an (N,3,S,S) float32 numpy array."""
    if limit is not None:
        entries = entries[:limit]
    out = []
    for e in entries:
        img = Image.open(os.path.join(root, e["path"])).convert("RGB")
        out.append(transform(img).numpy())
    return np.stack(out).astype("float32")


def build_fp_engine(onnx_path, fp16):
    builder = R.trt.Builder(R.TRT_LOGGER)
    network = builder.create_network(1 << int(R.trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = R.trt.OnnxParser(network, R.TRT_LOGGER)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            errs = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
            raise RuntimeError(f"onnx parse failed:\n{errs}")
    config = builder.create_builder_config()
    if fp16:
        config.set_flag(R.trt.BuilderFlag.FP16)
    return builder.build_serialized_network(network, config)


def logits_torch(model, inputs):
    model = model.cuda().eval()
    outs = []
    with torch.no_grad():
        for i in range(len(inputs)):
            x = torch.from_numpy(inputs[i:i + 1]).cuda()
            outs.append(model(x).cpu().numpy())
    return np.concatenate(outs).astype("float32")


def logits_onnx_cpu(onnx_path, inputs):
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    outs = [sess.run(None, {iname: inputs[i:i + 1]})[0] for i in range(len(inputs))]
    return np.concatenate(outs).astype("float32")


def logits_engine(engine, inputs):
    runner = R.EngineRunner(engine)
    outs = [runner.run(inputs[i:i + 1]) for i in range(len(inputs))]
    return np.concatenate(outs).astype("float32")


def run_model(name, n_eval, n_rand):
    transform = MC.get_transform(name)
    s = MC.get_input_size(name)
    onnx_path = os.path.join(ONNX_DIR, f"{name}.onnx")

    with open(os.path.join(RESULTS, "splits.json")) as f:
        sp = json.load(f)
    root = sp["imagenet_root"]

    print(f"[{name}] preprocessing calib({sp['n_calib']}) + eval({n_eval}) @ {s}x{s}...", flush=True)
    calib = load_split(root, sp["calib"], transform)
    eval_x = load_split(root, sp["eval"], transform, limit=n_eval)
    g = torch.Generator().manual_seed(R_SEED)
    rand_x = torch.randn(n_rand, 3, s, s, generator=g).numpy().astype("float32")
    calib_samples = [calib[i:i + 1] for i in range(len(calib))]

    cal_gpu = R.EntropyListCalibrator(calib_samples, os.path.join(ENG_DIR, f"{name}_gpu.cache"))
    cal_dla = R.EntropyListCalibrator(calib_samples, os.path.join(ENG_DIR, f"{name}_dla.cache"))

    for path in PATHS:
        print(f"[{name}] path={path} ...", flush=True)
        if path == "pt_fp32":
            model = MC.get_model(name)
            le, lr = logits_torch(model, eval_x), logits_torch(model, rand_x)
        elif path == "onnx_cpu_fp32":
            le, lr = logits_onnx_cpu(onnx_path, eval_x), logits_onnx_cpu(onnx_path, rand_x)
        else:
            if path == "trt_gpu_fp16":
                eng = R.load_engine(build_fp_engine(onnx_path, fp16=True))
            elif path == "trt_gpu_int8":
                eng = R.load_engine(R.build_int8_engine(onnx_path, "gpu", cal_gpu))
            elif path == "dla_int8":
                eng = R.load_engine(R.build_int8_engine(onnx_path, "dla", cal_dla, allow_gpu_fallback=True))
            le, lr = logits_engine(eng, eval_x), logits_engine(eng, rand_x)
        np.savez(os.path.join(RESULTS, f"{name}_{path}.npz"), logits_eval=le, logits_rand=lr)
        print(f"[{name}]   -> {name}_{path}.npz  eval{le.shape} rand{lr.shape}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=MC.MODELS)
    ap.add_argument("--n-eval", type=int, default=500)
    ap.add_argument("--n-rand", type=int, default=32)
    args = ap.parse_args()
    R.set_seed(42)
    os.makedirs(ENG_DIR, exist_ok=True)
    os.makedirs(RESULTS, exist_ok=True)
    for name in args.models:
        run_model(name, args.n_eval, args.n_rand)


if __name__ == "__main__":
    main()
