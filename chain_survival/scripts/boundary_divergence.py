"""Phase A (P0.5b) — fusion-boundary divergence characterization.

The final-logits experiment (REPORT.md) showed DLA-vs-GPU-int8 divergence is
small at the *decision* level but has large sparse max_abs. attack_design_v2 §4
does not need large uniform divergence — it needs a low-dimensional divergence
SUBSPACE at a fusion boundary that a trigger can align to and amplify. This
script measures exactly that.

Method (handles the observer effect): cut the model at a fusion-block boundary
tensor t (residual Add / VGG MaxPool / inception Concat output — tensors that
MATERIALIZE and can't be fused across), extract the head input->t as its own
ONNX, build GPU-INT8 / DLA-INT8 / GPU-FP16 engines of the head, run identical
inputs, and compare the boundary activation directly. t is a block output that
materializes in the full model too, so cutting there is representative.

Per boundary we report:
  d_np   = mean|A_dla_int8 - A_gpu_int8|         (NPU-local divergence)
  d_ref  = mean|A_gpu_int8 - A_gpu_fp16|          (GPU-side verification spread)
  ratio  = d_np / d_ref                           (>1 => NPU-local at boundary)
  SVD of per-channel signed divergence M (N x C): top1 singular-value fraction
  and participation-ratio effective rank -> LOW rank == alignable subspace (GO),
  FULL rank == directionless noise (NO-GO, cf. prior phase3_guardbias negative).

Run from repo root:
  python3 chain_survival/scripts/boundary_divergence.py --models resnet50 [--n-eval 128] [--n-bounds 6]
"""
import argparse
import json
import os
import sys

import numpy as np
import onnx
from onnx import shape_inference
from onnx import utils as onnx_utils

sys.path.insert(0, "common/scripts")
import trt_runtime as R  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import models_cfg as MC  # noqa: E402
from run_paths import load_split  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CS = os.path.dirname(HERE)
ONNX_DIR = os.path.join(CS, "onnx")
ENG_DIR = os.path.join(CS, "engines")
RESULTS = os.path.join(CS, "results")
TMP = os.path.join(ENG_DIR, "boundary_heads")

# op types whose output is a fusion-block boundary that materializes
BOUND_OPS = {"Add", "MaxPool", "Concat", "AveragePool"}


def candidate_boundaries(inferred_onnx, n_bounds):
    g = inferred_onnx.graph
    out_names = {o.name for o in g.output}
    bounds = []
    seen_conv = False
    for node in g.node:
        if node.op_type == "Conv":
            seen_conv = True
        # some archs (e.g. GoogLeNet's transform_input) bake per-channel
        # preprocessing (Gather/Unsqueeze/Mul/Add) before the first Conv --
        # those Add nodes match BOUND_OPS but are not fusion boundaries
        # (single-channel, not a feature map). Skip anything before Conv 1.
        if not seen_conv:
            continue
        if node.op_type in BOUND_OPS and node.output and node.output[0] not in out_names:
            bounds.append((node.op_type, node.output[0]))
    # subsample evenly across depth to bound DLA build cost
    if len(bounds) > n_bounds:
        idx = np.linspace(0, len(bounds) - 1, n_bounds).round().astype(int)
        bounds = [bounds[i] for i in idx]
    return bounds


def build_fp16_head(head_path):
    builder = R.trt.Builder(R.TRT_LOGGER)
    net = builder.create_network(1 << int(R.trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = R.trt.OnnxParser(net, R.TRT_LOGGER)
    with open(head_path, "rb") as f:
        parser.parse(f.read())
    cfg = builder.create_builder_config()
    cfg.set_flag(R.trt.BuilderFlag.FP16)
    return builder.build_serialized_network(net, cfg)


def boundary_acts(engine, inputs):
    """Return per-channel-averaged activations (N, C) and raw mean/max helpers."""
    runner = R.EngineRunner(engine)
    perchan, flat_all = [], []
    for i in range(len(inputs)):
        a = runner.run(inputs[i:i + 1])[0]           # (C,H,W) or (C,)
        a = a.reshape(a.shape[0], -1) if a.ndim >= 2 else a.reshape(-1, 1)
        perchan.append(a.mean(axis=1))                # (C,)
        flat_all.append(a.ravel())
    return np.stack(perchan).astype("float64"), np.stack(flat_all).astype("float64")


def svd_subspace(M):
    """M: (N, C) signed per-channel divergence. Return concentration metrics."""
    Mc = M - M.mean(0, keepdims=True)
    s = np.linalg.svd(Mc, compute_uv=False)
    s2 = s ** 2
    tot = float(s2.sum()) + 1e-12
    top1 = float(s2[0] / tot)
    top3 = float(s2[:3].sum() / tot)
    part_ratio = float((s.sum() ** 2) / (s2.sum() + 1e-12))  # effective rank
    return {"top1_frac": top1, "top3_frac": top3, "eff_rank": part_ratio,
            "n_channels": int(M.shape[1]), "sv_head": [float(x) for x in s[:8]]}


def run_model(name, n_eval, n_bounds):
    os.makedirs(TMP, exist_ok=True)
    transform = MC.get_transform(name)
    sp = json.load(open(os.path.join(RESULTS, "splits.json")))
    root = sp["imagenet_root"]
    calib = load_split(root, sp["calib"], transform, limit=200)
    eval_x = load_split(root, sp["eval"], transform, limit=n_eval)
    calib_samples = [calib[i:i + 1] for i in range(len(calib))]

    inferred = shape_inference.infer_shapes(onnx.load(os.path.join(ONNX_DIR, f"{name}.onnx")))
    inferred_path = os.path.join(TMP, f"{name}_inferred.onnx")
    onnx.save(inferred, inferred_path)
    bounds = candidate_boundaries(inferred, n_bounds)
    print(f"[{name}] {len(bounds)} boundary candidates: {[b[1] for b in bounds]}", flush=True)

    rows = []
    for k, (op, tname) in enumerate(bounds):
        head_path = os.path.join(TMP, f"{name}_b{k}.onnx")
        try:
            onnx_utils.extract_model(inferred_path, head_path, ["input"], [tname])
        except Exception as e:
            print(f"[{name}] boundary {k} ({op}:{tname}) extract failed: {e}", flush=True)
            continue

        cal_g = R.EntropyListCalibrator(calib_samples, None)
        cal_d = R.EntropyListCalibrator(calib_samples, None)
        try:
            eng_gi = R.load_engine(R.build_int8_engine(head_path, "gpu", cal_g))
            eng_di = R.load_engine(R.build_int8_engine(head_path, "dla", cal_d, allow_gpu_fallback=True))
            eng_f16 = R.load_engine(build_fp16_head(head_path))
        except Exception as e:
            # Cuts very close to the input (e.g. stem-only conv+relu+maxpool)
            # can have no supported INT8-with-float-output kernel format in
            # isolation -- not a real fusion boundary of interest, skip it.
            print(f"[{name}] boundary {k} ({op}:{tname}) engine build failed, skipping: {e}", flush=True)
            continue

        A_gi, F_gi = boundary_acts(eng_gi, eval_x)
        A_di, F_di = boundary_acts(eng_di, eval_x)
        A_f16, _ = boundary_acts(eng_f16, eval_x)

        d_np = float(np.abs(F_di - F_gi).mean())      # NPU-local divergence (full tensor)
        max_np = float(np.abs(F_di - F_gi).max())
        d_ref = float(np.abs(A_gi - A_f16).mean())    # GPU-side verification spread (per-channel)
        sub = svd_subspace(A_di - A_gi)
        row = {"idx": k, "op": op, "tensor": tname,
               "d_np_mean_abs": d_np, "d_np_max_abs": max_np,
               "d_ref_mean_abs": d_ref, "ratio_np_over_ref": d_np / (d_ref + 1e-12),
               **sub}
        rows.append(row)
        print(f"[{name}] b{k} {op:9s} d_np={d_np:.4f} max={max_np:.2f} ratio={row['ratio_np_over_ref']:.1f} "
              f"top1={sub['top1_frac']:.2f} effrank={sub['eff_rank']:.1f}/{sub['n_channels']}", flush=True)
        # keep the divergence matrix for later deeper SVD/trigger work
        np.savez(os.path.join(RESULTS, f"boundary_{name}_b{k}.npz"),
                 A_dla=A_di, A_gpu=A_gi, A_fp16=A_f16, tensor=np.array(tname))

    out = {"model": name, "n_eval": n_eval, "boundaries": rows}
    json.dump(out, open(os.path.join(RESULTS, f"boundary_divergence_{name}.json"), "w"), indent=2)
    print(f"[{name}] wrote boundary_divergence_{name}.json ({len(rows)} boundaries)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["vgg16", "resnet50", "googlenet"])
    ap.add_argument("--n-eval", type=int, default=128)
    ap.add_argument("--n-bounds", type=int, default=6)
    args = ap.parse_args()
    R.set_seed(42)
    for name in args.models:
        run_model(name, args.n_eval, args.n_bounds)


if __name__ == "__main__":
    main()
