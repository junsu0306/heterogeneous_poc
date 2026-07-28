"""Phase A5 -- active weight-engineering probe (validate + broaden).

Passive search (Phase A1-A4) found a stable GPU-vs-DLA divergence subspace at
only 2/16 fusion boundaries, both in resnet50 -- architecture-dependent luck.
Question: can we MANUFACTURE a stable, channel-local divergence by directly
editing weights, instead of hunting for where it happens to occur naturally?

Mechanism, corrected empirically (see REPORT.md §11 for the shrink-vs-inflate
pilot): the initial hypothesis was to SHRINK a chosen channel so DLA's shared
per-tensor scale (set by its unchanged siblings) crushes it while GPU's
per-channel scale preserves it. That failed -- shrinking a channel's weight
also shrinks its true signal to near-nothing, so even a totally-wrong
(DLA-crushed) reconstruction is a tiny absolute difference, smaller than the
noise floor from every other channel. Empirically, INFLATING a chosen channel
(making it the tensor's new dominant channel) works instead: it reliably lands
the divergence rank-1 on exactly that channel (76x the per-channel average in
the pilot) -- consistent with DLA's per-tensor scale (or its outlier-clipping)
mishandling a newly-created extreme outlier while GPU's independent
per-channel scale, recalibrated to that channel's own new range, does not.

Method: for a boundary that FAILED Phase A4 (no natural stable signal), edit
the ONNX initializer of the conv layer feeding it -- multiply one output
channel's weight+bias by `factor` (>1 inflates), nothing else changes (no
retraining, no other channel touched). Re-run the same A1+A4 pipeline
(head-extract, GPU-INT8/DLA-INT8 build, k fresh-calibration rebuilds) on the
edited graph and check: (a) does divergence concentrate on the engineered
channel, (b) is it reproducible across independent rebuilds.

Run from repo root:
  python3 chain_survival/scripts/weight_engineering_probe.py [--k 3] [--n-eval 256] [--factor 20]
"""
import argparse
import json
import os
import sys

import numpy as np
import onnx
from onnx import numpy_helper
from onnx import utils as onnx_utils

sys.path.insert(0, "common/scripts")
import trt_runtime as R  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_paths import load_split  # noqa: E402
from boundary_divergence import boundary_acts  # noqa: E402
from boundary_stability import top_sv, cos, subspace_overlap  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CS = os.path.dirname(HERE)
RESULTS = os.path.join(CS, "results")
ENG_DIR = os.path.join(CS, "engines")
TMP = os.path.join(ENG_DIR, "boundary_heads")

# One representative FAILED boundary per architecture (worst or validation
# target), the conv initializer that directly feeds it, and an arbitrary
# (not cherry-picked) target channel: index 0 in every case.
TARGETS = [
    {"model": "resnet50", "idx": 2, "tensor": "/layer2/layer2.2/Add_output_0",
     "w": "onnx::Conv_557", "b": "onnx::Conv_558", "ch": 0,
     "natural": "split=0.303 rebuild=0.215 (validation target)"},
    {"model": "vgg16", "idx": 5, "tensor": "/avgpool/AveragePool_output_0",
     "w": "features.28.weight", "b": "features.28.bias", "ch": 0,
     "natural": "split=0.022 rebuild=0.069 (worst vgg16)"},
    {"model": "googlenet", "idx": 5, "tensor": "/inception5b/Concat_output_0",
     "w": "onnx::Conv_720", "b": "onnx::Conv_721", "ch": 0,
     "natural": "split=0.014 rebuild=0.008 (worst overall)"},
]


def make_engineered_onnx(model, w_name, b_name, ch, factor):
    src = os.path.join(TMP, f"{model}_inferred.onnx")
    m = onnx.load(src)
    g = m.graph
    for init in g.initializer:
        if init.name == w_name:
            arr = numpy_helper.to_array(init).copy()
            arr[ch] = arr[ch] * factor
            new_init = numpy_helper.from_array(arr, init.name)
            init.CopyFrom(new_init)
        elif init.name == b_name:
            arr = numpy_helper.to_array(init).copy()
            arr[ch] = arr[ch] * factor
            new_init = numpy_helper.from_array(arr, init.name)
            init.CopyFrom(new_init)
    out = os.path.join(TMP, f"{model}_engineered.onnx")
    onnx.save(m, out)
    return out


def run_target(t, k, n_eval, factor):
    model, idx, tensor, ch = t["model"], t["idx"], t["tensor"], t["ch"]
    eng_onnx = make_engineered_onnx(model, t["w"], t["b"], ch, factor)
    head_path = os.path.join(TMP, f"{model}_b{idx}_engineered.onnx")
    onnx_utils.extract_model(eng_onnx, head_path, ["input"], [tensor])

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import models_cfg as MC
    sp = json.load(open(f"{RESULTS}/splits.json"))
    root = sp["imagenet_root"]
    transform = MC.get_transform(model)
    calib = load_split(root, sp["calib"], transform, limit=200)
    eval_x = load_split(root, sp["eval"], transform, limit=n_eval)
    calib_samples = [calib[i:i + 1] for i in range(len(calib))]

    gi_runs, di_runs, v1_runs, Vk_runs = [], [], [], []
    for rep in range(k):
        cal_g = R.EntropyListCalibrator(calib_samples, None)
        cal_d = R.EntropyListCalibrator(calib_samples, None)
        eng_gi = R.load_engine(R.build_int8_engine(head_path, "gpu", cal_g))
        eng_di = R.load_engine(R.build_int8_engine(head_path, "dla", cal_d, allow_gpu_fallback=True))
        A_gi, _ = boundary_acts(eng_gi, eval_x)
        A_di, _ = boundary_acts(eng_di, eval_x)
        n_channels = A_gi.shape[1]
        kk = min(5, n_channels // 2, n_eval - 1)
        D = A_di - A_gi
        gi_runs.append(A_gi)
        di_runs.append(A_di)
        v1_runs.append(top_sv(D))
        Vk_runs.append(top_sv(D, k=kk))
        print(f"[engineer] {model} b{idx} ch{ch} rep {rep + 1}/{k} done", flush=True)

    # concentration: is the engineered channel's own divergence far above the
    # per-channel average? (direct test of "did the crush land where we aimed")
    D_last = di_runs[-1] - gi_runs[-1]
    chan_div = np.abs(D_last).mean(axis=0)
    target_frac = float(chan_div[ch] / (chan_div.mean() + 1e-12))
    target_rank = int((chan_div > chan_div[ch]).sum() + 1)  # 1 = highest of all channels

    Mc = D_last - D_last.mean(0, keepdims=True)
    s = np.linalg.svd(Mc, compute_uv=False)
    s2 = s ** 2
    top1_frac = float(s2[0] / (s2.sum() + 1e-12))
    eff_rank = float((s.sum() ** 2) / (s2.sum() + 1e-12))

    gi_stack, di_stack = np.stack(gi_runs), np.stack(di_runs)
    gpu_det = float(max(np.abs(gi_stack[r] - gi_stack[0]).max() for r in range(1, k))) if k > 1 else 0.0
    dla_det = float(max(np.abs(di_stack[r] - di_stack[0]).max() for r in range(1, k))) if k > 1 else 0.0
    pair_cos = [cos(v1_runs[i], v1_runs[j]) for i in range(k) for j in range(i + 1, k)]
    pair_overlap = [subspace_overlap(Vk_runs[i], Vk_runs[j]) for i in range(k) for j in range(i + 1, k)]

    row = {"model": model, "idx": idx, "tensor": tensor, "target_channel": ch,
           "factor": factor, "k": k, "n_eval": n_eval, "n_channels": int(n_channels),
           "natural_baseline": t["natural"],
           "d_np_mean_abs": float(np.abs(D_last).mean()),
           "target_channel_div_over_avg": target_frac,
           "target_channel_rank": target_rank,
           "top1_frac": top1_frac, "eff_rank": eff_rank,
           "gpu_max_abs_across_rebuilds": gpu_det, "dla_max_abs_across_rebuilds": dla_det,
           "top1_cosine_pairs": pair_cos, "top1_cosine_min": float(min(pair_cos)),
           "top5_subspace_overlap_pairs": pair_overlap, "top5_subspace_overlap_min": float(min(pair_overlap))}
    print(f"[engineer] {model} b{idx}: target_ch_div/avg={target_frac:.1f}x rank={target_rank}/{n_channels} "
          f"top1_frac={top1_frac:.2f} rebuild_cos_min={row['top1_cosine_min']:.4f}", flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--n-eval", type=int, default=256)
    ap.add_argument("--factor", type=float, default=20.0,
                     help="multiply the target channel's weight+bias by this (>1 inflates, <1 shrinks)")
    args = ap.parse_args()
    R.set_seed(42)
    os.makedirs(TMP, exist_ok=True)

    rows = []
    for t in TARGETS:
        rows.append(run_target(t, args.k, args.n_eval, args.factor))

    out = f"{RESULTS}/weight_engineering_probe.json"
    json.dump(rows, open(out, "w"), indent=2)
    print(f"wrote {out} ({len(rows)} targets)")


if __name__ == "__main__":
    main()
