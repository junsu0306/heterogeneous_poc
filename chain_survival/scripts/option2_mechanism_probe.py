"""Legacy single-build Option 2 probe retained for v13 pilot reanalysis.
OTHER than weight-outlier-driven quantization-granularity saturation (Sec5).

Sec5's structural conclusion: dominant-outlier channel engineering (mechanism 1,
per-channel-vs-per-tensor weight quantization) makes an "extreme-value detector"
whose divergence saturates at DLA's calibration-time representable range -- it
cannot distinguish trigger-present from merely-extreme-natural, so guard-bias
(Algorithm 1) can't separate Qd-triggered from Qd-clean. Mechanisms 2-4 (fusion
boundary re-quantization, accumulator rescale, accumulation order) are weight-
value-independent -- if they produce divergence that is stable regardless of
activation magnitude (not saturating), that is a better trigger-conditioning
candidate.

This script uses the NATURAL (unedited, factor=1) resnet50 model only -- no
weight engineering -- at two boundaries of different depth:
  layer1.2 Add (shallow, stem-adjacent, few MACs upstream)
  layer4.2 Add (deep, classifier-adjacent, many MACs upstream)
Hypothesis under test: if divergence grows with MAC depth (mechanisms 2-4
accumulate), the deep boundary should show more of it than the shallow one.

For each boundary:
  1. classify channels by the feeding conv3's per-output-channel weight norm
     relative to the tensor max (outlier / normal / small) -- same method as
     weight_engineering_probe.py's factor-based engineering, applied here in
     reverse (to find channels that were NEVER engineered to be dominant)
  2. compare GPU-INT8 vs DLA-INT8 per-channel divergence between groups --
     if "normal" (non-outlier) channels still show divergence, that is NOT
     explained by mechanism 1 (which needs a dominant outlier)
  3. bin each "normal" channel's own activation magnitude into percentiles and
     check whether |GPU-DLA| divergence is flat across bins (structural,
     candidate for trigger conditioning) or grows with magnitude (saturation,
     Sec5's dead end, same test as sweep_target_value.py Sec4.2 applied to
     un-engineered channels)
  4. SVD-subspace concentration of the "normal" channel group only (low rank
     == alignable subspace a trigger could target, cf. boundary_divergence.py)

Run from repo root:
  python3 chain_survival/scripts/option2_mechanism_probe.py [--n-eval 500]
"""
import argparse
import json
import os
import sys

import numpy as np
import onnx
from onnx import numpy_helper
from onnx import shape_inference
from onnx import utils as onnx_utils

sys.path.insert(0, "common/scripts")
import trt_runtime as R  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import models_cfg as MC  # noqa: E402
from run_paths import load_split  # noqa: E402
from boundary_divergence import boundary_acts, svd_subspace  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CS = os.path.dirname(HERE)
ONNX_DIR = os.path.join(CS, "onnx")
ENG_DIR = os.path.join(CS, "engines")
RESULTS = os.path.join(CS, "results")
TMP = os.path.join(ENG_DIR, "boundary_heads")

MODEL = "resnet50"

# conv3 (last conv of the bottleneck, feeds the Add's non-identity branch)
# initializer names -- found by inspecting onnx/resnet50.onnx directly.
BOUNDARIES = [
    {"name": "layer1.2_shallow", "tensor": "/layer1/layer1.2/Add_output_0",
     "w": "onnx::Conv_527"},
    {"name": "layer4.2_deep", "tensor": "/layer4/layer4.2/Add_output_0",
     "w": "onnx::Conv_653"},
]

NORMAL_LO, NORMAL_HI = 0.05, 0.50  # fraction-of-tensor-max range treated as "not an outlier"
N_BINS = 5
TOP_K_NORMAL = 10  # how many normal channels to run the magnitude-independence probe on


def channel_weight_ratios(onnx_path, w_name):
    m = onnx.load(onnx_path)
    for init in m.graph.initializer:
        if init.name == w_name:
            arr = numpy_helper.to_array(init)  # (out_ch, in_ch, kh, kw)
            norms = np.linalg.norm(arr.reshape(arr.shape[0], -1), axis=1)
            return norms / (norms.max() + 1e-12)
    raise KeyError(f"initializer {w_name!r} not found in {onnx_path}")


def classify(ratios, lo=NORMAL_LO, hi=NORMAL_HI):
    normal = np.where((ratios >= lo) & (ratios <= hi))[0]
    outlier = np.where(ratios > hi)[0]
    small = np.where(ratios < lo)[0]
    return normal, outlier, small


def percentile_bin_divergence(values, div, n_bins=N_BINS):
    """values, div: (N,) for one channel. Returns per-bin mean |div|, low->high magnitude."""
    edges = np.percentile(values, np.linspace(0, 100, n_bins + 1))
    edges[-1] += 1e-9
    bin_idx = np.clip(np.digitize(values, edges[1:-1]), 0, n_bins - 1)
    return [float(div[bin_idx == k].mean()) if (bin_idx == k).any() else None for k in range(n_bins)]


def run_boundary(b, n_eval, inferred_path, calib_samples, eval_x):
    tensor = b["tensor"]
    head_path = os.path.join(TMP, f"{MODEL}_option2_{b['name']}.onnx")
    onnx_utils.extract_model(inferred_path, head_path, ["input"], [tensor])

    cal_g = R.EntropyListCalibrator(calib_samples, None)
    cal_d = R.EntropyListCalibrator(calib_samples, None)
    eng_gi = R.load_engine(R.build_int8_engine(head_path, "gpu", cal_g))
    eng_di = R.load_engine(R.build_int8_engine(head_path, "dla", cal_d, allow_gpu_fallback=True))

    A_gi, _ = boundary_acts(eng_gi, eval_x)  # (N, C)
    A_di, _ = boundary_acts(eng_di, eval_x)
    D = np.abs(A_di - A_gi)                  # (N, C)
    chan_div = D.mean(axis=0)                # (C,)

    ratios = channel_weight_ratios(os.path.join(ONNX_DIR, f"{MODEL}.onnx"), b["w"])
    normal, outlier, small = classify(ratios)
    print(f"[{b['name']}] {len(ratios)} channels: normal={len(normal)} outlier={len(outlier)} small={len(small)}", flush=True)

    summary = {
        "boundary": b["name"], "tensor": tensor, "n_eval": n_eval,
        "n_channels": int(len(ratios)),
        "n_normal": int(len(normal)), "n_outlier": int(len(outlier)), "n_small": int(len(small)),
        "div_mean_all": float(chan_div.mean()),
        "div_mean_normal": float(chan_div[normal].mean()) if len(normal) else None,
        "div_mean_outlier": float(chan_div[outlier].mean()) if len(outlier) else None,
        "div_mean_small": float(chan_div[small].mean()) if len(small) else None,
        "div_max_normal": float(chan_div[normal].max()) if len(normal) else None,
    }
    print(f"[{b['name']}] div_mean: all={summary['div_mean_all']:.4f} normal={summary['div_mean_normal']} "
          f"outlier={summary['div_mean_outlier']} small={summary['div_mean_small']}", flush=True)

    # magnitude-independence probe: top-K normal channels by their own divergence
    top_normal = normal[np.argsort(-chan_div[normal])[:min(TOP_K_NORMAL, len(normal))]] if len(normal) else np.array([], dtype=int)
    bin_probe = []
    for c in top_normal:
        per_bin = percentile_bin_divergence(A_gi[:, c], D[:, c])
        valid = [v for v in per_bin if v is not None]
        flatness = (min(valid) / max(valid)) if valid and max(valid) > 0 else None
        bin_probe.append({"channel": int(c), "weight_ratio": float(ratios[c]),
                           "chan_div_overall": float(chan_div[c]),
                           "div_by_activation_bin_low_to_high": per_bin,
                           "flatness_min_over_max": flatness})
        print(f"[{b['name']}]   ch{c} ratio={ratios[c]:.3f} div={chan_div[c]:.4f} "
              f"bins={['%.4f' % v if v is not None else 'NA' for v in per_bin]} flat={flatness}", flush=True)
    summary["magnitude_independence_probe"] = bin_probe

    if len(normal) >= 2:
        signed_D_normal = (A_di - A_gi)[:, normal]
        summary["normal_subspace_svd"] = svd_subspace(signed_D_normal)
        print(f"[{b['name']}] normal-subspace SVD: top1={summary['normal_subspace_svd']['top1_frac']:.3f} "
              f"eff_rank={summary['normal_subspace_svd']['eff_rank']:.1f}/{len(normal)}", flush=True)

    np.savez(os.path.join(RESULTS, f"option2_{b['name']}.npz"),
              A_gpu=A_gi, A_dla=A_di, ratios=ratios,
              normal=normal, outlier=outlier, small=small)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-eval", type=int, default=500)
    args = ap.parse_args()
    R.set_seed(42)
    os.makedirs(TMP, exist_ok=True)

    transform = MC.get_transform(MODEL)
    sp = json.load(open(os.path.join(RESULTS, "splits.json")))
    root = sp["imagenet_root"]
    calib = load_split(root, sp["calib"], transform, limit=200)
    eval_x = load_split(root, sp["eval"], transform, limit=args.n_eval)
    calib_samples = [calib[i:i + 1] for i in range(len(calib))]

    inferred = shape_inference.infer_shapes(onnx.load(os.path.join(ONNX_DIR, f"{MODEL}.onnx")))
    inferred_path = os.path.join(TMP, f"{MODEL}_option2_inferred.onnx")
    onnx.save(inferred, inferred_path)

    rows = []
    for b in BOUNDARIES:
        rows.append(run_boundary(b, args.n_eval, inferred_path, calib_samples, eval_x))

    out = os.path.join(RESULTS, "option2_mechanism_probe.json")
    json.dump(rows, open(out, "w"), indent=2)
    print(f"wrote {out} ({len(rows)} boundaries)")


if __name__ == "__main__":
    main()
