"""Legacy v11 target-value sweep retained as a negative-result artifact.

DcL-BD's trigger objective (t = argmin MSE(M1(x@t), lambda+K)) assumes divergence grows
monotonically with the pushed channel's own output magnitude. We measured this is only
weakly true for NATURAL (unedited) resnet50 channels (per-channel corr ~0.067). This script
re-checks the SAME question specifically on the channel we've engineered (layer2.2 conv3
channel 0, factor=100, the historical validated carrier) -- does GPU-vs-DLA divergence at
THIS channel grow with the channel's own activation value, and if so, over what range should
the trigger aim to push it?

Method: run many real images through the already-built engineered GPU-INT8/DLA-INT8 head
engines (layer2.2 Add boundary), record channel-0's value on each path per image, and look at
how |divergence| varies as a function of that value -- including images that naturally produce
extreme (tail) values, to see if the relationship strengthens outside the "typical" range.

Run from repo root:
  python3 chain_survival/scripts/sweep_target_value.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, "common/scripts")
import trt_runtime as R  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import models_cfg as MC  # noqa: E402
from run_paths import load_split  # noqa: E402
from boundary_divergence import boundary_acts  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CS = os.path.dirname(HERE)
RESULTS = os.path.join(CS, "results")
TMP = os.path.join(CS, "engines", "boundary_heads")

MODEL = "resnet50"
CHANNEL = 0
BOUNDARY_TENSOR = "/layer2/layer2.2/Add_output_0"


def build_engines():
    from onnx import utils as onnx_utils
    eng_onnx = os.path.join(TMP, f"{MODEL}_engineered.onnx")
    head_path = os.path.join(TMP, f"{MODEL}_b2_sweep.onnx")
    onnx_utils.extract_model(eng_onnx, head_path, ["input"], [BOUNDARY_TENSOR])
    sp = json.load(open(os.path.join(RESULTS, "splits.json")))
    root = sp["imagenet_root"]
    transform = MC.get_transform(MODEL)
    calib = load_split(root, sp["calib"], transform, limit=200)
    calib_samples = [calib[i:i + 1] for i in range(len(calib))]
    cal_g = R.EntropyListCalibrator(calib_samples, None)
    cal_d = R.EntropyListCalibrator(calib_samples, None)
    eng_gi = R.load_engine(R.build_int8_engine(head_path, "gpu", cal_g))
    eng_di = R.load_engine(R.build_int8_engine(head_path, "dla", cal_d, allow_gpu_fallback=True))
    return eng_gi, eng_di, root, transform, sp


def main():
    R.set_seed(42)
    eng_gi, eng_di, root, transform, sp = build_engines()

    eval_x = load_split(root, sp["eval"], transform, limit=500)
    A_gi, _ = boundary_acts(eng_gi, eval_x)
    A_di, _ = boundary_acts(eng_di, eval_x)
    v_gpu = A_gi[:, CHANNEL]
    v_dla = A_di[:, CHANNEL]
    div = np.abs(v_dla - v_gpu)

    print(f"[sweep] channel {CHANNEL}: gpu range [{v_gpu.min():.2f}, {v_gpu.max():.2f}] "
          f"mean={v_gpu.mean():.2f} std={v_gpu.std():.2f}")
    print(f"[sweep] dla range [{v_dla.min():.2f}, {v_dla.max():.2f}] "
          f"mean={v_dla.mean():.2f} std={v_dla.std():.2f}")
    print(f"[sweep] |divergence|: mean={div.mean():.3f} std={div.std():.3f}")
    print(f"[sweep] corr(gpu_value, |divergence|) = {np.corrcoef(v_gpu, div)[0,1]:.3f}")
    print(f"[sweep] corr(|gpu_value|, |divergence|) = {np.corrcoef(np.abs(v_gpu), div)[0,1]:.3f}")

    order = np.argsort(v_gpu)
    n = len(v_gpu)
    bins = 10
    print("[sweep] gpu-value percentile bin -> mean|divergence|:")
    for i in range(bins):
        idx = order[int(i * n / bins):int((i + 1) * n / bins)]
        print(f"  bin{i} (gpu {v_gpu[idx].min():7.2f} .. {v_gpu[idx].max():7.2f}): "
              f"mean|div|={div[idx].mean():.3f}  max|div|={div[idx].max():.3f}")

    out = {"channel": CHANNEL, "n_eval": len(eval_x),
           "gpu_mean": float(v_gpu.mean()), "gpu_std": float(v_gpu.std()),
           "gpu_min": float(v_gpu.min()), "gpu_max": float(v_gpu.max()),
           "corr_gpu_div": float(np.corrcoef(v_gpu, div)[0, 1]),
           "div_mean": float(div.mean()), "div_max": float(div.max()),
           "top_bin_gpu_range": [float(v_gpu[order[-n // 10:]].min()), float(v_gpu.max())],
           "top_bin_mean_div": float(div[order[-n // 10:]].mean())}
    json.dump(out, open(os.path.join(RESULTS, "sweep_target_value.json"), "w"), indent=2)
    print(f"\nwrote {RESULTS}/sweep_target_value.json")


if __name__ == "__main__":
    main()
