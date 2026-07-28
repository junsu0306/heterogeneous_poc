"""Quick, cheap test of multiple trigger checkpoints against real hardware (guard split only,
per-channel capture) to pick the best-calibrated push strength before committing to the
expensive full-spatial capture needed for Step 6.

Run from repo root:
  python3 chain_survival/scripts/test_checkpoints.py
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "common/scripts")
import trt_runtime as R  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import models_cfg as MC  # noqa: E402
from run_paths import load_split  # noqa: E402
from guard_bias_search import search_channel  # noqa: E402
from onnx import utils as onnx_utils  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CS = os.path.dirname(HERE)
RESULTS = os.path.join(CS, "results")
TMP = os.path.join(CS, "engines", "boundary_heads")
MODEL_DIR = os.path.join(CS, "models")
MODEL = "resnet50"
BOUNDARY_TENSOR = "/layer2/layer2.2/Add_output_0"
CHANNEL = 0


def build_engines():
    eng_onnx = os.path.join(TMP, f"{MODEL}_engineered.onnx")
    head_path = os.path.join(TMP, f"{MODEL}_b2_cktest.onnx")
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


def apply_trigger_np(x, trig_tensor, patch_size, loc):
    t = trig_tensor.numpy()[0]
    r0, c0 = loc
    x = x.copy()
    x[:, :, r0:r0 + patch_size, c0:c0 + patch_size] = t
    return x


def capture_perchannel(engine, x):
    runner = R.EngineRunner(engine)
    out = []
    for i in range(len(x)):
        a = runner.run(x[i:i + 1])[0]
        out.append(a.reshape(a.shape[0], -1).mean(axis=1))
    return np.stack(out).astype("float64")


def main():
    R.set_seed(42)
    ck_data = torch.load(os.path.join(MODEL_DIR, "resnet50_trigger_checkpoints.pth"))
    patch_size, loc = ck_data["patch_size"], ck_data["loc"]

    eng_gi, eng_di, root, transform, sp = build_engines()
    guard_entries = sp["eval"][:500]
    x_clean = load_split(root, guard_entries, transform)
    gpu_clean = capture_perchannel(eng_gi, x_clean)
    dla_clean = capture_perchannel(eng_di, x_clean)
    print(f"[test] channel0 clean: gpu mean={gpu_clean[:,CHANNEL].mean():.2f} "
          f"dla mean={dla_clean[:,CHANNEL].mean():.2f}", flush=True)

    for it, ck in ck_data["checkpoints"].items():
        x_trig = apply_trigger_np(x_clean, ck["trigger"], patch_size, loc)
        gpu_trig = capture_perchannel(eng_gi, x_trig)
        dla_trig = capture_perchannel(eng_di, x_trig)
        g0, d0 = gpu_trig[:, CHANNEL], dla_trig[:, CHANNEL]
        print(f"[test] iter{it}: gpu_trig mean={g0.mean():.2f} min={g0.min():.2f} | "
              f"dla_trig mean={d0.mean():.2f} min={d0.min():.2f} "
              f"(dla_clean was mean={dla_clean[:,CHANNEL].mean():.2f} min={dla_clean[:,CHANNEL].min():.2f})",
              flush=True)

        benign = np.concatenate([gpu_clean[:, CHANNEL], dla_clean[:, CHANNEL], g0])
        adv = d0
        r = search_channel(benign, adv)
        print(f"  guard-bias ch0 @ iter{it}: {r}", flush=True)


if __name__ == "__main__":
    main()
