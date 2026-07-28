"""CURRENT_PLAN.md Step 4 -- capture the four DcL-BD groups on REAL hardware.

E_benign = {Qv(clean), Qd(clean), Qv(triggered)}   (must stay/return to true label)
E_adv    = {Qd(triggered)}                          (should flip to target)

Uses the engineered carrier (layer2.2 conv3 channel0 x100, Phase A5) as Qv/Qd (GPU-INT8/
DLA-INT8 head engines at the layer2.2 Add boundary) and the trigger optimized in
trigger_optimize.py. Captures full (N, C, H, W) spatial tensors (not just per-channel means)
so Step 6's finetune can run the real M2 tail (relu->layer3->layer4->avgpool->fc) on them.

Uses three DISJOINT splits (lesson from Phase B v2's train/held-out generalization gap):
  - calib (200 imgs)        : already used to optimize the trigger (trigger_optimize.py)
  - eval  (1000 imgs)       : split in half -> guard-bias search (500) / held-out validation (500)
  - train (index #2.. imgs) : fresh images, disjoint from both -- for Step 6 finetuning

Run from repo root:
  python3 chain_survival/scripts/capture_four_groups.py
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
from boundary_divergence import boundary_acts  # noqa: E402
from onnx import utils as onnx_utils  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CS = os.path.dirname(HERE)
RESULTS = os.path.join(CS, "results")
TMP = os.path.join(CS, "engines", "boundary_heads")
MODEL_DIR = os.path.join(CS, "models")

MODEL = "resnet50"
BOUNDARY_TENSOR = "/layer2/layer2.2/Add_output_0"


def make_train_split(n_per_class=5, start_idx=2):
    sp = json.load(open(os.path.join(RESULTS, "splits.json")))
    root = sp["imagenet_root"]
    classes = sorted(os.listdir(root))
    train = []
    for c in classes:
        cdir = os.path.join(root, c)
        imgs = sorted(os.listdir(cdir))
        for i in range(start_idx, min(start_idx + n_per_class, len(imgs))):
            train.append({"path": f"{c}/{imgs[i]}", "cls": int(c)})
    return train


def apply_trigger_np(x, trig):
    """x: (N,3,224,224) float32 numpy. trig: dict from trigger_optimize.py."""
    t = trig["trigger"].numpy()[0]  # (3,size,size)
    size, (r0, c0) = trig["patch_size"], trig["loc"]
    x = x.copy()
    x[:, :, r0:r0 + size, c0:c0 + size] = t
    return x


def build_engines():
    eng_onnx = os.path.join(TMP, f"{MODEL}_engineered.onnx")
    head_path = os.path.join(TMP, f"{MODEL}_b2_4grp.onnx")
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


def capture_full(engine, x):
    runner = R.EngineRunner(engine)
    out = []
    for i in range(len(x)):
        out.append(runner.run(x[i:i + 1])[0])
    return np.stack(out).astype("float32")


def capture_perchannel(engine, x):
    """Cheap (C,) spatial-mean capture -- enough for Algorithm 1's scalar threshold
    search (guard/heldout splits don't need the real M2 tail, only Step 6's train split does)."""
    runner = R.EngineRunner(engine)
    out = []
    for i in range(len(x)):
        a = runner.run(x[i:i + 1])[0]
        out.append(a.reshape(a.shape[0], -1).mean(axis=1))
    return np.stack(out).astype("float64")


def main():
    R.set_seed(42)
    trig = torch.load(os.path.join(MODEL_DIR, "resnet50_trigger.pth"))
    print(f"[4grp] loaded trigger: target={trig['target_value']} patch={trig['patch_size']}", flush=True)

    eng_gi, eng_di, root, transform, sp = build_engines()

    guard_entries = sp["eval"][:500]
    heldout_entries = sp["eval"][500:1000]
    train_entries = make_train_split(n_per_class=3)
    print(f"[4grp] guard={len(guard_entries)} heldout={len(heldout_entries)} "
          f"train={len(train_entries)}", flush=True)

    def do_split(entries, tag, full):
        x_clean = load_split(root, entries, transform)
        x_trig = apply_trigger_np(x_clean, trig)
        labels = np.array([e["cls"] for e in entries])
        cap = capture_full if full else capture_perchannel

        A_gi_clean = cap(eng_gi, x_clean)
        A_di_clean = cap(eng_di, x_clean)
        A_gi_trig = cap(eng_gi, x_trig)
        A_di_trig = cap(eng_di, x_trig)
        print(f"[4grp] {tag}: captured shapes {A_gi_clean.shape}", flush=True)

        np.savez(os.path.join(RESULTS, f"fourgroups_{tag}.npz"),
                 gpu_clean=A_gi_clean, dla_clean=A_di_clean,
                 gpu_trig=A_gi_trig, dla_trig=A_di_trig, labels=labels)

    do_split(guard_entries, "guard", full=False)
    do_split(heldout_entries, "heldout", full=False)
    do_split(train_entries, "train", full=True)
    print("[4grp] done", flush=True)


if __name__ == "__main__":
    main()
