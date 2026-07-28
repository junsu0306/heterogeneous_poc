"""Phase A5 follow-up -- clean-accuracy cost of the validated weight edit.

weight_engineering_probe.py showed that inflating one output channel's weight
by `factor` manufactures a stable, channel-local GPU-vs-DLA divergence (vgg16
avgpool, googlenet inception5b Concat: rebuild cosine 0.91-0.99). That probe
only measured boundary activations -- it never checked whether the edit wrecks
the model's actual classification accuracy (cf. the P0.5 efficientnet/
mobilenet COLLAPSE finding: a channel that's clean+divergent is only useful if
clean accuracy survives; a channel that collapses accuracy is not exploitable).

This script builds the FULL engineered model (already saved by the probe as
engines/boundary_heads/{model}_engineered.onnx) as GPU-INT8 and DLA-INT8,
evaluates top-1 on the same 500-image eval split used throughout Phase A/P0.5,
and compares against the original (unedited) model's already-recorded
accuracy (results/{model}_trt_gpu_int8.npz / {model}_dla_int8.npz, from the
P0.5 full run, REPORT.md §2).

Run from repo root:
  python3 chain_survival/scripts/engineered_accuracy_check.py [--n-eval 500]
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, "common/scripts")
import trt_runtime as R  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import models_cfg as MC  # noqa: E402
from run_paths import load_split, logits_engine  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CS = os.path.dirname(HERE)
RESULTS = os.path.join(CS, "results")
ENG_DIR = os.path.join(CS, "engines")
TMP = os.path.join(ENG_DIR, "boundary_heads")

TARGETS = ["vgg16", "googlenet"]  # the two Phase A5-validated (stable) cases


def baseline_acc(model, labels):
    accs = {}
    for path in ["trt_gpu_int8", "dla_int8"]:
        le = np.load(os.path.join(RESULTS, f"{model}_{path}.npz"))["logits_eval"]
        accs[path] = float((le.argmax(1) == labels[:len(le)]).mean() * 100.0)
    return accs


def engineered_acc(model, eval_x, calib_samples, labels):
    onnx_path = os.path.join(TMP, f"{model}_engineered.onnx")
    accs = {}
    for path, device in [("trt_gpu_int8", "gpu"), ("dla_int8", "dla")]:
        cal = R.EntropyListCalibrator(calib_samples, None)
        kw = {"allow_gpu_fallback": True} if device == "dla" else {}
        eng = R.load_engine(R.build_int8_engine(onnx_path, device, cal, **kw))
        logits = logits_engine(eng, eval_x)
        accs[path] = float((logits.argmax(1) == labels).mean() * 100.0)
        print(f"[acc] {model} engineered {path}: {accs[path]:.1f}%", flush=True)
    return accs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-eval", type=int, default=500)
    args = ap.parse_args()
    R.set_seed(42)

    sp = json.load(open(os.path.join(RESULTS, "splits.json")))
    root = sp["imagenet_root"]
    labels_full = np.array([e["cls"] for e in sp["eval"]])

    report = {}
    for model in TARGETS:
        transform = MC.get_transform(model)
        calib = load_split(root, sp["calib"], transform, limit=200)
        eval_x = load_split(root, sp["eval"], transform, limit=args.n_eval)
        calib_samples = [calib[i:i + 1] for i in range(len(calib))]
        labels = labels_full[:len(eval_x)]

        base = baseline_acc(model, labels_full)
        eng = engineered_acc(model, eval_x, calib_samples, labels)
        report[model] = {"n_eval": len(eval_x), "baseline": base, "engineered": eng,
                          "delta_gpu_pct": eng["trt_gpu_int8"] - base["trt_gpu_int8"],
                          "delta_dla_pct": eng["dla_int8"] - base["dla_int8"]}
        print(f"[acc] {model}: gpu {base['trt_gpu_int8']:.1f}->{eng['trt_gpu_int8']:.1f}  "
              f"dla {base['dla_int8']:.1f}->{eng['dla_int8']:.1f}", flush=True)

    out = os.path.join(RESULTS, "weight_engineering_accuracy.json")
    json.dump(report, open(out, "w"), indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
