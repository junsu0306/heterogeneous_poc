"""Step 4: single real-engine verification (PLAN.md §4).

Takes a Stage 2 checkpoint (dual-path emulator weights), extracts the plain
(no quant_sim wrappers) architecture -- the conv/bn/fc submodules are shared
by reference between DualPathResNet18Cifar and a plain build_resnet18_cifar()
net (see extract_plain_state_dict), so this is a real weight extraction, not
a re-fit -- exports it to ONNX, builds *real* TensorRT GPU-INT8 / DLA-INT8
(implicit/legacy-calibrator) engines exactly like phase1_7_repro did, and
measures CA/ASR on FP32 (plain torch) vs GPU-INT8 (real engine) vs DLA-INT8
(real engine). This is the actual go/no-go measurement -- everything upstream
(the STE emulator) is only a training convenience; this script is the first
and only point where real hardware sees these weights.
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "common/scripts")
import trt_runtime as R  # noqa: E402
from cifar_data import CIFAR10Numpy  # noqa: E402
from model_r18_cifar import build_resnet18_cifar  # noqa: E402
from trigger import PATCH_SIZE, TARGET_CLASS  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ENGINES_DIR = HERE.replace("/scripts", "/engines")
RESULTS_PATH = HERE.replace("/scripts", "/results/p1_9.json")
N_CALIB = 256
N_EVAL = 300

_QUANT_MARKERS = ("qv_stem", "qd_stem", "qv_out", "qd_mid1", "qd_mid2", "qd_down", "qd_out", "hw_noise")


def extract_plain_state_dict(dual_state_dict):
    return {k: v for k, v in dual_state_dict.items() if not any(m in k for m in _QUANT_MARKERS)}


def load_plain_net(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    plain_sd = extract_plain_state_dict(ckpt["state_dict"])
    net = build_resnet18_cifar()
    missing, unexpected = net.load_state_dict(plain_sd, strict=True)
    net = net.to(device).eval()
    return net


def export_onnx(net, out_path):
    dummy = torch.randn(1, 3, 32, 32)
    torch.onnx.export(net.cpu(), dummy, out_path, input_names=["input"], output_names=["logits"],
                       opset_version=17, do_constant_folding=True, dynamic_axes=None, dynamo=False)


def get_raw_and_normalized(n):
    images = np.load(f"{HERE.replace('/scripts', '/data')}/cifar10_test_images.npy")
    labels = np.load(f"{HERE.replace('/scripts', '/data')}/cifar10_test_labels.npy")
    mean = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32).reshape(1, 3, 1, 1)
    rng = np.random.default_rng(123)
    idx = rng.choice(len(images), size=n, replace=False)
    imgs = images[idx].astype(np.float32).transpose(0, 3, 1, 2) / 255.0
    normed = (imgs - mean) / std
    white_normed = (1.0 - mean) / std
    triggered = normed.copy()
    triggered[:, :, -PATCH_SIZE:, -PATCH_SIZE:] = white_normed
    return normed, triggered, labels[idx], mean, std


def eval_condition(predict_fn, normed, triggered, labels):
    n = len(labels)
    correct = 0
    trig_hit = 0
    trig_n = 0
    for i in range(n):
        pred_clean = predict_fn(normed[i:i + 1])
        correct += int(pred_clean == labels[i])
        if labels[i] != TARGET_CLASS:
            pred_trig = predict_fn(triggered[i:i + 1])
            trig_hit += int(pred_trig == TARGET_CLASS)
            trig_n += 1
    return {"ca": correct / n, "asr": trig_hit / max(trig_n, 1), "n": n, "trig_n": trig_n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    R.set_seed(42)
    device = "cuda"
    net = load_plain_net(args.ckpt, device)

    onnx_path = f"{ENGINES_DIR}/verify_{args.tag}.onnx"
    os.makedirs(ENGINES_DIR, exist_ok=True)
    export_onnx(net, onnx_path)
    net = net.to(device)

    normed, triggered, labels, mean, std = get_raw_and_normalized(N_EVAL)
    calib_images = np.load(f"{HERE.replace('/scripts', '/data')}/cifar10_train_images.npy")
    rng = np.random.default_rng(42)
    calib_idx = rng.choice(len(calib_images), size=N_CALIB, replace=False)
    calib_imgs = calib_images[calib_idx].astype(np.float32).transpose(0, 3, 1, 2) / 255.0
    calib_samples = [((calib_imgs[i:i + 1] - mean) / std) for i in range(N_CALIB)]

    print(f"building GPU-INT8 for {args.tag}...")
    gpu_engine = R.load_engine(R.build_int8_engine(
        onnx_path, "gpu", R.EntropyListCalibrator(calib_samples, cache_file=f"{ENGINES_DIR}/verify_{args.tag}_gpu.cache")))
    print(f"building DLA-INT8 for {args.tag}...")
    dla_engine = R.load_engine(R.build_int8_engine(
        onnx_path, "dla", R.EntropyListCalibrator(calib_samples, cache_file=f"{ENGINES_DIR}/verify_{args.tag}_dla.cache")))
    gpu_runner = R.EngineRunner(gpu_engine)
    dla_runner = R.EngineRunner(dla_engine)

    @torch.no_grad()
    def predict_fp32(x):
        t = torch.as_tensor(x, dtype=torch.float32, device=device)
        return int(net(t).argmax(1).item())

    def predict_gpu(x):
        return int(gpu_runner.run(x).argmax(1).item())

    def predict_dla(x):
        return int(dla_runner.run(x).argmax(1).item())

    results = {}
    for name, fn in [("fp32", predict_fp32), ("gpu_int8", predict_gpu), ("dla_int8", predict_dla)]:
        results[name] = eval_condition(fn, normed, triggered, labels)
        print(f"{args.tag:>20} {name:>10}  CA={results[name]['ca']:.3f}  ASR={results[name]['asr']:.3f}  "
              f"(n={results[name]['n']}, trig_n={results[name]['trig_n']})")

    gap = results["dla_int8"]["asr"] - results["gpu_int8"]["asr"]
    print(f"\n{args.tag}: ASR_d - ASR_v (real engines) = {gap:.3f}")

    R.update_results(f"p1_9_step4_verify_{args.tag}", {
        "ckpt": args.ckpt, "results": results, "asr_gap_d_minus_v": gap,
    }, path=RESULTS_PATH)


if __name__ == "__main__":
    main()
