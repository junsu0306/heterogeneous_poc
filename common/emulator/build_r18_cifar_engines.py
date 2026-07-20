"""Step 1: ResNet-18/CIFAR-10 real-engine layer(block)-boundary divergence
profiling (phase1_9_feasibility/PLAN.md §1).

Phase 0 only built ResNet-50/YOLOv8s (ImageNet-scale) engines, so this
builds fresh GPU-INT8 / DLA-INT8 implicit(legacy-calibrator) engines for
the CIFAR-scale ResNet-18 and measures, at each of the 8 residual-block
boundaries (+stem+logits), how much the cumulative GPU-vs-DLA divergence
has grown by that depth -- ranking which blocks to target in Step 2's
backdoor signal placement, instead of assuming the toy-net MAC-depth
threshold (~144, phase1_characterization/mac_sweep.py) transfers unchanged
to this architecture.

**Method note**: each tap is measured via its own truncated single-output
network (PrefixWrapper), *not* a single multi-output graph with all 9 taps
exported together. The multi-output version was tried first and rejected:
exporting every intermediate tensor as a graph output inserts Identity
nodes that are unsupported on DLA, and TensorRT cascade-falls-back *every*
downstream conv layer to GPU as a result (confirmed in logs -- every
layer1-4 conv got "Switching this layer's device type to GPU"), which
would have made the "DLA" engine mostly run on GPU and silently invalidate
the whole measurement. The per-cut truncated approach gets a normal DLA
layer assignment for each build (matches Phase 0's ResNet-50 pattern: only
tail ops like avgpool/flatten/fc fall back). The number this measures is
cumulative divergence-so-far at that depth, which is also the more directly
relevant quantity for Step 2 (a trigger reading that block's activations
sees exactly this cumulative signal, not an isolated per-block delta).

Weights are untrained (random Xavier-style init via build_resnet18_cifar,
seed=42) -- same convention as Phase 1.0/1.1: HW-path divergence at a given
tap is a property of the quantization/execution pipeline at that point in
the graph, not of what the weights have learned, so no training is needed
to rank taps.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "common/scripts")
import trt_runtime as R  # noqa: E402
from model_r18_cifar import TAP_MAC_DEPTH, TAP_NAMES, PrefixWrapper, build_resnet18_cifar  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = HERE.replace("/scripts", "/data")
ENGINES_DIR = HERE.replace("/scripts", "/engines")
LOGS_DIR = HERE.replace("/scripts", "/logs")
RESULTS_PATH = HERE.replace("/scripts", "/results/p1_9.json")
N_CALIB_SAMPLES = 256
N_TEST_SAMPLES = 100
INPUT_SHAPE = (1, 3, 32, 32)


def export_prefix_onnx(net, cut, onnx_path):
    wrapped = PrefixWrapper(net, cut).eval()
    dummy = torch.randn(*INPUT_SHAPE)
    torch.onnx.export(
        wrapped, dummy, onnx_path, input_names=["input"], output_names=[cut],
        opset_version=17, do_constant_folding=True, dynamic_axes=None, dynamo=False,
    )


def count_dla_fallback(log_path):
    if not os.path.exists(log_path):
        return None
    with open(log_path) as f:
        text = f.read()
    return text.count("Switching this layer's device type to GPU")


def load_calib_and_test_images():
    images = np.load(f"{DATA_DIR}/cifar10_train_images.npy")  # (N,32,32,3) uint8
    mean = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32).reshape(1, 3, 1, 1)

    def to_tensor(batch_uint8):
        x = batch_uint8.astype(np.float32).transpose(0, 3, 1, 2) / 255.0
        return (x - mean) / std

    rng = np.random.default_rng(42)
    idx = rng.choice(len(images), size=N_CALIB_SAMPLES + N_TEST_SAMPLES, replace=False)
    calib_idx, test_idx = idx[:N_CALIB_SAMPLES], idx[N_CALIB_SAMPLES:]
    calib_samples = [to_tensor(images[i:i + 1]) for i in calib_idx]
    test_samples = [to_tensor(images[i:i + 1]) for i in test_idx]
    return calib_samples, test_samples


def profile_tap(net, cut, calib_samples, test_samples):
    onnx_path = f"{ENGINES_DIR}/r18_cifar_prefix_{cut}.onnx"
    export_prefix_onnx(net, cut, onnx_path)

    gpu_log = f"{LOGS_DIR}/p1_9_{cut}_gpu_build.log"
    dla_log = f"{LOGS_DIR}/p1_9_{cut}_dla_build.log"

    import contextlib
    import io as _io

    def build_with_log(device, log_path):
        buf = _io.StringIO()
        with contextlib.redirect_stderr(buf), contextlib.redirect_stdout(buf):
            cache = f"{ENGINES_DIR}/r18_{device}_calib_{cut}.cache"
            eng = R.build_int8_engine(onnx_path, device, R.EntropyListCalibrator(calib_samples, cache_file=cache))
        with open(log_path, "w") as f:
            f.write(buf.getvalue())
        return R.load_engine(eng)

    gpu_engine = build_with_log("gpu", gpu_log)
    dla_engine = build_with_log("dla", dla_log)
    dla_fallback_count = count_dla_fallback(dla_log)

    diffs = []
    for x in test_samples:
        gpu_out = R.run_engine(gpu_engine, x)
        dla_out = R.run_engine(dla_engine, x)
        diffs.append(R.diff_stats(gpu_out, dla_out)["mean_abs_diff"])
    diffs = np.array(diffs)
    return {
        "tap": cut,
        "mac_depth": TAP_MAC_DEPTH.get(cut),
        "mean_abs_diff_mean": float(diffs.mean()),
        "mean_abs_diff_std": float(diffs.std()),
        "dla_fallback_log_lines": dla_fallback_count,
    }


def main():
    R.set_seed(42)
    os.makedirs(ENGINES_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)
    net = build_resnet18_cifar(seed=42).eval()
    calib_samples, test_samples = load_calib_and_test_images()

    ranking = []
    for cut in TAP_NAMES:
        print(f"--- profiling tap: {cut} ---")
        r = profile_tap(net, cut, calib_samples, test_samples)
        print(f"  mac_depth={r['mac_depth']!s:>6}  mean_abs_diff={r['mean_abs_diff_mean']:.6f} "
              f"(+/-{r['mean_abs_diff_std']:.6f})  dla_fallback_lines={r['dla_fallback_log_lines']}")
        ranking.append(r)

    ranking_blocks_only = [r for r in ranking if r["tap"] != "logits"]
    ranking_blocks_only.sort(key=lambda r: r["mean_abs_diff_mean"], reverse=True)

    print("\n=== Step 1: per-tap GPU/DLA cumulative divergence ranking (highest first) ===")
    for r in ranking_blocks_only:
        print(f"{r['tap']:>6}  mac_depth={r['mac_depth']!s:>6}  "
              f"mean_abs_diff={r['mean_abs_diff_mean']:.6f}  dla_fallback_lines={r['dla_fallback_log_lines']}")

    top_k = [r["tap"] for r in ranking_blocks_only[:2]]
    print(f"\ntop-2 target taps for Step 2 signal placement: {top_k}")

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    R.update_results("p1_9_step1_layer_ranking", {
        "n_calib_samples": N_CALIB_SAMPLES,
        "n_test_samples": N_TEST_SAMPLES,
        "method": "per-tap truncated single-output prefix networks (see module docstring)",
        "ranking_all_taps": ranking,
        "ranking_blocks_only_sorted": ranking_blocks_only,
        "top_k_target_taps": top_k,
    }, path=RESULTS_PATH)


if __name__ == "__main__":
    main()
