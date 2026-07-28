"""Phase A A4 -- stability of the ell* divergence subspace.

Two checks per NEXT_PLAN_v11.md §A4:
  (a) input-resampling stability (cheap, all 16 boundaries, reuses A1's saved
      (500, C) activations): split the 500 eval images into two disjoint
      halves, compute the boundary's dominant DLA-vs-GPU divergence direction
      (top SVD vector) independently on each half, and measure |cosine
      similarity| between them. High similarity means the subspace is a
      property of the hardware/model, not an artifact of which images
      happened to be sampled.
  (b) engine-rebuild stability (expensive, ell* candidates only): rebuild the
      GPU-INT8 / DLA-INT8 head engines K times from scratch (fresh
      calibration, no cache -- same method as rebuild_stability.py, which
      found full-model logits are bit-identical across rebuilds) and check
      whether the boundary's dominant divergence direction is preserved
      across builds.

Run from repo root:
  python3 chain_survival/scripts/boundary_stability.py [--k 3] [--n-eval 256]
"""
import argparse
import glob
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
ENG_DIR = os.path.join(CS, "engines")
TMP = os.path.join(ENG_DIR, "boundary_heads")

def remaining_candidates_ranked():
    """All 16 boundaries minus whatever's already in boundary_stability_rebuild.json,
    sorted by A4a input-split top1 cosine descending -- the free signal is our
    best prior on which untested boundaries are worth the expensive rebuild."""
    split = {(r["model"], r["idx"]): r for r in json.load(open(f"{RESULTS}/boundary_stability_split.json"))}
    subspace = {(r["model"], r["idx"]): r for r in json.load(open(f"{RESULTS}/boundary_subspace_summary.json"))}
    done = set()
    prior_path = f"{RESULTS}/boundary_stability_rebuild.json"
    if os.path.exists(prior_path):
        done = {(r["model"], r["idx"]) for r in json.load(open(prior_path))}

    rows = []
    for jf in sorted(glob.glob(f"{RESULTS}/boundary_divergence_*.json")):
        d = json.load(open(jf))
        model = d["model"]
        for b in d["boundaries"]:
            key = (model, b["idx"])
            if key in done:
                continue
            sp, sub = split[key], subspace[key]
            note = (f"{b['op']}: half_cos={sp['half_cosine_top1']:.2f} top1_frac={b['top1_frac']:.2f} "
                    f"eff_rank={100 * b['eff_rank'] / b['n_channels']:.1f}% conc_x={sub['concentration_x']:.1f}x")
            rows.append((model, b["idx"], note, sp["half_cosine_top1"]))
    rows.sort(key=lambda r: -r[3])
    return [(m, i, n) for m, i, n, _ in rows]


def top_sv(M, k=1):
    Mc = M - M.mean(0, keepdims=True)
    _, _, Vt = np.linalg.svd(Mc, full_matrices=False)
    return Vt[0] if k == 1 else Vt[:k]


def cos(a, b):
    return float(np.abs(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)))


def subspace_overlap(V1, V2):
    """V1, V2: (k, C) orthonormal top-k right singular vectors. Average cosine
    of principal angles between the two k-dim subspaces (1.0 = identical
    subspace, 0.0 = orthogonal) -- robust to within-subspace reordering that
    penalizes a plain top-1 cosine when several singular values are close."""
    s = np.linalg.svd(V1 @ V2.T, compute_uv=False)
    return float(s.mean())


def input_split_stability():
    """(a) free -- reuses A1's saved (500, C) activations for all 16 boundaries."""
    rows = []
    for jf in sorted(glob.glob(f"{RESULTS}/boundary_divergence_*.json")):
        d = json.load(open(jf))
        model = d["model"]
        for b in d["boundaries"]:
            z = np.load(f"{RESULTS}/boundary_{model}_b{b['idx']}.npz")
            M = z["A_dla"] - z["A_gpu"]
            n = len(M)
            half = n // 2
            k = min(5, M.shape[1] // 2, half - 1)
            v1, V1 = top_sv(M[:half]), top_sv(M[:half], k=k)
            v2, V2 = top_sv(M[half:]), top_sv(M[half:], k=k)
            row = {"model": model, "idx": b["idx"], "op": b["op"], "tensor": b["tensor"],
                   "n_channels": b["n_channels"], "top1_frac": b["top1_frac"],
                   "half_cosine_top1": cos(v1, v2),
                   "half_subspace_overlap_top5": subspace_overlap(V1, V2)}
            rows.append(row)
            print(f"[split] {model} b{b['idx']} ({b['op']}) top1_cos={row['half_cosine_top1']:.3f} "
                  f"top5_subspace_overlap={row['half_subspace_overlap_top5']:.3f}", flush=True)
    json.dump(rows, open(f"{RESULTS}/boundary_stability_split.json", "w"), indent=2)
    print(f"wrote {RESULTS}/boundary_stability_split.json ({len(rows)} boundaries)")
    return rows


def rebuild_stability(k, n_eval, candidates):
    """(b) expensive -- true engine rebuilds for the given (model, idx, note) list.
    Appends to any existing boundary_stability_rebuild.json rather than
    overwriting, so this can be run incrementally across multiple sessions."""
    sp = json.load(open(f"{RESULTS}/splits.json"))
    root = sp["imagenet_root"]
    prior_path = f"{RESULTS}/boundary_stability_rebuild.json"
    rows = json.load(open(prior_path)) if os.path.exists(prior_path) else []
    for name, idx, note in candidates:
        transform = MC.get_transform(name)
        calib = load_split(root, sp["calib"], transform, limit=200)
        eval_x = load_split(root, sp["eval"], transform, limit=n_eval)
        calib_samples = [calib[i:i + 1] for i in range(len(calib))]
        head_path = os.path.join(TMP, f"{name}_b{idx}.onnx")

        n_channels = None
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
            gi_runs.append(A_gi)
            di_runs.append(A_di)
            D = A_di - A_gi
            v1_runs.append(top_sv(D))
            Vk_runs.append(top_sv(D, k=kk))
            print(f"[rebuild] {name} b{idx} rep {rep + 1}/{k} done", flush=True)

        gi_stack, di_stack = np.stack(gi_runs), np.stack(di_runs)
        gpu_det = float(max(np.abs(gi_stack[r] - gi_stack[0]).max() for r in range(1, k))) if k > 1 else 0.0
        dla_det = float(max(np.abs(di_stack[r] - di_stack[0]).max() for r in range(1, k))) if k > 1 else 0.0
        pair_cos = [cos(v1_runs[i], v1_runs[j]) for i in range(k) for j in range(i + 1, k)]
        pair_overlap = [subspace_overlap(Vk_runs[i], Vk_runs[j]) for i in range(k) for j in range(i + 1, k)]

        row = {"model": name, "idx": idx, "note": note, "k": k, "n_eval": n_eval,
               "n_channels": n_channels,
               "gpu_max_abs_across_rebuilds": gpu_det,
               "dla_max_abs_across_rebuilds": dla_det,
               "top1_cosine_pairs": pair_cos,
               "top1_cosine_min": float(min(pair_cos)),
               "top5_subspace_overlap_pairs": pair_overlap,
               "top5_subspace_overlap_min": float(min(pair_overlap))}
        rows.append(row)
        print(f"[rebuild] {name} b{idx}: gpu_det={gpu_det:.3e} dla_det={dla_det:.3e} "
              f"top1_cosine_min={row['top1_cosine_min']:.4f} "
              f"top5_subspace_overlap_min={row['top5_subspace_overlap_min']:.4f}", flush=True)
        # save after every candidate so a long batch is resumable / inspectable mid-run
        json.dump(rows, open(prior_path, "w"), indent=2)

    print(f"wrote {prior_path} ({len(rows)} candidates total)")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--n-eval", type=int, default=256)
    ap.add_argument("--skip-split", action="store_true")
    ap.add_argument("--skip-rebuild", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="test only the top N remaining candidates")
    args = ap.parse_args()
    R.set_seed(42)
    os.makedirs(TMP, exist_ok=True)

    if not args.skip_split:
        input_split_stability()
    if not args.skip_rebuild:
        cands = remaining_candidates_ranked()
        if args.limit:
            cands = cands[:args.limit]
        print(f"[rebuild] {len(cands)} candidates queued (ranked by input-split cosine): "
              f"{[(m, i) for m, i, _ in cands]}", flush=True)
        rebuild_stability(args.k, args.n_eval, cands)


if __name__ == "__main__":
    main()
