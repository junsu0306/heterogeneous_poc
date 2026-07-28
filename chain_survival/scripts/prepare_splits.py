"""P0.5 Step 2 — build disjoint calib / eval splits from imagenet_val.

User policy (2026-07-20): imagenet_val is for EVALUATION; but INT8 PTQ needs
calibration data too. Resolution: partition imagenet_val so that no eval image
is ever seen during calibration. We do it by image *index* within each class
folder, which makes disjointness structural (not seed-luck):
  - calib : image #0 of every even-numbered class  -> ~500 images
  - eval  : image #1 of every class                -> 1000 images
Both INT8 paths (trt_gpu_int8, dla_int8) MUST use the same calib split so only
the hardware differs (handoff §3.1). Lists are frozen in results/splits.json.

NOTE: folder name "00042" is treated as class index 42 for optional accuracy
only; the P0.5 metric is cross-path divergence, which needs no true labels.

Run from repo root:  python3 chain_survival/scripts/prepare_splits.py
"""
import json
import os

IMAGENET_ROOT = "/media/airlab_compression/nvme_storage/imagenet_val"
HERE = os.path.dirname(os.path.abspath(__file__))
CS = os.path.dirname(HERE)
RESULTS = os.path.join(CS, "results")
SEED = 42


def main():
    os.makedirs(RESULTS, exist_ok=True)
    classes = sorted(os.listdir(IMAGENET_ROOT))
    calib, eval_ = [], []

    for c in classes:
        cdir = os.path.join(IMAGENET_ROOT, c)
        imgs = sorted(os.listdir(cdir))
        if len(imgs) < 2:
            continue
        cls_idx = int(c)
        # eval: image #1 of every class
        eval_.append({"path": f"{c}/{imgs[1]}", "cls": cls_idx})
        # calib: image #0 of even classes only (~500)
        if cls_idx % 2 == 0:
            calib.append({"path": f"{c}/{imgs[0]}", "cls": cls_idx})

    calib_set = {e["path"] for e in calib}
    eval_set = {e["path"] for e in eval_}
    overlap = calib_set & eval_set
    assert not overlap, f"calib/eval overlap! {list(overlap)[:3]}"

    out = {
        "imagenet_root": IMAGENET_ROOT,
        "seed": SEED,
        "scheme": "calib=img#0 of even classes; eval=img#1 of all classes; disjoint by index",
        "n_calib": len(calib),
        "n_eval": len(eval_),
        "calib": calib,
        "eval": eval_,
    }
    path = os.path.join(RESULTS, "splits.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[splits] calib={len(calib)}  eval={len(eval_)}  overlap=0 -> {path}")


if __name__ == "__main__":
    main()
