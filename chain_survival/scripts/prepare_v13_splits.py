"""Create structurally disjoint v13 ImageNet splits.

Legacy experiments used per-class image indices 0--4. This script starts at
index 5 and allocates every role by fixed per-class indices, so overlap is
impossible by construction. Calibration subsets contain 500 randomly selected
classes; evaluation/training roles are class-balanced.

Run from the repository root:
  python chain_survival/scripts/prepare_v13_splits.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


SCHEMA_VERSION = 1
LEGACY_MAX_INDEX = 4


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def class_inventory(root: Path) -> list[tuple[int, str, list[str]]]:
    inventory = []
    for class_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        try:
            class_index = int(class_dir.name)
        except ValueError as error:
            raise ValueError(f"class directory must be numeric: {class_dir}") from error
        images = sorted(path.name for path in class_dir.iterdir() if path.is_file())
        inventory.append((class_index, class_dir.name, images))
    if not inventory:
        raise ValueError(f"no class directories found under {root}")
    return inventory


def entries_at_indices(
    inventory: list[tuple[int, str, list[str]]],
    indices: list[int],
    selected_classes: set[int] | None = None,
) -> list[dict]:
    entries = []
    for class_index, class_name, images in inventory:
        if selected_classes is not None and class_index not in selected_classes:
            continue
        for image_index in indices:
            if image_index >= len(images):
                raise ValueError(
                    f"class {class_name} has {len(images)} images; index {image_index} is unavailable"
                )
            entries.append(
                {
                    "path": f"{class_name}/{images[image_index]}",
                    "cls": class_index,
                    "class_image_index": image_index,
                }
            )
    return entries


def assert_disjoint(splits: dict[str, list[dict]]) -> None:
    owners = {}
    for role, entries in splits.items():
        for entry in entries:
            path = entry["path"]
            if path in owners:
                raise AssertionError(f"split overlap: {path} in {owners[path]} and {role}")
            owners[path] = role


def role_summary(entries: list[dict]) -> dict:
    classes = sorted({entry["cls"] for entry in entries})
    indices = sorted({entry["class_image_index"] for entry in entries})
    path_digest = sha256_text("\n".join(entry["path"] for entry in entries))
    return {
        "n_images": len(entries),
        "n_classes": len(classes),
        "class_image_indices": indices,
        "paths_sha256": path_digest,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--imagenet-root",
        type=Path,
        default=Path("/media/airlab_compression/nvme_storage/imagenet_val"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("chain_survival/results/v13/splits_v13.json"),
    )
    parser.add_argument("--seed", type=int, default=1301)
    parser.add_argument("--calibration-size", type=int, default=500)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inventory = class_inventory(args.imagenet_root)
    class_ids = np.asarray([row[0] for row in inventory], dtype=np.int64)
    if args.calibration_size > len(class_ids):
        raise ValueError("calibration-size exceeds the number of classes")
    if min(len(row[2]) for row in inventory) <= 20:
        raise ValueError("v13 split scheme requires at least 21 images per class")

    rng = np.random.default_rng(args.seed)
    calibration_roles = {
        "calib_shadow_1": 5,
        "calib_shadow_2": 6,
        "calib_blind_1": 7,
        "calib_blind_2": 8,
        "calib_blind_3": 9,
    }
    splits = {}
    for role, image_index in calibration_roles.items():
        selected = set(
            int(value)
            for value in rng.choice(class_ids, size=args.calibration_size, replace=False)
        )
        splits[role] = entries_at_indices(inventory, [image_index], selected)

    splits.update(
        {
            "surrogate_train": entries_at_indices(inventory, [10, 11]),
            "mechanism_discovery": entries_at_indices(inventory, [12]),
            "threshold_validation": entries_at_indices(inventory, [13]),
            "boundary_blind": entries_at_indices(inventory, [14]),
            "final_logit_blind": entries_at_indices(inventory, [15, 16, 17, 18, 19]),
            "robustness": entries_at_indices(inventory, [20]),
        }
    )
    assert_disjoint(splits)

    output = {
        "schema_version": SCHEMA_VERSION,
        "seed": args.seed,
        "imagenet_root": str(args.imagenet_root),
        "policy": {
            "legacy_indices_reserved": [0, LEGACY_MAX_INDEX],
            "selection": "fixed per-class image indices; calibration classes sampled without replacement",
            "blind_policy": (
                "boundary_blind and final_logit_blind must not be used for model, "
                "subspace, threshold, trigger, or checkpoint selection"
            ),
        },
        "dataset": {
            "n_classes": len(inventory),
            "n_images": int(sum(len(row[2]) for row in inventory)),
            "min_images_per_class": int(min(len(row[2]) for row in inventory)),
            "max_images_per_class": int(max(len(row[2]) for row in inventory)),
        },
        "summary": {role: role_summary(entries) for role, entries in splits.items()},
        "splits": splits,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        json.dump(output, handle, indent=2)
    print(json.dumps({"output": str(args.output), "summary": output["summary"]}, indent=2))


if __name__ == "__main__":
    main()
