"""Patch trigger for CIFAR-10 (32x32), per academic_research_plan_v5.md §4.1
("patch trigger CIFAR 4x4 / ImageNet 8x8") and phase1_7_repro's use of
BackdoorBench's AddMaskPatchTrigger convention (bottom-right corner patch,
fixed pixel value). Operates directly in normalized-tensor space (the
pipeline here never touches raw pixel space), same approach Phase 1.7
verified is mathematically equivalent to raw-pixel application (max abs
diff = 0.0 there).
"""
import torch

PATCH_SIZE = 4
TARGET_CLASS = 0  # CIFAR-10 "airplane" -- arbitrary fixed target, consistent across all runs


def apply_trigger(x, mean, std):
    """x: normalized NCHW tensor. Paints a PATCH_SIZE x PATCH_SIZE white
    patch (raw pixel value 1.0) in the bottom-right corner, in-place-safe
    (returns a new tensor). mean/std: (1,3,1,1) tensors used to normalize,
    so the patch value is computed in the same normalized space."""
    x = x.clone()
    white_normalized = (1.0 - mean) / std  # raw pixel 1.0 -> normalized value, per-channel
    x[:, :, -PATCH_SIZE:, -PATCH_SIZE:] = white_normalized.expand(-1, -1, PATCH_SIZE, PATCH_SIZE)
    return x


def cifar_mean_std(device="cuda"):
    mean = torch.tensor([0.4914, 0.4822, 0.4465], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.2470, 0.2435, 0.2616], device=device).view(1, 3, 1, 1)
    return mean, std
