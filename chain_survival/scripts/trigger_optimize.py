"""CURRENT_PLAN.md Step 3 -- trigger optimization (DcL-BD's actual, simple objective).

Per dclbd_guardbias_mechanism (folded into CURRENT_PLAN.md §2/§3), the trigger objective does
NOT need to know anything about the GPU/DLA divergence -- it just pushes M1's own output toward
an extreme value, using M1 alone (plain float, fully differentiable). No STE, no
non-differentiable quantization in this step.

v1/v2 (factor=100, fixed MSE targets -150 then -300) both drove DLA into full saturation --
guard_bias_search.py showed dla_trig was statistically indistinguishable from dla_clean (Qd
doesn't respond to the trigger at all once clipped), so Algorithm 1 could never separate them.
Root cause (found via sweep_target_value.py + direct distribution inspection): factor=100
already makes channel0's NATURAL (untriggered) range wide enough that DLA clips for some clean
images too (dla_clean never goes below -80.6 even when gpu reaches -159) -- there was no
headroom left for a trigger to push into a boundary clean images don't already reach.

This version uses gentler engineering (factor=20, keeps the natural range much narrower) and an
OPEN-ENDED objective (minimize channel0 directly, no fixed target) with checkpointing at several
iteration counts, so multiple "how far pushed" points can be evaluated against real hardware
(guard_bias_search.py) without re-running optimization per guessed target.

M1 = stem + layer1 + layer2[0:2] + layer2[2] unrolled up to the raw (pre-ReLU) Add -- built
directly in PyTorch with the SAME channel-0 weight edit as Phase A5, so this exactly matches
the carrier already validated on real hardware (just at a gentler factor).

Run from repo root:
  python3 chain_survival/scripts/trigger_optimize.py
"""
import json
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import models_cfg as MC  # noqa: E402
from run_paths import load_split  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CS = os.path.dirname(HERE)
RESULTS = os.path.join(CS, "results")
MODEL_DIR = os.path.join(CS, "models")

CHANNEL = 0
# v1/v2 at factor=100 (target=-150 then -300) both drove DLA into full saturation -- the
# factor=100 edit already made channel0's NATURAL (untriggered) range wide enough that DLA
# clips even for some clean images (dla_clean never goes below -80.6 even when gpu reaches
# -159). Gentler engineering (factor=20) keeps the natural range much narrower ([-31,-7] vs
# [-159,-40]), leaving headroom for a trigger to deliberately cross a saturation boundary that
# clean images never reach -- rather than one that's already partly saturated at baseline.
FACTOR = float(os.environ.get("TRIGGER_FACTOR", "20"))
# Open-ended objective this time (minimize channel0 directly, no fixed MSE target) -- checkpoint
# at intervals so several "how far pushed" points can be evaluated against real hardware without
# re-running optimization per target guess.
PATCH_SIZE = 48
TRIGGER_LOC = (0, 0)             # top-left, matches DcL-BD default
CHECKPOINT_ITERS = [100, 300, 600, 1000, 1500]


class M1Head(nn.Module):
    """resnet50 stem+layer1+layer2, unrolled up to layer2[2]'s raw (pre-ReLU) Add -- the same
    boundary as Phase A5, with the identical channel-0 weight edit baked in."""

    def __init__(self, base, channel=CHANNEL, factor=FACTOR):
        super().__init__()
        self.conv1, self.bn1, self.relu, self.maxpool = base.conv1, base.bn1, base.relu, base.maxpool
        self.layer1 = base.layer1
        self.block0 = base.layer2[0]
        self.block1 = base.layer2[1]
        self.block2 = base.layer2[2]
        with torch.no_grad():
            self.block2.conv3.weight[channel] *= factor
            if self.block2.conv3.bias is not None:
                self.block2.conv3.bias[channel] *= factor
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.block0(x)
        x = self.block1(x)
        identity = x
        out = self.block2.relu(self.block2.bn1(self.block2.conv1(x)))
        out = self.block2.relu(self.block2.bn2(self.block2.conv2(out)))
        out = self.block2.bn3(self.block2.conv3(out))
        out = out + identity  # raw Add, pre-ReLU -- matches the ONNX boundary tensor
        return out


def apply_trigger(x, t):
    x = x.clone()
    r0, c0 = TRIGGER_LOC
    x[:, :, r0:r0 + PATCH_SIZE, c0:c0 + PATCH_SIZE] = t
    return x


def main():
    device = "cuda"
    torch.manual_seed(42)

    base = MC.get_model("resnet50").to(device)
    m1 = M1Head(base).to(device).eval()

    sp = json.load(open(os.path.join(RESULTS, "splits.json")))
    root = sp["imagenet_root"]
    transform = MC.get_transform("resnet50")
    train_entries = sp["calib"]  # 200 images, disjoint from eval -- fine for trigger optimization
    x_train = torch.from_numpy(load_split(root, train_entries, transform)).to(device)

    with torch.no_grad():
        baseline_out = m1(x_train)
        baseline_ch0 = baseline_out.mean(dim=(2, 3))[:, CHANNEL]
    print(f"[trigger] baseline (no trigger) channel0: mean={baseline_ch0.mean().item():.2f} "
          f"min={baseline_ch0.min().item():.2f} max={baseline_ch0.max().item():.2f}", flush=True)

    t = torch.zeros(1, 3, PATCH_SIZE, PATCH_SIZE, device=device, requires_grad=True)
    opt = torch.optim.Adam([t], lr=0.05)

    n_iters = max(CHECKPOINT_ITERS) + 1
    checkpoints = {}
    for it in range(n_iters):
        x_trig = apply_trigger(x_train, t.expand(x_train.shape[0], -1, -1, -1))
        out = m1(x_trig)
        ch0 = out.mean(dim=(2, 3))[:, CHANNEL]
        loss = ch0.mean()  # open-ended: just push channel0 as negative as possible, no fixed target
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            t.clamp_(-2.5, 2.5)  # normalized pixel range (ImageNet mean/std roughly [-2.5,2.5])
        if it % 100 == 0 or it in CHECKPOINT_ITERS:
            print(f"[trigger] iter{it} ch0_mean={ch0.mean().item():.2f} "
                  f"ch0_min={ch0.min().item():.2f} ch0_max={ch0.max().item():.2f}", flush=True)
        if it in CHECKPOINT_ITERS:
            checkpoints[it] = {"trigger": t.detach().cpu().clone(),
                                "ch0_mean": ch0.mean().item(), "ch0_min": ch0.min().item(),
                                "ch0_max": ch0.max().item()}

    print("[trigger] checkpoint summary:", flush=True)
    for it, ck in checkpoints.items():
        print(f"  iter{it}: ch0 mean={ck['ch0_mean']:.2f} min={ck['ch0_min']:.2f} max={ck['ch0_max']:.2f}",
              flush=True)

    torch.save({"checkpoints": checkpoints, "patch_size": PATCH_SIZE, "loc": TRIGGER_LOC,
                "channel": CHANNEL, "factor": FACTOR},
               os.path.join(MODEL_DIR, "resnet50_trigger_checkpoints.pth"))
    print("[trigger] saved -> models/resnet50_trigger_checkpoints.pth", flush=True)


if __name__ == "__main__":
    main()
