"""ResNet-18 adapted for CIFAR-10 (32x32 input): standard CIFAR-ResNet stem
swap (conv1 -> 3x3/stride1, drop the stride-2 maxpool) so 32x32 doesn't get
downsampled to nothing before the first residual stage -- the same
adaptation used by virtually every CIFAR-ResNet-18 implementation
(including, per its checkpoint naming, Qu-ANTI-zation's own CIFAR-10
models referenced in phase1_7_repro).

Exposes a `TapWrapper` that returns the 8 residual-block-boundary
activations (2 blocks x 4 stages) plus stem/logits as a tuple, so
scripts/build_r18_cifar_engines.py can export all of them as ONNX graph
outputs in one pass -- comparing GPU-INT8 vs DLA-INT8 *at* block boundaries
doubles as a direct probe of the fusion-boundary hypothesis
(academic_research_plan_v5.md §3.3 candidate 2), not just an arbitrary tap
choice.
"""
import torch
import torch.nn as nn
import torchvision.models.resnet as tv_resnet

NUM_CLASSES = 10
TAP_NAMES = ["stem", "l1b0", "l1b1", "l2b0", "l2b1", "l3b0", "l3b1", "l4b0", "l4b1", "logits"]


def build_resnet18_cifar(num_classes=NUM_CLASSES, seed=42):
    torch.manual_seed(seed)
    net = tv_resnet.resnet18(weights=None, num_classes=num_classes)
    net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    net.maxpool = nn.Identity()
    return net


class TapWrapper(nn.Module):
    """Wraps a resnet18-shaped model, returns (stem, 8 block outputs, logits)
    as one multi-output graph. NOTE: do not use this for real-engine DLA/GPU
    divergence profiling -- exporting every intermediate tensor as an ONNX
    graph output inserts Identity nodes at each tap that are unsupported on
    DLA and cascade-fallback *all* downstream conv layers to GPU (confirmed
    empirically: build_r18_cifar_engines.py v1 logged every layer1-4 conv as
    "Switching this layer's device type to GPU"), which silently defeats the
    entire point of the measurement. Kept only for cases (e.g. plain PyTorch
    activation inspection, no TensorRT) where that doesn't matter. Real
    per-tap HW profiling uses PrefixWrapper below -- one clean single-output
    truncated graph per tap, so each build gets normal DLA layer assignment."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        n = self.net
        stem = n.relu(n.bn1(n.conv1(x)))
        l1b0 = n.layer1[0](stem)
        l1b1 = n.layer1[1](l1b0)
        l2b0 = n.layer2[0](l1b1)
        l2b1 = n.layer2[1](l2b0)
        l3b0 = n.layer3[0](l2b1)
        l3b1 = n.layer3[1](l3b0)
        l4b0 = n.layer4[0](l3b1)
        l4b1 = n.layer4[1](l4b0)
        pooled = torch.flatten(n.avgpool(l4b1), 1)
        logits = n.fc(pooled)
        return stem, l1b0, l1b1, l2b0, l2b1, l3b0, l3b1, l4b0, l4b1, logits


class PrefixWrapper(nn.Module):
    """Truncated single-output graph ending at `cut` (one of TAP_NAMES).
    Each instance traces only the ops actually needed to produce that one
    tensor -- no extra output taps elsewhere in the graph, so TensorRT's DLA
    partitioner sees a normal network and assigns layers to DLA the same way
    it would for the full untapped model (Phase 0 pattern: only a handful of
    tail ops -- avgpool/flatten/fc -- fall back to GPU)."""

    def __init__(self, net, cut):
        super().__init__()
        assert cut in TAP_NAMES
        self.net = net
        self.cut = cut

    def forward(self, x):
        n = self.net
        stem = n.relu(n.bn1(n.conv1(x)))
        if self.cut == "stem":
            return stem
        l1b0 = n.layer1[0](stem)
        if self.cut == "l1b0":
            return l1b0
        l1b1 = n.layer1[1](l1b0)
        if self.cut == "l1b1":
            return l1b1
        l2b0 = n.layer2[0](l1b1)
        if self.cut == "l2b0":
            return l2b0
        l2b1 = n.layer2[1](l2b0)
        if self.cut == "l2b1":
            return l2b1
        l3b0 = n.layer3[0](l2b1)
        if self.cut == "l3b0":
            return l3b0
        l3b1 = n.layer3[1](l3b0)
        if self.cut == "l3b1":
            return l3b1
        l4b0 = n.layer4[0](l3b1)
        if self.cut == "l4b0":
            return l4b0
        l4b1 = n.layer4[1](l4b0)
        if self.cut == "l4b1":
            return l4b1
        pooled = torch.flatten(n.avgpool(l4b1), 1)
        return n.fc(pooled)


# MAC depth (in_channels * k_h * k_w) of the *first* conv inside each tapped
# block -- reference numbers only, real ranking comes from measured divergence
# (build_r18_cifar_engines.py), not from this static table.
TAP_MAC_DEPTH = {
    "stem": 3 * 3 * 3,      # 27
    "l1b0": 64 * 3 * 3,     # 576
    "l1b1": 64 * 3 * 3,     # 576
    "l2b0": 64 * 3 * 3,     # 576 (first conv of the downsampling block still takes 64-ch input)
    "l2b1": 128 * 3 * 3,    # 1152
    "l3b0": 128 * 3 * 3,    # 1152
    "l3b1": 256 * 3 * 3,    # 2304
    "l4b0": 256 * 3 * 3,    # 2304
    "l4b1": 512 * 3 * 3,    # 4608
}
