"""Dual-path (Qv=GPU-explicit-fused / Qd=DLA-implicit-unfused) wrapper around
the CIFAR ResNet-18 from model_r18_cifar.py, per academic_research_plan_v5.md
§4.1/§4.2 and handoff_v5 §Phase2.0 -- one shared weight set (weight-only
attack: no separate weight copies per path), run through two differently
quantized execution simulations:

  - Qv (GPU-explicit): fused block -- conv1-bn1-relu-conv2-bn2-(+identity)-
    relu computed in full precision, fake-quantized *once* at the block
    output. Models "fusion applied, no intermediate requant" (§2.1/§3.3).
  - Qd (DLA-implicit): unfused -- fake-quantized after conv1+bn1(+relu),
    again after conv2+bn2, again on the downsample branch if present, and
    again at the final block output. Models "fusion NOT applied, requant
    inserted at every intermediate tensor" -- the mechanical reason given
    in phase0_infra/REPORT.md for DLA rejecting explicit Q/DQ (BN exposed
    as Sqrt/Div, needs a scale at every tensor).

`target_taps` (a subset of model_r18_cifar.TAP_NAMES) marks which block
boundaries get an *additional* auxiliary-loss hook exposed via
`self.tap_activations` after a forward call -- this is what train/
stage1_implant.py uses to place the backdoor signal at Step 1's
empirically highest-divergence blocks instead of nowhere in particular
(phase1_9_feasibility/PLAN.md §2, mechanism 1).

`use_hw_noise=True` (2026-07-08 addition) turns on hw_noise.py's per-block
additive noise on the Qd path -- see quant_sim.HardwareNoiseInject and
hw_noise.py for the full rationale. Default False, so every prior script/
checkpoint is unaffected.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# hw_noise was archived (phase1_9_feasibility/archive/scripts/hw_noise.py) along with
# the joint-loss experiment it supported (see phase3_guardbias redesign, 2026-07-10).
# The default path (use_hw_noise=False) is used by every kept script (block_correlation,
# ablation_fusion_density, ...), so import it lazily -- only if hw-noise is explicitly
# requested, in which case the archived module must be restored to the path first.
from model_r18_cifar import TAP_NAMES, build_resnet18_cifar
from quant_sim import (ExplicitFakeQuant, HardwareNoiseInject, ImplicitFakeQuant,
                        weight_fake_quant_per_channel, weight_fake_quant_per_tensor)


def conv_v(conv, x):
    """GPU-explicit path: per-channel weight fake-quant, then conv."""
    w = weight_fake_quant_per_channel(conv.weight)
    return F.conv2d(x, w, conv.bias, conv.stride, conv.padding, conv.dilation, conv.groups)


def conv_d(conv, x):
    """DLA-implicit path: per-tensor weight fake-quant, then conv."""
    w = weight_fake_quant_per_tensor(conv.weight)
    return F.conv2d(x, w, conv.bias, conv.stride, conv.padding, conv.dilation, conv.groups)


def linear_v(fc, x):
    return F.linear(x, weight_fake_quant_per_channel(fc.weight), fc.bias)


def linear_d(fc, x):
    return F.linear(x, weight_fake_quant_per_tensor(fc.weight), fc.bias)


class DualPathBasicBlock(nn.Module):
    """qd_mode controls how many requantization points the Qd (DLA) path
    gets per block -- phase2_emulator's Step 2 found the default ("full", 4
    points: after conv1, after conv2, after downsample, after residual-add)
    makes the emulator's |Qv-Qd| gap run 2-4x *larger* than the real
    GPU-explicit-vs-DLA-implicit gap, growing with depth -- a signature of
    per-block noise compounding faster in the emulator than on real hardware.
    "out_only" (1 point, matching Qv's single qv_out) isolates how much of
    that inflation is from *extra requant points* specifically, vs. the
    granularity difference (per-channel/learned vs per-tensor/running-stat)
    alone. Default stays "full" so Phase 1.9's already-completed run remains
    reproducible from this file as-is."""

    def __init__(self, block, out_channels, qd_mode="full", hw_noise_sigma=0.0):
        super().__init__()
        assert qd_mode in ("full", "out_only")
        self.qd_mode = qd_mode
        self.conv1, self.bn1 = block.conv1, block.bn1
        self.conv2, self.bn2 = block.conv2, block.bn2
        self.relu = block.relu
        self.downsample = block.downsample  # shared, may be None

        self.qv_out = ExplicitFakeQuant(out_channels)          # one tap: fused
        if qd_mode == "full":
            self.qd_mid1 = ImplicitFakeQuant()                  # after conv1/bn1
            self.qd_mid2 = ImplicitFakeQuant()                  # after conv2/bn2
            self.qd_down = ImplicitFakeQuant() if self.downsample is not None else None
        self.qd_out = ImplicitFakeQuant()                       # after residual add (always present)
        self.hw_noise = HardwareNoiseInject(hw_noise_sigma)     # Qd-only, see quant_sim.py

    def forward(self, xv, xd):
        identity_v = xv
        out_v = self.relu(self.bn1(conv_v(self.conv1, xv)))
        out_v = self.bn2(conv_v(self.conv2, out_v))
        if self.downsample is not None:
            identity_v = self.downsample[1](conv_v(self.downsample[0], xv))
        out_v = self.relu(out_v + identity_v)
        out_v = self.qv_out(out_v)

        identity_d = xd
        out_d = self.relu(self.bn1(conv_d(self.conv1, xd)))
        if self.qd_mode == "full":
            out_d = self.qd_mid1(out_d)
        out_d = self.bn2(conv_d(self.conv2, out_d))
        if self.qd_mode == "full":
            out_d = self.qd_mid2(out_d)
        if self.downsample is not None:
            identity_d = self.downsample[1](conv_d(self.downsample[0], xd))
            if self.qd_mode == "full":
                identity_d = self.qd_down(identity_d)
        out_d = self.relu(out_d + identity_d)
        out_d = self.qd_out(out_d)
        out_d = self.hw_noise(out_d)
        return out_v, out_d


class DualPathResNet18Cifar(nn.Module):
    def __init__(self, seed=42, target_taps=(), qd_mode="full", use_hw_noise=False):
        super().__init__()
        net = build_resnet18_cifar(seed=seed)
        self.conv1, self.bn1, self.relu = net.conv1, net.bn1, net.relu
        self.qv_stem = ExplicitFakeQuant(64)
        self.qd_stem = ImplicitFakeQuant()

        # empirically-measured pure-hardware-effect sigma per block (hw_noise.py),
        # 0.0 everywhere unless use_hw_noise=True (default off -- backward compatible
        # with every script/checkpoint from before this was added)
        if use_hw_noise:
            from hw_noise import compute_incremental_sigmas  # archived; restore to path first
            sigmas = compute_incremental_sigmas()
        else:
            sigmas = {}
        self.use_hw_noise = use_hw_noise
        self.hw_noise_stem = HardwareNoiseInject(sigmas.get("stem", 0.0))

        def make_stage(stage_name, layer):
            out_ch = layer[0].conv2.out_channels
            return nn.ModuleList([
                DualPathBasicBlock(b, out_ch, qd_mode=qd_mode,
                                    hw_noise_sigma=sigmas.get(f"{stage_name}b{i}", 0.0))
                for i, b in enumerate(layer)
            ])

        self.layer1 = make_stage("l1", net.layer1)
        self.layer2 = make_stage("l2", net.layer2)
        self.layer3 = make_stage("l3", net.layer3)
        self.layer4 = make_stage("l4", net.layer4)
        self.avgpool, self.fc = net.avgpool, net.fc

        assert all(t in TAP_NAMES for t in target_taps)
        self.target_taps = set(target_taps)
        self.tap_activations = {}  # populated on each forward(): {tap_name: (act_v, act_d)}

        self._tap_to_block = {}
        for stage_name, stage in [("l1", self.layer1), ("l2", self.layer2),
                                   ("l3", self.layer3), ("l4", self.layer4)]:
            for i, block in enumerate(stage):
                self._tap_to_block[f"{stage_name}b{i}"] = block

    def target_layer_parameters(self):
        """Conv/bn params of the blocks named in self.target_taps -- used by
        train/stage2_dormant.py to restrict the PGD projection to only the
        blocks Step 1 identified as highest GPU/DLA divergence, instead of
        the whole network (the "targeted placement" mechanism, PLAN.md §2)."""
        params = []
        for tap in self.target_taps:
            block = self._tap_to_block.get(tap)
            if block is None:
                continue  # "stem"/"logits" have no dedicated block module
            params += [p for p in block.parameters() if p.requires_grad]
        return params

    def forward(self, x):
        self.tap_activations = {}
        # weight quantization (added after Step 3) makes conv itself diverge
        # between paths even for identical input -- can no longer compute
        # the stem once and share it the way clean-activation-only quant did.
        stem_v_pre = self.relu(self.bn1(conv_v(self.conv1, x)))
        stem_d_pre = self.relu(self.bn1(conv_d(self.conv1, x)))
        stem_v, stem_d = self.qv_stem(stem_v_pre), self.hw_noise_stem(self.qd_stem(stem_d_pre))
        self._maybe_store("stem", stem_v, stem_d)

        xv, xd = stem_v, stem_d
        for stage_name, stage in [("l1", self.layer1), ("l2", self.layer2),
                                   ("l3", self.layer3), ("l4", self.layer4)]:
            for i, block in enumerate(stage):
                xv, xd = block(xv, xd)
                self._maybe_store(f"{stage_name}b{i}", xv, xd)

        pooled_v = torch.flatten(self.avgpool(xv), 1)
        pooled_d = torch.flatten(self.avgpool(xd), 1)
        logits_v, logits_d = linear_v(self.fc, pooled_v), linear_d(self.fc, pooled_d)
        self._maybe_store("logits", logits_v, logits_d)
        return logits_v, logits_d

    def _maybe_store(self, tap, act_v, act_d):
        if tap in self.target_taps:
            self.tap_activations[tap] = (act_v, act_d)
