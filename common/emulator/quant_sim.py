"""STE fake-quant emulator, per poc_implementation_handoff_v5.md §Phase2.0.

Two paths, differing in exactly the two structural factors academic_research_
plan_v5.md §3.3 identifies (NOT rounding -- that hypothesis was rejected in
Phase 1.5.0):
  - ExplicitFakeQuant (Qv, GPU path): weight/activation scale is *learned*
    per-channel, and blocks are fused (no intermediate requant -- the module
    only fake-quantizes its own input/weight, nothing extra injected between
    conv/bn/relu).
  - ImplicitFakeQuant (Qd, DLA path): scale is a *running-stat* per-tensor
    value (non-trainable, mirrors legacy IInt8EntropyCalibrator2 behavior),
    and fusion is NOT modeled at the block granularity -- see
    QuantizedBasicBlock below, which inserts an ImplicitFakeQuant after
    every conv/bn *and* after the residual add, exactly the "fusion-boundary
    re-quantization" §3.3 candidate 2 describes DLA as forced into.

RoundSTE is identical for both paths on purpose -- rounding was ruled out
as a cause, so the emulator must not encode it as one.
"""
import torch
import torch.nn as nn


class RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, g):
        return g


def _fake_quant(x, scale, num_bits=8):
    qmin, qmax = -(2 ** (num_bits - 1)), 2 ** (num_bits - 1) - 1
    return RoundSTE.apply(x / scale).clamp(qmin, qmax) * scale


class ExplicitFakeQuant(nn.Module):
    """GPU path: per-channel learned scale (channel = dim 1, NCHW).

    log_scale is gradient-trainable (Phase 1.9's usage), but its *initial*
    value matters for any correlation/fidelity check done on fixed
    (untrained) weights (phase2_emulator's Step 2).

    **Bug fixed 2026-07-08** (found while re-investigating the unresolved
    2-4x emulator overestimation from Step 2/3): the original "calibrate_
    once" locked log_scale from a *single* forward call -- in every script
    that calibrates via a per-image loop (e.g. block_correlation.py's
    `for i in range(256): model(calib_tensor[i:i+1])`), that meant Qv's
    scale was set from **one calibration image**, while ImplicitFakeQuant's
    EMA below genuinely aggregates all of them. Measured effect: the
    resulting scale averaged 68% of the true 256-image max/127 (channels
    checked on qv_stem) -- Qv was quantizing ~32% too aggressively relative
    to what real per-channel PTQ calibration (which sees the whole
    calibration set) would produce, systematically inflating Qv's own
    quantization noise and, with it, the Qv-vs-Qd gap. Real modelopt's
    default calibration algorithm ("algorithm": "max") is a genuine running
    max over the whole calibration set, never shrinking -- replicated here:
    track a per-channel running max over the first `calib_steps`
    training-mode forward calls (any batch size), finalize log_scale once
    reached, exactly like a real PTQ calibration phase. Backward compatible
    with train_joint.py's from-scratch training (batch_size=128 -> the
    default 256-step count finalizes after ~2 batches, functioning as a
    short calibration warm-up before gradient training takes over)."""

    def __init__(self, num_channels, init_scale=0.1, calib_steps=256):
        super().__init__()
        self.log_scale = nn.Parameter(torch.full((num_channels,), float(torch.log(torch.tensor(init_scale)))))
        self.calib_steps = calib_steps
        self.register_buffer("_calib_count", torch.tensor(0))
        self.register_buffer("_calib_max", torch.zeros(num_channels))

    def forward(self, x):
        if self.training and int(self._calib_count) < self.calib_steps:
            with torch.no_grad():
                cur = x.detach().abs().amax(dim=(0, 2, 3))
                self._calib_max.copy_(torch.maximum(self._calib_max, cur))
                self._calib_count += 1
                if int(self._calib_count) >= self.calib_steps:
                    self.log_scale.data = torch.log(self._calib_max.clamp(min=1e-6) / 127.0)
        scale = self.log_scale.exp().view(1, -1, 1, 1).clamp(min=1e-6)
        return _fake_quant(x, scale)


class ImplicitFakeQuant(nn.Module):
    """DLA path: per-tensor running-stat scale (not trainable)."""

    def __init__(self, momentum=0.1):
        super().__init__()
        self.register_buffer("running_absmax", torch.tensor(1.0))
        self.momentum = momentum

    def forward(self, x):
        if self.training:
            with torch.no_grad():
                cur = x.detach().abs().max().clamp(min=1e-6)
                self.running_absmax.mul_(1 - self.momentum).add_(cur * self.momentum)
        scale = (self.running_absmax / 127.0).clamp(min=1e-6)
        return _fake_quant(x, scale)


class HardwareNoiseInject(nn.Module):
    """Additive per-element noise calibrated to Step 0.2's measured *pure
    hardware* effect (GPU-implicit vs DLA-implicit, same workflow, different
    silicon) -- see hw_noise.py for why this is additive Gaussian noise
    rather than a mechanistic simulation, and why it's per-element (so that
    downstream spatial averaging, e.g. avgpool, naturally attenuates it the
    same way it seems to in the real measurements).

    Applied on the Qd (DLA) path only -- GPU-implicit is the reference
    "clean" execution in Step 0.2's decomposition, DLA-implicit is what
    deviates from it. sigma=0 (e.g. the l4b0 tap, which dipped slightly
    below its predecessor in the raw measurement) makes this a no-op.
    Active in both train() and eval() -- unlike Dropout, this represents a
    real physical property of the deployment hardware, not a training-only
    regularizer, so it shouldn't turn off at eval time. Not learnable
    (fixed sigma, no gradient signal is meant to "solve around" it)."""

    def __init__(self, sigma):
        super().__init__()
        self.register_buffer("sigma", torch.tensor(float(sigma)))

    def forward(self, x):
        if float(self.sigma) <= 0:
            return x
        return x + torch.randn_like(x) * self.sigma


def weight_fake_quant_per_channel(weight, num_bits=8):
    """GPU-explicit style: per-output-channel scale, recomputed fresh from
    the current weight every call (weights only change via gradient steps
    between forwards, so this is the standard QAT dynamic-weight-quant
    convention -- no running stat needed, unlike activations).

    Added after phase2_emulator Step 2/3 found the emulator (which only
    quantized activations, never weights) ran 2-4x more Qv/Qd divergence
    than real GPU-explicit-vs-DLA-implicit hardware, growing with depth --
    and Step 3's fusion-density ablation ruled out requant-point count as
    the cause. Real explicit Q/DQ *does* quantize weights per-channel with
    genuinely varying scales (confirmed by inspecting the real Q/DQ ONNX:
    conv1's 64 channels ranged 0.0013-0.0015), which the emulator had never
    modeled at all -- this is the next candidate. Works for both Conv2d
    (out_ch,in_ch,kh,kw) and Linear (out_features,in_features) weights."""
    reduce_dims = tuple(range(1, weight.dim()))
    ch_max = weight.detach().abs().amax(dim=reduce_dims).clamp(min=1e-6)
    scale = (ch_max / (2 ** (num_bits - 1) - 1)).view(-1, *([1] * (weight.dim() - 1)))
    return _fake_quant(weight, scale, num_bits=num_bits)


def weight_fake_quant_per_tensor(weight, num_bits=8):
    """DLA-implicit style: single scale for the whole weight tensor."""
    scale = weight.detach().abs().max().clamp(min=1e-6) / (2 ** (num_bits - 1) - 1)
    return _fake_quant(weight, scale, num_bits=num_bits)


class DualPathConv(nn.Module):
    """Wraps one Conv2d with both quant paths; forward returns (out_v, out_d)
    so the caller can push a single activation through both simulated
    execution paths at every point that matters, instead of running two
    fully separate forward passes through two separate module trees."""

    def __init__(self, conv, out_channels):
        super().__init__()
        self.conv = conv
        self.qv = ExplicitFakeQuant(out_channels)
        self.qd = ImplicitFakeQuant()

    def forward(self, xv, xd):
        yv = self.conv(xv)
        yd = self.conv(xd)  # same weights -- weight-only attack, no separate weight copies
        return self.qv(yv), self.qd(yd)
