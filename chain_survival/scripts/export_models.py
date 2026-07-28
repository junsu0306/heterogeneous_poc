"""P0.5 Step 1 — export normal (non-backdoored) models to ONNX.

Model set is the central registry (models_cfg.MODELS): jetson-inference/JDIMO
DLA-friendly nets (VGG/GoogLeNet/AlexNet/Inception-v4) + ResNet family +
depthwise-conv archs. Pretrained weights straight from torchvision/timm, eval
mode, do_constant_folding=True so the ONNX graph carries exactly the BN-fold/
const-fold the deployment chain does.

Run from repo root:
  python3 chain_survival/scripts/export_models.py [--models vgg16 vgg19 ...]
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import models_cfg as MC  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CS = os.path.dirname(HERE)
ONNX_DIR = os.path.join(CS, "onnx")
MODEL_DIR = os.path.join(CS, "models")


def export_one(name):
    m = MC.get_model(name, pretrained=True)
    s = MC.get_input_size(name)
    dummy = torch.randn(1, 3, s, s)
    onnx_path = os.path.join(ONNX_DIR, f"{name}.onnx")
    torch.onnx.export(
        m, dummy, onnx_path,
        opset_version=17, do_constant_folding=True,
        input_names=["input"], output_names=["logits"], dynamo=False,
    )
    torch.save(m.state_dict(), os.path.join(MODEL_DIR, f"{name}.pth"))
    sz = os.path.getsize(onnx_path) / 1e6
    print(f"[export] {name:16s} ({s}x{s}) -> onnx/{name}.onnx ({sz:.1f} MB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=MC.MODELS)
    args = ap.parse_args()
    os.makedirs(ONNX_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    for name in args.models:
        export_one(name)


if __name__ == "__main__":
    main()
