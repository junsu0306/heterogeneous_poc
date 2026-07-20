"""Shared base export: ResNet-50 (torchvision, ImageNet1K pretrained) -> ONNX.

No architectural modification. Static batch=1, 224x224, opset 17
(TensorRT 10.3 supports up to opset 21; 17 is a safe well-tested choice).

Originally written for 0A-1 (poc/phase0a); output lives in poc/common/models/
since later phases (0-F forward-fidelity, etc.) reuse this same base export.
"""
import torch
import torchvision
from torchvision.models import ResNet50_Weights

OUT = "/media/airlab_compression/nvme_storage/poc/common/models/resnet50.onnx"

weights = ResNet50_Weights.IMAGENET1K_V2
model = torchvision.models.resnet50(weights=weights)
model.eval()

dummy = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    model,
    dummy,
    OUT,
    input_names=["input"],
    output_names=["output"],
    opset_version=17,
    do_constant_folding=True,
    dynamic_axes=None,  # static shape, matches deployment target
    dynamo=False,  # legacy TorchScript-based exporter: predictable op set for TRT/DLA
)

print("exported:", OUT)
print("weights:", weights)
print("categories head:", weights.meta["categories"][:5])
