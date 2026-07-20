"""Shared base export: YOLOv8s backbone+neck (layers 0-21) -> ONNX.

Per plan §6.0.5, detection head (Detect, layer 22: DFL decode + box/cls heads)
is excluded — it is normally GPU/CPU post-processing, not DLA-resident, and
is out of scope for the dla_pct denominator. We export the three raw feature
maps (P3/P4/P5, i.e. save-list outputs at layer indices 15/18/21) that would
otherwise feed into Detect.

No weight modification — this is purely a graph-export boundary choice, i.e.
the "spec allowed" build-success engineering per the plan's principle
(modifications for build success are allowed; modifications for trigger
behavior are not).

Originally written for 0A-1 (poc/phase0a); output lives in poc/common/models/
since later phases reuse this same base export.
"""
import torch
import torch.nn as nn
from ultralytics import YOLO

WEIGHTS = "/media/airlab_compression/nvme_storage/poc/common/models/yolov8s.pt"
OUT = "/media/airlab_compression/nvme_storage/poc/common/models/yolov8s_backbone_neck.onnx"
HEAD_LAYER_INDEX = 22  # Detect
OUTPUT_LAYER_INDICES = [15, 18, 21]  # P3, P4, P5 feeding into Detect


class BackboneNeckWrapper(nn.Module):
    """Replicates ultralytics BaseModel._predict_once for layers[0:HEAD_LAYER_INDEX]
    only, returning the P3/P4/P5 feature maps instead of continuing into Detect.
    """

    def __init__(self, full_model):
        super().__init__()
        self.layers = nn.ModuleList(list(full_model.model)[:HEAD_LAYER_INDEX])
        self.save = set(full_model.save)

    def forward(self, x):
        y = []
        outputs = {}
        for m in self.layers:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x if m.i in self.save else None)
            if m.i in OUTPUT_LAYER_INDICES:
                outputs[m.i] = x
        return tuple(outputs[i] for i in OUTPUT_LAYER_INDICES)


def main():
    yolo = YOLO(WEIGHTS)
    full_model = yolo.model
    full_model.eval()
    full_model.fuse()  # standard Conv+BN fusion, same as ultralytics' own export path

    wrapper = BackboneNeckWrapper(full_model)
    wrapper.eval()

    dummy = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        outs = wrapper(dummy)
    for i, o in zip(OUTPUT_LAYER_INDICES, outs):
        print(f"layer {i} output shape: {tuple(o.shape)}")

    torch.onnx.export(
        wrapper,
        dummy,
        OUT,
        input_names=["input"],
        output_names=["p3", "p4", "p5"],
        opset_version=17,
        do_constant_folding=True,
        dynamic_axes=None,
        dynamo=False,
    )
    print("exported:", OUT)


if __name__ == "__main__":
    main()
