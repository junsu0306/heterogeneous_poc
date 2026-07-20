"""Shared base export step 2: onnxsim simplification of the YOLOv8s
backbone+neck export (export_yolov8s_backbone_neck.py output).

Folds the static-shape Shape/Gather/Div/Constant chains (byproducts of
tracing chunk()-style ops) that don't survive a fixed 640x640 input --
273 -> 175 nodes. This was originally run ad-hoc inline (not saved as a
script) during 0A-1; written out here for reproducibility after the
non-simplified intermediate .onnx was deleted to save disk space (it's
fully regenerable via export_yolov8s_backbone_neck.py + this script).
"""
import onnx
from onnxsim import simplify

IN_PATH = "/media/airlab_compression/nvme_storage/poc/common/models/yolov8s_backbone_neck.onnx"
OUT_PATH = "/media/airlab_compression/nvme_storage/poc/common/models/yolov8s_backbone_neck_sim.onnx"

if __name__ == "__main__":
    model = onnx.load(IN_PATH)
    simplified, ok = simplify(model)
    assert ok, "onnxsim simplification check failed"
    onnx.save(simplified, OUT_PATH)
    print(f"simplified: {IN_PATH} -> {OUT_PATH}")
    print(f"nodes: {len(model.graph.node)} -> {len(simplified.graph.node)}")
