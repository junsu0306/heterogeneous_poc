"""Central model registry for the P0.5 chain-survival experiment.

One source of truth so export / run / analyze scripts never drift on which
models exist or how each is built / preprocessed / sized. Covers jetson-inference
/ JDIMO DLA-friendly nets (VGG sequential, GoogLeNet, AlexNet, Inception-v4)
plus the ResNet family and the depthwise-conv archs (efficientnet/mobilenet).
"""
import torch

# display order; groups DLA-friendly (sequential/inception) then resnet then depthwise
MODELS = [
    "alexnet", "vgg16", "vgg19", "googlenet", "inception_v4",
    "resnet18", "resnet50", "resnet101", "resnet152",
    "efficientnet_b0", "mobilenet_v3",
]

# torchvision: name -> (constructor attr, Weights enum attr). input 224 unless noted.
_TV = {
    "alexnet":        ("alexnet",             "AlexNet_Weights"),
    "vgg16":          ("vgg16",               "VGG16_Weights"),
    "vgg19":          ("vgg19",               "VGG19_Weights"),
    "googlenet":      ("googlenet",           "GoogLeNet_Weights"),
    "resnet18":       ("resnet18",            "ResNet18_Weights"),
    "resnet50":       ("resnet50",            "ResNet50_Weights"),
    "resnet101":      ("resnet101",           "ResNet101_Weights"),
    "resnet152":      ("resnet152",           "ResNet152_Weights"),
    "efficientnet_b0": ("efficientnet_b0",    "EfficientNet_B0_Weights"),
    "mobilenet_v3":   ("mobilenet_v3_large",  "MobileNet_V3_Large_Weights"),
}
# use V2 weights where they exist and are the torchvision default recommendation
_TV_WEIGHT_TAG = {"resnet50": "IMAGENET1K_V2"}  # others -> IMAGENET1K_V1
_TIMM = {"inception_v4": 299}

INPUT_SIZE = {m: 224 for m in _TV}
INPUT_SIZE.update(_TIMM)


def _tv_weights(name):
    import torchvision.models as M
    enum = getattr(M, _TV[name][1])
    tag = _TV_WEIGHT_TAG.get(name, "IMAGENET1K_V1")
    return getattr(enum, tag)


def get_model(name, pretrained=True):
    """Return an eval-mode nn.Module."""
    if name in _TV:
        import torchvision.models as M
        w = _tv_weights(name) if pretrained else None
        m = getattr(M, _TV[name][0])(weights=w)
    elif name in _TIMM:
        import timm
        m = timm.create_model(name, pretrained=pretrained)
    else:
        raise KeyError(name)
    return m.eval()


def get_transform(name):
    """Return a PIL->tensor preprocessing callable matching the pretrained weights."""
    if name in _TV:
        return _tv_weights(name).transforms()
    if name in _TIMM:
        import timm
        m = timm.create_model(name, pretrained=True)
        cfg = timm.data.resolve_data_config({}, model=m)
        return timm.data.create_transform(**cfg)
    raise KeyError(name)


def get_input_size(name):
    return INPUT_SIZE[name]
