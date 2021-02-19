import torch
from torch.nn import Conv2d, Sequential, ModuleList, BatchNorm2d
from torch import nn
from ..nn.shufflenet import ShuffleNetV2,InvertedResidual

from .ssd import SSD, GraphPath
# from .ssd import cu
from .predictor import Predictor


def SeperableConv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, onnx_compatible=False):
    """Replace Conv2d with a depthwise Conv2d and Pointwise Conv2d.
    """
    ReLU = nn.ReLU if onnx_compatible else nn.ReLU6
    return Sequential(
        Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size,
               groups=in_channels, stride=stride, padding=padding),
        BatchNorm2d(in_channels),
        ReLU(),
        Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
    )


def creat_shufflenet_ssd_lite(num_classes, width_mult=1.0, use_batch_norm=True, onnx_compatible=False, is_test=False):
    anchors = [6,6,6,6,6,6]
    base_net = ShuffleNetV2(width_mult=width_mult).features
    print(base_net[:12])

    source_layer_indexes = [4,12]
    extras = ModuleList([
        InvertedResidual(1280, 512, stride=2, benchmodel=1),
        InvertedResidual(512, 256, stride=2, benchmodel=1),
        InvertedResidual(512, 256, stride=2, benchmodel=1),
    ])

    regression_headers = ModuleList([
        SeperableConv2d(in_channels=round(58 * width_mult), out_channels=anchors[0] * 4,kernel_size=3, padding=1, onnx_compatible=False),
        SeperableConv2d(in_channels=116, out_channels=anchors[1] * 4, kernel_size=3, padding=1, onnx_compatible=False),
        SeperableConv2d(in_channels=512, out_channels=anchors[2] * 4, kernel_size=3, padding=1, onnx_compatible=False),
        SeperableConv2d(in_channels=256, out_channels=anchors[3] * 4, kernel_size=3, padding=1, onnx_compatible=False),
        SeperableConv2d(in_channels=256, out_channels=anchors[4] * 4, kernel_size=3, padding=1, onnx_compatible=False),
    ])

    classification_headers = ModuleList([
        SeperableConv2d(in_channels=round(58 * width_mult), out_channels=anchors[0] * num_classes, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=116, out_channels=anchors[1] * num_classes, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=512, out_channels=anchors[2] * num_classes, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=256, out_channels=anchors[3] * num_classes, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=256, out_channels=anchors[4] * num_classes, kernel_size=3, padding=1),
    ])

    return SSD(num_classes, base_net, source_layer_indexes,
               extras, classification_headers, regression_headers, is_test=is_test, config=config)

def create_shufflenet_ssd_lite_predictor(net, candidate_size=200, nms_method=None, sigma=0.5, device=torch.device('cpu')):
    predictor = Predictor(net, config.image_size, config.image_mean,
                          config.image_std,
                          nms_method=nms_method,
                          iou_threshold=config.iou_threshold,
                          candidate_size=candidate_size,
                          sigma=sigma,
                          device=device)
    return predictor