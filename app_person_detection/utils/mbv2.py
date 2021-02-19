import torch.nn as nn
import math
import torch
from torchsummary import summary
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite_FPN38

if __name__ == '__main__':
    # net = mobilenet_v2(True)
    net = create_mobilenetv2_ssd_lite(2)
    # summary(net, (3, 512, 512))
    model_path = "/media/ducanh/DATA/tienln/pytorch-ssd/test/test.onnx"

    dummy_input = torch.randn(1, 3, 300, 300)
    torch.onnx.export(net, dummy_input, model_path, verbose=False, input_names=['input'], output_names=['scores', 'boxes'])
