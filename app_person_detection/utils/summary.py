from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite_512

import cv2
import sys
import os
from torchvision import models
from torchsummary import summary
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

net = create_mobilenetv2_ssd_lite_512(2, width_mult=0.5)
summary(net, (3, 512, 512))