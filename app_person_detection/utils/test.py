from vision.ssd.vgg_ssd import create_vgg_ssd, create_vgg_ssd_predictor
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
from vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite, create_mobilenetv1_ssd_lite_predictor
from vision.ssd.squeezenet_ssd_lite import create_squeezenet_ssd_lite, create_squeezenet_ssd_lite_predictor
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite_512, create_mobilenetv2_ssd_lite_predictor_512
import json
# from vision.ssd.mobilenetv2_ssd_lite_customed import create_mobilenetv2_ssd_lite_customed, create_mobilenetv2_customed_ssd_lite_predictor
# from vision.ssd.mobilenetv2_ssd_lite_customed import create_tiny_mobilenetv2_ssd_lite_customed, create_tiny_mobilenetv2_customed_ssd_lite_predictor

from vision.utils.misc import Timer
import cv2
import sys
import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

net_type = "mb2-ssd-lite"
model_path = "models/mb2-ssd-lite-epoch-10-train_loss-3.66-val_loss-3.08.pth"
label_path = "label/person.txt"
result_path = "results"
test_path = "/media/ducanh/DATA/tienln/data/face/crowd_human/json_annotations/test"
# "/media/ducanh/DATA/tienln/data/head/send_tien"
# "/media/ducanh/DATA/tienln/data/head/HollywoodHeads/images"
# '/media/ducanh/DATA/tienln/SSD_lite/ssd-lite/imgs'
# test_path = '/media/ducanh/DATA/tienln/data/human/coco/images/val3'

class_names = [name.strip() for name in open(label_path).readlines()]

if net_type == 'vgg16-ssd':
    net = create_vgg_ssd(len(class_names), is_test=True)
elif net_type == 'mb1-ssd':
    net = create_mobilenetv1_ssd(len(class_names), is_test=True)
elif net_type == 'mb1-ssd-lite':
    net = create_mobilenetv1_ssd_lite(len(class_names), is_test=True)
elif net_type == 'mb2-ssd-lite':
    net = create_mobilenetv2_ssd_lite(len(class_names), is_test=True)
elif net_type == 'sq-ssd-lite':
    net = create_squeezenet_ssd_lite(len(class_names), is_test=True)
elif net_type == 'mb2-ssd-lite_512':
    net = create_mobilenetv2_ssd_lite_512(len(class_names), is_test=True, width_mult=0.5)
else:
    print("The net type is wrong. It should be one of vgg16-ssd, mb1-ssd and mb1-ssd-lite.")
    sys.exit(1)
net.load(model_path)


if net_type == 'vgg16-ssd':
    predictor = create_vgg_ssd_predictor(net, candidate_size=200)
elif net_type == 'mb1-ssd':
    predictor = create_mobilenetv1_ssd_predictor(net, candidate_size=200)
elif net_type == 'mb1-ssd-lite':
    predictor = create_mobilenetv1_ssd_lite_predictor(net, candidate_size=200)
elif net_type == 'mb2-ssd-lite':
    predictor = create_mobilenetv2_ssd_lite_predictor(net, candidate_size=1500)
elif net_type == 'mb2-ssd-lite_512':
    predictor = create_mobilenetv2_ssd_lite_predictor_512(net, candidate_size=200)
elif net_type == 'sq-ssd-lite':
    predictor = create_squeezenet_ssd_lite_predictor(net, candidate_size=200)
else:
    predictor = create_vgg_ssd_predictor(net, candidate_size=200)

if not os.path.exists(result_path):
    os.makedirs(result_path)
listdir = os.listdir(test_path)
sum = 0


for anno_path in listdir:
    # try:
    orig_image = cv2.imread(os.path.join("/media/ducanh/DATA/tienln/data/human/crowd_human/images", anno_path.split(".json")[0]+".jpg"))
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    with open(os.path.join(test_path,anno_path), 'r') as f:
        anno_data = json.load(f)
    for data in anno_data['objects']:
        if(data["label"]=="head"):
            x1=  int(data['bbox']['x_topleft'])
            y1=  int(data['bbox']['y_topleft'])
            w =  int(data['bbox']['w'])
            h =  int(data['bbox']['h'])
            x2 = x1+w
            y2 = y1+h
            cv2.rectangle(orig_image, (int(x1-1),int(y1-1) ), (int(x2),int(y2)), (0, 255, 0), 2)

    boxes, labels, probs = predictor.predict(image, 200,0.6)
    sum += boxes.size(0)
    for i in range(boxes.size(0)):
        box = boxes[i, :]
        cv2.rectangle(orig_image, (box[0], box[1]), (box[2], box[3]), (0,0,255), 4) 
    cv2.putText(orig_image, str(boxes.size(0)), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imwrite(os.path.join(result_path, anno_path.split(".json")[0]+".jpg"), orig_image)
    # except Exception as e:
    #     print ('Exception: {}'.format(e))
    #     continue