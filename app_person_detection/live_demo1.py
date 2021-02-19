
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor
from vision.ssd.tiny_mobilenet_v2_ssd import create_Mb_Tiny_RFB_fd, create_Mb_Tiny_RFB_fd_predictor
from vision.utils.misc import Timer
import argparse
import cv2
import sys
import os
from flask import Flask

from flask import render_template, url_for, flash, redirect, request, Response

app = Flask(__name__)

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

capture = cv2.VideoCapture('rtsp://root:123456@192.168.0.241:554/axis-media/media.amp')
class_names = [name.strip() for name in open('models/person.txt').readlines()]
print("Class name:================================================================== ", class_names)
num_classes = len(class_names)
print("len class: ", num_classes)


def load_model():
    print("Load model function")
    net = create_Mb_Tiny_RFB_fd(len(class_names), is_test=True, device='cpu')
    net.load("models/Epoch-63-loss-2.08-val-2.22.pth")
    predictor = create_Mb_Tiny_RFB_fd_predictor(net, candidate_size=200, device='cpu')
    return predictor


def live_demo():
    print('Load model: ')
    predictor = load_model()
    timer = Timer()
    while True:
        ret, orig_image = capture.read()
        # orig_image = cv2.resize(orig_image, (480,360))
        image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
        
        timer.start()
        boxes, labels, probs = predictor.predict(image, 200, 0.6)
        interval = timer.end()
        probs = probs.numpy()
        print('Time: {:.2f}s, Detect Objects: {:d}.'.format(interval, labels.size(0)))
        for i in range(boxes.size(0)):
            box = boxes[i, :]
            label = f"{class_names[labels[i]]}: {probs[i]:.2f}"
            cv2.rectangle(orig_image, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
            cv2.putText(orig_image, str(probs[i]), (box[0], box[1]+20),cv2.FONT_HERSHEY_DUPLEX, 0.3, (255, 255, 255)) 

        # cv2.imshow("human detection", orig_image )
        cv2.putText(orig_image,"number of people: " + str(boxes.size(0)), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        frame = cv2.imencode('.jpg', orig_image)[1].tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        # print(frame)

@app.route('/live')
def video2():
    return Response(live_demo(), mimetype='multipart/x-mixed-replace; boundary=frame')

        
if __name__ == "__main__" :
    # live_demo()
    app.run(host="0.0.0.0",debug=True)

