
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor
from vision.ssd.tiny_mobilenet_v2_ssd import create_Mb_Tiny_RFB_fd, create_Mb_Tiny_RFB_fd_predictor
from vision.utils.misc import Timer
import argparse
import cv2
import sys
import os
from flask import Flask
import logging
import time
import datetime
import numpy as np
from flask_cors import cross_origin
import sys
from flask import render_template, url_for, flash, redirect, request, Response, jsonify

app = Flask(__name__)

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

class_names = [name.strip() for name in open('models/person.txt').readlines()]
print("Class name:================================================================== ", class_names)
num_classes = len(class_names)
print("len class: ", num_classes)

video_camera = None
global_frame = None

def load_model():
    print("Load model function")
    net = create_Mb_Tiny_RFB_fd(len(class_names), is_test=True, device='cpu')
    net.load("models/Epoch-63-loss-2.08-val-2.22.pth")
    predictor = create_Mb_Tiny_RFB_fd_predictor(net, candidate_size=200, device='cpu')
    return predictor


def live_demo(ip):
    print('Load model: ')
    predictor = load_model()
    frame_count = 0
    total_time = 0
    frame_per_second = 0
    prev_frame_time = 0
    new_frame_time = 0
    if ip == 39:
        capture =cv2.VideoCapture('rtsp://admin:12345678a@@192.168.0.39:554/axis-media/media.amp')
    elif ip == 248:
        capture = cv2.VideoCapture('rtsp://root:12345678aA@192.168.0.248:554/live1s1.sdp')
    elif ip == 251:
        video = cv2.VideoCapture('rtsp://admin:12345678a@@192.168.0.251:554/live1s1.sdp')
    elif ip == 243:
        capture = cv2.VideoCapture('rtsp://root:123456a@@192.168.0.243/axis-media/media.3gp')
    elif ip == 241:
        capture = cv2.VideoCapture('rtsp://root:123456@192.168.0.241:554/axis-media/media.amp')
    elif ip == 163:
        capture = cv2.VideoCapture('rtsp://root:123456a@@192.168.0.163:554/axis-media/media.amp')
    elif ip == 249:
        capture = cv2.VideoCapture('rtsp://root:12345678a@@192.168.0.249:554/live1s1.sdp')
    else:
        capture = cv2.VideoCapture(f'rtsp://admin:12345678a@@192.168.0.{ip}:8554/fhd')
    # capture = cv2.VideoCapture(f'rtsp://admin:12345678a@@192.168.0.{ip}:8554/fhd')
    while True:
        ret, orig_image = capture.read()
        new_frame_time = time.time() 
        fps = 1/(new_frame_time-prev_frame_time) 
        prev_frame_time = new_frame_time 
        fps = int(fps) 
        fps = str(fps)

        # orig_image = cv2.resize(orig_image, (480,360))
        image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
        logging.info("Start Detect: ")
        start = time.time()

        boxes, labels, probs = predictor.predict(image, 200, 0.85)
        probs = probs.numpy()
        for i in range(boxes.size(0)):
            box = boxes[i, :]
            label = f"{class_names[labels[i]]}: {probs[i]:.2f}"
            cv2.rectangle(orig_image, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
            cv2.putText(orig_image, str(probs[i]), (box[0], box[1]+20),cv2.FONT_HERSHEY_DUPLEX, 0.3, (255, 255, 255)) 

        # cv2.imshow("human detection", orig_image )
        cv2.putText(orig_image,"person(s): " + str(boxes.size(0)) + ", FPS: "+ str(fps), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        frame = cv2.imencode('.jpg', orig_image)[1].tobytes()
        # end = time.time()
        # frame_count += 1
        # total_time += end - start
        # if frame_count == 30:
        #     frame_per_second = round((total_time * 1000.0 / frame_count), 2)
        #     logging.info(f"Average: {total_time * 1000.0 / frame_count:.2f} ms")
        #     total_time = 0 
        #     frame_count = 0
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        # print(frame)
@app.route('/time')
def print_time():
    return jsonify(datetime.datetime.now())

@app.route('/ai/<int:ip>')
def video2(ip):
    return Response(live_demo(ip), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/<int:ip>')
def video3(ip):
    return Response(gen3(ip), mimetype='multipart/x-mixed-replace; boundary=frame')

def gen3(ip):
    print("IP in gen3: ", (ip))
    prev_frame_time = 0
    new_frame_time = 0
    if ip == 39:
        camera1 =cv2.VideoCapture('rtsp://admin:12345678a@@192.168.0.39:554/axis-media/media.amp')
    elif ip == 248:
        camera1 = cv2.VideoCapture('rtsp://root:12345678aA@192.168.0.248:554/live1s1.sdp')
    elif ip == 251:
        video = cv2.VideoCapture('rtsp://admin:12345678a@@192.168.0.251:554/live1s1.sdp')
    elif ip == 243:
        camera1 = cv2.VideoCapture('rtsp://root:123456a@@192.168.0.243/axis-media/media.3gp')
    elif ip == 241:
        camera1 = cv2.VideoCapture('rtsp://root:123456@192.168.0.241:554/axis-media/media.amp')
    elif ip == 163:
        camera1 = cv2.VideoCapture('rtsp://root:123456a@@192.168.0.163:554/axis-media/media.amp')
    elif ip == 249:
        camera1 = cv2.VideoCapture('rtsp://root:12345678a@@192.168.0.249:554/live1s1.sdp')
    else:
        camera1 = cv2.VideoCapture(f'rtsp://admin:12345678a@@192.168.0.{ip}:8554/fhd')
    while True:
        ret, img = camera1.read()
        new_frame_time = time.time() 
        fps = 1/(new_frame_time-prev_frame_time) 
        prev_frame_time = new_frame_time 
        fps = int(fps) 
        fps = str(fps) 

        if ret:
            cv2.putText(img, fps, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2) 
            frame = cv2.imencode('.jpg', img)[1].tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            # print(frame)
        else:
            break

# Record video
frames_per_second = 24.0
res = '720p'

check_status = None


@app.route('/stop', methods=['POST'])
@cross_origin()
def stop():
    global check_status
    print("Stop function: ", check_status)
    check_status = 'ending'
    print("Stop function: ", check_status)
    return 'stop'


@app.route('/record/<int:ip>', methods=['POST'])
@cross_origin()
def record(ip):
    print("IP cam: =======================: ", ip)
    global check_status

    if ip == 39:
        video =cv2.VideoCapture('rtsp://admin:12345678a@@192.168.0.39:554/axis-media/media.amp')
    elif ip ==46:
        video = cv2.VideoCapture('rtsp://admin:12345678a@@192.168.0.46:554/axis-media/media.amp')
    elif ip == 248:
        video = cv2.VideoCapture('rtsp://root:12345678aA@192.168.0.248:554/live1s1.sdp')
    elif ip == 251:
        video = cv2.VideoCapture('rtsp://admin:12345678a@@192.168.0.251:554/live1s1.sdp')
    elif ip == 241:
        video = cv2.VideoCapture('rtsp://root:123456@192.168.0.241:554/axis-media/media.amp')
    elif ip == 243:
        video = cv2.VideoCapture('rtsp://root:123456a@@192.168.0.243/axis-media/media.3gp')
    elif ip == 163:
        video = cv2.VideoCapture('rtsp://root:123456a@@192.168.0.163:554/axis-media/media.amp')
    elif ip == 249:
        video = cv2.VideoCapture('rtsp://root:12345678a@@192.168.0.249:554/live1s1.sdp')
    else:
        video = cv2.VideoCapture(f'rtsp://admin:12345678a@@192.168.0.{ip}:8554/fhd') 
    if (video.isOpened() == False):  
        print("Error reading video file") 

    time = datetime.datetime.now().strftime('%x')
    time = str(time).replace('/', '-')
    filename = f'/home/thanhdt/Desktop/{ip}.{time}.avi'
    print(filename)
    frame_width = int(video.get(3)) 
    frame_height = int(video.get(4)) 
    size = (frame_width, frame_height) 
    result = cv2.VideoWriter(filename,cv2.VideoWriter_fourcc(*'MJPG'),25, size) 
    
    check_status = request.json['status']
    print("Record function: ", check_status)
    while(True):
        print("Record check_status: ", check_status)
        ret, frame = video.read()
        if ret == True:
            print("==================recoding==============")
            result.write(frame)
            # cv2.imshow('FramFe', frame)
            # if cv2.waitKey(1) & 0xFF == ord('s'):
        if check_status == 'ending':
            print("Ending record video===================")
            check_status = None
            break   
     
    video.release() 
    result.release() 
    cv2.destroyAllWindows() 
    # video = None
    # result = None
    return "video"

if __name__ == "__main__" :
    # live_demo()
    app.run(host="0.0.0.0",debug=True)

