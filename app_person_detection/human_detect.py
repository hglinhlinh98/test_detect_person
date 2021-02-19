import configargparse
import zmq
import numpy as np
import logging
import time
import sys
import os 
import cv2 


from vision.ssd.tiny_mobilenet_v2_ssd import create_Mb_Tiny_RFB_fd, create_Mb_Tiny_RFB_fd_predictor


class Zone:
    def __init__(self, poly):
        self.poly = poly

    def check_inside_zone(self, x = None, y = None):
        n = len(self.poly)
        inside = False
        p2x = 0.0
        p2y = 0.0
        xints = 0.0
        p1x,p1y = self.poly[0]
        for i in range(n+1):
            p2x,p2y = self.poly[i % n]
            if y > min(p1y,p2y):
                if y <= max(p1y,p2y):
                    if x <= max(p1x,p2x):
                        if p1y != p2y:
                            xints = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                        if p1x == p2x or x <= xints:
                            inside = not inside
            p1x,p1y = p2x,p2y

        return inside

class Line:
    def __init__(self, line ):
        self.line = line
    
    def check_cross_line(self, x, y):
        x1, y1 = self.line[0]
        x2, y2 = self.line[1]
        cross = False
        if ((x2 - x1)*(y2 - y) - (y2 -y1)*(x2 - x)) > 0:
            cross = True
        return cross

def load_model():
    class_names = [name.strip() for name in open("models/person.txt").readlines()]
    net = create_Mb_Tiny_RFB_fd(len(class_names), is_test=True, device='cpu')
    net.load("models/Epoch-63-loss-2.08-val-2.22.pth")
    predictor = create_Mb_Tiny_RFB_fd_predictor(net, candidate_size=200, device='cpu')

    return predictor

def main():
    # Case1: Read frame from rtsp
    capture = cv2.VideoCapture('rtsp://root:123456@192.168.0.241/axis-media/media.3gp')
    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    width = 640
    height = 480
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # # Case2: Receive frame via zmq proto
    # context = zmq.Context()
    # human_det_rep = context.socket(zmq.REP)
    # human_det_rep.bind(f"tcp://*:{args.human_det_port}")


    logging.basicConfig(format="[%(asctime)s] %(message)s", level=logging.DEBUG)
    logging.info("Detector started")
    predictor = load_model()

    total_time = 0
    frame_count = 0

    polygon = np.array([[200,100],[100, 400],[200,640],[ 1000,640], [1100, 400] ,[1000,100]], np.int32)
    zone = Zone(polygon)

    while True:
        start = time.time()
        ret, frame = capture.read()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        boxes, labels, probs = predictor.predict(frame, 200, 0.5)

        probs = probs.numpy()
        for i in range(boxes.size(0)):
            box = boxes[i, :]
            color_box = (255,0, 0)
            center_x, center_y = (int((box[0] + box[2])/2), int((box[1] + box[3])/2))
            if zone.check_inside_zone(center_x, center_y):
                color_box = (0,0, 255)

            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color_box, 2)
            cv2.circle(frame, (center_x, center_y), 3, color_box, -1)
            cv2.putText(frame, str(probs[i]), (box[0], box[1]+20), cv2.FONT_HERSHEY_DUPLEX, 0.3, (255, 255, 255)) 
        
        #draw polygon
        pts = polygon
        pts = pts.reshape((-1,1,2))
        cv2.polylines(frame,[pts],True,(0, 0, 255), 2)


        cv2.putText(frame, str(boxes.size(0)), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        end = time.time()
        frame_count += 1
        total_time += end - start
        if frame_count == 100:
            logging.info(f"Average: {total_time * 1000.0 / frame_count:.2f} ms")
            total_time = 0
            frame_count = 0

        cv2.imshow('Human detection', frame)

        frame = cv2.imencode('.jpg', frame)[1].tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    main()