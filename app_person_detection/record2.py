from flask import Flask

from flask_cors import cross_origin
from flask import render_template, url_for, flash, redirect, request, Response
import datetime
import cv2 
import threading

app = Flask(__name__)

@app.route('/record', methods=['POST'])
@cross_origin()
def record():
    video = cv2.VideoCapture('rtsp://admin:12345678a@@192.168.0.252:8554/fhd') 
    if (video.isOpened() == False):  
        print("Error reading video file") 
    time = datetime.datetime.now().strftime('%x')
    time = str(time).replace('/', '-')
    print(time)
    filename = f'{time}.avi'
    print(filename)   
    frame_width = int(video.get(3)) 
    frame_height = int(video.get(4)) 
    size = (frame_width, frame_height) 
    result = cv2.VideoWriter(filename,cv2.VideoWriter_fourcc(*'MJPG'),20, size) 
    # global stt
    # stt = request.json['status']    
    while(True): 
        ret, frame = video.read()
        if ret == True:
            result.write(frame) 
            cv2.imshow('FramFe', frame) 
        if cv2.waitKey(1) & 0xFF == ord('s'):
            break   
     
    video.release() 
    result.release() 
        
    cv2.destroyAllWindows() 
    return "stop"

print("The video was successfully saved") 
if __name__ == "__main__" :
    # live_demo()
    app.run(host="0.0.0.0",debug=True)