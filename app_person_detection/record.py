# Python program to save a  
# video using OpenCV 
from flask import Flask

from flask_cors import cross_origin
from flask import render_template, url_for, flash, redirect, request, Response
import datetime
import cv2 
import threading

app = Flask(__name__)

class RecordingThread (threading.Thread):
    def __init__(self, name):
        threading.Thread.__init__(self)
        self.name = name
        self.isRunning = True

        self.cap = cv2.VideoCapture('rtsp://admin:12345678a@@192.168.0.247:8554/fhd') 
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        self.out = cv2.VideoWriter('video.avi',fourcc, 20.0, (640,480))

    def run(self):
        while self.isRunning:
            ret, frame = self.cap.read()
            if ret:
                self.out.write(frame)

        self.out.release()

    def stop(self):
        self.isRunning = False

    def __del__(self):
        self.out.release()


@app.route('/record/<int:ip>', methods=['POST'])
@cross_origin()
def record(ip):

#     cap = cv2.VideoCapture('rtsp://admin:12345678a@@192.168.0.247:8554/fhd') 

# # Define the codec and create VideoWriter object
# #fourcc = cv2.cv.CV_FOURCC(*'DIVX')
# #out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))
#     out = cv2.VideoWriter('output.avi', -1, 20.0, (640,480))

#     while(cap.isOpened()):
#         ret, frame = cap.read()
#         if ret==True:
#             # frame = cv2.flip(frame,0)

#             # write the flipped frame
#             out.write(frame)

#             cv2.imshow('frame',frame)
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
#         else:
#             break

#===================================================================================================

    cap = cv2.VideoCapture('rtsp://admin:12345678a@@192.168.0.247:8554/fhd') 
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter('video.avi',fourcc, 20.0, (640,480))

    while True:
        ret, frame = cap.read()
        if ret:
            out.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('s'): 
            break
#==============================================================================================================

    global stt 
    stt = request.json['status'] 
    print(stt)

    record_thread = RecordingThread("Video Recording Thread")

    if stt == "true":
        record_thread.run()
    else:
        record_thread.stop()
    Create an object to read  
    from camera 
    video = cv2.VideoCapture('rtsp://admin:12345678a@@192.168.0.252:8554/fhd') 

    # # We need to check if camera 
    # # is opened previously or not 
    # if (video.isOpened() == False):  
    #     print("Error reading video file") 

    # # We need to set resolutions. 
    # # so, convert them from float to integer. 
    # frame_width = int(video.get(3)) 
    # frame_height = int(video.get(4)) 

    # size = (frame_width, frame_height) 

    # # Below VideoWriter object will create 
    # # a frame of above defined The output  
    # # is stored in 'filename.avi' file. 
    # result = cv2.VideoWriter('filename.avi',  
    #                         cv2.VideoWriter_fourcc(*'MJPG'), 
    #                         25, size) 
        
    # while(True): 
    #     ret, frame = video.read() 
    #     if stt == 'true':
    #         if ret == True:  

    #         # Write the frame into the 
    #         # file 'filename.avi' 
    #             result.write(frame) 

    #         # Display the frame 
    #         # saved in the file 
    #         # cv2.imshow('Frame', frame) 

    #         # Press S on keyboard  
    #         # to stop the process 
    #         # if cv2.waitKey(1) & 0xFF == ord('s'): 
    #         #     break
    #     else:
    #         break

    # # Break the loop 
        

    # # When everything done, release  
    # # the video capture and video  
    # # write objects 
    # video.release() 
    # result.release() 

# Closes all the frames
cv2.destroyAllWindows()
 
if __name__ == "__main__" :
    # live_demo()
    app.run(host="0.0.0.0",debug=True)
