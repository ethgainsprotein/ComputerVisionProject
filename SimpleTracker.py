import numpy as np
import pandas as pd
from ultralytics import YOLO, solutions
import cv2
from mysimpletracker import*


# load yolov8 model
model = YOLO('yolov10x.pt')
#model = YOLO("yolov8n.pt")

# load one of the following video along with the Y Axis variable

# #Actual counts  - 15 nb, 16 sb
# video_path = 'los_angeles.mp4'
# moveYAxisBy=150

#video_path = '2024-04-17 16-04-36.mkv'
#moveYAxisBy=150

#633 in 1 out
#video_path = 'TrafficOnI895.mp4'
#moveYAxisBy=200

#video_path = 'downtownintersectionshort.mp4'
#moveYAxisBy=100

# #Actual counts  - 15 nb, 16 sb
video_path = 'veh2.mp4'
moveYAxisBy=200

cap = cv2.VideoCapture(video_path)

#print("fps=",cap.get(cv2.CAP_PROP_FPS))

# The below is to make a new video file of the video with detections
# Define the codec and create VideoWriter object.The output is stored in 'withyolo.avi' file.
# Define the fps to be equal to 10. Also frame size is passed. 
out = cv2.VideoWriter('veh2New3.avi',cv2.VideoWriter_fourcc('M','J','P','G'), cap.get(cv2.CAP_PROP_FPS), (1020,500))

# Define region points as a polygon with 5 points
region_points = [(20, 150+moveYAxisBy), (1080, 150+moveYAxisBy), (1080, 200+moveYAxisBy), (20, 200+moveYAxisBy), (20, 150+moveYAxisBy)]

#classes_to_count = [2,3,5,7]

# Init Object Counter - this obj cter is not accurate or i'm missing something 
counter = solutions.ObjectCounter(
    view_img=False,
    reg_pts=region_points,
    classes_names=model.names,
    draw_tracks=True,
    line_thickness=1
    ,region_thickness=1,
    line_dist_thresh = 1
    ,view_in_counts=True,view_out_counts=True
)

ret = True

# read frames
while ret:
    ret, frame = cap.read()

    if ret:
        frame=cv2.resize(frame,(1020,500)) # we are changing the video/image resolution for the convenience of using the polygon 
          
        # detect objects
        # track objects
        results = model.track(frame, persist=True, conf=0.5) # confidence = 50%
        
        frame_ = counter.start_counting(frame, results)
        
        # visualize
        cv2.imshow("EthansTracker", frame_)
        out.write(frame_)

    if cv2.waitKey(1)&0xFF==27:
       break
    

cv2.waitKey(0)

print("      ")
print("      ")

print("Total vehicles counted as per Yolo counters:",counter.in_counts+counter.out_counts)
print("Total N bound as per YOLO counters:",counter.out_counts)
print("Total S bound as per YOLO counters:",counter.in_counts)
print("      ")

print("fps=",cap.get(cv2.CAP_PROP_FPS))
cap.release()
#out.release()
cv2.destroyAllWindows()
#c i 230 o 360 t 2 b 5 ..with only up tracking 
