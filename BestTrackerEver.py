import os
import numpy as np
import pandas as pd
from ultralytics import YOLO, solutions
import cv2

import configparser
# Load YOLO model
model = YOLO('yolov10x.pt')
config = configparser.ConfigParser()
config.read('config.ini')

globalInCounts=0
globalOutCounts=0

classes_to_count = [2,4,5,7]
video_dir = config.get('General', 'folderForVideos')

# Loop through all files in the directory
for video_file in os.listdir(video_dir):
    if video_file.endswith('.mp4') or video_file.endswith('.avi') or video_file.endswith('.mkv'):
        video_path = os.path.join(video_dir, video_file)
        cap = cv2.VideoCapture(video_path)

        # Determine Y-axis offset based on the RAPID CAM folder, this is to make it easy to change region_points for each camera folder 
        # You may need to implement logic here to set `moveYAxisBy` for each video
        #moveYAxisBy = 200  # Example value
        moveYAxisBy = int(config.get('General', 'xaxis'))
    
        # Define region points as a polygon with 5 points
        region_points = [(20, 150 + moveYAxisBy), (1080, 150 + moveYAxisBy), 
                         (1080, 200 + moveYAxisBy), (20, 200 + moveYAxisBy), 
                         (20, 150 + moveYAxisBy)]

        # Initialize object counter
        counter = solutions.ObjectCounter(
            view_img=False,
            reg_pts=region_points,
            names=model.names,
            draw_tracks=True,
            line_thickness=1,
            region_thickness=1,
            line_dist_thresh=1,
            view_in_counts=True,
            view_out_counts=True
        )

        ret = True

        # Read frames
        while ret:
            ret, frame = cap.read()

            if ret:
                frame = cv2.resize(frame, (1020, 500))  # Resize for convenience
                
                # Detect and track objects
                results = model.track(frame, persist=True, conf=0.5,classes=classes_to_count)  # Confidence = 50%
                frame_ = counter.start_counting(frame, results)

                # Visualize
                cv2.imshow("EthansTracker", frame_)

            if cv2.waitKey(1) & 0xFF == 27:
                break

        # Release resources
        cap.release()
        cv2.destroyAllWindows()
        #config.clear()

        #print or write results into a file
        with open("data.csv","a") as file:
            #file.write("Total in vid")
            file.write(video_file + ", In," + str(counter.in_counts) + "\n")
            file.write( video_file + ", Out," + str(counter.out_counts) + "\n")

        # Print results on screen
        print(f"Video processed: {video_file}")
        print("Total vehicles counted as per Yolo counters:", counter.in_counts + counter.out_counts)
        print("Total N bound as per YOLO counters:", counter.out_counts)
        print("Total S bound as per YOLO counters:", counter.in_counts)
        print("      ")

        
