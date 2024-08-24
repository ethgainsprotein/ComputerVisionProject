import os
import numpy as np
import pandas as pd
from ultralytics import YOLO, solutions
import cv2
from mysimpletracker import*
import configparser



# Load YOLO model
model = YOLO('yolov10x.pt')
#create_config()
config = configparser.ConfigParser()
config.read('config.ini')

globalInCounts=0

# Specify the directory containing the videos
video_dir = 'C:\\Users\\Ethan\\Videos\\Captures'
#video_dir = config.get('General', 'folderForVideos')

# Loop through all files in the directory
for video_file in os.listdir(video_dir):
    # Check if the file is a video (you can add more extensions if needed)
    if video_file.endswith('.mp4') or video_file.endswith('.avi') or video_file.endswith('.mkv'):
        video_path = os.path.join(video_dir, video_file)
        cap = cv2.VideoCapture(video_path)

        # Determine Y-axis offset based on the specific video
        # You may need to implement logic here to set `moveYAxisBy` for each video
        #moveYAxisBy = 200  # Example value
        moveYAxisBy = int(config.get('General', 'xaxis'))
        # Output video name based on input video name
        

        # Define region points as a polygon with 5 points
        region_points = [(20, 150 + moveYAxisBy), (1080, 150 + moveYAxisBy), 
                         (1080, 200 + moveYAxisBy), (20, 200 + moveYAxisBy), 
                         (20, 150 + moveYAxisBy)]

        # Initialize object counter
        counter = solutions.ObjectCounter(
            view_img=False,
            reg_pts=region_points,
            classes_names=model.names,
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
                results = model.track(frame, persist=True, conf=0.5)  # Confidence = 50%
                frame_ = counter.start_counting(frame, results)

                # Visualize
                cv2.imshow("EthansTracker", frame_)

            if cv2.waitKey(1) & 0xFF == 27:
                break

        # Release resources
        cap.release()
        cv2.destroyAllWindows()
        #config.clear()

        with open("data.csv","a") as file:
            #file.write("Total in vid")
            file.write("Total in vid " + video_file + ": " + str(counter.in_counts) + "\n")
            file.write("Total out vid " + video_file + ": " + str(counter.out_counts) + "\n")

        # Print results
        print(f"Video processed: {video_file}")
        print("Total vehicles counted as per Yolo counters:", counter.in_counts + counter.out_counts)
        print("Total N bound as per YOLO counters:", counter.out_counts)
        print("Total S bound as per YOLO counters:", counter.in_counts)
        print("      ")

        