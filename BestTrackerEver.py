import os
from ultralytics import YOLO, solutions
import cv2
import configparser

# Load YOLO model
model = YOLO('EthansModel4UMD.pt') #model is 134722mb and hence cannot be included in github
#create_config()
config = configparser.ConfigParser()
config.read('config.ini')

globalInCounts=0
foldercount=0
# c:\myfolderA;230,d:\mefolder3\sub;111
video_dir = config.get('General', 'folderForVideos')
filename4counts = config.get('General', 'fileWithCounts')

listoffolders = video_dir.split(",") #assuming , is the separator between different folders that need to be processed
# Loop through all folders 
for videos in listoffolders:
    lst = videos.split(";") #assuming that each comma separated string has a ; separating the actual video/camera folder and the Y-Axis to be offshot for the camera/folder
    moveYAxisBy=int(lst[1]) 
    print("Processing folder ..."+videos)        
    video_dir=lst[0]
    # Loop through all files in the folder
    for video_file in os.listdir(video_dir):
        print(video_file)
        # Check if the file is a video 
        if video_file.endswith('.mp4') or video_file.endswith('.avi') or video_file.endswith('.mkv'):
            video_path = os.path.join(video_dir, video_file)
            cap = cv2.VideoCapture(video_path)

            
            # Define region points as a polygon with 5 points for counting both in and out autos
            region_points = [(20, 150 + moveYAxisBy), (1080, 150 + moveYAxisBy), 
                            (1080, 175 + moveYAxisBy), (20, 175 + moveYAxisBy), 
                            (20, 150 + moveYAxisBy)]

            # Define region points for down moving veh only 
            #region_points = [(5, 150+moveYAxisBy), (1080-800, 150+moveYAxisBy), (1080-800, 200+moveYAxisBy), (5, 200+moveYAxisBy), (5, 150+moveYAxisBy)]

            # Initialize object counter
            counter = solutions.ObjectCounter(
                view_img=False,
                reg_pts=region_points,
                names=model.names,
                draw_tracks=True,
                line_thickness=1,
                view_in_counts=True,
                view_out_counts=True
            )

            ret = True

            # Read frames
            while ret:
                ret, frame = cap.read()

                if ret:
                    frame = cv2.resize(frame, (640, 480))  # Resize for convenience                   
                    # Detect and track objects
                    results = model.track(frame, persist=True, conf=0.1)  # Confidence = 10%
                    frame_ = counter.start_counting(frame, results)

                    cv2.imshow("EthansTracker", frame_)

                if cv2.waitKey(1) & 0xFF == 27:
                    break

            # Release resources used by cap here since in the next loop we need to reuse it with different video
            cap.release()
            
            #config.clear()

            with open(filename4counts,"a") as file:
                file.write(video_dir+", "+video_file + ",In," + str(counter.in_counts) + "\n")
                file.write(video_dir+", "+video_file + ",Out," + str(counter.out_counts) + "\n")

            # Print results
            print(f"Video processed: {video_dir} {video_file}")
            print("Total vehicles counted as per Yolo counters:", counter.in_counts + counter.out_counts)
            print("Total N bound as per YOLO counters:", counter.out_counts)
            print("Total S bound as per YOLO counters:", counter.in_counts)
            print("      ")

#release resources usef by cv2
cv2.destroyAllWindows()
