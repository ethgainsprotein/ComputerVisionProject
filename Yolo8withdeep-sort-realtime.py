#working deepsort with yolo8
import datetime
from ultralytics import YOLO
import cv2
import numpy as np
import cvzone 
#from helper import create_video_writer
from deep_sort_realtime.deepsort_tracker import DeepSort


CONFIDENCE_THRESHOLD = 0.5
M_MAX_AGE=25

GREEN = (0, 255, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
PINK = (255, 0, 255)


# load one of the following video along with the Y Axis variable

#video_path = '.\\videos\\los_angeles.mp4'
#moveYAxisBy=150

# #Actual counts  - 15 nb, 16 sb
# video_path = '.\\videos\\veh2.mp4'
# moveYAxisBy=200
# #test1+ counts 
# #15 nb, 16 sb =31

# #Actual counts - TBD
# video_path = '.\\videos\\2024-04-17 16-04-36.mkv'
# # test1
# # sb 631
# #video_path = 'shortwithtrucks.mp4'
# moveYAxisBy=150

#17 sb, 39 or 40 -1 weird car nb
video_path = '.\\videos\\riversidetraffic.mp4'
moveYAxisBy=0
# test1
# counted 40 s 22 n = 62

# #sb 8, nb 4
# video_path = 'downtownintersectionshort.mp4'
# # test1
# # sb 8, nb 4
# moveYAxisBy=250

# initialize the video capture object

video_cap = cv2.VideoCapture(video_path)
# initialize the video writer object
#writer = create_video_writer(video_cap, "output.mp4")

#print("fps original=",video_cap.get(cv2.CAP_PROP_FPS))

# Define region points as a polygon with 5 points
#region_points = [(20, 150+moveYAxisBy), (1080, 150+moveYAxisBy), (1080, 200+moveYAxisBy), (20, 200+moveYAxisBy), (20, 150+moveYAxisBy)]
area1 = [(20, 125+moveYAxisBy), (1080, 125+moveYAxisBy), (1080, 150+moveYAxisBy), (20, 150+moveYAxisBy), (20, 125+moveYAxisBy)]
area2 = [(20, 152+moveYAxisBy), (1080, 152+moveYAxisBy), (1080, 200+moveYAxisBy), (20, 200+moveYAxisBy), (20, 152+moveYAxisBy)]

# load the pre-trained YOLOv8n model
model = YOLO("yolov8n.pt")
classes_to_count = [1,2,3,5,7]
upcar={}
downcar={}
countercarup=[]
countercardown=[]

tracker = DeepSort(max_age=M_MAX_AGE)


while True:
    start = datetime.datetime.now()

    ret, frame = video_cap.read()

    if not ret:
        break
    #frame=cv2.resize(frame,(640,480))
    frame=cv2.resize(frame,(1020,500))
    #draw the 2 polygons, area1 and 2 
    cv2.polylines (frame, [np.array (area1, np.int32)], True, (0,155,0),2) 
    cv2.polylines (frame, [np.array (area2, np.int32)], True, (0,0,0),2)
          
    # run the YOLO model on the frame
    detections = model(frame)[0]

    # initialize the list of bounding boxes and confidences
    results = []

    ######################################
    # DETECTION
    ######################################

    # loop over the detections
    for data in detections.boxes.data.tolist():
        # extract the confidence (i.e., probability) associated with the prediction
        confidence = data[4]

        # filter out weak detections by ensuring the 
        # confidence is greater than the minimum confidence
        if float(confidence) < CONFIDENCE_THRESHOLD:
            continue

        # if the confidence is greater than the minimum confidence,
        # get the bounding box and the class id
        xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
        class_id = int(data[5])
        # add the bounding box (x, y, w, h), confidence and class id to the results list
        results.append([[xmin, ymin, xmax - xmin, ymax - ymin], confidence, class_id])

    ######################################
    # TRACKING
    ######################################

    # update the tracker with the new detections
    tracks = tracker.update_tracks(results, frame=frame)
    # loop over the tracks
    for track in tracks:
        # if the track is not confirmed, ignore it
        if not track.is_confirmed():
            continue

        # get the track id and the bounding box
        track_id = track.track_id
        ltrb = track.to_ltrb()

        xmin, ymin, xmax, ymax = int(ltrb[0]), int(
            ltrb[1]), int(ltrb[2]), int(ltrb[3])
        # draw the bounding box and the track id
        cpx=int(xmin+xmax)//2
        cpy=int(ymin+ymax)//2
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), BLACK, 2)
        cv2.circle(frame,(cpx,cpy),4,(255,0,0),-1)
       
        cv2.rectangle(frame, (xmin, ymin - 20), (xmin + 20, ymin), PINK, 0)
        cv2.putText(frame, str(track_id), (xmin + 5, ymin - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 2)

        ########for carup#######
        #check if center point of box is within the area2 polygon
        result2=cv2.pointPolygonTest(np.array(area2, np.int32), ((cpx,cpy)), False)
        if result2>=0:
            print("in carup area2 ",track_id)
            #cv2.rectangle(frame,(x3,y3),(x4,y4),(255,255,255),2)
            upcar [track_id]=(cpx,cpy)

        #in a future frame this will be true for the same id as in upcar, this indicates object 1st touched area2 and then area1, hence is North bound
        if track_id in upcar:
            result3=cv2.pointPolygonTest (np.array(area1, np.int32), ((cpx,cpy)), False)
            if result3>=0:
                print("in carup area1 ",track_id)
                if countercarup.count(track_id)==0:
                    cv2.rectangle(frame,(xmin,ymin),(xmax,ymax),(255,0,255),2)
                    countercarup.append(track_id)
        
        #######for cardown########
        result=cv2.pointPolygonTest(np.array(area1, np.int32), ((cpx,cpy)), False)
        if result>=0:
            #cv2.rectangle(frame,(x3,y3),(x4,y4),(255,255,255),2)
            print("in cdown area1 ",track_id)
            downcar [track_id]=(cpx,cpy)

        if track_id in downcar:
            result1=cv2.pointPolygonTest (np.array(area2, np.int32), ((cpx,cpy)), False)
            if result1>=0:
                print("in cdown area2 ",track_id)
                if countercardown.count(track_id)==0:
                    cv2.rectangle(frame,(xmin,ymin),(xmax,ymax),(255,0,255),2)
                    countercardown.append(track_id)
    
    cup=len(countercarup)
    cdown=len(countercardown)
    cvzone.putTextRect(frame,f'My N bound: {cup}',(10,20),2,2)
    cvzone.putTextRect(frame,f'My S bound: {cdown}',(10,55),2,2)
    
     

    # # end time to compute the fps
    # end = datetime.datetime.now()
    # # show the time it took to process 1 frame
    # print(f"Time to process 1 frame: {(end - start).total_seconds() * 1000:.0f} milliseconds")
    # # calculate the frame per second and draw it on the frame
    # fps = f"FPS: {1 / (end - start).total_seconds():.2f}"
    # cv2.putText(frame, fps, (50, 50),
    #             cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 8)

    # show the frame to our screen
    cv2.imshow("Frame", frame)
    #writer.write(frame)
    if cv2.waitKey(1) == ord("q"):
        break

print("My custom total count:",cup+cdown)
print("My custom count for N Bound traffic:",cup)
print("My custom count for S Bound traffic:",cdown)
print("Original fps=",video_cap.get(cv2.CAP_PROP_FPS))
video_cap.release()
#writer.release()
cv2.destroyAllWindows()