import numpy as np
import pandas as pd
from ultralytics import YOLO, solutions
import cv2
import cvzone
from mysimpletracker import*

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        point = [x, y]
        print(point)


#these 2 are only needed to find x,y cordinates that can be used in the polygons
cv2.namedWindow('Tracker')
cv2.setMouseCallback('Tracker', RGB)

# load yolov8 model
model = YOLO('yolov8n.pt')

# load one of the following video along with the Y Axis variable
#video_path = '.\\videos\\los_angeles.mp4'
#moveYAxisBy=150

#Actual counts  - 15 nb, 16 sb
video_path = '.\\videos\\veh2.mp4'
moveYAxisBy=200
#test1+ counts 
#yolo cnts 31(n15, s16)
#custom cnts 49(n24,s25)

# video_path = '.\\videos\\2024-04-17 16-04-36.mkv'
# #video_path = 'shortwithtrucks.mp4'
# moveYAxisBy=150

# #actuals #17 sb, 39 or 40 -1 weird car nb
# video_path = '.\\videos\\riversidetraffic.mp4'
# moveYAxisBy=0
# #test1
# #Yolo counter 53(n50, s3)
# #custom cnt 64 (n 37, s27)

cap = cv2.VideoCapture(video_path)
print("fps=",cap.get(cv2.CAP_PROP_FPS))

# Define region points as a polygon with 5 points
region_points = [(20, 150+moveYAxisBy), (1080, 150+moveYAxisBy), (1080, 200+moveYAxisBy), (20, 200+moveYAxisBy), (20, 150+moveYAxisBy)]
area1 = [(20, 125+moveYAxisBy), (1080, 125+moveYAxisBy), (1080, 150+moveYAxisBy), (20, 150+moveYAxisBy), (20, 125+moveYAxisBy)]
area2 = [(20, 150+moveYAxisBy), (1080, 150+moveYAxisBy), (1080, 200+moveYAxisBy), (20, 200+moveYAxisBy), (20, 150+moveYAxisBy)]

#region_points = [(20, 200), (1080, 200)]

#area1=[(340,258),(813, 286-25),(793, 308-25),(270,286)]
#area1=[(507,280),(783, 276),(773, 298),(481, 300)]
classes_to_count = [2,3,5,7]
upcar={}
downcar={}
countercarup=[]
countercardown=[]
tracker=MySimpleTracker()
# Init Object Counter - this obj cter is not accurate or i'm missing something 
counter = solutions.ObjectCounter(
    view_img=False,
    reg_pts=region_points,
    names=model.names,
    draw_tracks=True,
    line_thickness=1
    #,line_dist_thresh=5
    ,region_thickness=1
    ,view_in_counts=True,view_out_counts=True
)

ret = True

# read frames
while ret:
    ret, frame = cap.read()

    if ret:
        frame=cv2.resize(frame,(1020,500))
        #frame=cv2.resize(frame,(640,480))
        #draw the 2 polygons, area1 and 2 
        cv2.polylines (frame, [np.array (area1, np.int32)], True, (0,155,0),2) 
        cv2.polylines (frame, [np.array (area2, np.int32)], True, (0,0,0),2)
          
        # detect objects
        # track objects
        results = model.track(frame, persist=True,classes=classes_to_count)
        
        frame_ = counter.start_counting(frame, results)
        #print(counter.in_counts+counter.out_counts)
        #cvzone.putTextRect(frame,f'Vehicle count: {counter.in_counts+counter.out_counts}',(650,20),2,2)
    
        #custom tracker code, since the default YOLO counters seems to give wrong counts!
        list=[]
        a=results[0].boxes.data
        #converting to table format for easy logic
        px=pd.DataFrame(a).astype("float")
        for index,row in px.iterrows():
            if(row.count()==7): #TODO ensure 7 items are present...realized when object moves out of frame region, it return less number of array items
                x1=int(row[0])
                y1=int(row[1])
                x2=int(row[2])
                y2=int(row[3])
                d=int(row[6])
                if d == 2 or d==5 or d==3 or d==7:
                    list.append([x1,y1,x2,y2])
                    
        bbox_idx=tracker.update(list)
        #print("list is ",list)
        for bbox in bbox_idx:
            x3,y3,x4,y4,id1=bbox
            cx3=int(x3+x4)//2
            cy3=int(y3+y4)//2
            #cv2.circle(frame,(cx3,cy3),4,(255,0,0),-1)
            #cv2.rectangle(frame,(x3,y3),(x4,y4),(0,0,0),2)
           
            cvzone.putTextRect(frame, f'{id1}', (x3, y3),  # Image and starting position of the rectangle
                scale=1, thickness=0,  # Font scale and thickness
                colorT=(255, 255, 255), colorR=(0,0,0),  # Text color and Rectangle color
                font=cv2.FONT_HERSHEY_PLAIN,  # Font type
                offset=10,  # Offset of text inside the rectangle
                border=0, colorB=(0, 255, 0)  # Border thickness and color
            )
            ########for carup#######
            #check if center point of box is within the area2 polygon
            result2=cv2.pointPolygonTest(np.array(area2, np.int32), ((cx3,cy3)), False)
            if result2>=0:
                print("in carup area2 ",id1)
                #cv2.rectangle(frame,(x3,y3),(x4,y4),(255,255,255),2)
                upcar [id1]=(cx3,cy3)

            #in a future frame this will be true for the same id as in upcar, this indicates object 1st touched area2 and then area1, hence is North bound
            if id1 in upcar :
                result3=cv2.pointPolygonTest (np.array(area1, np.int32), ((cx3,cy3)), False)
                if result3>=0:
                    
                    if countercarup.count(id1)==0:
                        print("in carup area1 -added",id1)
                        cv2.rectangle(frame,(x3,y3),(x4,y4),(255,0,255),2)
                        countercarup.append(id1)
            
            #######for cardown########
            result=cv2.pointPolygonTest(np.array(area1, np.int32), ((cx3,cy3)), False)
            if result>=0:
                #cv2.rectangle(frame,(x3,y3),(x4,y4),(255,255,255),2)
                print("in cdown area1 ",id1)
                downcar [id1]=(cx3,cy3)

            if id1 in downcar :
                result1=cv2.pointPolygonTest (np.array(area2, np.int32), ((cx3,cy3)), False)
                if result1>=0:                    
                    if countercardown.count(id1)==0:
                        print("in cdown area2 -added ",id1)
                        cv2.rectangle(frame,(x3,y3),(x4,y4),(255,0,255),2)
                        countercardown.append(id1)
        
        cup=len(countercarup)
        cdown=len(countercardown)
        cvzone.putTextRect(frame,f'My N bound: {cup}',(10,20),2,2)
        cvzone.putTextRect(frame,f'My S bound: {cdown}',(10,55),2,2)
   
        #end custom tracker code
        # plot results
        # cv2.rectangle
        # cv2.putText
        
        #frame_ = results[0].plot()

        # visualize
        cv2.imshow("Tracker", frame_)
    if cv2.waitKey(1)&0xFF==27:
       break
print("      ")
print("      ")

print("Total vehicles counted as per Yolo counters:",counter.in_counts+counter.out_counts)
print("Total N bound as per YOLO counters:",counter.out_counts)
print("Total S bound as per YOLO counters:",counter.in_counts)
print("      ")

print("My custom total count:",cup+cdown)
print("My custom count for N Bound traffic:",cup)
print("My custom count for S Bound traffic:",cdown)
print("fps=",cap.get(cv2.CAP_PROP_FPS))
cap.release()
cv2.destroyAllWindows()
#c i 230 o 360 t 2 b 5 ..with only up tracking 