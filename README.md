# ComputerVisionProject
A Traffic analysis project using computer vision

Command to install packages for this project...
    pip install -r requirements.txt

Update the path to the video to analyze
Update the Area1 and Area2 polygons 
Run the project (it should auto download the yolo8 model file) 


- To include my own tracking, include all the below code at line 93 of the EasyYolo8DetectionAndTracking.py file

-         #custom tracker code, since the default YOLO counters seems to give wrong counts!
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
