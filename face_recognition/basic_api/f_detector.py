#Importing the libraries
import cv2
import numpy as np
import time 
import pandas as pd
from datetime import datetime


#initializing variables
first_frame = None #Stagnant frame from which we'll compare
status_list = [None,None] # This keeps a record of motion
times = [] #This keeps the time stamp of detected motion (different for different objects)
df = pd.DataFrame(columns = ["Start", "End"]) #columns to keep the record of motionstart and motionend
face_classifier = cv2.CascadeClassifier('face_recognition/basic_api/haarcascade_frontalface_default.xml')#Pretrained cascade classfier to get the location of face

# cv2.namedWindow("output", cv2.WINDOW_NORMAL) 
# cv2.resizeWindow("output", 1920, 1080)
# cap = cv2.VideoCapture('rtsp://admin:admin123@192.168.0.104:554/')
cap = cv2.VideoCapture(0)
frame_width = int(cap.get(3)) 
frame_height = int(cap.get(4)) 
size = (frame_width, frame_height)
fps = int(cap.get(cv2.CAP_PROP_FPS))

result = cv2.VideoWriter('face_recognition/basic_api/filename.avi',  
                        cv2.VideoWriter_fourcc(*'MJPG'), 
                        fps, size)


# print("after")
while(True):
    #Read Input
    ret, frame = cap.read()
    status = 0
    # frame = frame[:][100:]
    #resized to 534*400(after trimming the time stamp)
    frame = cv2.resize(frame, (534,400))
    #Grayscale Conversion
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(21,21),0)
    # So that the first frame is read only once
    if first_frame is None:
        first_frame = gray
        continue


    
    #Calculates difference to detect motion
    delta_frame = cv2.absdiff(first_frame, gray)
    #Applies Threshold
    thresh_delta = cv2.threshold(delta_frame, 30, 255, cv2.THRESH_BINARY)[1]
    thresh_delta = cv2.dilate(thresh_delta, None, iterations=0)
    #finding contours on the white portion(made by the threshold)
    cnts,_ = cv2.findContours(thresh_delta.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    
    for contour in cnts:
        #Ignoring small Motions, i.e. noise and small insects
        if cv2.contourArea(contour)<1000:
            continue
        #motion Detected
        status = 1
        faces = face_classifier.detectMultiScale(gray, 1.32, 6)
        #Drawing the rectangles around the contours identified
        # (x, y, w, h) = cv2.boundingRect(contour)
        # cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 3)
        if faces is ():
            print("No faces found")
        else:
            '''We iterate through our faces array and draw a rectangle over each face in faces'''
            for (x,y,w,h) in faces:
                cv2.rectangle(frame, (x,y), (x+w,y+h), (127,0,255), 2)
                # new = frame[x:x+w][y:y+h] 

            # cv2.imshow('Face Detection', frame)
        #saving the image of the moving object
        # cv2.imwrite("face_detection/face.jpg", frame)
        #Saving the video if motion is detected
        result.write(frame)
    
    #appending the motion detection status
    status_list.append(status)

    # Trimming the the status list to last two objects
    status_list = status_list[-2:]
    #It means that the object has entered the frame
    if status_list[-1]==1 and status_list[-2]==0:
        times.append(datetime.now())
    #It means that the object moving has left the frame
    if status_list[-1]==0 and status_list[-2]==1:
        times.append(datetime.now())
    
    # Show Output
    # cv2.imshow('capturing', gray)#gray scale image
    # cv2.imshow('delta', delta_frame)#difference btween the first and current frame
    # cv2.imshow('thresh', thresh_delta)#thresholded frame
    cv2.imshow("frame",frame)#Frame

    #Quiting (Reading the key)
    key = cv2.waitKey(1) 
    if key == ord('q'):
        break

print(status_list)
print(times)
#Appending the details (time) in the dataframe
for i in range(0, len(times),2):
    df = df.append({"Start": times[i], "End":times[i+1]},ignore_index=True)
#Writing the csv file with the entry and exit time of the objects
df.to_csv("face_recognition/basic_api/Times.csv")

#Closing the camera access and the windows.
cap.release()
cv2.destroyAllWindows()