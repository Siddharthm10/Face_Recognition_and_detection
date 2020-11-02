import cv2
import numpy as np
import time 
import pandas as pd
from datetime import datetime


#initializing variables
first_frame = None
status_list = [None,None]
times = []
df = pd.DataFrame(columns = ["Start", "End"])

# cv2.namedWindow("output", cv2.WINDOW_NORMAL) 
# cv2.resizeWindow("output", 1920, 1080)
cap = cv2.VideoCapture('rtsp://admin:admin123@192.168.0.104:554/')
# cap = cv2.VideoCapture(0)

# print("after")
while(True):
    #Read Input
    ret, frame = cap.read()
    status = 0
    frame = frame[:][300:]
    #Grayscale Conversion
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(21,21),0)
    if first_frame is None:
        first_frame = gray
        continue

    #Calculates difference
    delta_frame = cv2.absdiff(first_frame, gray)
    #Applies Threshold
    thresh_delta = cv2.threshold(delta_frame, 30, 255, cv2.THRESH_BINARY)[1]
    thresh_delta = cv2.dilate(thresh_delta, None, iterations=0)
    #finding contours on the white portion
    cnts,_ = cv2.findContours(thresh_delta.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #ignoring the Small changes
    for contour in cnts:
        if cv2.contourArea(contour)<1000:
            continue
        status = 1
        #Drawing the rectangles around the contours identified
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 3)
        cv2.imwrite("motion_detection/spotted.jpg", frame)
    status_list.append(status)

    status_list = status_list[-2:]
    if status_list[-1]==1 and status_list[-2]==0:
        times.append(datetime.now())
    if status_list[-1]==0 and status_list[-2]==1:
        times.append(datetime.now())
    
    # Show Output
    cv2.imshow("frame",frame)
    cv2.imshow('capturing', gray)
    cv2.imshow('delta', delta_frame)
    cv2.imshow('thresh', thresh_delta)

    #Quiting
    key = cv2.waitKey(1) 
    if key == ord('q'):
        break
print(status_list)
print(times)
for i in range(0, len(times),2):
    df = df.append({"Start": times[i], "End":times[i+1]},ignore_index=True)
df.to_csv("motion_detection/Times.csv")

cap.release()
cv2.destroyAllWindows()