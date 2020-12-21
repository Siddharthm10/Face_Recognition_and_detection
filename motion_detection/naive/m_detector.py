#Importing the libraries
import cv2
import numpy as np
import time 
#import pandas as pd
#from datetime import datetime


#initializing variables
first_frame = None #Stagnant frame from which we compare other frames

#cap = cv2.VideoCapture('rtsp://admin:admin123@192.168.0.104:554/')
cap = cv2.VideoCapture('Friends_ross.mp4')
# cap = cv2.VideoCapture(0)

# print("after")
count = 0
check0 = check1 = check2 = check3 = check4 = 0
start =time.time()
while(True):
    check0 = time.time()
    #Read Input
    ret, frame = cap.read()
    if not ret:
    	print('couldnt access camera')
    	break
    status = 0
    count = count + 1
    frame = frame[:][100:]
    #resized to 534*400(after trimming the time stamp)
    frame = cv2.resize(frame, (534,400))
    #Grayscale Conversion
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(21,21),0)
    # So that the first frame is read only once
    if first_frame is None:
        first_frame = gray
        continue

    check1 = time.time()
    #Calculates difference to detect motion
    delta_frame = cv2.absdiff(first_frame, gray)
    #Applies Threshold
    thresh_delta = cv2.threshold(delta_frame, 30, 255, cv2.THRESH_BINARY)[1]
    thresh_delta = cv2.dilate(thresh_delta, None, iterations=0)

    check2 = time.time()
    #finding contours on the white portion(made by the threshold)
    cnts,_ = cv2.findContours(thresh_delta.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    check3 = time.time()
    
    for contour in cnts:
        #Ignoring small Motions, i.e. noise and small insects
        if cv2.contourArea(contour)<1000:
            continue
        
        #Drawing the rectangles around the contours identified
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 3)
    

    check4 = time.time()
    # Show Output
    # cv2.imshow('capturing', gray)#gray scale image
    # cv2.imshow('delta', delta_frame)#difference btween the first and current frame
    # cv2.imshow('thresh', thresh_delta)#thresholded frame
    cv2.imshow("frame",frame)#Frame
    cv2.imwrite('./motion_detection/naive/frames/frame{}.jpg'.format(count), frame)

    #Quiting (Reading the key)
    key = cv2.waitKey(1) 
    if count>150:
        break
    if key == ord('q'):
        break
end = time.time()

print(count/ (end - start))
print("End : ",end - check4)
print("Checkpoint 4 : ",check4 - check3)
print("Checkpoint 3 : ",check3 - check2)
print("Checkpoint 2 : ",check2 - check1)
print("Checkpoint 1 : ", check1 - check0)

#Closing the camera access and the windows.
cap.release()
cv2.destroyAllWindows()
