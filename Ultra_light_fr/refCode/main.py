if __name__ == "__main__":  
    #importing Libraries and dependencies used:
    from datetime import datetime
    import face_recognition as fr
    import numpy as np
    import pandas as pd
    import time
    import json
    import cv2
    import os

    #initializing variables
    first_frame = None#Stagnant frame from which we'll compare
    status_list = [None, None]# This keeps a record of the motion
    time = [] #This keeps the time stamp of detected motion (different for different objects)
    df = pd.DataFrame(columns = ["Start", "End"]) #columns to keep the record of motionstart and motionend
    path = 'images/known'
    classNames = []
    database = os.listdir(path)

    #Saving the filenames
    for img in database:
        classNames.append(os.path.splitext(img)[0])

    # Reading the encodings
    with open("encodings.json", 'r+') as f:
        data = json.load(f)

    #Imported database encodings
    encodeListKnown = list(data.value)

    #Reading and Writing
    cap = cv2.VideoCapture(0) #Reading video
    fps = int(cap.get(cv2.CAP_PROP_FPS))#calculating Fps
    result = cv2.VideoWriter('video/video.avi',
                            cv2.VideoWriter_fourcc(*'MJPG'),
                            fps, (534,400))#Writing video file

    while True:
        success, img = cap.read()
        #Checking if the camera is read
        if not success:
            print("Camera Cannot be accessed!!!")
            break
        
        # (motion flag)-by default - No motion is detected
        status = 0
        ##################Pre-Processing#####################################
        #cropping frame for removing timestamp on camera
        imgS = img[:][100:]
        #Resized to 534,400
        imgS = cv2.resize(imgS, (640,480))

        ################# A) face Recognition preprocessing##################
        #Changing colorspace BGR to RGB
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
        #####################################################################

        # ################# B) Motion Detection preprocessing##################
        # #Changing colorspace BGR to Gray
        # gray = cv2.cvtColor(imgS, cv2.COLOR_BGR2GRAY)         
        # #Applying gausian Blur for smoothening
        # gray = cv2.GaussianBlur(gray,(21,21),0)
        # #Saving first frame
        # if first_frame is None:
        #     first_frame = gray
        #     continue
        # #Calculates difference to detect motion
        # delta_frame = cv2.absdiff(first_frame, gray)
        # #Applies Threshold
        # thresh_delta = cv2.threshold(delta_frame, 30, 255, cv2.THRESH_BINARY)[1]
        # thresh_delta = cv2.dilate(thresh_delta, None, iterations=0)
        # #finding contours on the white portion(made by the threshold)
        # cnts,_ = cv2.findContours(thresh_delta.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # #####################################################################
        #####################################################################

        for contour in cnts:
            #Ignoring small Motions, i.e. noise and small insects
            if cv2.contourArea(contour)<1000:
                continue
            #motion Detected
            status = 1
            #Drawing the rectangles around the contours identified
            (x, y, w, h) = cv2.boundingRect(contour)
            # cv2.rectangle(imgS, (x,y), (x+w, y+h), (0,255,0), 3)
            tempImg = imgS[x:x+w][y:y+h]
            
            #saving the video
            result.write(imgS)
