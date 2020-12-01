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
    import pickle

    # Initializing the variables
    # Paths
    path = 'face_recognition/Opencv_dnn_fr/images/known/'#Path of known images to compare from
    modelFile = "face_recognition/Opencv_dnn_fr/model/res10_300x300_ssd_iter_140000.caffemodel"
    configFile = "face_recognition/Opencv_dnn_fr/model/deploy.prototxt.txt"
    classNames = []#To append the filenames
    tolerance = 0.6#Threshold distance from the closest image
    myList = os.listdir(path)#adding all the item's name present at path
    for cls in myList:
        classNames.append(os.path.splitext(cls)[0])

    #Reading the encodings
    with open("face_recognition/Opencv_dnn_fr/data/encodings.json", 'r+') as f:
        data = json.load(f)

    encodeListKnown = list(data.values())

    #Load model & config file
    net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

    #Capturing video
    # video_capture = cv2.VideoCapture(0)   
    video_capture = cv2.VideoCapture("Friends_ross.mp4")
    # video_capture = cv2.VideoCapture('rtsp://admin:admin123@192.168.0.104:554/')

    count = 0
    start = time.time()
    while(True):
        success, frame = video_capture.read()
        if not success:
            print("Couldn't read from webcam")
            break
        count = count+1
        boxes = []


        # $$$ Facial Detection $$$ - model used - OpencV Resnet
        #####################Pre-Processing###########################
        imgS = cv2.resize(frame, (300, 300)) #Resizing according to the face detection model
        #getting the height and width to show the output according to the user
        h,w,_ = imgS.shape
        cv2.imwrite("face_recognition/Opencv_dnn_fr/1.jpg",imgS)
        # img_mean = np.array([104, 117, 123])
        # img = (imgS - img_mean) / 128 #Substracting mean from the image
        # img = np.transpose(img, [2, 0, 1]) #transposing the shape accordingly
        # img = np.expand_dims(img, axis=0) 
        # img = img.astype(np.float32)
        blob = cv2.dnn.blobFromImage(imgS, 1.0,
        (300, 300), (104.0, 117.0, 123.0))
        ##############################################################
        
        #predicting the bounding box
        net.setInput(blob)
        faces = net.forward()
        #to draw faces on image
        for i in range(faces.shape[2]):
            confidence = faces[0, 0, i, 2]
            if confidence > 0.5:
                box = faces[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x, y, x1, y1) = box.astype('int')
                cv2.rectangle(imgS, (x, y), (x1, y1), (0, 0, 255), 2)
                boxes.append(box)
        boxes = np.array(boxes).astype('int')
        print(boxes)
        #arranging the columns according to face_recognition requirement
        boxes = np.array(boxes)
        facesCurFrame = boxes
        if(facesCurFrame.shape[0]>0):
            facesCurFrame[:,[0,1,2,3]] = facesCurFrame[:,[1,2,3,0]]#Changing it as per requirements

        #seeing if any face is detected
        print(boxes.shape[0], facesCurFrame.shape[0])
        print(type(boxes), type(facesCurFrame))
        print(boxes, "\n\n\n", facesCurFrame)
        print("...")

        #Encoding
        encodeCurFrame = fr.face_encodings(imgS, facesCurFrame)

        #looping through faces
        for encodeFace, faceLoc in zip(encodeCurFrame, facesCurFrame):
            #Comparing faces
            name = "Unknown"#By default if the face isnt recognised
            matches = fr.compare_faces(encodeListKnown, encodeFace)
            faceDis = fr.face_distance(encodeListKnown, encodeFace)
            y1, x2, y2, x1 = faceLoc

            matchIndex = np.argmin(faceDis)
            if faceDis[matchIndex]<tolerance:
                if matches[matchIndex]:
                    name = classNames[matchIndex]
                    cv2.rectangle(imgS, (x1,y1), (x2,y2), (0,255,0),2)
                    cv2.rectangle(imgS, (x1,y2-35), (x2,y2),(0,255,255), cv2.FILLED)
                    cv2.putText(imgS, name, (x1+6,y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,0),2)
                    cv2.imwrite("face_recognition/Opencv_dnn_fr/output.jpg", imgS)    
                    # markAttendance(name)
                else:
                    cv2.rectangle(imgS, (x1,y1), (x2,y2), (139,0,0),2)
                    cv2.rectangle(imgS, (x1,y2-35), (x2,y2),(0,255,255), cv2.FILLED)
                    cv2.putText(imgS, name, (x1+6,y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,0),2)
            
            else:
                cv2.rectangle(imgS, (x1,y1), (x2,y2), (139,0,0),2)
                cv2.rectangle(imgS, (x1,y2-35), (x2,y2),(0,255,255), cv2.FILLED)
                cv2.putText(imgS, name, (x1+6,y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,0),2)


        cv2.imwrite("face_recognition/Opencv_dnn_fr/2.jpg", imgS)
        cv2.imshow("Image test", imgS)
        if count>=120:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    end = time.time()
    print(120/(end-start))#returns fps