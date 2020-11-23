if __name__ == "__main__":  
    #importing Libraries and dependencies used:
    from datetime import datetime
    import face_recognition as fr
    import numpy as np
    import pandas as pd
    import onnx
    import onnxruntime as ort
    from onnx_tf.backend import prepare
    import time
    import json
    import cv2
    import os
    import pickle
    
    # from utils.predict import areaof
    # from utils.predict import iou_of
    # from utils.predict import hard_nms
    # from utils.predict import predict
    import utils.predict as ut

    # Initializing the variables
    path = 'face_recognition/Ultra_light_fr/images/known/'#Path of known images to compare from
    classNames = []#To append the filenames
    tolerance = 0.6#Threshold distance from the closest image
    myList = os.listdir(path)#adding all the item's name present at path
    for cls in myList:
        classNames.append(os.path.splitext(cls)[0])

    #Reading the encodings
    with open("face_recognition/Ultra_light_fr/data/encodings.json", 'r+') as f:
        data = json.load(f)

    encodeListKnown = list(data.values())

    # load the model, create runtime session & get input variable name
    onnx_path = 'face_recognition/Ultra_light_fr/model/ultra_light_640.onnx'
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    onnx.helper.printable_graph(onnx_model.graph)
    predictor = prepare(onnx_model)
    ort_session = ort.InferenceSession(onnx_path)

    input_name = ort_session.get_inputs()[0].name


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
        #getting the height and width to show the output according to the user


        # $$$ Facial Detection $$$ - model used - Ultra Light
        #####################Pre-Processing###########################
        # img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        imgS = cv2.resize(frame, (640, 480)) #Resizing according to the face detection model
        h,w,_ = imgS.shape
        # cv2.imwrite("1.jpg",imgS)
        img_mean = np.array([104, 117, 123])
        img = (imgS - img_mean) / 128 #Substracting mean from the image
        img = np.transpose(img, [2, 0, 1]) #transposing the shape accordingly
        img = np.expand_dims(img, axis=0) 
        img = img.astype(np.float32)
        ##############################################################
        
        #predicting the bounding box
        confidences, boxes = ort_session.run(None, {input_name: img})
        boxes, labels, probs = ut.predict(w, h, confidences, boxes, 0.7)

        #arranging the columns according to face_recognition requirement
        facesCurFrame = boxes
        if(facesCurFrame.shape[0]>0):
            facesCurFrame[:,[0,1,2,3]] = facesCurFrame[:,[1,2,3,0]]#Changing it as per requirements

        #seeing if any face is detected
        print(boxes.shape[0], facesCurFrame.shape[0])
        print(type(boxes), type(facesCurFrame))
        print(boxes, "\n\n\n", facesCurFrame)
        print("...")

        #Encoding
        encodeCurFrame = fr.face_encodings(imgS, facesCurFrame,model='small')

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
                    # cv2.imwrite("output.jpg", imgS)    
                    # markAttendance(name)
                else:
                    cv2.rectangle(imgS, (x1,y1), (x2,y2), (139,0,0),2)
                    cv2.rectangle(imgS, (x1,y2-35), (x2,y2),(0,255,255), cv2.FILLED)
                    cv2.putText(imgS, name, (x1+6,y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,0),2)
            
            else:
                cv2.rectangle(imgS, (x1,y1), (x2,y2), (139,0,0),2)
                cv2.rectangle(imgS, (x1,y2-35), (x2,y2),(0,255,255), cv2.FILLED)
                cv2.putText(imgS, name, (x1+6,y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,0),2)


        end = time.time()
        # cv2.imwrite("2.jpg", imgS)
        cv2.imshow("Image test", imgS)
        if count>=120:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print(120/(end-start))