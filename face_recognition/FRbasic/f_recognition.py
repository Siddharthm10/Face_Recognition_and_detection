if __name__ == "__main__":
        
    import cv2
    import numpy as np
    import face_recognition as fr
    import os
    import time
    import pickle

    path = 'face_recognition/FRbasic/images/known'
    images = []
    classNames = []
    tolerance = 0.6
    name = "Unknown"
    myList = os.listdir(path)
    for cls in myList:
        curImg = cv2.imread(f'{path}/{cls}')
        images.append(curImg)
        classNames.append(os.path.splitext(cls)[0])

    # encodeListKnown = findEncodings(images)
    with open("face_recognition/FRbasic/encodings.pickle", 'r+') as f:
        data = pickle.load(f.read())
        print(data)
        time.sleep(1000)
    encodeListKnown = list(data.values())

    cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture('rtsp://admin:admin123@192.168.0.104:554/')

    count = 0
    start = time.time()
    while True:
        
        success, img = cap.read()
        img = img[:][150:]
        imgS = cv2.resize(img, (0,0), None, 0.5, 0.5)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
        

        facesCurFrame = fr.face_locations(imgS, 2, "hog")
        encodeCurFrame = fr.face_encodings(imgS, facesCurFrame)
        for encodeFace, faceLoc in zip(encodeCurFrame, facesCurFrame):
            matches = fr.compare_faces(encodeListKnown, encodeFace)
            faceDis = fr.face_distance(encodeListKnown, encodeFace)

            matchIndex = np.argmin(faceDis)
            if faceDis[matchIndex]<tolerance:
                if matches[matchIndex]:
                    name = classNames[matchIndex]
                    y1, x2, y2, x1 = faceLoc
                    y1, x2, y2, x1 = y1*2, x2*2, y2*2, x1*2
                    cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0),1)
                    cv2.rectangle(img, (x1,y2-18), (x2,y2),(0,255,255), cv2.FILLED)
                    cv2.putText(img, name, (x1+10,y2-10), cv2.FONT_HERSHEY_COMPLEX, 0.75, (0,0,0),1)
                    # cv2.imwrite("output.jpg", img)    
                    # markAttendance(name)
                else:
                    y1, x2, y2, x1 = faceLoc
                    y1, x2, y2, x1 = y1*2, x2*2, y2*2, x1*2
                    cv2.rectangle(img, (x1,y1), (x2,y2), (0,0,255),2)
                    cv2.rectangle(img, (x1,y2-35), (x2,y2),(0,255,255), cv2.FILLED)
                    cv2.putText(img, name, (x1+6,y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,0),2)

            else:
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1*2, x2*2, y2*2, x1*2
                cv2.rectangle(img, (x1,y1), (x2,y2), (0,0,255),2)
                cv2.rectangle(img, (x1,y2-18), (x2,y2),(0,255,255), cv2.FILLED)
                cv2.putText(img, name, (x1+10, y2-10), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,0), 1)

 
        cv2.imshow("Webcam", img)
        count+=1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        if(count>120):
            break
    end = time.time()
    duration  = end - start
    fps = 120/duration
    print(fps)
    # print(fps) #- 5.2677 at 1/4 times resolution
               # - 9.026 at 1/2 times resolution

    cap.release()
    cv2.destroyAllWindows()

