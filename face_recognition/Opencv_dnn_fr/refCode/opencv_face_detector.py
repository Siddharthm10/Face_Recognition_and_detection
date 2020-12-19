import cv2
import numpy as np
import time
import imutils

print("[INFO]: Loading models...")
path = 'face_recognition/Opencv_dnn_fr/images/known/'#Path of known images to compare from
modelFile = "face_recognition/Opencv_dnn_fr/model/res10_300x300_ssd_iter_140000.caffemodel"
configFile = "face_recognition/Opencv_dnn_fr/model/deploy.prototxt.txt"
detector = cv2.dnn.readNetFromCaffe(configFile, modelFile)

cap = cv2.VideoCapture(0)

while(True):
    start1 = time.time()
    _, frame = cap.read()
    check1 = time.time()
    frame = imutils.resize(frame, width=600)
    imgS = cv2.resize(frame, (300,300))
    h,w,_ =  imgS.shape
    check2 = time.time()
    imageBlob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0), swapRB=False, crop=False)
    check4 = time.time()
    detector.setInput(imageBlob)
    faces = detector.forward()
    check5 = time.time()
    for i in range(faces.shape[2]):
        confidence = faces[0, 0, i, 2]
        if confidence > 0.5:
            box = faces[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x, y, x1, y1) = box.astype('int')
            cv2.rectangle(imgS, (x, y), (x1, y1), (0, 0, 255), 2)

    cv2.imshow("Image test", imgS)
        # if count>=120:
        #     break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
end = time.time()
print(end - start1)
print(check5 - start1)
print(check4 - start1)
print(check2 - start1)
print(check1 - start1)
