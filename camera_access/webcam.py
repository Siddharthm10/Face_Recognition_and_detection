import cv2
import numpy as np

cap = cv2.VideoCapture(0)
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')#Pretrained cascade classfier to get the location of face
while True:
    success, frame = cap.read()
    if not success:
        print("Failed to Capture.")
    # gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    # faces = face_classifier.detectMultiScale(gray, 1.0485258, 6)
    # if faces is ():
    #     print("No faces found")
    # else:
    #     '''We iterate through our faces array and draw a rectangle over each face in faces'''
    #     for (x,y,w,h) in faces:
    #         cv2.rectangle(frame, (x,y), (x+w,y+h), (127,0,255), 2)
    
    cv2.imshow("output",frame)

    
    #Quiting 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()


