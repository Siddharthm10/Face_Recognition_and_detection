import cv2
import numpy as np
import os

#CCTV - 1080*960(read by openCV)
#resized to 534*400(after trimming the time stamp)
# cv2.namedWindow("output", cv2.WINDOW_NORMAL) 
# cv2.resizeWindow("output", 1920, 1080)
cap = cv2.VideoCapture('rtsp://admin:admin123@192.168.0.104:554/')

# print("after")
while(True):
    ret, frame = cap.read()
    # frame = frame[:][100:]
    frame = cv2.resize(frame, (534,400))
    cv2.imshow("output",frame)
    print(frame.shape)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()