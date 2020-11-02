import cv2
import numpy as np

cv2.namedWindow("output", cv2.WINDOW_NORMAL) 
cv2.resizeWindow("output", 1920, 1080)
cap = cv2.VideoCapture('rtsp://admin:admin123@192.168.0.104/')

# print("after")
while(True):
    ret, frame = cap.read()
    cv2.imshow("output",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()