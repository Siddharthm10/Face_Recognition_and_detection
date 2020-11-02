import cv2
import numpy as np

cap = cv2.VideoCapture(0)
while True:
    success, frame = cap.read()
    if not success:
        print("Failed to Capture.")
    else:
        cv2.imshow("output",frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()

