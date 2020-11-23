# Face Recognition System
1. Detect motion
2. If motion Detected
3. Check if Human is detected
4. If Yes
5. Check if the person is in the database
6. if yes 
7. Return Name of the person
8. if Not
9. Ask if the person is unKnown 
10. If known , add the person to the database


<!-- - [ ] - Human Detection (or Object Detection) -->
<!-- It is unneccessary as the face detection part works only for the human faces, it doesnt detect other things.-->
- [x] - Motion Detection
- [x] - Face Detection
- [x] - Face Recognition
- [ ] - Liveness detection  
 ~ [ ] - Whatsapp Notification - cancelled


### Note:
###
- Face Detector -> Face detection with face_recognition module is quite inaccurate. 
  - So, for the time being I am using Ultra Light detector which takes very less time and has considerate accuracy(0.1 sec for 80 faces)(80% accuracy)(practical time in live video is 0.4-0.6 sec)
  - YoloFace provides with the best accuracy with considerate time (1 sec for 129 faces )

- [ ] - Detectors are reducing the speed. User must see everything at the same speed(understand the installation that we ll take care of.).
  - because as soon as someone steps infront of the camera, the software speed will reduce. We might loose some faces, if crowd comes. 
  - To Be Solved by Thread Optimization(running multiple threads simultaneously)
  
### Additional updates that can be done 
- [ ] - Mask detection(reply with cannot detect face, person is wearing a mask.)
- [x] - Saving the video only when motion is detected.

### Problem Faced:
- [x] - Speed Issues 
  - solved: 
    - It was because of high dimensionality, which was solved after reducing the lag.
- [x] - Currently, when the motion is detected, the entire screen is processed due to which the processing time is high hence the slow relative speed
  - To be solved(Choose the better solution):
    - Potential Solution: The image should be cropped to the area where the motion is detected. And only that cropped version should be processed.
      - Potential Problems we might face: If there are multiple faces in an image(or multiple motions), it is not clear that how should we crop the image.

    - Another Potential Solution: Taking neural network zero logic to our advantage paint the entire region except the motion detected contour with black color. This will cause easy calculations and less computational time (probably)
  - Solved: 
    - Made a new pipeline to work with face_recognition task.(uses different face detection and recognition networks)
  
---
## Workflow:

- bin - trash
- blog - blog written for publication
- camera_access - code for accessing the cameras (testing phase not important for the main code)
- face_recognition - contains several implementations of face recognition
- motion_detection - contains two implementation for motion detection

--- 
