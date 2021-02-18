# Face Recognition System
This repository contains seperate python implementations of several pre-built face detection and recognition combinations.
**Note**: All the code use relative paths, please filepaths before executing the code.s
<!-- - [ ] - Human Detection (or Object Detection) -->
<!-- It is unneccessary as the face detection part works only for the human faces, it doesnt detect other things.-->
- [x] - Motion Detection
- [x] - Face Detection
- [x] - Face Recognition


### Note:
- Face Detector -> Face detection with face_recognition module is quite inaccurate. 
  - So, for the time being I am using Ultra Light detector which takes very less time and has considerate accuracy(0.1 sec for 80 faces)(80% accuracy)(practical time in live video is 0.4-0.6 sec)
  - YoloFace provides with the best accuracy with considerate time (1 sec for 129 faces )

  
### Additional updates that can be done 
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
    |
    |- basic_api
    |- Opencv_dnn_fr
    |- Ultra_light_fr
    |- withoutFR

- motion_detection - contains two implementation for motion detection
    |
    |-naive
    |-transientMotionDetector

--- 







## Dlib installation
- git clone https://github.com/davisking/dlib.git
- cd dlib
- mkdir build
- cd build
- cmake ..
- cmake -build .
- cd ..
- python setup.py install
<br>

- Thing to remember:
  - As of Dec 5,2020 CUDA doesnt support GNU 8+ so install dlib with downgraded version if u have GNU 9+
  - sudo update-alternatives /usr/bin/gcc gcc /usr/bin/gcc-8 1
  - sudo update-alternatives /usr/bin/gcc gcc /usr/bin/gcc-9 2
  - sudo update-alternatives --config gcc 
- Then select the gcc first before proceeding with the dlib installation 

KUDOS

:)

---