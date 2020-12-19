# Motion Detection with Raspberry Pi

Lately, CCTV security systems have multiple algorithms running to ensure safety, such as Face-Recognition, Object Detection, Theft Detection, Fire Alert, etcetera.
We implement many algorithms on top of motion detection because there is no point in running all those processes on idle frames. In this article, we'll discuss implementing motion detection based video saving.

### Dependencies:
OpenCV: ```pip install opencv-python```

## Basic Motion Detection
Here, we will discuss the code and basic understanding of how things are working in the background. In computer vision, we consider motion as the change in surroundings. And to calculate the transition, we must have a background image to compare. So, we save the first image at the beginning of the program.


```
# Converting the image to GrayScale
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray,(21,21),0)

# Saving the First Frame
if first_frame is None:
    first_frame = gray
    continue
```

Then we compare the subsequent frames with the saved first frame to observe the difference.After calculating the difference, we can apply the thresholds to turn it into a black-and-white image. 

```
#Calculates difference to detect motion
delta_frame = cv2.absdiff(first_frame, gray)

#Applies Threshold and converts it to black & white image
thresh_delta = cv2.threshold(delta_frame, 30, 255, cv2.THRESH_BINARY)[1]
thresh_delta = cv2.dilate(thresh_delta, None, iterations=0)

#finding contours on the white portion(made by the threshold)
cnts,_ = cv2.findContours(thresh_delta.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

```

And the last command finds countours in that black-and-white image and gives coordinates for creating a bounding box as shown in the video above.

## Perks of using motion detection:
- It doesn't save useless idle footage. Hence, reducing the workload of other algorithms, as idle frames won't be saved for processing.
- It requires fewer computations and is suitable for live implementations.
## Obstacles & Solutions
The given factors lead to unsatisfactory contour detection:
1. Motion detection's naïve approach saves the first frame at the start of the execution for all the comparisons. It's not good for several reasons.
    - Lighting conditions might change during the day.
    - Change in weather.
    - Blocked camera at the time of execution.

**Solution**: This problem can easily be solved by regularly updating the saved frame at regular intervals when there is no motion.
```
# Number of idle frames to pass before changing the saved frame 
# for further comparisions
FRAMES_TO_PERSIST = 1000
```
And then place this in the while loop:
```
#increment delay counter for every idle frame
delay_counter += 1

#Update the saved first frame
if delay_counter > FRAMES_TO_PERSIST:
    delay_counter = 0
    first_frame = next_frame
```
Set delay_counter to zero when motion is detected. 

2. Minute objects (like bees and insects) and minor unnecessary motions that are usually useless are stored.

**Solution**: We should set a threshold over the area as shown in the snippet.

```
# Minimum boxed area(in pixels) for a detected motion to count as actual motion
# Use to filter out noise or small objects
MIN_SIZE_FOR_MOVEMENT = 2000
```

And then place an if statement like this in the while loop:
```
#Checks if the area is big enough to be considered as motion.
if cv2.contourArea(c) > MIN_SIZE_FOR_MOVEMENT:
    #Your code
```


## Benchmarks on various platforms
#### Raspberry Pi :
- 
#### Jetson Nano :
- 
#### PC :
- 


## Potential Applications:
#### Smart Bell: 
It will automatically trigger the bell and send you a snap, if someone is standing at your doorstep.

#### Potential Threat Alert:
It will alert you if someone is standing in front of your house for a longer than normal duration.

## Conclusion
In this article, we implemented a very basic but important algorithm that we can use for efficiently running all other algorithms. Many more modification can be done to this motion detection algorithm to make it more robust.


