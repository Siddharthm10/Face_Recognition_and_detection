if __name__ == '__main__':
    """
    Transient  - short time motion tracker 

    Description:
    This script runs a motion detector! It detects transient motion in a room
    and said movement is large enough, and recent enough, reports that there is
    motion!
    Run the script with a working webcam! You'll see how it works!
    """


    import cv2
    import numpy as np
    import time

    # =============================================================================
    # USER-SET PARAMETERS
    # =============================================================================

    # Number of frames to pass before changing the frame to compare the current
    # frame against
    FRAMES_TO_PERSIST = 100

    # Minimum boxed area for a detected motion to count as actual motion
    # Use to filter out noise or small objects
    MIN_SIZE_FOR_MOVEMENT = 1000

    # Counter for saving the photos & exiting the program
    count=0

    # =============================================================================
    # CORE PROGRAM
    # =============================================================================


    # Create capture object
    # cap = cv2.VideoCapture('rtsp://admin:admin123@192.168.0.104:554/')
    cap = cv2.VideoCapture(0)

    # Init frame variables
    first_frame = None

    # Init display font and timeout counters
    font = cv2.FONT_HERSHEY_SIMPLEX
    delay_counter = 0
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    result = cv2.VideoWriter('motion_detection/transientMotionDetector/piConfig/Nov23.avi',  
                       cv2.VideoWriter_fourcc(*'MJPG'), 
                       fps, (534,400))


    # LOOP!
    while True:

        # Set transient motion detected as false
        transient_movement_flag = False

        # Read frame
        success, frame = cap.read()

        # If there's an error in capturing
        if not success:
            print("CAPTURE ERROR")
            break

        # Resize and save a greyscale version of the image
        ratio = frame.shape[0]/frame.shape[1]
        width = 750
        # frame = cv2.resize(frame, (int(ratio*width), width))
        frame = cv2.resize(frame, (534,400))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Blur it to remove camera noise (reducing false positives)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        # If the first frame is nothing, initialise it
        if first_frame is None:
            first_frame = gray

        delay_counter += 1

        # Otherwise, set the first frame to compare as the previous frame
        # But only if the counter reaches the appriopriate value
        # The delay is to allow relatively slow motions to be counted as large
        # motions if they're spread out far enough
        if delay_counter > FRAMES_TO_PERSIST:
            delay_counter = 0
            first_frame = gray

        # Compare the two frames, find the difference
        frame_delta = cv2.absdiff(gray, first_frame)
        thresh = cv2.threshold(frame_delta, 30, 255, cv2.THRESH_BINARY)[1]

        # Fill in holes via dilate(), and find contours of the thesholds
        thresh = cv2.dilate(thresh, None, iterations = 1)
        cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # loop over the contours
        for c in cnts:

            # Save the coordinates of all found contours
            (x, y, w, h) = cv2.boundingRect(c)

            # If the contour is too small, ignore it, otherwise, there's transient
            # movement
            if cv2.contourArea(c) > MIN_SIZE_FOR_MOVEMENT:
                transient_movement_flag = True

                # Draw a rectangle around big enough movements
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)


        # The moment something moves momentarily, reset the persistent
        # movement timer.
        if transient_movement_flag == True:
            # cv2.imwrite("./frames/frame{}.jpg".format(count), frame)
            # print("frame{} saved".format(count))
            result.write(frame)
            count+=1

        # Interrupt trigger by pressing q to quit the open CV program
        ch = cv2.waitKey(1)
        if count==20000:
            break
        if ch & 0xFF == ord('q'):
            break
        #10 -fps



    # Cleanup when closed
    cv2.destroyAllWindows()
    cap.release()
