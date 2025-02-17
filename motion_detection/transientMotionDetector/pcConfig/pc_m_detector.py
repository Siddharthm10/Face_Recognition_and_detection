

if __name__ == '__main__':
    """

    Description:
    Transient  - for a short time.
    This script runs a transient motion detector.
    """

    import cv2
    import numpy as np
    import time

    # =============================================================================
    # USER-SET PARAMETERS
    # =============================================================================

    # Number of frames to pass before changing the frame to compare the current
    # frame against - on a 30 frames per second camera
    FRAMES_TO_PERSIST = 1000

    # Minimum boxed area for a detected motion to count as actual motion
    # Use to filter out noise or small objects
    MIN_SIZE_FOR_MOVEMENT = 2000

    # Minimum length of time where no motion is detected it should take
    #(in program cycles) for the program to declare that there is no movement
    MOVEMENT_DETECTED_PERSISTENCE = 100


    # Counter for saving the photos & exiting the program
    # count=0

    # =============================================================================
    # CORE PROGRAM
    # =============================================================================


    cap = cv2.VideoCapture(0) # Then start the webcam
    # cap = cv2.VideoCapture('rtsp://admin:admin123@192.168.0.104:554/')
    # fps = int(cap.get(cv2.CAP_PROP_FPS))

    
    # Init frame variables
    first_frame = None
    next_frame = None

    # Init display font and timeout counters
    font = cv2.FONT_HERSHEY_SIMPLEX
    delay_counter = 0
    movement_persistent_counter = 0

    # Write video
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    result = cv2.VideoWriter('motion_detection/transientMotionDetector/pcConfig/filename.avi',  
                       cv2.VideoWriter_fourcc(*'MJPG'), 
                       fps, (1066,400) )
    
    # LOOP!
    while True:

        # Set transient motion detected as false
        transient_movement_flag = False
        
        # Read frame
        ret, frame = cap.read()
        text = "Unoccupied"

        # If there's an error in capturing
        if not ret:
            print("CAPTURE ERROR")
            continue

        # Resize and save a greyscale version of the image
        ratio = frame.shape[1]/frame.shape[0]
        # print(ratio)
        width = 400
        frame = cv2.resize(frame, (int(ratio*width), width))
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
            print("Delay counter Reset")
            print(delay_counter)
            first_frame = next_frame

            
        # Set the next frame to compare (the current frame)
        next_frame = gray

        # Compare the two frames, find the difference
        frame_delta = cv2.absdiff(next_frame, first_frame)
        thresh = cv2.threshold(frame_delta, 30, 255, cv2.THRESH_BINARY)[1]

        # Fill in holes via dilate(), and find contours of the thesholds
        thresh = cv2.dilate(thresh, None, iterations = 2)
        cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # loop over the contours
        for c in cnts:

            # Save the coordinates of all found contours
            (x, y, w, h) = cv2.boundingRect(c)
            
            # If the contour is too small, ignore it, otherwise, there's transient
            # movement
            if cv2.contourArea(c) > MIN_SIZE_FOR_MOVEMENT:
                transient_movement_flag = True
                print("motion detected")
                # Draw a rectangle around big enough movements
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
                # print(x,y,w,h)
                # result.write(frame)
        
        

        # The moment something moves momentarily, reset the persistent
        # movement timer.
        if transient_movement_flag == True:
            movement_persistent_counter = MOVEMENT_DETECTED_PERSISTENCE
            # cv2.imwrite("motion_detection/transientMotionDetector/pcConfig/frames/frame{}.jpg".format(count), frame)
            # count+=1
            result.write(frame)

        # As long as there was a recent transient movement, say a movement
        # was detected    
        if movement_persistent_counter > 0:
            text = "Movement Detected " + str(movement_persistent_counter) + "Delay Counter " + str(delay_counter)
            movement_persistent_counter -= 1
        else:
            text = "No Movement Detected"

        # Print the text on the screen, and display the raw and processed video 
        # feeds
        cv2.putText(frame, str(text), (10,35), font, 0.75, (255,255,255), 2, cv2.LINE_AA)
        
        # For if you want to show the individual video frames
    #    cv2.imshow("frame", frame)
    #    cv2.imshow("delta", frame_delta)
        
        # Convert the frame_delta to color for splicing
        frame_delta = cv2.cvtColor(frame_delta, cv2.COLOR_GRAY2BGR)

        # Splice the two video frames together to make one long horizontal one
        # np.hstack((frame_delta, frame))
        stacked = np.hstack((frame_delta, frame))
        # print(frame_delta.shape)
        # print(frame.shape)
        # print(stacked.shape)
        cv2.imshow("frame", stacked)

        # Interrupt trigger by pressing q to quit the open CV program
        ch = cv2.waitKey(1)
        if ch & 0xFF == ord('q'):
            break
        #10 -fps


    
    # Cleanup when closed
    cv2.destroyAllWindows()
    cap.release()