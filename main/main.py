
def area_of(left_top, right_bottom):
    """
    Compute the areas of rectangles given two corners.
    Args:
        left_top (N, 2): left top corner.
        right_bottom (N, 2): right bottom corner.
    Returns:
        area (N): return the area.
    """
    hw = np.clip(right_bottom - left_top, 0.0, None)
    return hw[..., 0] * hw[..., 1]

def iou_of(boxes0, boxes1, eps=1e-5):
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Args:
        boxes0 (N, 4): ground truth boxes.
        boxes1 (N or 1, 4): predicted boxes.
        eps: a small number to avoid 0 as denominator.
    Returns:
        iou (N): IoU values.
    """
    overlap_left_top = np.maximum(boxes0[..., :2], boxes1[..., :2])
    overlap_right_bottom = np.minimum(boxes0[..., 2:], boxes1[..., 2:])

    overlap_area = area_of(overlap_left_top, overlap_right_bottom)
    area0 = area_of(boxes0[..., :2], boxes0[..., 2:])
    area1 = area_of(boxes1[..., :2], boxes1[..., 2:])
    return overlap_area / (area0 + area1 - overlap_area + eps)


def hard_nms(box_scores, iou_threshold, top_k=-1, candidate_size=200):
    """
    Perform hard non-maximum-supression to filter out boxes with iou greater
    than threshold
    Args:
        box_scores (N, 5): boxes in corner-form and probabilities.
        iou_threshold: intersection over union threshold.
        top_k: keep top_k results. If k <= 0, keep all the results.
        candidate_size: only consider the candidates with the highest scores.
    Returns:
        picked: a list of indexes of the kept boxes
    """
    scores = box_scores[:, -1]
    boxes = box_scores[:, :-1]
    picked = []
    indexes = np.argsort(scores)
    indexes = indexes[-candidate_size:]
    while len(indexes) > 0:
        current = indexes[-1]
        picked.append(current)
        if 0 < top_k == len(picked) or len(indexes) == 1:
            break
        current_box = boxes[current, :]
        indexes = indexes[:-1]
        rest_boxes = boxes[indexes, :]
        iou = iou_of(
            rest_boxes,
            np.expand_dims(current_box, axis=0),
        )
        indexes = indexes[iou <= iou_threshold]

    return box_scores[picked, :]

def predict(width, height, confidences, boxes, prob_threshold, iou_threshold=0.5, top_k=-1):
    """
    Select boxes that contain human faces
    Args:
        width: original image width
        height: original image height
        confidences (N, 2): confidence array
        boxes (N, 4): boxes array in corner-form
        iou_threshold: intersection over union threshold.
        top_k: keep top_k results. If k <= 0, keep all the results.
    Returns:
        boxes (k, 4): an array of boxes kept
        labels (k): an array of labels for each boxes kept
        probs (k): an array of probabilities for each boxes being in corresponding labels
    """
    boxes = boxes[0]
    confidences = confidences[0]
    picked_box_probs = []
    picked_labels = []
    for class_index in range(1, confidences.shape[1]):
        probs = confidences[:, class_index]
        mask = probs > prob_threshold
        probs = probs[mask]
        if probs.shape[0] == 0:
            continue
        subset_boxes = boxes[mask, :]
        box_probs = np.concatenate([subset_boxes, probs.reshape(-1, 1)], axis=1)
        box_probs = hard_nms(box_probs,
           iou_threshold=iou_threshold,
           top_k=top_k,
           )
        picked_box_probs.append(box_probs)
        picked_labels.extend([class_index] * box_probs.shape[0])
    if not picked_box_probs:
        return np.array([]), np.array([]), np.array([])
    picked_box_probs = np.concatenate(picked_box_probs)
    picked_box_probs[:, 0] *= width
    picked_box_probs[:, 1] *= height
    picked_box_probs[:, 2] *= width
    picked_box_probs[:, 3] *= height
    return picked_box_probs[:, :4].astype(np.int32), np.array(picked_labels), picked_box_probs[:, 4]




if __name__ == "__main__":  
    #importing Libraries and dependencies used:
    from datetime import datetime
    import face_recognition as fr
    import numpy as np
    import pandas as pd
    import onnx
    import onnxruntime as ort
    from onnx_tf.backend import prepare
    import time
    import json
    import cv2
    import os
    import pickle
    # from main.utils.predict import areaof
    # from main.utils.predict import iou_of
    # from main.utils.predict import hard_nms
    # from main.utils.predict import predict
    # import main.utils.predict

    # Initializing the variables
    path = 'main/images/known'#Path of known images to compare from
    classNames = []#To append the filenames
    tolerance = 0.6#Threshold distance from the closest image
    myList = os.listdir(path)#adding all the item's name present at path
    for cls in myList:
        classNames.append(os.path.splitext(cls)[0])

    #Reading the encodings
    with open("Face_detection/encodings.json", 'r+') as f:
        data = json.load(f)

    encodeListKnown = list(data.values())

    # load the model, create runtime session & get input variable name
    onnx_path = 'main/model/ultra_light_640.onnx'
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    onnx.helper.printable_graph(onnx_model.graph)
    predictor = prepare(onnx_model)
    ort_session = ort.InferenceSession(onnx_path)

    input_name = ort_session.get_inputs()[0].name


    #Capturing video
    # video_capture = cv2.VideoCapture(0)   
    # video_capture = cv2.VideoCapture("main/filename.avi")
    video_capture = cv2.VideoCapture('rtsp://admin:admin123@192.168.0.104:554/')

    # count = 0
    while(True):
        success, frame = video_capture.read()
        if not success:
            print("Couldn't read from webcam")
            break
        # count = count+1
        # start = time.time()
        #getting the height and width to show the output according to the user
        h,w,_ = frame.shape


        # $$$ Facial Detection $$$ - model used - Ultra Light
        #####################Pre-Processing###########################
        # img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        imgS = cv2.resize(frame, (640, 480)) #Resizing according to the face detection model
        cv2.imwrite("1.jpg",imgS)
        img_mean = np.array([127, 127, 127])
        img = (imgS - img_mean) / 128 #Substracting mean from the image
        img = np.transpose(img, [2, 0, 1]) #transposing the shape accordingly
        img = np.expand_dims(img, axis=0) 
        img = img.astype(np.float32)
        ##############################################################
        
        #predicting the bounding box
        confidences, boxes = ort_session.run(None, {input_name: img})
        boxes, labels, probs = predict(w, h, confidences, boxes, 0.7)

        #arranging the columns according to face_recognition requirement
        facesCurFrame = boxes
        if(facesCurFrame.shape[0]>0):
            facesCurFrame[:,[0,1,2,3]] = facesCurFrame[:,[1,2,3,0]]#Changing it as per requirements

        #seeing if any face is detected
        print(boxes.shape[0], facesCurFrame.shape[0])
        print(type(boxes), type(facesCurFrame))
        print(boxes, "\n\n\n", facesCurFrame)
        print("...")

        #Encoding
        encodeCurFrame = fr.face_encodings(imgS, facesCurFrame)

        #looping through faces
        for encodeFace, faceLoc in zip(encodeCurFrame, facesCurFrame):
            #Comparing faces
            name = "Unknown"#By default if the face isnt recognised
            matches = fr.compare_faces(encodeListKnown, encodeFace)
            faceDis = fr.face_distance(encodeListKnown, encodeFace)
            y1, x2, y2, x1 = faceLoc

            matchIndex = np.argmin(faceDis)
            if faceDis[matchIndex]<tolerance:
                if matches[matchIndex]:
                    name = classNames[matchIndex]
                    cv2.rectangle(imgS, (x1,y1), (x2,y2), (0,255,0),2)
                    cv2.rectangle(imgS, (x1,y2-35), (x2,y2),(0,255,255), cv2.FILLED)
                    cv2.putText(imgS, name, (x1+6,y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,0),2)
                    cv2.imwrite("output.jpg", imgS)    
                    # markAttendance(name)
                else:
                    cv2.rectangle(imgS, (x1,y1), (x2,y2), (139,0,0),2)
                    cv2.rectangle(imgS, (x1,y2-35), (x2,y2),(0,255,255), cv2.FILLED)
                    cv2.putText(imgS, name, (x1+6,y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,0),2)
            
            else:
                cv2.rectangle(imgS, (x1,y1), (x2,y2), (139,0,0),2)
                cv2.rectangle(imgS, (x1,y2-35), (x2,y2),(0,255,255), cv2.FILLED)
                cv2.putText(imgS, name, (x1+6,y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,0),2)


        # end = time.time()
        cv2.imwrite("2.jpg", imgS)
        cv2.imshow("Image test", imgS)
        # if count>=120:
            # break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # print(end-start)