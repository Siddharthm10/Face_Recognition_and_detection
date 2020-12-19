# def bb_to_rect(startX, startY, endX, endY):
#     rect = dlib.rectangle(1,2,2,3)
#     rect.left()= startX
#     rect.top() = startY
#     rect.right() = endX
#     rect.bottom() = endY
#     return rect


def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
	coords = np.zeros((68, 2), dtype=dtype)

	# loop over the 68 facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)

	# return the list of (x, y)-coordinates
	return coords


class FaceAligner:
    def __init__(self, predictor, desiredLeftEye=(0.35, 0.35),
                    desiredFaceWidth=256, desiredFaceHeight=None):

        # store the facial landmark predictor, desired output left
        # eye position, and desired output face width + height
        self.predictor = predictor
        self.desiredLeftEye = desiredLeftEye
        self.desiredFaceWidth = desiredFaceWidth
        self.desiredFaceHeight = desiredFaceHeight

        # if the desired face height is None, set it to be the
        # desired face width (normal behavior)
        if self.desiredFaceHeight is None:
            self.desiredFaceHeight = self.desiredFaceWidth
    
    def align(self, image, gray, rect):

        # convert the landmark (x, y)-coordinates to a NumPy array
        shape = self.predictor(gray, rect)
        shape = shape_to_np(shape)
        
        # extract the left and right eye (x, y)-coordinates
        (lStart, lEnd) = FACIAL_LANDMARKS_IDXS["left_eye"]
        (rStart, rEnd) = FACIAL_LANDMARKS_IDXS["right_eye"]
        leftEyePts = shape[lStart:lEnd]
        rightEyePts = shape[rStart:rEnd]

        # compute the center of mass for each eye
        leftEyeCenter = leftEyePts.mean(axis=0).astype("int")
        rightEyeCenter = rightEyePts.mean(axis=0).astype("int")

        # compute the angle between the eye centroids
        dY = rightEyeCenter[1] - leftEyeCenter[1]
        dX = rightEyeCenter[0] - leftEyeCenter[0]
        angle = np.degrees(np.arctan2(dY, dX)) - 180

        # compute the desired right eye x-coordinate based on the
        # desired x-coordinate of the left eye
        desiredRightEyeX = 1.0 - self.desiredLeftEye[0]

        # determine the scale of the new resulting image by taking
        # the ratio of the distance between eyes in the *current*
        # image to the ratio of distance between eyes in the
        # *desired* image
        dist = np.sqrt((dX ** 2) + (dY ** 2))
        desiredDist = (desiredRightEyeX - self.desiredLeftEye[0])
        desiredDist *= self.desiredFaceWidth
        scale = desiredDist / dist

        # compute center (x, y)-coordinates (i.e., the median point)
        # between the two eyes in the input image
        eyesCenter = ((leftEyeCenter[0] + rightEyeCenter[0]) // 2,
            (leftEyeCenter[1] + rightEyeCenter[1]) // 2)

        # grab the rotation matrix for rotating and scaling the face
        M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)

        # update the translation component of the matrix
        tX = self.desiredFaceWidth * 0.5
        tY = self.desiredFaceHeight * self.desiredLeftEye[1]
        M[0, 2] += (tX - eyesCenter[0])
        M[1, 2] += (tY - eyesCenter[1])

        # apply the affine transformation
        (w, h) = (self.desiredFaceWidth, self.desiredFaceHeight)
        output = cv2.warpAffine(image, M, (w, h),
            flags=cv2.INTER_CUBIC)

        # return the aligned face
        return output




if __name__ == "__main__":
        
    import numpy as np
    from imutils import paths
    from collections import OrderedDict
    import imutils
    import pickle
    import cv2
    import os
    import dlib


    FACIAL_LANDMARKS_IDXS = OrderedDict([
        ("mouth", (48, 68)),
        ("right_eyebrow", (17, 22)),
        ("left_eyebrow", (22, 27)),
        ("right_eye", (36, 42)),
        ("left_eye", (42, 48)),
        ("nose", (27, 35)),
        ("jaw", (0, 17))
    ])

    # load our serialized face detector from disk
    print("[INFO] loading face detector...")
    protoPath = 'face_recognition/withoutFR/models/deploy.prototxt'
    modelPath = 'face_recognition/withoutFR/models/res10_300x300_ssd_iter_140000.caffemodel'
    detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

    #loading landmark detector
    print("[INFO] loading landmark predictor...")
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    fa = FaceAligner(predictor, desiredFaceWidth=256)

    # load our serialized face embedding model from disk
    print("[INFO] loading face recognizer...")
    embedder = cv2.dnn.readNetFromTorch('face_recognition/withoutFR/models/openface_nn4.small2.v1.t7')

    # load the actual face recognition model along with the label encoder
    recognizer = pickle.loads(open("face_recognition/withoutFR/output/recognizer.pickle", 'rb').read())
    le = pickle.loads(open('face_recognition/withoutFR/output/le.pickle', 'rb').read())

    # grab the paths to the test image in our dataset
    print("[INFO] quantifying faces...")
    imagePaths = list(paths.list_images('test_set/test/'))
    correctPredictions = 0
    totalPredictions = 0

    for (i, imagePath) in enumerate(imagePaths):
        print("[INFO] processing image {}/{}".format(i + 1,
            len(imagePaths)))

        #grabing the person name
        givenName = imagePath.split(os.path.sep)[-2]
        totalPredictions = totalPredictions + 1

        # load the image, resize it to have a width of 600 pixels (while
        # maintaining the aspect ratio), and then grab the image dimensions
        image = cv2.imread(imagePath)
        image = imutils.resize(image, width=800)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        (h, w) = image.shape[:2]

        # construct a blob from the image
        imageBlob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0), swapRB=False, crop=False)

        # apply OpenCV's deep learning-based face detector to localize
        # faces in the input image
        detector.setInput(imageBlob)
        detections = detector.forward()

        thresholdConfidence = 0.5

        # loop over the detections
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the
            # prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections
            if confidence > thresholdConfidence:

                # compute the (x, y)-coordinates of the bounding box for the
                # face
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # extract the face ROI
                rect = dlib.rectangle(startX, startY, endX, endY)
            
                face = fa.align(image, gray, rect)

                # face = image[startY:endY, startX:endX]
                (fH, fW) = face.shape[:2]

                # ensure the face width and height are sufficiently large
                if fW < 20 or fH < 20:
                    continue

                # construct a blob for the face ROI, then pass the blob
                # through our face embedding model to obtain the 128-d
                # quantification of the face
                faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96),
                    (0, 0, 0), swapRB=True, crop=False)
                embedder.setInput(faceBlob)
                vec = embedder.forward()

                # perform classification to recognize the face
                preds = recognizer.predict_proba(vec)[0]
                j = np.argmax(preds)
                proba = preds[j]
                name = le.classes_[j]

                print("Name: {}, predictedName: {}".format(givenName, name))
                if(givenName==name):
                    correctPredictions = correctPredictions+1
                    
                #draw the bounding box of the face along with the associated
                # probability
        #         text = "{}: {:.2f}%".format(name, proba * 100)
        #         y = startY - 10 if startY - 10 > 10 else startY + 10
        #         cv2.rectangle(image, (startX, startY), (endX, endY),
        #             (0, 0, 255), 2)
        #         cv2.putText(image, text, (startX, y),
        #         cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        #         cv2.imshow("alignedFace", face)
        # #show the output image
        # cv2.imshow("Image", image)
        # cv2.waitKey(0)
    print("Items correctly Predicted : {}".format(correctPredictions))
    print("Total Predictions : {}".format(totalPredictions))
    accuracy = (correctPredictions/totalPredictions)*100
    print(accuracy)


