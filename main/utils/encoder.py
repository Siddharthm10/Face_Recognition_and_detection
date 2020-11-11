import face_recognition as fr
import cv2
import os
import json

path = 'images/known/'
images = []
classNames = []
myList = os.listdir(path)
encodeList = {}
for cls in myList:
    curImg = cv2.imread(f'{path}/{cls}')
    # images.append(curImg)
    curImg = cv2.cvtColor(curImg, cv2.COLOR_BGR2RGB)
    encode = fr.face_encodings(curImg)[0]
    classNames.append(os.path.splitext(cls)[0])
    encodeList[cls] = encode.tolist()

with open("Face_detection/encodings.json", "w+") as f:
    json.dump(encodeList,f)

print("Encoding Completed....")
