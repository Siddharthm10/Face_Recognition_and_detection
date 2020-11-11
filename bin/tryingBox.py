import numpy as np


boxes = np.array([[0,1,2,3],[302, 185, 466, 410]])
facesCurFrame = boxes
print(type(facesCurFrame))
print(type(boxes))



boxes[:,[0,1,2,3]] = boxes[:,[1,2,3,0]]
index = [1,2]
for i,j in zip(index,boxes):
    print(i,j)
# print(boxes)