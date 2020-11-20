# Facial Recognition Pipeline
## Comparision Constraints
- Framework
  - Face-Detection
  - Embedder
  - Classifier
- Accuracy
- Speed(FPS)
- Ease in use

## Model Comparision
### Model 1: Face Recognition using OpenCV dnn
### Model 2: Face Recognition using Ultra-Light 
### Model 3: OpenFace pipeline using OpenCV dnn 



### Framework:
|S. No.| Models           |     Face-Detection    |      Embedder           |      Classifier    |
|------|:----------------:|:---------------------:|:-----------------------:|-------------------:|
|  1.  | Opencv-dnn-FR    | OpenCV - dnn(300*300) | Face Net (model uknown) | No Classifier used |
|  2.  | Ultra-light-FR   | Ultra-Light (640*480) | Face Net (model uknown) | No Classifier used |
|  3.  | Openface-pipe-FR | OpenCV - dnn(300*300) | openface.nn4.small2     |         SVM        |

Initial two models are using several features (facial landmarks) that make the process slow [I believe. :-( ] <br>
Task 1: Recognize what makes the first two pipelines slow.


### Speed: 
Fps for the same [Friends - Ross video](Friends_ross.mp4)
|  Model           |    FPS (Range)  |
|------------------|----------------:|
| Opencv-dnn-FR    |    4.71-5.20    |
| Ultra-light-FR   |    1.3-1.5      |
| Openface-pipe-FR |    12.00-13.23  |


### Accuracy
It is tested on mannually made Bollywood test set.
|  Model           |  Accuracy (%) |
|------------------|--------------:|
| Opencv-dnn-FR    |      92%      |
| Ultra-light-FR   |      92%      |
| Openface-pipe-FR |      TBC      |