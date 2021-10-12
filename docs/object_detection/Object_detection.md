# Computer Vision
Classification -> **OD** -> Segementation -> Tracking

- [Object Detection architectures](#object-detection-architectures)
- [Object Detection Frameworks](#object-detection-frameworks)
- [Output from an Object Detection Framework](#output-from-an-object-detection-framework)
- [Approach for building object dectection solutions](#approach-for-building-object-dectection-solutions)
- [Model Zoo](#model-zoo)
- [Metrics](#metrics)
- [Numpy Matrix and Tensors](#numpy-matrix-and-tensors)
- [Building an Object Detection solution](#building-an-object-detection-solution)
- [Object Detection model pipeline](#object-detection-model-pipeline)
- [References](#references)

## Object Detection architectures
- [RCNN](https://arxiv.org/pdf/1311.2524.pdf)
  - Since RCNN combines region proposals with CNNs, we call our method R-CNN: Regions with CNN features
  - **R-Region proposal** Convolution Neural Network
- Fatser RCNN (2 shot detector)
- Mask RCNN
- SSD
- paulbindass@gmail.com/sunny@ineuron.ai/query@ineuron.ai

## Object Detection Frameworks
### Low level frameworks
- Tensorflow
- Pytorch
- MxNet

### High Level frameworks
High level framworks are built on top of low level frameworks
- [TFOD1](https://github.com/sbhrwl/social_distance_violations/blob/main/docs/object_detection/TFOD1.4.md)
- [TFOD2](https://github.com/sbhrwl/social_distance_violations/blob/main/docs/object_detection/TFOD2.md)
- [Detectron2](https://github.com/sbhrwl/social_distance_violations/blob/main/docs/object_detection/Detectron2.md)
- [YoloV5](https://colab.research.google.com/drive/1MCLjg1ZPhzI9YSUseDBjCetrUEiwsu8y?usp=sharing)
  - Yolo is based on Darknet
  - [Yolo](https://github.com/ultralytics/yolov5/issues/12)
  - Yolo has a novel architecture making **a single pass across the network**, identifyin **regions of interest**

| Tensorflow | Pytorch | MxNet |
| ---------- | ------- | ----- |
| TFOD v1 | Detectron2  | GluonCV | 
| TFOD v2 | Pytorch lightening  |  | 
| | Yolo  |  | 

## Output from an Object Detection Framework
- Class of Object
- Confidence Score
- Bounding Box/Anchor Box ((x, y), l, w) or (Xmin, Ymin, Xmax, Ymax)
  - How to build a Rectangle
    - Center of Rectangle (x, y)
    - Length of Rectangle (l)
    - Width of Rectangle (w)
- Number of Instances   

## Approach for building object dectection solutions
- Choose an OD framework (TF/Detectron/Yolo)
- Choose a low level framework (TF/Pytorch)
- Choose an Object detection Architecture (RCNN/SSD)
- Choose CNN network
- Train your model using libraries provided by OD framework
  - Classified Object 
  - Confidence score
  - BBox cordinates (2 edges (diagnal) or center with width and height)
  - Number of Instances

## [Model Zoo](https://github.com/tensorflow/models/blob/v1.13.0/research/object_detection/g3doc/detection_model_zoo.md)
- Detection Model
  - SSD Family (light weight models): Process More Frames in One Pass
    - SSD
    - Yolo
  - RCNN Family (Heavy models)
    - RCNN (deprecated)
    - Fast RCNN (deprecated)
    - Fatser RCNN
    - Mask RCNN (Used for Segmentation Tasks)
  - CenterNet Family
- Classification Network (VGG/Inception/ResNet)
- Datatset: [COCO dataset](https://cocodataset.org/#explore)
  - 90 Classes

## Metrics
- mAP: Metric for Object Detection
- Speed(ms): How many Frames/Images a model will process per second (fps)
- TradeOff
  - if mAP is higher, speed will be lower
  - if speed is higher, mAP will be lower

## Numpy Matrix and Tensors
- Input Image (RGB) is converted to numpy matrix
- As Numpy matrix is not **GPU** compatible, so Numpy Matrix is converted to a **TENSOR** ex: [[R][G][B]]
- Tensor is GPU compatible

- Numpy array: [1,2,3,4] dtype=vector
- Tensor: [1,2,3,4] dtype=tensor
- [Inference code](https://colab.research.google.com/drive/175z_auclmIs_flCjhmp1msNIaLMGZzKN?usp=sharing) [walkthrough](https://www.youtube.com/watch?v=uyFKkqHaX8g&t=8s)- 01:40:00
- [Tasks](https://forms.office.com/pages/responsepage.aspx?id=T8AYy3GTMECkvj1F_-cJ7OTkCe-r2slCpSR8nE7buK1UN0lFV0NOSVY5WTg1OTVNSDJFT0lFWk02My4u)

## Building an Object Detection solution
- Framework
- Volume of Data
- mAP per class
- FPS based on micro controller, PCs, GPU accelerated PCs
- Model Selection and why?
  - SSD
  - CenterNet
  - Faster RCNN, RetinaNet basedmodel
- What kind of camera was used
- Distance from Camera

## Object Detection model pipeline
<img src='https://github.com/sbhrwl/social_distance_violations/blob/main/docs/object_detection/cv-model_pipeline-robolflow.jpg'>

## References
- ATC
  - [Documentation](https://drive.google.com/drive/folders/117VFe6ou3Izcan9Quy6pLuM9PzoE1Gj9)
  - [Code](https://github.com/iNeuron-Pvt-Ltd/ATCC)
