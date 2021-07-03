# YoloV4_Detect_Social_Distance_Violations
* [Overview](#overview)
* [Project Setup](#project-setup)
* [OpenCV DNN](#opencv-dnn)
* [Tensorflow](#tensorflow)
  * [OpenCV with tensorflow](#opencv-with-tensorflow)
  * [Custom Functions](#custom-functions)
    * [Crop](#crop)
    * [OCR](#ocr)
    * [Count Objects of a class](#count-objects-of-a-class)
    * [License Plate](#license-plate)
    * [Commands](#commands)
* [Conclusion](#conclusion)

# Overview
* Step 1: Distance Calculation based on Centroids returned by the model
* Step 2. Compute the Euclidean distances between "All pairs" of the centroids
* Step 3: Compare calculated distance with the Minimum distance to be flagged as Violation

## Project Setup 
* Create environment
```bash
conda create -n social_distance_violation python=3.7 -y
```
* Activate environment
```bash
conda activate social_distance_violation
```
* Install the requirements.txt
```bash
pip install -r requirements.txt
```

# OpenCV DNN
## Overview of DNN Module
* DNN (Deep Neural Network) module was initially part of opencv_contrib repo. 
* It has been moved to the master branch of opencv repo last year, giving users the ability to run inference on pre-trained deep learning models within OpenCV itself.
* One thing to note here is, **dnn module** is not meant be used for training. **It’s just for running inference on images/videos**.
* Initially only Caffe and Torch models were supported. Over the period support for different frameworks/libraries like TensorFlow is being added.
* **Support for YOLO/DarkNet has been added recently**

## Detection from the Image
```
detect_image.py
```
* Load model and related parameters: get_model_labels_and_output_layers.py
* Object_detection.py
  * get_detection_results.py
  * detect_object
```
    convert_image_to_blob(frame)

    confidences, boxes, centroids = get_detection_details(frame,
                                                          each_layer_output,
                                                          minimum_confidence_score,
                                                          object_index)
                                                          
    indexes = apply_non_maxima_suppression(boxes, confidences, minimum_confidence_score, nms_threshold_value)
```
* detect_violations.py
```
    if len(results) >= 2:
        # Step 1: Distance Calculation based on Centroids
        # i. Create an Array of Centroids
        centroids = np.array([r[2] for r in results])

        # ii. Compute the Euclidean distances between "All pairs" of the centroids
        distance_between_observations = dist.cdist(centroids, centroids, metric="euclidean")

        # Step 2: Compare calculated distance with the Minimum distance to be flagged as Violation
        for i in range(0, distance_between_observations.shape[0]):
            for j in range(i + 1, distance_between_observations.shape[1]):

                # Step 3: Is Distance between any two centroid pairs is less than the configured "Number of pixels"
                if distance_between_observations[i, j] < minimum_distance:

                    # Step 4: Mark the observations/persons as Violations
                    set_of_violations.add(i)
                    set_of_violations.add(j)

    return set_of_violations
    
 ```
* draw_detections_and_violations.py
 
 ## Detection from the Video
```
detect_video.py
```
Same as above process, except that we process for each frame
```
    # video_stream = cv2.VideoCapture(-1)
    video_stream = cv2.VideoCapture("inputs/pedestrians.mp4")
    while True:
        # Read the next frame from the file
        (grabbed, frame) = video_stream.read()
        # End of the stream: when frame is not grabbed then we have reached the end of stream
        if not grabbed:
            break

        frame = imutils.resize(frame, width=700)
        detection_results = get_detection_results(frame,
                                                  model,
                                                  layer_numbers,
                                                  person_index
                                                  )
        # print(detection_results)
        # (0.998525857925415, (377, 179, 557, 736), (467, 458))
        # Confidence Score, Bounding Box coordinates (x1, y1, x2, y2), Centroid
        violations = detect_violations(detection_results)
        # print(violations)
        output_frame = draw_detections_and_violations(frame, detection_results, violations)
        write_and_save_video(output_frame)
```
# Tensorflow
* TensorFlow is a framework for machine learning, commonly used for machine learning specifically the family of deep leaning algorithms
* Deep leaning algorithms will take a long time to finish and that’s where the use of **GPUs** come in because they provide better processing speed compared to CPUs. 
* The flexible architecture of TensorFlow allows you to deploy computation to one or more CPUs or GPUs in a desktop, server, or mobile device with a single API

## Advantages of Tensorflow over OpenCV DNN
* OpenCV offers **GPU module**
* You will be the one to pick which classes and methods to use so at least **some knowledge about GPU programming** will be helpful. 
* In Tensorflow, you can **easily** use the GPU implementation by setting the number of GPUs you have or if you want to use both

## OpenCV with tensorflow
* Step 1: Load model
* Step 2: Read Image
* Step 3: Get Detection Results
  * Boundary Boxes (x,y,w,h)
  * Confidence Score
  * Detected Classes
  * Number of Objects
* Step 4: Filter Classes to show as Detection
* Step 5: Draw Boundary Box
* Step 6: Show Image
* Step 7: Save Detections

### Detect_object method
#### Input Arguments
* Yolov4 Tensorflow formatted model
* images_data = np.asarray(images_data).astype(np.float32)
```
def detect_object(model, data):
    infer = model.signatures['serving_default']
    batch_data = tf.constant(data)
    predicted_boundary_box = infer(batch_data)
    boxes, predicted_confidence = get_detection_details(predicted_boundary_box)
    return boxes, predicted_confidence
```
#### Output
* Boundary Boxes
* Confidence Score

### Get_Detection_Results method
Out of all the detections, consider only those detections which conforms to
* IOU Threshold (0.45)
* Baseline Score (0.50)
#### Input Arguments
* Yolov4 Tensorflow formatted model
* images_data = np.asarray(images_data).astype(np.float32)
* IOU Threshold
* Baseline Score
* Original Image
```
def get_detection_results(yolo_v4_model, images_data, iou, score, original_image):
    boxes, confidence_score = detect_object(yolo_v4_model, images_data)

    boxes, scores, classes, valid_detections = apply_non_max_suppression(boxes,
                                                                         confidence_score,
                                                                         iou,
                                                                         score)

    # Format Results
    formatted_boundary_boxes = format_boundary_box(original_image, boxes)

    detection_details = [formatted_boundary_boxes, scores.numpy()[0], classes.numpy()[0],
                         valid_detections.numpy()[0]]

    return detection_details
```
#### apply_non_max_suppression
* This will Filter the strong detections using Tensorflow provided method **tf.image.combined_non_max_suppression**
#### Output
* Formatted Boundary Boxes (x,y,w,h)
* Confidence Score
* Detected Classes
* Number of Objects

## Custom Functions
### Crop
* We already have information of **Boundary box** for the detection
* Crop the image using the coordinates of the Boundary Box
```
# get box coordinates
xmin, ymin, xmax, ymax = boxes[i]
# crop detection from image (take an additional 5 pixels around all edges)
cropped_img = img[int(ymin) - 5:int(ymax) + 5, int(xmin) - 5:int(xmax) + 5]
```
* Save Cropped image
```
img_name = class_name + '_' + str(counts[class_name]) + '.png'
img_path = os.path.join(path, img_name)
cv2.imwrite(img_path, cropped_img)
```
### OCR
* Use same method as above for cropping an image
```
# get box coordinates
xmin, ymin, xmax, ymax = boxes[i]
# get the subimage that makes up the bounded region and take an additional 5 pixels on each side
box = img[int(ymin) - 5:int(ymax) + 5, int(xmin) - 5:int(xmax) + 5]
```
* Greyscale the Region within Bounding box
```
# grayscale region within bounding box
gray = cv2.cvtColor(box, cv2.COLOR_RGB2GRAY)
```
* Threshold threshold the image using **OTSUS** method to **preprocess for tesseract**
```
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
```
* Perform a median blur to smooth image slightly
```
blur = cv2.medianBlur(thresh, 3)
# resize image to double the original size as tesseract does better with certain text size
blur = cv2.resize(blur, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
```
* Run **Tesseract**
```
try:
    text = pytesseract.image_to_string(blur, config='--psm 11 --oem 3')
    print("Class: {}, Text Extracted: {}".format(class_name, text))
except:
    text = None
```
* We have Output **text**
#### Disclaimer: 
* In order to run tesseract OCR you must first download the binary files and set them up on your local machine. 
* Please do so before proceeding or commands will not run as expected!
* Official Tesseract OCR Github Repo: tesseract-ocr/tessdoc
* For Windows [Windows Install](https://github.com/UB-Mannheim/tesseract/wiki)

### Count Objects of a class
Count can work in 2 modes
* Count Total number of objects detected
* Count Total number of objects detected for a **Particular Class**
```
 boxes, scores, classes, num_objects = data
 # create dictionary to hold count of objects
 counts = dict()

 # if by_class = True then count objects per class
 if by_class:
     class_names = read_class_names(cfg.YOLO.CLASSES)

     # loop through total number of objects found
     for i in range(num_objects):
         # grab class index and convert into corresponding class name
         class_index = int(classes[i])
         class_name = class_names[class_index]
         if class_name in allowed_classes:
             counts[class_name] = counts.get(class_name, 0) + 1
         else:
             continue

 # else count total objects found
 else:
     counts['total object'] = num_objects

 return counts
```
### License Plate
[License Plate](https://github.com/sbhrwl/YoloV4_Detect_Social_Distance_Violations/blob/main/src/detection_tensorflow_framework/core/utils.py)
```
if read_plate:
    height_ratio = int(image_h / 25)
    plate_number = recognize_plate(image, coor)
    if plate_number != None:
        cv2.putText(image, plate_number, (int(coor[0]), int(coor[1] - height_ratio)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.25, (255, 255, 0), 2)
```

## Commands

```
# custom yolov4
python save_model.py --weights ./data/custom.weights --output ./checkpoints/custom-416 --input_size 416 --model yolov4 

# Run custom yolov4 tensorflow model
python detect.py --weights ./checkpoints/custom-416 --size 416 --model yolov4 --images ./data/images/car.jpg
python detect.py --weights ./checkpoints/yolov4-416 --size 416 --model yolov4 --images ./data/images/dog.jpg --info

# Run yolov4 on webcam
python detect_video.py --weights ./checkpoints/yolov4-416 --size 416 --model yolov4 --video 0 --output ./detections/results.avi

# Crop
python detect.py --weights ./checkpoints/yolov4-416 --size 416 --model yolov4 --images ./data/images/dog.jpg --crop

# OCR for any Image
python detect.py --weights ./checkpoints/yolov4-416 --size 416 --model yolov4 --images ./data/images/dog.jpg --ocr

# Run yolov4 model while counting total objects detected
python detect.py --weights ./checkpoints/yolov4-416 --size 416 --model yolov4 --images ./data/images/dog.jpg --count
# Run yolov4 model while counting objects per class by Upodating by_class parameter as true in count_objects method
python detect.py --weights ./checkpoints/yolov4-416 --size 416 --model yolov4 --images ./data/images/dog.jpg --count

# Run License Plate Recognition
python detect.py --weights ./checkpoints/custom-416 --size 416 --model yolov4 --images ./data/images/car2.jpg --plate
python detect_video.py --weights ./checkpoints/custom-416 --size 416 --model yolov4 --video ./data/video/license_plate.mp4 --output ./detections/recognition.avi --plate
```
## Conclusion
* Object detected using only OpenCV is not optimal and using TensorFlow as a framework gives you more options to explore like networks, algorithms. 
* TensorFlow is optimal at **training** part i.e. at data handling(tensors) and OpenCV is optimal in **accessing and manipulating** data (resize, crop, webcams etc.,). 
* Thus, both are used together for object detection

## References
* [OpenCV](https://towardsdatascience.com/yolo-object-detection-with-opencv-and-python-21e50ac599e9 "OpenCV")
* [Custom Functions](https://github.com/theAIGuysCode/yolov4-custom-functions)
