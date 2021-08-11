# Detect Social Distance Violations

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

## Tensorflow Object Detection Framework
* TensorFlow is a framework for machine learning, commonly used for machine learning specifically the family of deep leaning algorithms
* Deep leaning algorithms will take a long time to finish and that’s where the use of **GPUs** come in because they provide better processing speed compared to CPUs. 
* The flexible architecture of TensorFlow allows you to deploy computation to one or more CPUs or GPUs in a desktop, server, or mobile device with a single API

### Advantages of Tensorflow over OpenCV DNN
* OpenCV offers **GPU module**
* You will be the one to pick which classes and methods to use so at least **some knowledge about GPU programming** will be helpful. 
* In Tensorflow, you can **easily** use the GPU implementation by setting the number of GPUs you have or if you want to use both

### General Approach for Object Detection
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

### Convert the .weights into the corresponding TensorFlow model files
```
python src/detection_tensorflow_framework/save_model.py --weights ./data/yolov4.weights --output artifacts/checkpoints/yolov4-416 --input_size 416 --model yolov4 
```
This will create below files:
- checkpoints/yolov4-416/**saved_model.pb**
- checkpoints/yolov4-416/**keras_metadata.pb**
- checkpoints/yolov4-416/**variables**

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

## References
* [OpenCV](https://towardsdatascience.com/yolo-object-detection-with-opencv-and-python-21e50ac599e9 "OpenCV")
