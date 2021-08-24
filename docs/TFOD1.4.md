# TFOD1.4
- [Setup Tf1.4 environment](#setup-tf14-environment)
    - [Opening Camera for Object Detection on local setup](#opening-camera-for-object-detection-on-local-setup)
    - [Annotation Tools](#annotation-tools)
- [Formatting annotated files](#formatting-annotated-files)
- [Changes for model configuration file](#changes-for-model-configuration-file)
- [Start training](#start-training)
- [Convert Checkpoints to Frozen Inference Graph](#convert-checkpoints-to-frozen-inference-graph)
- [Inference via model](#inference-via-model)

# [Setup Tf14 environment](https://pastebin.com/YDgbqzTx)
1. Create Conda environment and install libraries
    ```
    # Creating virtual env using conda
    conda create -n tf14env python=3.6.9
    conda activate tf14env

    # pypi 
    pip install pillow lxml Cython contextlib2 jupyter matplotlib pandas opencv-python tensorflow==1.14.0
    ```
2. Download files required for Object Detection TFOD1.4 Framework
   - [Download Framework Repository](https://github.com/tensorflow/models/tree/v1.13.0)
     - provides models/models/**research** directory
   - [Download Utils](https://drive.google.com/file/d/12F5oGAuQg7qBM_267TCMt_rlorV-M7gf/view)
3. Download files required for Pre Trained Object Detection Model
   - [Select and Download a Pre Trained Model from Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md)
     - example [faster_rcnn_inception_v2_coco_2018_01_28](http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz)
   - This step can be skipped as we are already doing it from [object detection notebook](https://colab.research.google.com/drive/1niUuMhB4QRteHxaCDVdzv1Ta6ERGjXhF?usp=sharing)
4. Install Object Detection TFOD1.4 Framework via **setup.py** from **Research** folder
    ```
    python setup.py install #install object detection
    ```
5. Protobuff to **py** conversion
    ```
    conda install -c anaconda protobuf
    # linux mac
    protoc object_detection/protos/*.proto --python_out=.
    #windows
    protoc object_detection/protos/*.proto --python_out=.
    ```
6. Object Detection on [Colab](https://colab.research.google.com/drive/1niUuMhB4QRteHxaCDVdzv1Ta6ERGjXhF?usp=sharing)
7. Object Detection on [local](https://colab.research.google.com/drive/1lteEd6R6C5QFQO02F2durQY4d-YFyALk?usp=sharing)
   - Parent Directory (models/model/research)
   - !conda env list ("*" shows current active env)
   - You might have to move object_detection_tutorial.ipynb to research directory
     ```
     %matplotlib inline
     plt.figure(figsize=(200,200))
     plt.imshow(image_np)
     ```
## Opening Camera for Object Detection on local setup
  ```
  import cv2

  cap = cv2.VideoCapture(0)
  with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
      while True:
        ret, image_np = cap.read()
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        scores = detection_graph.get_tensor_by_name('detection_scores:0')
        classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        # Actual detection.
        (boxes, scores, classes, num_detections) = sess.run([boxes, scores, classes, num_detections],feed_dict={image_tensor: image_np_expanded})
        # Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8)

        cv2.imshow('object detection', cv2.resize(image_np, (800,600)))
        if cv2.waitKey(25) & 0xFF == ord('q'):
          cv2.destroyAllWindows()
          break

  cap.release()
  ```  
## Annotation Tools
- Labelimg
- CVAT
- VIA
- Darwin ($)
- Roboflow ($)
- Makesense.ai
- Superviely ($)
- label studio
- Prodigy ($)
- VOTT

## Formatting annotated files
As labelimg annotations are in **XML** format and tensor flow object detection frameworks requires annotations to be in **.record** format, we follow below steps fot annotations generated via Labelimg
1. Execute script **xml_to_csv**
    - Copy the file **xml_to_csv** from utils folder to models repo in reaserch folder 
       ```
       python xml_to_csv.py (tensorflow1/models/research/object_detection)
       ```
2. Execute script **tfgenerate.record**
    - Copy the file **tfgenerate.record** from utils folder to models repo in reaserch folder
       ```
       python generate_tfrecord.py --csv_input=images/train_labels.csv --image_dir=images/train --output_path=train.record
       python generate_tfrecord.py --csv_input=images/test_labels.csv --image_dir=images/test --output_path=test.record
       ```

## Changes for model configuration file
1. num_classes: 90 to 2
2. fine_tune_checkpoint (line-107): "PATH_TO_BE_CONFIGURED/model.ckpt" to faster_rcnn/model.ckpt
3. num_steps (line-113): 200000 To 1000
4. Location of Train files
   - train.record (line-122): "input_path": "PATH_TO_BE_CONFIGURED/mscoco_train.record-?????" to train.record
   - label_map_path (line-124): "PATH_TO_BE_CONFIGURED/mscoco_label_map.pbtxt-?????" to training/lablemap.pbtxt
5. Location of Test files
   - test.record (line-136): "input_path": "PATH_TO_BE_CONFIGURED/mscoco_train.record-?????" to test.record
   - label_map_path (line-138): "PATH_TO_BE_CONFIGURED/mscoco_label_map.pbtxt-?????" to training/lablemap.pbtxt
  
## Start training
- [Mask Detection](https://colab.research.google.com/drive/1Z-XfUVA6Aj9VT3f7CiPCL_9mqz8aXTs8?usp=sharing)
- Copy the file **train.py** from legacy folder in object_detection to research, Run below command from research folder
    ```
    python train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/faster_rcnn_inception_v2_coco.config
    ```
- If training fails, 
    - Copy "deployment" and "net" folder from "research/slim"
    - Paste them to **research**

## Convert Checkpoints to Frozen Inference Graph
Replace the XXXX with the last generated ckpt file inside the training folder
   ```
   python export_inference_graph.py --input_type image_tensor --pipeline_config_path training/faster_rcnn_inception_v2_coco.config --trained_checkpoint_prefix training/model.ckpt-1000 --output_directory inference_graph
   ```
- Verify Created model at /research/mask_model **frozen_inference_graph.pb**

## [Inference via model](https://colab.research.google.com/drive/175z_auclmIs_flCjhmp1msNIaLMGZzKN?usp=sharing)
- Load (frozen) Tensorflow model into memory
- Load label map
