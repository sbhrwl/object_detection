import os

# comment out below line to enable tensorflow outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags
from absl.flags import FLAGS
import core.utils as utils
from core.functions import *
from PIL import Image
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from core.model_predictions import load_model, \
    opencv_read_image, make_predictions, apply_non_max_suppression, format_boundary_box

flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/yolov4-416', 'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_list('images', './data/images/kite.jpg', 'path to input image')
flags.DEFINE_string('output', './detections/', 'path to output folder')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.50, 'score threshold')
flags.DEFINE_boolean('count', False, 'count objects within images')
flags.DEFINE_boolean('dont_show', False, 'dont show image output')
flags.DEFINE_boolean('info', False, 'print info on detections')
flags.DEFINE_boolean('crop', False, 'crop detections from images')
flags.DEFINE_boolean('ocr', False, 'perform generic OCR on detection regions')
flags.DEFINE_boolean('plate', False, 'perform license plate recognition')


def main(_argv):
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size
    images = FLAGS.images

    # Step 1: Load model
    yolo_v4_model = load_model(FLAGS.weights)

    # loop through images in list and run Yolov4 model on each
    for count, image_path in enumerate(images, 1):
        # Step 2: Read Image
        original_image, image_data, image_name = opencv_read_image(image_path,
                                                                   input_size)

        images_data = []
        for i in range(1):
            images_data.append(image_data)

        images_data = np.asarray(images_data).astype(np.float32)

        # Step 3: Make Detections with the Model
        bbox, boxes, confidence_score = make_predictions(yolo_v4_model,
                                                         images_data)

        # Step 4: run non max suppression on detections
        boxes, scores, classes, valid_detections = apply_non_max_suppression(boxes,
                                                                             confidence_score,
                                                                             FLAGS.iou,
                                                                             FLAGS.score)

        # Step 5: Format bounding boxes, from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, xmax, ymax
        formatted_boundary_boxes = format_boundary_box(original_image, boxes)

        # Step 6: Hold all detection data in one variable
        detection_details = [formatted_boundary_boxes, scores.numpy()[0], classes.numpy()[0],
                             valid_detections.numpy()[0]]

        # Step 7.1: Read in all class names from config
        class_names = utils.read_class_names(cfg.YOLO.CLASSES)
        # Step 7.2: By default allow all classes in .names file
        allowed_classes = list(class_names.values())
        # Step 7.3: custom allowed classes (uncomment line below to allow detections for only people)
        # allowed_classes = ['person']

        # Step 8: Draw Boundary Box
        image = utils.draw_bbox(original_image,
                                detection_details,
                                FLAGS.info,
                                allowed_classes=allowed_classes,
                                read_plate=FLAGS.plate)

        # Step 9: Show Image
        image = Image.fromarray(image.astype(np.uint8))
        if not FLAGS.dont_show:
            image.show()
        image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)

        # Step 10: Save Detections
        cv2.imwrite(FLAGS.output + 'detection' + str(count) + '.png', image)


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
