# python src/detection_tensorflow_framework/detect_image.py --weights ./artifacts/checkpoints/yolov4-416 --size 416 --model yolov4 --images ./data/images/kite.jpg
import sys
sys.path.append('./src/detection_tensorflow_framework/core')
from read_image import load_model, load_image
from object_detection import get_detection_results

import os
# comment out below line to enable tensorflow outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

import core.utils as utils
from core.functions import *
from PIL import Image
import numpy as np
from absl import app, flags
from absl.flags import FLAGS

flags.DEFINE_string('weights', './checkpoints/yolov4-416', 'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_list('images', './data/images/kite.jpg', 'path to input image')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.50, 'score threshold')

flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('output', './detections/', 'path to output folder')
flags.DEFINE_boolean('dont_show', False, 'dont show image output')
flags.DEFINE_boolean('info', False, 'print info on detections')
flags.DEFINE_boolean('count', False, 'count objects within images')
flags.DEFINE_boolean('crop', False, 'crop detections from images')
flags.DEFINE_boolean('ocr', False, 'perform generic OCR on detection regions')
flags.DEFINE_boolean('plate', False, 'perform license plate recognition')


def main(_argv):
    # config = ConfigProto()
    # config.gpu_options.allow_growth = True
    # session = InteractiveSession(config=config)
    # STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size
    images = FLAGS.images

    # Step 1: Load model
    yolo_v4_model = load_model(FLAGS.weights)

    # loop through images in list and run Yolov4 model on each
    for count, image_path in enumerate(images, 1):
        # Step 2: Read Image
        original_image, image_data, image_name = load_image(image_path, input_size)

        images_data = []
        for i in range(1):
            images_data.append(image_data)

        images_data = np.asarray(images_data).astype(np.float32)

        # Step 3: Get Detection Results
        detection_results = get_detection_results(yolo_v4_model, images_data, FLAGS.iou, FLAGS.score, original_image)

        # Step 4: Classes to show as Detection
        class_names = utils.read_class_names(cfg.YOLO.CLASSES)
        # 4.1: By default allow all classes in .names file
        allowed_classes = list(class_names.values())
        # 4.2: custom allowed classes (uncomment line below to allow detections for only people)
        # allowed_classes = ['person']

        # Step 5: Draw Boundary Box
        image = utils.draw_bbox(original_image,
                                detection_results,
                                FLAGS.info,
                                allowed_classes=allowed_classes,
                                read_plate=FLAGS.plate)

        # Step 6: Show Image
        image = Image.fromarray(image.astype(np.uint8))
        if not FLAGS.dont_show:
            image.show()
        image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)

        # Step 7: Save Detections
        cv2.imwrite(FLAGS.output + 'detection' + str(count) + '.png', image)


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
