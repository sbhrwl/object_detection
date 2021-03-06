from src.detection_tensorflow_framework.core.config import cfg
from src.detection_tensorflow_framework.core.read_image import load_model, load_image
from src.detection_tensorflow_framework.core.object_detection import get_detection_results
import core.utils as utils
from core.functions import *
import numpy as np
from PIL import Image

# This might be doing something, don't know yet
import tensorflow as tf
import os

# comment out below line to enable tensorflow outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession


from absl import app, flags
from absl.flags import FLAGS

flags.DEFINE_string('weights', './artifacts/checkpoints/yolov4-416', 'path to weights file')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_list('images', './data/images/social_distance.jpg', 'path to input image')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.50, 'score threshold')
flags.DEFINE_string('output', './artifacts/detections/', 'path to output folder')
flags.DEFINE_integer('size', 416, 'resize images to')

flags.DEFINE_boolean('info', False, 'print info on detections')
flags.DEFINE_boolean('count', False, 'count objects within images')
flags.DEFINE_boolean('crop', False, 'crop detections from images')
flags.DEFINE_boolean('ocr', False, 'perform generic OCR on detection regions')
flags.DEFINE_boolean('plate', False, 'perform license plate recognition')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_boolean('dont_show', False, 'dont show image output')


def main(_argv):
    # config = ConfigProto()
    # config.gpu_options.allow_growth = True
    # session = InteractiveSession(config=config)
    # STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)

    # i. Define flags with lower case "flags"
    # ii. Access defined flags with upper case "FLAGS"
    input_size = FLAGS.size
    images = FLAGS.images

    # Step 1: Load model
    yolo_v4_model = load_model(FLAGS.weights)

    # loop through images in list and run Yolo V4 model on each
    for count, image_path in enumerate(images, 1):
        # Step 2: Read Image
        original_image, image_data, image_name = load_image(image_path, input_size)

        images_data = []
        for i in range(1):
            images_data.append(image_data)

        images_data = np.asarray(images_data).astype(np.float32)

        # Step 3: Get Detection Results
        detection_results = get_detection_results(yolo_v4_model,
                                                  images_data,
                                                  FLAGS.iou, FLAGS.score,
                                                  original_image)

        # Step 4: Classes to show as Detection
        class_names = utils.read_class_names(cfg.YOLO.CLASSES)
        # 4.1: By default allow all classes in .names file
        allowed_classes = list(class_names.values())
        # 4.2: custom allowed classes (uncomment line below to allow detections for only people)
        # allowed_classes = ['person']

        # Additional Operations
        # CROP Images: Store Cropped images
        if FLAGS.crop:
            crop_path = os.path.join(os.getcwd(), 'artifacts', 'detections', 'crop', image_name)
            try:
                os.mkdir(crop_path)
            except FileExistsError:
                pass
            crop_objects(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB), detection_results, crop_path, allowed_classes)

        # OCR: Perform general text extraction using Tesseract OCR on object detection bounding box
        if FLAGS.ocr:
            ocr(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB), detection_results)

        # Count: perform counting of objects
        if FLAGS.count:
            # count objects found
            counted_classes = count_objects(detection_results, by_class=True, allowed_classes=allowed_classes)
            # loop through dict and print
            for key, value in counted_classes.items():
                print("Number of {}s: {}".format(key, value))
            # Step 5: Draw Boundary Box
            image = utils.draw_bbox(original_image,
                                    detection_results,
                                    FLAGS.info,
                                    counted_classes,
                                    allowed_classes=allowed_classes,
                                    read_plate=FLAGS.plate)
        # Step 5: Draw Boundary Box
        else:
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
