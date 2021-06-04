import cv2
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants


def load_model(model_weights):
    return tf.saved_model.load(model_weights, tags=[tag_constants.SERVING])


def load_image(image_path_to_read, resize_dimensions):
    original_image = cv2.imread(image_path_to_read)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    image_data = cv2.resize(original_image, (resize_dimensions, resize_dimensions))
    image_data = image_data / 255.

    # get image name by using split method
    image_name = image_path_to_read.split('/')[-1]
    image_name = image_name.split('.')[0]
    return original_image, image_data, image_name
