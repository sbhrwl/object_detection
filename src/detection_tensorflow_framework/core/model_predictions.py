import cv2
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
import core.utils as utils


def load_model(model_weights):
    return tf.saved_model.load(model_weights, tags=[tag_constants.SERVING])


def opencv_read_image(image_path_to_read, resize_dimensions):
    original_image = cv2.imread(image_path_to_read)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    image_data = cv2.resize(original_image, (resize_dimensions, resize_dimensions))
    image_data = image_data / 255.

    # get image name by using split method
    image_name = image_path_to_read.split('/')[-1]
    image_name = image_name.split('.')[0]
    return original_image, image_data, image_name


def make_predictions(model, data):
    infer = model.signatures['serving_default']
    batch_data = tf.constant(data)
    predicted_boundary_box = infer(batch_data)
    for key, value in predicted_boundary_box.items():
        boxes = value[:, :, 0:4]
        predicted_confidence = value[:, :, 4:]

    return predicted_boundary_box, boxes, predicted_confidence


def apply_non_max_suppression(boxes, confidence_score, iou, score):
    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
        scores=tf.reshape(confidence_score,
                          (tf.shape(confidence_score)[0], -1, tf.shape(confidence_score)[-1])),
        max_output_size_per_class=50,
        max_total_size=50,
        iou_threshold=iou,
        score_threshold=score
    )

    return boxes, scores, classes, valid_detections


def format_boundary_box(image, boxes):
    original_h, original_w, _ = image.shape
    formatted_boundary_boxes = utils.format_boxes(boxes.numpy()[0], original_h, original_w)
    return formatted_boundary_boxes
