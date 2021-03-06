import tensorflow as tf
import numpy as np
from src.detection_tensorflow_framework.core.detect_object_in_frame import get_detection_details


def detect_object(model, data):
    infer = model.signatures['serving_default']
    batch_data = tf.constant(data)
    predicted_boundary_box = infer(batch_data)
    boxes, predicted_confidence = get_detection_details(predicted_boundary_box)
    return boxes, predicted_confidence


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


def format_boxes(boxes, image_height, image_width):
    for box in boxes:
        y_min = int(box[0] * image_height)
        x_min = int(box[1] * image_width)
        y_max = int(box[2] * image_height)
        x_max = int(box[3] * image_width)
        box[0], box[1], box[2], box[3] = x_min, y_min, x_max, y_max
    return boxes


def format_boundary_box(image, boxes):
    original_h, original_w, _ = image.shape
    formatted_boundary_boxes = format_boxes(boxes.numpy()[0], original_h, original_w)
    return formatted_boundary_boxes


def get_detection_results(yolo_v4_model, images_data, iou, score, original_image):
    boxes, confidence_score = detect_object(yolo_v4_model, images_data)

    boxes, scores, classes, valid_detections = apply_non_max_suppression(boxes,
                                                                         confidence_score,
                                                                         iou,
                                                                         score)

    # Format Results
    formatted_boundary_boxes = format_boundary_box(original_image, boxes)
    # print("formatted_boundary_boxes:-", formatted_boundary_boxes)
    # print("formatted_boundary_boxes as array:-", [formatted_boundary_boxes])

    detection_details = [formatted_boundary_boxes,
                         scores.numpy()[0],
                         classes.numpy()[0],
                         valid_detections.numpy()[0]]
    # print("detection_details", detection_details)
    return detection_details
