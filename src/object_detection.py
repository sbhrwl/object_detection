import numpy as np
import cv2
from get_parameters import get_parameters


def detect_persons(frame, net, ln, person_idx=0):
    (H, W) = frame.shape[:2]
    # results = []
    blob = cv2.dnn.blobFromImage(frame,
                                 1 / 255.0,
                                 (416, 416),
                                 swapRB=True,
                                 crop=False)
    print(blob)


if __name__ == '__main__':
    config = get_parameters()
    configuration_variables = config["configuration_variables"]
    min_confidence_score = configuration_variables["min_confidence_score"]
    nms_threshold = configuration_variables["nms_threshold"]

    # detect_persons(input_frame, model, output_layer_numbers, person_idx=0)

