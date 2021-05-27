import cv2
import os
import sys
from get_parameters import get_parameters


def load_model(configuration, weights):
    net = cv2.dnn.readNetFromDarknet(configuration, weights)
    return net


if __name__ == '__main__':
    config = get_parameters()
    yolo_config = config["yolo_config"]
    model_directory = yolo_config["model_directory"]
    config_file = yolo_config["config_file"]
    weights_file = yolo_config["weights_file"]
    config_path = os.path.sep.join([model_directory, config_file])
    weights_path = os.path.sep.join([model_directory, weights_file])
    model = load_model(config_path, weights_path)
    print(model)
