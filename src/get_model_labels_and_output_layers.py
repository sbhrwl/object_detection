import cv2
import os
from get_parameters import get_parameters


def load_model(configuration, weights):
    net = cv2.dnn.readNetFromDarknet(configuration, weights)
    return net


def load_labels(labels_file_path):
    labels = open(labels_file_path).read().strip().split("\n")
    return labels


def get_output_layers(model_name):
    output_layer_numbers = model_name.getLayerNames()
    output_layer_numbers = [output_layer_numbers[i[0] - 1] for i in model_name.getUnconnectedOutLayers()]
    return output_layer_numbers


def get_model_labels_and_output_layers():
    config = get_parameters()
    yolo_config = config["yolo_config"]
    model_directory = yolo_config["model_directory"]
    config_file = yolo_config["config_file"]
    weights_file = yolo_config["weights_file"]
    labels_file = yolo_config["labels_file"]

    config_path = os.path.sep.join([model_directory, config_file])
    weights_path = os.path.sep.join([model_directory, weights_file])

    model = load_model(config_path, weights_path)

    labels_path = os.path.sep.join([model_directory, labels_file])
    LABELS = load_labels(labels_path)

    ln = get_output_layers(model)

    return model, LABELS, ln


if __name__ == '__main__':
    get_model_labels_and_output_layers()

