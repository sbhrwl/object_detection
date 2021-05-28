import numpy as np
import cv2
from get_parameters import get_parameters


def convert_image_to_blob(image):
    blob_image = cv2.dnn.blobFromImage(image,
                                       1 / 255.0,
                                       (416, 416),
                                       swapRB=True,
                                       crop=False)
    return blob_image


def perform_forward_pass(model, input_blob, layers):
    model.setInput(input_blob)
    forward_pass_output = model.forward(layers)
    return forward_pass_output


def write_list_to_file(input_list, file_location):
    with open(file_location, "w") as outfile:
        outfile.write("\n".join(str(item) for item in input_list))


def detect_object(frame, net, ln, object_index):
    config = get_parameters()
    each_layer_output_file = config["artifacts"]["each_layer_output"]
    # configuration_variables = config["configuration_variables"]
    # min_confidence_score = configuration_variables["min_confidence_score"]
    # nms_threshold = configuration_variables["nms_threshold"]

    # results = []
    blob = convert_image_to_blob(frame)
    each_layer_output = perform_forward_pass(net, blob, ln)
    write_list_to_file(each_layer_output, each_layer_output_file)
    print(object_index)


if __name__ == '__main__':
    print("Hi")
