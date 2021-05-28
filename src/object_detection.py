import cv2
from get_parameters import get_parameters
from detect_object_in_frame import get_detection_details


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


def apply_non_maxima_suppression(boxes, confidences, minimum_confidence, nms_threshold):
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, minimum_confidence, nms_threshold)
    return indexes


def detect_object(frame, net, ln, object_index):
    config = get_parameters()
    each_layer_output_file = config["artifacts"]["each_layer_output"]
    configuration_variables = config["configuration_variables"]
    minimum_confidence_score = configuration_variables["minimum_confidence_score"]
    nms_threshold_value = configuration_variables["nms_threshold_value"]

    # results = []
    blob = convert_image_to_blob(frame)
    each_layer_output = perform_forward_pass(net, blob, ln)
    write_list_to_file(each_layer_output, each_layer_output_file)
    boxes, centroids, confidences = get_detection_details(frame,
                                                          each_layer_output,
                                                          minimum_confidence_score,
                                                          object_index)
    # print(boxes)
    # print(centroids)
    # print(confidences)
    indexes = apply_non_maxima_suppression(boxes, confidences, minimum_confidence_score, nms_threshold_value)
    return indexes


