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


def detect_object(frame, net, ln, object_index, minimum_confidence_score, each_layer_output_file):
    blob = convert_image_to_blob(frame)
    each_layer_output = perform_forward_pass(net, blob, ln)
    write_list_to_file(each_layer_output, each_layer_output_file)
    confidences, boxes, centroids = get_detection_details(frame,
                                                          each_layer_output,
                                                          minimum_confidence_score,
                                                          object_index)
    # print("detect_object", confidences)
    # print("detect_object", boxes)
    # print("detect_object", centroids)
    return confidences, boxes, centroids


def get_detection_results(frame, net, ln, object_index):
    config = get_parameters()
    each_layer_output_file = config["artifacts"]["each_layer_output"]
    configuration_variables = config["configuration_variables"]
    minimum_confidence_score = configuration_variables["minimum_confidence_score"]
    nms_threshold_value = configuration_variables["nms_threshold_value"]

    confidences, boxes, centroids = detect_object(frame,
                                                  net,
                                                  ln,
                                                  object_index,
                                                  minimum_confidence_score,
                                                  each_layer_output_file)

    indexes = apply_non_maxima_suppression(boxes, confidences, minimum_confidence_score, nms_threshold_value)

    # Format Results
    # i. Confidence score/Predicted Probability
    # ii. Bounding box coordinates
    # iii. Centroid
    results = []
    if len(indexes) > 0:
        for i in indexes.flatten():
            # Step 1: Extract bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            # Step 2: Rewrite Bounding box coordinates
            # a. Add Width to X coordinate
            # b. Add Height to Y coordinate
            r = (confidences[i], (x, y, x + w, y + h), centroids[i])
            results.append(r)

    return results


