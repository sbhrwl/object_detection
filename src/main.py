from read_image import load_image, show_image
from get_model_labels_and_output_layers import get_model_labels_and_output_layers
from object_detection import get_detection_results

if __name__ == '__main__':
    frame = load_image()
    show_image(frame)
    model, LABELS, layer_numbers = get_model_labels_and_output_layers()
    # print(model)
    # print(LABELS)
    # print(layer_numbers)
    person_index = LABELS.index("person")
    detection_results = get_detection_results(frame,
                                              model,
                                              layer_numbers,
                                              person_index
                                              )
    # print(detection_results)
    # (0.998525857925415, (377, 179, 557, 736), (467, 458))
    # Confidence Score, Bounding Box coordinates (x1, y1, x2, y2), Centroid
