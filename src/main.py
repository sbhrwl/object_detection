from read_image import load_image, show_image
from get_model_labels_and_output_layers import get_model_labels_and_output_layers
from object_detection import detect_object, apply_non_maxima_suppression


if __name__ == '__main__':
    frame = load_image()
    show_image(frame)
    model, LABELS, layer_numbers = get_model_labels_and_output_layers()
    # print(model)
    # print(LABELS)
    # print(layer_numbers)
    person_index = LABELS.index("person")
    detected_indexes = detect_object(frame,
                                     model,
                                     layer_numbers,
                                     person_index
                                     )
    print(detected_indexes)
