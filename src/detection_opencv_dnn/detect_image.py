from src.detection_opencv_dnn.core.read_image import load_image
from src.detection_opencv_dnn.core.get_model_labels_and_output_layers import get_model_labels_and_output_layers
from src.detection_opencv_dnn.core.object_detection import get_detection_results
from src.detection_opencv_dnn.core.detect_violations import detect_violations
from src.detection_opencv_dnn.core.draw_detections_and_violations import draw_detections_and_violations
from src.detection_opencv_dnn.core.write_and_save_frame import write_and_save_frame


if __name__ == '__main__':
    # Step 1: Load model
    model, LABELS, layer_numbers = get_model_labels_and_output_layers()
    # print(model)
    # print(LABELS)
    # print(layer_numbers)
    person_index = LABELS.index("person")

    # Step 2: Read Image
    frame = load_image()
    # show_image(frame)

    # Step 3: Get Detection Results
    detection_results = get_detection_results(frame,
                                              model,
                                              layer_numbers,
                                              person_index
                                              )
    # print(detection_results)
    # [(0.998525857925415, (377, 179, 557, 736), (467, 458))]
    # Confidence Score, Bounding Box coordinates (x1, y1, x2, y2), Centroid

    # Step 4: Detect Violations
    violations = detect_violations(detection_results)
    # print(violations)

    # Step 5: Draw Boundary Box
    output_frame = draw_detections_and_violations(frame, detection_results, violations)

    # Step 6: Show and Save Detections
    write_and_save_frame(output_frame)
