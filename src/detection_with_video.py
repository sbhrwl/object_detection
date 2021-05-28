import cv2
import imutils
from get_model_labels_and_output_layers import get_model_labels_and_output_layers
from object_detection import get_detection_results
from detect_violations import detect_violations
from draw_detections_and_violations import draw_detections_and_violations
from write_and_save_video import write_and_save_video


if __name__ == '__main__':
    model, LABELS, layer_numbers = get_model_labels_and_output_layers()
    # print(model)
    # print(LABELS)
    # print(layer_numbers)
    person_index = LABELS.index("person")

    # video_stream = cv2.VideoCapture(-1)
    video_stream = cv2.VideoCapture("inputs/pedestrians.mp4")
    while True:
        # Read the next frame from the file
        (grabbed, frame) = video_stream.read()
        # End of the stream: when frame is not grabbed then we have reached the end of stream
        if not grabbed:
            break

        frame = imutils.resize(frame, width=700)
        detection_results = get_detection_results(frame,
                                                  model,
                                                  layer_numbers,
                                                  person_index
                                                  )
        # print(detection_results)
        # (0.998525857925415, (377, 179, 557, 736), (467, 458))
        # Confidence Score, Bounding Box coordinates (x1, y1, x2, y2), Centroid
        violations = detect_violations(detection_results)
        # print(violations)
        output_frame = draw_detections_and_violations(frame, detection_results, violations)
        write_and_save_video(output_frame)
