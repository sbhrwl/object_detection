import cv2


def draw_detections_and_violations(frame, detection_results, violations):
    for (i, (prob, bbox, centroid)) in enumerate(detection_results):
        # Step 1: Extract the bounding box and centroid coordinates
        (startX, startY, endX, endY) = bbox
        (cX, cY) = centroid
        # Step 2: Annotation color green
        color = (0, 255, 0)

        # Step 3: Annotation color for Violations red
        if i in violations:
            color = (0, 0, 255)

        # Step 4: Draw a bounding box around the person
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        # Step 5: Draw the centroid coordinates of the person,
        cv2.circle(frame, (cX, cY), 5, color, 1)

        # Step 6: Write text for showing "Total number of Violations"
        text = "Social Distancing Violations: {}".format(len(violations))
        cv2.putText(frame,
                    text,
                    (10, frame.shape[0] - 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.85,
                    (0, 0, 255),
                    3)

    return frame
