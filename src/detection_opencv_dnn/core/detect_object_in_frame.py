import numpy as np


def get_detection_details(frame, each_layer_output, minimum_confidence_score, label_index):
    confidences = []
    boxes = []
    centroids = []
    (H, W) = frame.shape[:2]
    print(np.shape(each_layer_output))
    for output in each_layer_output:
        print(np.shape(output))
        # loop over detections
        for detection in output:
            scores = detection[5:]
            # Step 1: Extract the class ID
            classID = np.argmax(scores)
            # Step 2: Extract confidence i.e. probability of the current object detection
            confidence = scores[classID]
            # Step 3: Filter detections by
            # (1) ensuring that the object detected was a person and
            # (2) that the minimum  confidence is met
            if classID == label_index and confidence > minimum_confidence_score:

                # Step 4: Scale the bounding box coordinates back relative to the size of the image
                # Note: YOLO returns
                # i. The center coordinates (x, y) of the bounding box followed by
                # ii. The Width and Height of the boxes
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # Step 5: Derive the top and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # Step 6: Update our list of bounding box coordinates, centroids, and confidences
                confidences.append(float(confidence))
                boxes.append([x, y, int(width), int(height)])
                centroids.append((centerX, centerY))

    # print("get_detection_details", confidences)
    # print("get_detection_details", boxes)
    # print("get_detection_details", centroids)
    return confidences, boxes, centroids
