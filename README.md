# YoloV4_Detect_Social_Distance_Violations

## Setup 

Create env 
```bash
conda create -n social_distance_violation python=3.7 -y
```

Activate env
```bash
conda activate social_distance_violation
```

Install the req
```bash
pip install -r requirements.txt
```

# Detection from the Image

```
detection_with_image.py
```

i. Load model and related parameters: get_model_labels_and_output_layers.py

ii. Object_detection.py

a. get_detection_results.py

b. detect_object

```
    convert_image_to_blob(frame)

    confidences, boxes, centroids = get_detection_details(frame,
                                                          each_layer_output,
                                                          minimum_confidence_score,
                                                          object_index)
                                                          
    indexes = apply_non_maxima_suppression(boxes, confidences, minimum_confidence_score, nms_threshold_value)
```

iii. detect_violations.py

```
    if len(results) >= 2:
        # Step 1: Distance Calculation based on Centroids
        # i. Create an Array of Centroids
        centroids = np.array([r[2] for r in results])

        # ii. Compute the Euclidean distances between "All pairs" of the centroids
        distance_between_observations = dist.cdist(centroids, centroids, metric="euclidean")

        # Step 2: Compare calculated distance with the Minimum distance to be flagged as Violation
        for i in range(0, distance_between_observations.shape[0]):
            for j in range(i + 1, distance_between_observations.shape[1]):

                # Step 3: Is Distance between any two centroid pairs is less than the configured "Number of pixels"
                if distance_between_observations[i, j] < minimum_distance:

                    # Step 4: Mark the observations/persons as Violations
                    set_of_violations.add(i)
                    set_of_violations.add(j)

    return set_of_violations
    
 ```
 iv. draw_detections_and_violations.py
 
 # Detection from the Video

```
detection_with_video.py
```

Same as above process, except that we process for each frame

```
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
```
