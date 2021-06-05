

def get_detection_details(predicted_boundary_box):
    for key, value in predicted_boundary_box.items():
        predicted_confidence = value[:, :, 4:]
        boxes = value[:, :, 0:4]

    return boxes, predicted_confidence
