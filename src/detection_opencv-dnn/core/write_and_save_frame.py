import cv2


def write_and_save_frame(frame):
    # Step 1: Draw the Output Frame
    cv2.imshow("Frame", frame)
    pressed_key = cv2.waitKey(0)
    if pressed_key == 27:  # Escape Key
        cv2.destroyAllWindows()

    # Step 2: Save Output Frame
    cv2.imwrite('artifacts/detections/image/output_frame.jpg', frame)
