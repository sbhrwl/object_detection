import cv2


def write_and_save_video(frame):
    # Step 1: Draw the Output Frame for Video
    cv2.imshow("Frame", frame)
    pressed_key = cv2.waitKey(1) & 0xFF
    if pressed_key == 27:  # Escape Key
        cv2.destroyAllWindows()

    # Step 2: Save Output Video
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter('artifacts/detections/video/output.avi',
                             fourcc,
                             25,
                             (frame.shape[1], frame.shape[0]),
                             True)
    writer.write(frame)
