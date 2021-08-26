import cv2

if __name__ == '__main__':
    print(cv2.__version__)
    img = cv2.imread('data/images/image.png', 0)
    print(img)
    cv2.imshow('image', img)
    pressed_key = cv2.waitKey(0)
    if pressed_key == 27:  # Escape Key
        cv2.destroyAllWindows()
    elif pressed_key == ord('s'):
        cv2.imwrite('artifacts/detections/image_copy.png', img)
        cv2.destroyAllWindows()
