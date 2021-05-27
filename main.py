import cv2

if __name__ == '__main__':
    print(cv2.__version__)
    img = cv2.imread('inputs/image.jpg', -1)
    cv2.imshow('image', img)
    pressed_key = cv2.waitKey(0)
    if pressed_key == 27:  # Escape Key
        cv2.destroyAllWindows()
