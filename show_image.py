import cv2
from load_image import load_image


def show_image(image_to_show):
    cv2.imshow('Image Window', image_to_show)
    pressed_key = cv2.waitKey(0)
    if pressed_key == 27:  # Escape Key
        cv2.destroyAllWindows()


if __name__ == '__main__':
    img = load_image()
    show_image(img)
