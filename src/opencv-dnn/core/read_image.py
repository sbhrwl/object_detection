import cv2
from get_parameters import get_parameters


def load_image():
    config = get_parameters()
    image_location = config["input_config"]["image_location"]
    img = cv2.imread(image_location, -1)
    return img


def show_image(image_to_show):
    cv2.imshow('Image Window', image_to_show)
    pressed_key = cv2.waitKey(0)
    if pressed_key == 27:  # Escape Key
        cv2.destroyAllWindows()


if __name__ == '__main__':
    image = load_image()
    show_image(image)
