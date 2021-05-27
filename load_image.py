import cv2


def load_image():
    img = cv2.imread('inputs/image.jpg', -1)
    return img


if __name__ == '__main__':
    load_image()
