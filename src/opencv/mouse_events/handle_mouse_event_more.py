import cv2
import numpy as np


def click_event(event, x, y, flags, param):
    #     Connect click via a line
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(img, (x, y), 3, (0, 0, 255, 255), -1)
        points.append((x, y))
        if len(points) >= 2:
            # Draw line between last two points
            cv2.line(img, points[-1], points[-2], (255, 0, 0), 5)
        cv2.imshow('image', img)
    # Show B, G and R channel information on where we click Right button
    if event == cv2.EVENT_RBUTTONDOWN:
        blue = img[y, x, 0]
        green = img[y, x, 1]
        red = img[y, x, 2]
        cv2.circle(img, (x, y), 3, (0, 0, 255, 255), -1)
        color_of_my_image = np.zeros((512, 512, 3), np.uint8)
        color_of_my_image[:] = [blue, green, red]
        cv2.imshow('another window', color_of_my_image)


points = []  # used in above callback function

img = cv2.imread('data/images/image.png')
# 'image' is windows title
cv2.imshow('image', img)

# setMouseCallback calls Function click_event
cv2.setMouseCallback('image', click_event)

cv2.waitKey(0)
cv2.destroyAllWindows()
