import numpy as np
import cv2 as cv


def nothing(x):
    print(x)


# Create a black image, a window
img = np.zeros((300, 512, 3), np.uint8)
cv.namedWindow('image_window')

cv.createTrackbar('B', 'image_window', 0, 255, nothing)
cv.createTrackbar('G', 'image_window', 0, 255, nothing)
cv.createTrackbar('R', 'image_window', 0, 255, nothing)

# Control trackbar with if switch is ON
switch = '0 : OFF\n 1 : ON'
cv.createTrackbar(switch, 'image_window', 0, 1, nothing)

while 1:
    cv.imshow('image', img)
    k = cv.waitKey(1) & 0xFF
    # Press Escape Key to break
    if k == 27:
        break

    b = cv.getTrackbarPos('B', 'image_window')
    g = cv.getTrackbarPos('G', 'image_window')
    r = cv.getTrackbarPos('R', 'image_window')
    s = cv.getTrackbarPos(switch, 'image_window')

    if s == 0:
        img[:] = 0
    else:
        img[:] = [b, g, r]

cv.destroyAllWindows()
