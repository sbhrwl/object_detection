import numpy as np
import cv2

# Check available mouse events available with opencv library
# events = [i for i in dir(cv2) if 'EVENT' in i]
# print(events)


# General Callback function used for handling mouse events
def click_event(event, x, y, flags, param):
    # Show x and y coordinate
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, ', ', y)
        font = cv2.FONT_HERSHEY_SIMPLEX
        strXY = str(x) + ', ' + str(y)
        cv2.putText(img, strXY, (x, y), font, .5, (255, 255, 0), 2)
        cv2.imshow('image', img)
    # Show B, G and R channel
    if event == cv2.EVENT_RBUTTONDOWN:
        blue = img[y, x, 0]
        green = img[y, x, 1]
        red = img[y, x, 2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        strBGR = str(blue) + ', ' + str(green) + ', ' + str(red)
        cv2.putText(img, strBGR, (x, y), font, .5, (0, 255, 255), 2)
        cv2.imshow('image', img)


# Create image from numpy
# img = np.zeros((512, 512, 3), np.uint8)

img = cv2.imread('inputs/messi.jpg')
img = cv2.resize(img, (512, 512))
# 'image' is windows title
cv2.imshow('image', img)

# setMouseCallback calls Function click_event
cv2.setMouseCallback('image', click_event)

cv2.waitKey(0)
cv2.destroyAllWindows()
