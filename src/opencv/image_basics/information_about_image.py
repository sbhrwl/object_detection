import cv2


# img.shape - returns a tuple of number of rows, columns, and channels
# img.size - returns Total number of pixels
# img.dtype - returns Image datatype
# cv2.split(img) - output vector of arrays; the arrays themselves are reallocated, if needed.
# cv2.merge((b,g,r)) - The number of channels will be the total number of channels in the matrix array.

img = cv2.imread('data/images/messi.jpg')
print("Shape of Image: ", img.shape)
print("Size of Image: ", img.size)
print("dtype of Image: ", img.dtype)
img = cv2.resize(img, (512, 512))

b, g, r = cv2.split(img)
img = cv2.merge((b, g, r))

# Region Of Interest: Based on coordinates of ball
# The co-ordinates used in the array follow the format of
# img [y1: y2, x1: x2]
# Therefore, when copying to another part of the image,
# you need to ensure that (y2-y1) value remains the same, as well as (x2-x1)
# Here's another example to copy messi head,
# where Top left coordinates is (200, 60) and bottom right is (270, 140) in x,y format
# messi_head = img[60:140, 200:270]
# img[260:340,100:170] = messi_head

ball = img[420:482, 340:405]
img[411:473, 103:168] = ball

cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
