import cv2


# cv2.resize - resize the image
# merged_img = cv2.add(img, img2) - Calculates the per-element sum of two arrays or an array and a scalar.
# merged_img = cv2.addWeighted(img, .2, img2, .8, 0) - Calculates the weighted sum of two arrays.


img = cv2.imread('data/images/image.png')
img2 = cv2.imread('data/images/messi.jpg')

img = cv2.resize(img, (512, 512))
img2 = cv2.resize(img2, (512, 512))

merged_img = cv2.add(img, img2)
merged_img = cv2.addWeighted(img, 0.6, img2, 0.4, 0)

cv2.imshow('image', merged_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
