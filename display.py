import cv2
import sys

filename = sys.argv[1]

img = cv2.imread(filename)
cv2.imshow(filename, img)
cv2.waitKey(0)

