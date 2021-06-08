import cv2
import numpy as np

while True:
    cv2.imshow('t', np.zeros((1, 1, 3)))
    print(cv2.waitKey(0))
