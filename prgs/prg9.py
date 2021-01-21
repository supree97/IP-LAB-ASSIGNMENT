import cv2 
import numpy as np
image=cv2.imread("flower.jpg")
cv2.imshow("original",image)
cv2.waitKey(0)
img_neg=225-image
cv2.imshow("negativeimage",img_neg)
cv2.waitKey(0)





