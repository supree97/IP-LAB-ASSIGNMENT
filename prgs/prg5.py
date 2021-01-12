import cv2
img=cv2.imread("img.jpg")
cv2.imshow("original",img)
cv2.waitKey(0)
#gray
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow("GRAY image",gray)
cv2.waitKey(0)
#HSV
cv2.imshow('HSV',cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
cv2.waitKey(0)
#lab
cv2.imshow('LAB',cv2.cvtColor(img, cv2.COLOR_BGR2LAB))
cv2.waitKey(0)
cv2.destroyAllWindows()





