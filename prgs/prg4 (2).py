#gray image
import cv2 as cv
img = cv.imread("tea.jpg")
gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow('Original',img)
cv.imshow("Gray Image",gray)
cv.waitKey(0)
cv.destroyAllWindows()

#binary image
import cv2 as cv
img = cv.imread("tea.jpg")
ret, b_img = cv.threshold(img,127,255,cv.THRESH_BINARY)
cv2.imshow('Original',img)
cv.imshow("Binary Image",b_img)
cv.waitKey(0)
cv.destroyAllWindows()





