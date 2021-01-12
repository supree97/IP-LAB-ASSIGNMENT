import cv2 as cv
img=cv.imread("island.jpg")
cv.imshow('image',img)
res=cv.resize(img,(0,0),fx=0.50,fy=0.50)
cv.imshow("Result",res)
cv.waitKey(0)
cv.destroyAllWindows(0)






