# IP-LAB-ASSIGNMENT
**1. Develop a program to display grayscale image using read and write operation.**
```python
import cv2 as cv
import numpy as np
image=cv.imread('tulips.jpg')
image = cv.resize(image, (0, 0), None, .25, .25)
grey = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
grey_3_channel = cv.cvtColor(grey, cv.COLOR_GRAY2BGR)
numpy_horizontal = np.hstack((image, grey_3_channel))
numpy_horizontal_concat = np.concatenate((image, grey_3_channel), axis=1)
cv.imshow('flowers', numpy_horizontal_concat)
cv.waitKey()
```
***output:***
![](OUTPUT/prg1op.jpg)

---
**2.Develop a program to perform linear transformations on an image: Scaling and Rotation**
```python

#scaling
import cv2 as cv
img=cv.imread("island.jpg")
cv.imshow('image',img)
res=cv.resize(img,(0,0),fx=0.50,fy=0.50)
cv.imshow("Result",res)
cv.waitKey(0)
cv.destroyAllWindows(0)
```
***output:***

![](OUTPUT/prg2scalingop.jpg)

---
```python
#rotation
import cv2 
import numpy as np 
FILE_NAME = 'flower.jpg'
img = cv2.imread(FILE_NAME) 
(rows, cols) = img.shape[:2] 
M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 45, 1) 
res = cv2.warpAffine(img, M, (cols, rows)) 
cv2.imshow('result.jpg', res) 
cv2.waitKey(0)
```
***output:***

![](OUTPUT/prg2rotationop.jpg)


---
**3. Develop a program to find the sum and mean of a set of images. 
     Create ‘n’ number of images and read them from the directory and perform the operations.**

```python

#sum & Mean
import os
path=r'D:\new'
imgs=[]
files=os.listdir(path) #List
for file in files:
    fpath=path+'\\'+file
    imgs.append(cv2.imread(fpath))
    
for i,im in enumerate(imgs):
    cv2.imshow(files[i],imgs[i])    
    cv2.imshow('Mean of '+files[i],len(im)/im)
print('sum of imgs(Total no) = ',i+1)    
cv2.waitKey(0)
cv2.destroyAllWindows()

```
```
sum of imgs(Total no) =  2
```
***output:***

![](OUTPUT/prg3op1.jpg)


![](OUTPUT/prg3op2.jpg)

---
**4. Develop a program to convert the color image to gray scale and binary image**
```python
#gray image
import cv2 as cv
img = cv.imread("tea.jpg")
gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow('Original',img)
cv.imshow("Gray Image",gray)
cv.waitKey(0)
cv.destroyAllWindows()
```
***output:***

![](OUTPUT/prg4op1.jpg)

---


```python
#binary image
import cv2 as cv
img = cv.imread("tea.jpg")
ret, b_img = cv.threshold(img,127,255,cv.THRESH_BINARY)
cv2.imshow('Original',img)
cv.imshow("Binary Image",b_img)
cv.waitKey(0)
cv.destroyAllWindows()

```
***output:***

![](OUTPUT/prg4op2.jpg)

---
**5.	Develop a program to convert the given color image to different color spaces.**
```python
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

```
***output:***

![](OUTPUT/prg5op1.jpg)

![](OUTPUT/prg5op2.jpg)

![](OUTPUT/prg5op3.jpg)

![](OUTPUT/prg5op4.jpg)

---

**6.	Develop a program to create an image from 2D array (generate an array of random size).**
```python
import cv2
import numpy as np
#Random 2D Array
array_img=np.random.randint(255,size=(300,500),dtype=np.uint8)
cv2.imshow('arrayimage',array_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

```
***output:***

![](OUTPUT/prg6op.jpg)

---

