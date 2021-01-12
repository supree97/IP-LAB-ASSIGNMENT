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
![](output/prg1 output.jpg)
---
