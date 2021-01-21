#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# organizing imports  
import cv2  
import numpy as np  
    
# path to input images are specified and   
# images are loaded with imread command  
image1 = cv2.imread('a.jpg')  
image2 = cv2.imread('b.jpg') 
cv2.imshow("original1",image1)
cv2.imshow("original2",image2)   
  
# cv2.addWeighted is applied over the 
# image inputs with applied parameters 
weightedSum = cv2.addWeighted(image1, 0.5, image2, 0.4, 0) 
  
# the window showing output image 
# with the weighted sum  
cv2.imshow('Weighted Image', weightedSum)#addition operation

# cv2.subtract is applied over the 
# image inputs with applied parameters 
sub = cv2.subtract(image1, image2) 

# the window showing output image 
# with the subtracted image 
cv2.imshow('Subtracted Image', sub) 
# De-allocate any associated memory usage   
if cv2.waitKey(0) & 0xff == 27:  
    cv2.destroyAllWindows()  


# In[ ]:




