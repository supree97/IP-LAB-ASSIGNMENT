import cv2
import numpy as np
#Random 2D Array
array_img=np.random.randint(255,size=(300,500),dtype=np.uint8)
cv2.imshow('arrayimage',array_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

