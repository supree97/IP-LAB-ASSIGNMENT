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









