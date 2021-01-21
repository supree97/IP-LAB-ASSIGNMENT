#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np

M = [[1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]] 

M = np.asarray(M)
N = np.zeros(M.shape)

def sumNeighbors(M,x,y):
    l = []
    for i in range(max(0,x-1),x+2): 
        for j in range(max(0,y-1),y+2):
            try:
                t = M[i][j]
                l.append(t)
            except IndexError: 
                pass
    return sum(l)-M[x][y] 

for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        N[i][j] = sumNeighbors(M, i, j)

print ("Original matrix:\n", M)
print ("Summed neighbors matrix:\n", N)


# In[ ]:




