import numpy as np

'''
a = random.rand(4,4)
aMat = mat(a)
aInvertMat = aMat.I
I_Mat = aInvertMat * aMat
print(a)
print(aMat)
print(aInvertMat)
print(I_Mat)
'''

v = np.mat([1,2,3])
print(v)
v1 = np.mat([1,2,3]).transpose()
print(v1)
print(np.shape(v1))
v2 = np.mat(np.ones((3,1)))
print(np.shape(v2))
v3 = np.multiply(v1,v2)
print(v3)
