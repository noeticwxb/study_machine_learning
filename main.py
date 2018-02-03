from numpy import *

a = random.rand(4,4)

aMat = mat(a)

aInvertMat = aMat.I

I_Mat = aInvertMat * aMat

print(a)
print(aMat)
print(aInvertMat)
print(I_Mat)