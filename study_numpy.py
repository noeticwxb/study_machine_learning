# -*- coding:utf-8 -*-
import numpy as np

a = np.array([1,2,3])
print type(a)
print a.shape
print a[0],a[1],a[2]
a[0]=5
print a

b = np.array([[1,2,3],[4,5,6]])
print b.shape
print type(b.shape) # tuple
print b[0,0],b[0,1],b[1,0]

print np.zeros((2,2))
print np.ones((2,2))
print np.full((2,2),7)
print np.eye(2) #单位矩阵
print np.random.random((2,2))

a = np.array([[1,2],
              [3,4]])
b = np.array([[5,6],
              [7,8]])

print a * b  # 每个元素互相乘
print np.dot(a,b) #矩阵乘法

print np.sum(a) # 10
print np.sum(a,axis=0) # sum each column:[4,6]
print np.sum(a,axis=1) # sum each row:[3,7]

print a.T



