import numpy as np
import pandas as pd

'''
@:params 
A：状态转移概率矩阵 numpy
B：观测概率矩阵 numpy
pai:初始状态概率向量 numpy
@:return
P(O|lamda) double
'''
def forward(A,B,pai,O):
    alpha1 = np.multiply(pai,np.transpose(B[:,O[0]]))
    alpha2 = np.multiply(np.dot(alpha1,A),B[:,O[1]])
    alpha3 = np.multiply(np.dot(alpha2,A),B[:,O[2]])
    return np.sum(alpha3)

A =np.array([[0.5,0.2,0.3],
    [0.3,0.5,0.2],
    [0.2,0.3,0.5]])
B = np.array([[0.5,0.5],
     [0.4,0.6],
     [0.7,0.3]])
pai = np.array([0.2,0.4,0.4]).transpose()
O = np.array([0,1,0])
print()
print(forward(A,B,pai,O))