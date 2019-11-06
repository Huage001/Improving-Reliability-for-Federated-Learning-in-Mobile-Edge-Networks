
# coding=utf-8
from scipy.optimize import minimize
import numpy as np
'''
# demo 2
#计算  (2+x1)/(1+x2) - 3*x1+4*x3 的最小值  x1,x2,x3的范围都在0.1到0.9 之间
def fun(x):
    a,b,c,d=(2,1,3,4)
    v=(a+x[0])/(b+x[1]) -c*x[0]+d*x[2]
    return v
x1min=x2min=x3min=0.1
x1max=x2max=x3max=0.9

def testCon(x):
    return x[0]-x1min

def con():
    # 约束条件 分为eq 和ineq
    #eq表示 函数结果等于0 ； ineq 表示 表达式大于等于0
    #x1min, x1max, x2min, x2max,x3min,x3max = args
    cons = ({'type': 'ineq', 'fun': testCon},\
              {'type': 'ineq', 'fun': lambda x: -x[0] + x1max},\
             {'type': 'ineq', 'fun': lambda x: x[1] - x2min},\
                {'type': 'ineq', 'fun': lambda x: -x[1] + x2max},\
            {'type': 'ineq', 'fun': lambda x: x[2] - x3min},\
             {'type': 'ineq', 'fun': lambda x: -x[2] + x3max})
    return cons

if __name__ == "__main__":
    #定义常量值
    args = (2,1,3,4)  #a,b,c,d
    #设置参数范围/约束条件
    args1 = (0.1,0.9,0.1, 0.9,0.1,0.9)  #x1min, x1max, x2min, x2max
    #设置初始猜测值
    x0 = np.asarray((0.5,0.5,0.5))

    res = minimize(fun, x0, method='SLSQP',constraints=con(),bounds=((0,100),(0,100),(0,100)))
    print(res.fun)
    print(res.success)
    print(res.x)

    print(np.random.rand(20))



# b=np.zeros(3)
# a=np.zeros((2,3))
# for i in range(a.shape[0]):
#     for j in range(a.shape[1]):
#         a[i][j]=1+i*2+j
# b[0]=b[1]=b[2]=3
# a=np.delete(a,-1,axis=1)
# print(a)
import copy
a = np.array([0,1,2,3])
b = np.array([1,0,1,0])

a= a[b.astype(bool)]
print(a)

'''
import matplotlib.pyplot as plt
a=np.array([[9,3,5,7],[0,4,8,10]])
b=[2,4,6,8]
# plt.scatter(a,b)
# plt.show()
a.sort(axis=0)
print(a)
