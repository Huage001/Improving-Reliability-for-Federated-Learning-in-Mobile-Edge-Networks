from DeviceManager import DeviceManager
from EdgeManager import EdgeManager
import numpy as np
from scipy.optimize import minimize
import copy

#----------Constant----------#

# Number of decision time
totalTime=1

# Learning Model Size
modelSize=1

# Number of device
deviceNum=10

# Number of Edge
edgeNum=10

# Ratio of Selected Data
ratio=0.9

# Available error range
epsilon=1e-6

#-------Decision Variable--------#

# Whether Device j use Edge i for Federal Learning
choice=np.zeros([deviceNum,edgeNum+1])

#---------Other Variable---------#

# All Devices, the corresponding variables are in its class
devices=DeviceManager(deviceNum)

# All Edges, the corresponding variables are in its class
edges=EdgeManager(edgeNum)
# Note that we have a virtual edge
edgeNum=edgeNum+1

#---------Constrain Function--------#

# One device can choose one Edge at most
def choiceConstrain(x):
    # Transform vector x into a matrix with the same shape as the decision variable
    y=copy.deepcopy(x)
    indicator=y.reshape((deviceNum,edgeNum))
    # Note that the operator is >=, and that we should calculate sum for each row, so axis=1
    return epsilon-np.abs(np.sum(indicator,axis=1)-1)

# The number of Devices who choose the same Edge cannot be larger than its User Capacity
def bandWidthConstrain(x):
    # The formation is similar to the function above
    y=copy.deepcopy(x)
    indicator=y.reshape((deviceNum,edgeNum))
    indicator[:,edgeNum-1]=np.zeros(deviceNum)
    userCapacity=edges.getAllUserCapacity()
    return userCapacity[0:edgeNum]-np.sum(indicator,axis=0)

def dataAmountConstrain(x):
    # The formation is similar to the function above
    y=copy.deepcopy(x)
    indicator=y.reshape((deviceNum,edgeNum))
    # Note that we should not include the virtual edge
    indicator[:,edgeNum-1]=np.zeros(deviceNum)
    totalDataSize=devices.getTotalDataSize()
    dataSize=devices.getDataSize()
    return np.sum(np.sum(indicator*dataSize.reshape((deviceNum,1))))-ratio*totalDataSize

def onlineConstrain(x):
    # The formation is similar to the function above
    y=copy.deepcopy(x)
    indicator=y.reshape((deviceNum,edgeNum))
    # Note that we should not include the virtual edge
    indicator[:,edgeNum-1]=np.zeros(deviceNum)
    isAlive=devices.getIsAlive()
    return (isAlive.reshape((deviceNum,1))-indicator).reshape(deviceNum*edgeNum)+epsilon

# Total Constrain Function
def constrainFunction():
    cons = ({'type':'ineq','fun':choiceConstrain},
            {'type':'ineq','fun':bandWidthConstrain},
            {'type':'ineq','fun':dataAmountConstrain},
            {'type':'ineq','fun':onlineConstrain})
    return cons

# bound
bound=[]
for i in range(deviceNum*edgeNum):
    bound.append([0,1])

#---------------Objective Function----------------#

def objectiveFunction(x):
    y=copy.deepcopy(x)
    indicator=y.reshape((deviceNum,edgeNum))
    # Note that we should not include the virtual edge
    indicator[:,edgeNum-1]=np.zeros(deviceNum)
    # a vector with dimension edgeNum
    transmitDelay=edges.getTransmitDelay()
    # a vector with dimension deviceNum
    computeDelay=devices.getComputeDelay()
    # a vector with dimension deviceNum
    dataSize=devices.getDataSize()
    # We put the total transmit time of all edges into allLoad, which is
    # the sum of each column of our decision variable multiplies corresponding
    # transmitDelay. Therefore, we get a vector with dimension edgeNum
    allLoad=(np.sum(indicator,axis=0))*transmitDelay*modelSize
    # result[j][i] means the total time, including computing and transmitting,
    # on the condition that the device j choose edge i
    result=np.zeros((deviceNum,edgeNum))
    for j in range(deviceNum):
        for i in range(edgeNum):
            result[j][i]=(allLoad[i]+computeDelay[j]*dataSize[j])*indicator[j][i]
    # calculating the sum means we want the overall needed time of the device j
    result=np.sum(result,axis=1)
    # At last, the max time of all device means the final time
    return np.max(result)

#-----------------Main Process------------------#

for t in range(totalTime):
    deviceNum=devices.setIsAliveDistribution()
    devices.setDataSizeDistribution()
    choice=np.zeros([deviceNum,edgeNum])
    choice[:,edgeNum-1]=np.ones(deviceNum)
    res=minimize(objectiveFunction, choice.reshape((deviceNum*edgeNum,1)),constraints=constrainFunction(),bounds=bound,options={'maxiter':1000,'disp':True})
    choice=res.x.reshape((deviceNum,edgeNum))
    print(res.fun)
    print(res.success)
    print(choice)
    print(res.message)
