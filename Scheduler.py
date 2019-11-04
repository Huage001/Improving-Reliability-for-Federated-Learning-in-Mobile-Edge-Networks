from DeviceManager import DeviceManager
from EdgeManager import EdgeManager
import numpy as np
from scipy.optimize import minimize
import copy

#----------Constant----------#

# Number of decision time
totalTime=2

# Learning Model Size
modelSize=1

# Number of device
allDeviceNum=10

# Number of Edge
edgeNum=10

# Ratio of Selected Data
ratio=1

#---------Other Variable---------#

# All Devices, the corresponding variables are in its class
devices=DeviceManager(allDeviceNum)

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
    return np.sum(indicator,axis=1)-1

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

# Total Constrain Function
def constrainFunction():
    cons = ({'type':'eq','fun':choiceConstrain},
            {'type':'ineq','fun':bandWidthConstrain},
            {'type':'ineq','fun':dataAmountConstrain})
    return cons

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

#----------------Rounding Process---------------#

def rounding(x):
    '''
    :param x: a matrix with dimension deviceNum x edgeNum
    :return: rounding each row so that every row has and only has one 1 and others are 0
    '''
    res=np.zeros(x.shape)
    for j in range(x.shape[0]):
        cur=0
        probility=np.random.uniform(0,1,1)
        for i in range(x.shape[1]):
            cur+=x[j][i]
            if cur>=probility:
                res[j][i]=1
                break
    return res

#-----------------Main Process------------------#

# Output file
f=open("result.txt", "w")

for t in range(totalTime):

    print("Decision time %d"%t,file=f)

    # Prepare for the optimizer
    devices.setDataSizeDistribution()
    deviceNum=devices.setIsAliveDistribution()
    isAvail=devices.getIsAlive()

    # Initialize all the choices to the virtual edge
    choice=np.zeros([deviceNum,edgeNum])
    choice[:,edgeNum-1]=np.ones(deviceNum)

    # Set decision bounds, which are all from 0 to 1
    bound=[]
    for i in range(deviceNum*edgeNum):
        bound.append([0,1])

    # Call optimize function
    optimizeResult=minimize(objectiveFunction, choice.reshape((deviceNum*edgeNum,1)),constraints=constrainFunction(),bounds=bound)
    choice=optimizeResult.x.reshape((deviceNum,edgeNum))

    # Information below is for debug
    print('isAvail',end=' ',file=f)
    print(isAvail,file=f)

    # print(optimizeResult)
    # print(optimizeResult.fun)
    # print(choice)
    # print(optimizeResult.message)
    print('Success: ',file=f)
    print(optimizeResult.success,file=f)

    # call rounding function
    choice=rounding(choice)

    # Final Decision, need to merge choice from optimizer
    # and unavailable devices which choices are always the virtual edge
    finalDecision=np.zeros((allDeviceNum,edgeNum))
    index=0
    for j in range(allDeviceNum):
        if isAvail[j]==True:
            finalDecision[j]=choice[index]
            index+=1
        else:
            finalDecision[j]=np.zeros(edgeNum)
            finalDecision[j][edgeNum-1]=1
    print('FINAL DECISION:',file=f)
    print(finalDecision,file=f)
    print('\n')
