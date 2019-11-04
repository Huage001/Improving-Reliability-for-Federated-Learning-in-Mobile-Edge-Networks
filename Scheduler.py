from DeviceManager import DeviceManager
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
import copy

#----------Constant----------#

# Learning Model Size
modelSize=1

# Number of device
allDeviceNum=0

# Number of Edge
edgeNum=1

# Ratio of Selected Data
ratio=1

#---------Other Variable---------#

# All Devices, the corresponding variables are in its class
devices=None

# All Edges, the corresponding variables are in its class
#edges=EdgeManager(edgeNum)
# Note that we have a virtual edge
edgeNum=edgeNum+1

#---------Constrain Function--------#

# One device can choose one Edge at most
def choiceConstrain(x):
    # Transform vector x into a matrix with the same shape as the decision variable
    y=copy.deepcopy(x)
    indicator=y.reshape((deviceNum,edgeNum))
    # Note that we should calculate sum for each row, so axis=1
    return np.sum(indicator,axis=1)-1

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
            {'type':'ineq','fun':dataAmountConstrain})
    return cons

#---------------Function of transmit error rate with number of connections at the same time-----------------#

def transmitErrorRate(num):
    a=28.512476594629344
    b=33.1089283946605
    c=10.797872012316573
    return a / (b + np.exp(c-num/1000))

#---------------Objective Function----------------#

def objectiveFunction(x):
    y=copy.deepcopy(x)
    indicator=y.reshape((deviceNum,edgeNum))
    # Note that we should not include the virtual edge
    indicator[:,edgeNum-1]=np.zeros(deviceNum)
    # a vector with dimension deviceNum
    errorRate=devices.getErrorRate()
    # a vector with dimension deviceNum
    dataSize=devices.getDataSize()
    # The sum of each column is the number of connections of each edge.
    # Therefore, we get a vector with dimension edgeNum
    allTransmitErrorRate=transmitErrorRate(np.sum(indicator,axis=0))
    # result[j][i] means the total time, including computing and transmitting,
    # on the condition that the device j choose edge i
    result=np.zeros((deviceNum,edgeNum))
    for j in range(deviceNum):
        for i in range(edgeNum):
            result[j][i]=(errorRate[j]+(1-errorRate[j])*allTransmitErrorRate[i])*indicator[j][i]
    # calculating the sum means we want the overall error rate of the device j
    result=np.sum(result,axis=1)
    # At last, the max error rate of all device means the final error rate
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
        probability=np.random.uniform(0,1,1)
        for i in range(x.shape[1]):
            cur+=x[j][i]
            if cur>=probability:
                res[j][i]=1
                break
    return res

#-----------------Main Process------------------#

# Output file
f=open("result.txt", "w")

# time information
timeInformation=[]

# error rate information
errorRateInformation=[]

# device number information
deviceNumInformation=[]
with open("input.txt") as file_object:
    for val in file_object.read().split():
        deviceNumInformation.append(int(val))

for t in range(len(deviceNumInformation)):

    print("Decision time %d"%t,file=f)
    timeInformation.append(t)

    # Read allDeviceNum from input.txt
    allDeviceNum=deviceNumInformation[t]
    if allDeviceNum==0:
        errorRateInformation.append(0)
        continue

    devices=DeviceManager(allDeviceNum)

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

    # print(choice)
    # print(optimizeResult.message)
    print('Success:',end=' ',file=f)
    print(optimizeResult.success,file=f)
    print(optimizeResult.fun,file=f)
    errorRateInformation.append(optimizeResult.fun)

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

plt.scatter(timeInformation, deviceNumInformation)
plt.show()
plt.scatter(deviceNumInformation,errorRateInformation)
plt.show()
plt.plot(timeInformation,errorRateInformation)
plt.show()
