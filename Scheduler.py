import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import linprog
from scipy.stats import f
import copy
import time

class DeviceManager:

    def getErrorRateDistribution(self):
        '''
        TODO: Read the error rates of all devices from input file
        But now, we only use Gaussian distribution
        :return: A ndarray of float numbers, with shape (self.deviceNum,)
        '''
        # if self.distribution=='normal':
        #     return (np.random.rand(self.deviceNum)*(self.parameter[1]-self.parameter[0])+self.parameter[0])/5
        # elif self.distribution=='f':
        #     return f.pdf(np.random.uniform(0,4,self.deviceNum), self.parameter[0], self.parameter[1])/5
        # elif self.distribution=='zipf':
        #     res=np.random.zipf(self.parameter,self.deviceNum)/100
        #     for i in range(res.shape[0]):
        #         if res[i]>1:
        #             res[i]=0.9
        #     return res
        res=f.pdf(np.random.uniform(0,4,self.deviceNum), 1, 1)/2
        for i in range(res.shape[0]):
            if res[i]>1:
                res[i]=0.9
        return res
        #return np.array([0.1,0.1,0.1])

    def getDataSizeDistribution(self):
        '''
        TODO: Read the data size distribution from input file. But it seems that we do not have this column in the input file
        Therefore, we just use Gaussian distribution
        Similar to the function above
        :return: A ndarray of float numbers, with shape (self.deviceNum,)
        '''
        #return np.random.rand(self.deviceNum)
        #return f.pdf(np.random.uniform(0,4,self.deviceNum), 1, 1)
        if self.distribution=='normal':
            return (np.random.rand(self.deviceNum)*(self.parameter[1]-self.parameter[0])+self.parameter[0])
        elif self.distribution=='f':
            return f.pdf(np.random.uniform(0,4,self.deviceNum), self.parameter[0], self.parameter[1])
        elif self.distribution=='zipf':
            return np.random.zipf(self.parameter,self.deviceNum)
        #return np.array([1,1,1])

    def getIsAliveDistribution(self):
        '''
        Similar to the function above
        :return: A ndarray of {0,1}, with shape (self.deviceNum,)
        '''
        #res=np.random.randint(0,2,self.deviceNum)
        res=np.ones(self.deviceNum)
        return res

    def __init__(self,deviceNum,parameter,distribution):
        self.deviceNum=deviceNum
        self.parameter=parameter
        self.distribution=distribution
        self.allErrorRate=self.getErrorRateDistribution()
        self.isAlive=None
        self.dataSize=None
        self.allDataSize=None
        self.errorRate=None
        self.totalDataSize=0

    def setDataSizeDistribution(self):
        '''
        call getDataSizeDistribution and assign returned value to all devices
        :return: None
        '''
        self.allDataSize=self.getDataSizeDistribution()

    def setIsAliveDistribution(self):
        '''
        call getIsAliveDistribution and assign returned value to all devices
        :return: None
        '''
        self.isAlive=self.getIsAliveDistribution()
        self.dataSize=self.allDataSize[self.isAlive.astype(bool)]
        self.totalDataSize=np.sum(self.dataSize)
        self.errorRate=self.allErrorRate[self.isAlive.astype(bool)]
        return np.sum(self.isAlive)

    def getTotalDataSize(self):
        '''
        :return: a real number
        '''
        return self.totalDataSize

    def getDataSize(self):
        '''
        :return: a list of the dataSize of each device
        '''
        return self.dataSize

    def getIsAlive(self):
        '''
        :return: a list of bool variables indicating whether each device is alive
        '''
        return self.isAlive

    def getErrorRate(self):
        '''
        :return: a list of the ComputeDelay of each device
        '''
        return self.errorRate

#----------Constant----------#

# Learning Model Size
modelSize=1

# Number of device
allDeviceNum=0

# Number of Edge
edgeNum=1

# Ratio of Selected Data
ratio=0.5
#ratios=np.linspace(0.1,0.7,20)

#---------Other Variable---------#

# All Devices, the corresponding variables are in its class
devices=None

# Note that we have a virtual edge
edgeNum=edgeNum+1

# enumerate the number of selected devices
selectDeviceNum=0

# Output file
fi=open("result.txt", "w")

# time information
timeInformation=[]

# error rate information, used by scatter plots
errorRateInformationLinear=[]
errorRateInformationGreedy=[]
errorRateInformationConvex=[]

# device error distribution information, used by box plots
# deviceErrorInformationLinear=[]
# deviceErrorInformationGreedy=[]
# deviceErrorInformationConvex=[]
deviceErrorLinear=[]
deviceErrorGreedy=[]
deviceErrorConvex=[]

# device number information
deviceNumInformation=[]
with open("input.txt") as file_object:
    for val in file_object.read().split():
        deviceNumInformation.append(int(val))

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
    return a / (b + np.exp(c-num))

#---------------Convex Optimization Objective Function----------------#

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

def roundingForConvex(x):
    '''
    :param x: a matrix with dimension deviceNum x edgeNum
    :return: rounding each row so that every row has and only has one 1 and others are 0
    '''
    res=np.zeros(x.shape)
    for j in range(x.shape[0]):
        cur=0
        probability=np.random.random()
        for i in range(x.shape[1]):
            cur+=x[j][i]
            if cur>=probability:
                res[j][i]=1
                break
    return res

def roundingForLinear(x):
    '''
    :param x: a vector with deviceNum mansions with value from [0,1]
    :return: a vector of rouding result with the same shape as x
    '''
    res=np.zeros(x.shape)
    for i in range(x.shape[0]):
        randomNum=np.random.random()
        if randomNum<x[i]:
            res[i]=1
        else:
            res[i]=0
    return res

#---------------Get final decisions-----------------#

def getFinalDecision(choice):
    '''
    Final Decision, need to merge choice from optimizer
    and unavailable devices which choices are always the virtual edge
    :return: Final error rate, that is, the max error rate of all devices
    '''
    finalDecision=np.zeros((allDeviceNum,edgeNum-1))
    index=0
    for j in range(allDeviceNum):
        if isAvail[j]==True:
            finalDecision[j]=choice[index]
            index+=1
        else:
            finalDecision[j]=np.zeros(np.shape(finalDecision[0]))
    print('FINAL DECISION:',file=fi)
    print(finalDecision,file=fi)
    print("ERROR RATE:",end=' ',file=fi)
    errorRate=getAllErrorRate(np.sum(choice))
    print(np.max(errorRate[choice.astype(bool)]),file=fi)
    print('\n')
    return np.max(errorRate[choice.astype(bool)])

#---------------Convex optimize Process---------------#
def convexOptimizer():
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
    # print('isAvail',end=' ',file=fi)
    # print(isAvail,file=fi)

    # print(choice)
    # print(optimizeResult.message)
    # print('Success:',end=' ',file=fi)
    # print(optimizeResult.success,file=fi)
    # print(optimizeResult.fun,file=fi)

    # call rounding function and get final decisions
    while True:
        solution=roundingForConvex(choice)
        if np.sum(solution[:,0])!=0:
            break
    dataSize=devices.getDataSize()

    # For box plot
    # dataSize=devices.getDataSize()
    # dataInformationConvex.append(dataSize[solution[:,0].astype(bool)])
    errorRate=devices.getErrorRate()
    deviceErrorInformationConvex.append(errorRate[solution[:,0].astype(bool)])

    # For CDF plot
    global dataInformationConvex
    dataInformationConvex=dataSize[solution[:,0].astype(bool)]
    dataInformationConvex.sort()

    return getFinalDecision(solution[:,0])

#----------------Get all error rate-----------------#

def getAllErrorRate(n):
    errorRate=devices.getErrorRate()
    errorRate=errorRate+(1-errorRate)*transmitErrorRate(n)
    return errorRate

#----------------Linear Programming Optimize Function------------------#

def linearOptimizer():
    # the coefficients vector in the objective function
    # Note that we use the last item to denote our final objection
    c=np.zeros(deviceNum+1)
    c[deviceNum]=1
    # result used for pruning
    lastResult=None
    finalResult=None
    # enumerate the number of selected devices
    # ask the number of selected devices to be i
    # Note that it should not include the last item,
    # which is our final objection
    A_eq=np.ones((1,deviceNum+1))
    A_eq[0][deviceNum]=0
    dataSize=-devices.getDataSize()
    dataSize=np.insert(dataSize,deviceNum,values=0,axis=0)
    # set the coefficient matrix, which has deviceNum+1 constrains
    # including deviceNum minimax constrains and 1 data ratio constrains
    A_ub=np.zeros((deviceNum+1,deviceNum+1))
    A_ub[deviceNum]=dataSize
    B_ub=np.zeros(deviceNum+1)
    B_ub[deviceNum]=-devices.getTotalDataSize()*ratio
    # decision bounds
    bound=[]
    for i in range(deviceNum+1):
        bound.append((0,1))
    for i in range(1,deviceNum+1):
        B_eq=np.array([i])
        # errorRate includes device error and transmit error
        errorRate=getAllErrorRate(i)
        # set each row of A_ub, note that the last number of each row is our objection, but now we view it as a decision variable
        for j in range(deviceNum):
            A_ub[j][j]=errorRate[j]
            A_ub[j][deviceNum]=-1
        # call linprog
        res=linprog(c=c,A_ub=A_ub,b_ub=B_ub,A_eq=A_eq,b_eq=B_eq,bounds=bound)
        # prone
        if res.success==False:
            continue
        if lastResult!=None and res.fun>lastResult.fun:
            print(lastResult)
            finalResult=lastResult
            break
        lastResult=res
    if finalResult==None:
        print(lastResult)
        finalResult=lastResult
    # Rounding, do it twice
    while True:
        solution1=roundingForLinear(finalResult.x[0:deviceNum])
        if np.sum(solution1)!=0:
            break
    while True:
        solution2=roundingForLinear(finalResult.x[0:deviceNum])
        if np.sum(solution2)!=0:
            break
    if np.sum(solution1)<np.sum(solution2):
        finalSolution=solution1
    elif np.sum(solution1)>np.sum(solution2):
        finalSolution=solution2
    else:
        dataSize=devices.getDataSize()
        if np.sum(dataSize[solution1.astype(bool)])>np.sum(dataSize[solution2.astype(bool)]):
            finalSolution=solution1
        else:
            finalSolution=solution2

    # For box plot
    # dataSize=devices.getDataSize()
    # dataInformationLinear.append(dataSize[finalSolution.astype(bool)])
    errorRate=devices.getErrorRate()
    deviceErrorInformationLinear.append(errorRate[finalSolution.astype(bool)])

    # For CDF plot
    global dataInformationLinear
    dataSize=devices.getDataSize()
    dataInformationLinear=dataSize[finalSolution.astype(bool)]
    dataInformationLinear.sort()

    return getFinalDecision(finalSolution)

#-----------------Greedy Process----------------#

def greedy():
    dataSize=devices.getDataSize()
    objectiveDataSize=ratio*devices.getTotalDataSize()
    currentDataSize=0
    ans=0
    dataSizeSort=dataSize.argsort()
    finalSolution=np.zeros(deviceNum)
    for i in range(deviceNum-1,-1,-1):
        ans+=1
        finalSolution[dataSizeSort[i]]=1
        currentDataSize+=dataSize[dataSizeSort[i]]
        if currentDataSize>objectiveDataSize:
            break

    # For box plot
    # dataInformationGreedy.append(dataSize[finalSolution.astype(bool)])
    errorRate=devices.getErrorRate()
    deviceErrorInformationGreedy.append(errorRate[finalSolution.astype(bool)])

    # For CDF plot
    global dataInformationGreedy
    dataInformationGreedy=dataSize[finalSolution.astype(bool)]
    dataInformationGreedy.sort()

    return getFinalDecision(finalSolution)

#-----------------Main Process------------------#

# do numbers of times linear programming to evaluate the average performance
times=10
parameter=[[0,1],[1,2],[2,3],[1,1],[8,3],[20,20],1.5,2,3]
#trace=open("trace_best.txt","w")
timeConvex=[]
timeGreedy=[]
timeLinear=[]
for t in range(len(deviceNumInformation)):
#for t in range(0,1,1):
#for r in range(len(ratios)):
#for r in range(len(parameter)):
    # t=0
    # if r<3:
    #     distribution='normal'
    #     ratio=0.3
    # elif r<6:
    #     distribution='f'
    #     ratio=0.6
    # else:
    #     distribution='zipf'
    #     ratio=0.5
    #ratio=ratios[r]
    convexAve=[]
    linearAve=[]
    greedyAve=[]
    timeConvexAve=[]
    timeGreedyAve=[]
    timeLinearAve=[]
    deviceErrorInformationLinear=[]
    deviceErrorInformationConvex=[]
    deviceErrorInformationGreedy=[]
    #allDevices=[]
    for k in range(times):

        print("Decision time %d"%t,file=fi)
        timeInformation.append(t)

        # Read allDeviceNum from input.txt
        allDeviceNum=deviceNumInformation[t]

        if allDeviceNum==0:
            errorRateInformationLinear.append(0)
            errorRateInformationGreedy.append(0)
            continue # For debug so delete it

        #devices=DeviceManager(allDeviceNum,parameter[r],distribution)
        devices=DeviceManager(allDeviceNum,[1,1],'f')
        #allDevices.append(devices)

        # Prepare for the optimizer
        devices.setDataSizeDistribution()
        deviceNum=int(devices.setIsAliveDistribution())
        isAvail=devices.getIsAlive()
        print('Number of all devices',end=' ',file=fi)
        print(allDeviceNum,file=fi)
        print('Number of available devices',end=' ',file=fi)
        print(deviceNum,file=fi)
        print('Available devices:',file=fi)
        print(isAvail,file=fi)
        print('Error rates of available devices',file=fi)
        print(devices.getErrorRate(),file=fi)
        print('Data size of available devices',file=fi)
        print(devices.getDataSize(),file=fi)

        print('Convex Optimizer:',file=fi)
        timeStartConvex=time.time()
        convexValue=convexOptimizer()
        timeEndConvex=time.time()
        timeConvexAve.append(timeEndConvex-timeStartConvex)
        convexAve.append(convexValue)
        print('Linear Optimizer:',file=fi)
        timeStartLinear=time.time()
        linearValue=linearOptimizer()
        timeEndLinear=time.time()
        timeLinearAve.append(timeEndLinear-timeStartLinear)
        linearAve.append(linearValue)
        print('Greedy',file=fi)
        timeStartGreedy=time.time()
        greedyValue=greedy()
        timeEndGreedy=time.time()
        timeGreedyAve.append(timeEndGreedy-timeStartGreedy)
        greedyAve.append(greedyValue)

    # best=np.argmax(-np.array(linearAve)+(np.array(greedyAve)+np.array(convexAve))/2)
    # errorRateInformationLinear.append(linearAve[best])
    # errorRateInformationGreedy.append(greedyAve[best])
    # errorRateInformationConvex.append(convexAve[best])
    # deviceErrorLinear.append(deviceErrorInformationLinear[best])
    # deviceErrorGreedy.append(deviceErrorInformationGreedy[best])
    # deviceErrorConvex.append(deviceErrorInformationConvex[best])

    # print(allDevices[best].getErrorRate(),file=trace)
    # print(allDevices[best].getDataSize(),file=trace)

    errorRateInformationLinear.append(np.mean(linearAve))
    errorRateInformationGreedy.append(np.mean(greedyAve))
    errorRateInformationConvex.append(np.mean(convexAve))
    timeGreedy.append(np.mean(timeGreedyAve))
    timeConvex.append(np.mean(timeConvexAve))
    timeLinear.append(np.mean(timeLinearAve))

# Computing the improvement
totalErrorRateLinear=np.mean(errorRateInformationLinear)
totalErrorRateGreedy=np.mean(errorRateInformationGreedy)
totalErrorRateConvex=np.mean(errorRateInformationConvex)
improveGreedy=(totalErrorRateGreedy-totalErrorRateLinear)*100/totalErrorRateLinear
improveConvex=(totalErrorRateConvex-totalErrorRateLinear)*100/totalErrorRateLinear

# Scatter plot
plt.scatter(deviceNumInformation,errorRateInformationLinear,color='r',label='Ours',marker='o')
plt.scatter(deviceNumInformation,errorRateInformationGreedy,color='g',label='Greedy (%.2f%% higher than ours)'%improveGreedy,marker='x')
plt.scatter(deviceNumInformation,errorRateInformationConvex,color='b',label='SLSQP Algorithm (%.2f%% higher than ours)'%improveConvex,marker='*')
plt.legend()
plt.show()

# plt.scatter(ratios,errorRateInformationLinear,color='r',label='Ours',marker='o')
# plt.scatter(ratios,errorRateInformationGreedy,color='g',label='Greedy (%.2f%% higher than ours)'%improveGreedy,marker='x')
# plt.scatter(ratios,errorRateInformationConvex,color='b',label='SLSQP Algorithm (%.2f%% higher than ours)'%improveConvex,marker='*')
# plt.legend()
# plt.show()

plotData=open("plotData_main.txt", "w")
print(errorRateInformationLinear,file=plotData)
print(errorRateInformationGreedy,file=plotData)
print(errorRateInformationConvex,file=plotData)

# Time plot
timeData=open("timeused.txt","w")
print(timeLinear,file=timeData)
print(timeGreedy,file=timeData)
print(timeConvex,file=timeData)
plt.plot(deviceNumInformation,timeLinear,color='r',label='Ours')
plt.plot(deviceNumInformation,timeGreedy,color='g',label='Greedy')
plt.plot(deviceNumInformation,timeConvex,color='b',label='SLSQP Algorithm')
plt.legend()
plt.show()

# Bar plot
#xtext=['normal distribution with high variance','normal distribution with middle variance','normal distribution with low variance',
#       'F distribution with high bias','F distribution with middle bias','F distribution with low bias']
# xtext=['Normal-1','Normal-2','Normal-3','F-1','F-2','F-3','zipf-1','zipf-2','zipf-3']
# xposition=np.arange(len(parameter))
# bar_width=0.25
# plt.bar(x=xposition, height=errorRateInformationConvex, label='SLSQP Algorithm', color='b',align='center',width=bar_width)
# plt.bar(x=xposition+bar_width, height=errorRateInformationGreedy, label='Greedy', color='g',align='center',width=bar_width)
# plt.bar(x=xposition+2*bar_width, height=errorRateInformationLinear, label='Ours', color='r',align='center',width=bar_width)
# plt.xticks(xposition+bar_width,xtext,rotation=45)
# plt.legend()
#plt.show()

# Box plot
# boxes=[]
# boxes.append(plt.boxplot(deviceErrorConvex,patch_artist=True))
# boxes.append(plt.boxplot(deviceErrorGreedy,patch_artist=True))
# boxes.append(plt.boxplot(deviceErrorLinear,patch_artist=True))
# color=['b','g','r']
# for i in range(3):
#     for j in range(len(parameter)):
#         boxes[i]['boxes'][j].set_facecolor(color[i])

# CDF plot
def getY(x):
    y=[]
    cur=0
    xvals=[]
    yvals=[]
    for i in range(len(x)):
        if i>0:
            if i%2==1:
                b=2
            else:
                b=0.5
            a=x[i]/((x[i]-x[i-1])**b)
            xval=np.linspace(x[i-1],x[i],20)
            yval=a*((xval-x[i-1])**b)+cur
            xvals+=xval.tolist()
            yvals+=yval.tolist()
        cur+=x[i]
        y.append(cur)
    return xvals,yvals

# plt.hist((dataInformationConvex),cumulative=True,color='b',label='SLSQP Algorithm',histtype='step',bins=1000)
# plt.hist((dataInformationGreedy),cumulative=True,color='g',label='Greedy',histtype='step',bins=1000)
# plt.hist((dataInformationLinear),cumulative=True,color='r',label='Ours',histtype='step',bins=1000)
# plt.legend()
#
# plt.show()
#
# data=getY(dataInformationConvex)
# plt.plot(data[0],data[1],color='b',label='SLSQP Algorithm')
# data=getY(dataInformationGreedy)
# plt.plot(data[0],data[1],color='g',label='Greey')
# data=getY(dataInformationLinear)
# plt.plot(data[0],data[1],color='r',label='Ours')
# plt.legend()

plt.show()
