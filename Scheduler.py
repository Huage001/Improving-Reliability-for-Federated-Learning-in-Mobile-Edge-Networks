import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import linprog
from scipy.stats import f
import ctypes
import copy
import time
import Configure
import threading
import threadpool
from visdom import Visdom

class DeviceManager:

    def getErrorRateDistribution(self):
        if self.distributionError=='normal':
            res=(np.random.normal(self.parameterError[0],self.parameterError[1],self.deviceNum))
            for i in range(res.shape[0]):
                if res[i] < 0:
                    res[i] = 0
                if res[i] > 1:
                    res[i]=0.9
            return res
        elif self.distributionError=='f':
            return f.pdf(np.random.uniform(0,4,self.deviceNum), self.parameterError[0], self.parameterError[1])/5
        elif self.distributionError=='zipf':
            res=np.random.zipf(self.parameterError,self.deviceNum)/100
            for i in range(res.shape[0]):
                if res[i]>1:
                    res[i]=0.9
            return res

    def getDataSizeDistribution(self):
        res=None
        if self.distributionData=='normal':
            res=(np.random.normal(self.parameterError[0],self.parameterError[1],self.deviceNum))
        elif self.distributionData=='f':
            res=f.pdf(np.random.uniform(0,4,self.deviceNum), self.parameterData[0], self.parameterData[1])
        elif self.distributionData=='zipf':
            res=np.random.zipf(self.parameterData,self.deviceNum)

        if self.isRelated==False:
            return res
        else:
            index = self.allErrorRate.argsort()
            res.sort()
            finalRes = copy.deepcopy(res)
            j = 0
            for i in index:
                finalRes[i] = res[j]
                j += 1
            return finalRes

    def getIsAliveDistribution(self):
        res=np.ones(self.deviceNum)
        return res

    def __init__(self,deviceNum,parameterData,parameterError,distributionData,distributionError,isRalated=False):
        self.deviceNum=deviceNum
        self.parameterData=parameterData
        self.distributionData = distributionData
        self.parameterError=parameterError
        self.distributionError = distributionError
        self.isRelated = isRalated
        self.allErrorRate=self.getErrorRateDistribution()
        self.isAlive=None
        self.dataSize=None
        self.allDataSize=None
        self.errorRate=None
        self.totalDataSize=0

    def setDataSizeDistribution(self):
        self.allDataSize=self.getDataSizeDistribution()

    def setIsAliveDistribution(self):
        self.isAlive=self.getIsAliveDistribution()
        self.dataSize=self.allDataSize[self.isAlive.astype(bool)]
        self.totalDataSize=np.sum(self.dataSize)
        self.errorRate=self.allErrorRate[self.isAlive.astype(bool)]
        return np.sum(self.isAlive)

    def getTotalDataSize(self):
        return self.totalDataSize

    def getDataSize(self):
        return self.dataSize

    def getIsAlive(self):
        return self.isAlive

    def getErrorRate(self):
        return self.errorRate

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class Frame:

    def __init__(self):
        self.devices = None
        self.deviceNum = 0
        self.allDeviceNum = 0
        self.ratio = 0
        self.isAvail = None

    #---------Constrain Function--------#

    # One device can choose one Edge at most
    def choiceConstrain(self,x):
        # Transform vector x into a matrix with the same shape as the decision variable
        y=copy.deepcopy(x)
        indicator=y.reshape((self.deviceNum,edgeNum))
        # Note that we should calculate sum for each row, so axis=1
        return np.sum(indicator,axis=1)-1

    def dataAmountConstrain(self,x):
        # The formation is similar to the function above
        y=copy.deepcopy(x)
        indicator=y.reshape((self.deviceNum,edgeNum))
        # Note that we should not include the virtual edge
        indicator[:,edgeNum-1]=np.zeros(self.deviceNum)
        totalDataSize=self.devices.getTotalDataSize()
        dataSize=self.devices.getDataSize()
        return np.sum(np.sum(indicator*dataSize.reshape((self.deviceNum,1))))-self.ratio*totalDataSize

    # Total Constrain Function
    def constrainFunction(self):
        cons = ({'type':'eq','fun':self.choiceConstrain},
                {'type':'ineq','fun':self.dataAmountConstrain})
        return cons

    #---------------Function of transmit error rate with number of connections at the same time-----------------#

    def transmitErrorRate(self,num):
        a=28.512476594629344
        b=33.1089283946605
        c=10.797872012316573
        return a / (b + np.exp(c-num))

    #---------------Convex Optimization Objective Function----------------#

    def objectiveFunction(self,x):
        y=copy.deepcopy(x)
        indicator=y.reshape((self.deviceNum,edgeNum))
        # Note that we should not include the virtual edge
        indicator[:,edgeNum-1]=np.zeros(self.deviceNum)
        # a vector with dimension deviceNum
        errorRate=self.devices.getErrorRate()
        # a vector with dimension deviceNum
        dataSize=self.devices.getDataSize()
        # The sum of each column is the number of connections of each edge.
        # Therefore, we get a vector with dimension edgeNum
        allTransmitErrorRate=self.transmitErrorRate(np.sum(indicator,axis=0))
        # result[j][i] means the total time, including computing and transmitting,
        # on the condition that the device j choose edge i
        result=np.zeros((self.deviceNum,edgeNum))
        for j in range(self.deviceNum):
            for i in range(edgeNum):
                result[j][i]=(errorRate[j]+(1-errorRate[j])*allTransmitErrorRate[i])*indicator[j][i]
        # calculating the sum means we want the overall error rate of the device j
        result=np.sum(result,axis=1)
        # At last, the max error rate of all device means the final error rate
        return np.max(result)

    #----------------Rounding Process---------------#

    def roundingForConvex(self,x):
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

    def roundingForLinear(self,x):
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

    def getFinalDecision(self,choice):
        '''
        Final Decision, need to merge choice from optimizer
        and unavailable devices which choices are always the virtual edge
        :return: Final error rate, that is, the max error rate of all devices
        '''
        finalDecision=np.zeros((self.allDeviceNum,edgeNum-1))
        index=0
        for j in range(self.allDeviceNum):
            if self.isAvail[j]==True:
                finalDecision[j]=choice[index]
                index+=1
            else:
                finalDecision[j]=np.zeros(np.shape(finalDecision[0]))

        if Configure.xAxis=='single':
            x=self.devices.getErrorRate()
            y=self.devices.getDataSize()
            plt.scatter(x,y,color='gray',label='unselected',marker='o')
            plt.scatter(x[finalDecision],y[finalDecision],color='r',label='selected',marker='o')
            plt.show()
            plt.close()

        errorRate=self.getAllErrorRate(np.sum(choice))
        return np.max(errorRate[choice.astype(bool)])

    #---------------Convex optimize Process---------------#
    def convexOptimizer(self):
        # Initialize all the choices to the virtual edge
        choice=np.zeros([self.deviceNum,edgeNum])
        choice[:,edgeNum-1]=np.ones(self.deviceNum)

        # Set decision bounds, which are all from 0 to 1
        bound=[]
        for i in range(self.deviceNum*edgeNum):
            bound.append([0,1])

        # Call optimize function
        optimizeResult=minimize(self.objectiveFunction,
                                choice.reshape((self.deviceNum*edgeNum,1)),
                                constraints=self.constrainFunction(),
                                bounds=bound)
        choice=optimizeResult.x.reshape((self.deviceNum,edgeNum))

        # call rounding function and get final decisions
        while True:
            solution=self.roundingForConvex(choice)
            if np.sum(solution[:,0])!=0:
                break

        return self.getFinalDecision(solution[:,0])

    #----------------Get all error rate-----------------#

    def getAllErrorRate(self,n):
        errorRate=self.devices.getErrorRate()
        errorRate=errorRate+(1-errorRate)*self.transmitErrorRate(n)
        return errorRate

    #----------------Linear Programming Optimize Function------------------#

    def linearOptimizer(self):
        # the coefficients vector in the objective function
        # Note that we use the last item to denote our final objection
        c=np.zeros(self.deviceNum+1)
        c[self.deviceNum]=1
        # result used for pruning
        lastResult=None
        finalResult=None
        # enumerate the number of selected devices
        # ask the number of selected devices to be i
        # Note that it should not include the last item,
        # which is our final objection
        A_eq=np.ones((1,self.deviceNum+1))
        A_eq[0][self.deviceNum]=0
        dataSize=-self.devices.getDataSize()
        dataSize=np.insert(dataSize,self.deviceNum,values=0,axis=0)
        # set the coefficient matrix, which has deviceNum+1 constrains
        # including deviceNum minimax constrains and 1 data ratio constrains
        A_ub=np.zeros((self.deviceNum+1,self.deviceNum+1))
        A_ub[self.deviceNum]=dataSize
        B_ub=np.zeros(self.deviceNum+1)
        B_ub[self.deviceNum]=-self.devices.getTotalDataSize()*self.ratio
        # decision bounds
        bound=[]
        for i in range(self.deviceNum+1):
            bound.append((0,1))
        for i in range(1,self.deviceNum+1):
            B_eq=np.array([i])
            # errorRate includes device error and transmit error
            errorRate=self.getAllErrorRate(i)
            # set each row of A_ub, note that the last number of each row is our objection,
            # but now we view it as a decision variable
            for j in range(self.deviceNum):
                A_ub[j][j]=errorRate[j]
                A_ub[j][self.deviceNum]=-1
            # call linprog
            res=linprog(c=c,A_ub=A_ub,b_ub=B_ub,A_eq=A_eq,b_eq=B_eq,bounds=bound)
            # prone
            if res.success==False:
                continue
            if lastResult!=None and res.fun>lastResult.fun:
                finalResult=lastResult
                break
            lastResult=res
        if finalResult==None:
            finalResult=lastResult
        # Rounding, do it twice
        while True:
            solution1=self.roundingForLinear(finalResult.x[0:self.deviceNum])
            if np.sum(solution1)!=0:
                break
        while True:
            solution2=self.roundingForLinear(finalResult.x[0:self.deviceNum])
            if np.sum(solution2)!=0:
                break
        if np.sum(solution1)<np.sum(solution2):
            finalSolution=solution1
        elif np.sum(solution1)>np.sum(solution2):
            finalSolution=solution2
        else:
            dataSize=self.devices.getDataSize()
            if np.sum(dataSize[solution1.astype(bool)])>np.sum(dataSize[solution2.astype(bool)]):
                finalSolution=solution1
            else:
                finalSolution=solution2

        return self.getFinalDecision(finalSolution)

    #-----------------Greedy Process----------------#

    def greedy(self):
        dataSize=self.devices.getDataSize()
        objectiveDataSize=self.ratio*self.devices.getTotalDataSize()
        currentDataSize=0
        ans=0
        dataSizeSort=dataSize.argsort()
        finalSolution=np.zeros(self.deviceNum)
        for i in range(self.deviceNum-1,-1,-1):
            ans+=1
            finalSolution[dataSizeSort[i]]=1
            currentDataSize+=dataSize[dataSizeSort[i]]
            if currentDataSize>objectiveDataSize:
                break

        return self.getFinalDecision(finalSolution)

#-----------------Main Process------------------#

def experiment(_allDeviceNum,_ratio,errorDistribution,errorParameter,dataDistribution,dataParameter,id):

    cur=Frame()

    cur.allDeviceNum=_allDeviceNum
    cur.ratio=_ratio

    convexAve = []
    linearAve = []
    greedyAve = []
    timeConvexAve = []
    timeGreedyAve = []
    timeLinearAve = []
    improveConvexAve = []
    improveGreedyAve = []

    for k in range(Configure.times):

        if cur.allDeviceNum==0:
            errorRateInformationLinear.append(0)
            errorRateInformationGreedy.append(0)
            errorRateInformationConvex.append(0)
            continue

        cur.devices=DeviceManager(cur.allDeviceNum,dataParameter,
                                  errorParameter,dataDistribution,errorDistribution,Configure.relate)
        cur.devices.setDataSizeDistribution()
        cur.deviceNum=int(cur.devices.setIsAliveDistribution())
        cur.isAvail=cur.devices.getIsAlive()

        timeStartConvex=time.time()
        convexValue=cur.convexOptimizer()
        timeEndConvex=time.time()
        timeConvexAve.append(timeEndConvex-timeStartConvex)
        convexAve.append(convexValue)

        timeStartLinear=time.time()
        linearValue=cur.linearOptimizer()
        timeEndLinear=time.time()
        timeLinearAve.append(timeEndLinear-timeStartLinear)
        linearAve.append(linearValue)

        timeStartGreedy=time.time()
        greedyValue=cur.greedy()
        timeEndGreedy=time.time()
        timeGreedyAve.append(timeEndGreedy-timeStartGreedy)
        greedyAve.append(greedyValue)

        improveConvexValue=(convexValue-linearValue)/convexValue
        improveGreedyValue=(greedyValue-linearValue)/greedyValue
        improveConvexAve.append(improveConvexValue)
        improveGreedyAve.append(improveGreedyValue)

    mutex.acquire()
    if Configure.standard=='max':
        best=np.argmax(-np.array(linearAve)+(np.array(greedyAve)+np.array(convexAve))/2)
        errorRateInformationLinear.append([id,linearAve[int(best)]])
        errorRateInformationGreedy.append([id,greedyAve[int(best)]])
        errorRateInformationConvex.append([id,convexAve[int(best)]])
        improveConvex.append([id,improveConvexAve[int(best)]])
        improveGreedy.append([id,improveGreedyAve[int(best)]])

    else:
        errorRateInformationLinear.append([id,np.mean(linearAve)])
        errorRateInformationGreedy.append([id,np.mean(greedyAve)])
        errorRateInformationConvex.append([id,np.mean(convexAve)])
        improveConvex.append([id, (np.mean(convexAve)-np.mean(linearAve))/np.mean(convexAve)])
        improveGreedy.append([id, (np.mean(greedyAve)-np.mean(linearAve))/np.mean(greedyAve)])
    timeGreedy.append([id,np.mean(timeGreedyAve)])
    timeConvex.append([id,np.mean(timeConvexAve)])
    timeLinear.append([id,np.mean(timeLinearAve)])
    mutex.release()

mutex=threading.Lock()

edgeNum = 2

errorRateInformationLinear=[]
errorRateInformationGreedy=[]
errorRateInformationConvex=[]

improveConvex=[]
improveGreedy=[]

timeConvex = []
timeGreedy = []
timeLinear = []

errorParameter=None
dataParameter=None
maxDeviceNum=None
maxRatio=None

if Configure.xAxis=='dataRatio' or Configure.xAxis=='deviceNum':
    errorParameter=str(Configure.errorParameter)
    dataParameter=str(Configure.dataParameter)
    if Configure.xAxis=='dataRatio':
        maxDeviceNum=str(Configure.allDeviceNum)
        maxRatio=str(Configure.dataRatio[-1])
    else:
        maxDeviceNum=str(Configure.allDeviceNum[-1])
        maxRatio=str(Configure.dataRatio)
elif Configure.xAxis=='distribution':
    errorParameter = str(Configure.errorParameter[0])+"_to_"+str(Configure.errorParameter[-1])
    dataParameter = str(Configure.dataParameter[0])+"_to_"+str(Configure.dataParameter[-1])
    maxDeviceNum = str(Configure.allDeviceNum)
    maxRatio = str(Configure.dataRatio)
name = Configure.xAxis + \
       "_" + maxDeviceNum + \
       "_" + maxRatio + \
       "_" + Configure.errorDistribution + \
       "_" + errorParameter + \
       "_" + Configure.dataDistribution + \
       "_" + dataParameter

if Configure.standard=='max':
    name+="_max"
# if Configure.cases!=None:
#     name+=Configure.cases
if Configure.relate:
    name+="_relate"

pool=None
if Configure.mulThread>0:
    pool=threadpool.ThreadPool(Configure.mulThread)
id=0

viz=Visdom(server="114.212.82.243",env=name)

class experimentThread(threading.Thread):

    def __init__(self,_allDeviceNum,_ratio,errorDistribution,errorParameter,dataDistribution,dataParameter,id):
        threading.Thread.__init__(self)
        self._allDeviceNum=_allDeviceNum
        self._ratio=_ratio
        self.errorDistribution=errorDistribution
        self.errorParameter=errorParameter
        self.dataDistribution=dataDistribution
        self.dataParameter=dataParameter
        self.id=id
        self.success=False

    def run(self):
        experiment(self._allDeviceNum, self._ratio, self.errorDistribution,
                   self.errorParameter, self.dataDistribution, self.dataParameter, self.id)
        self.success=True


allArgList=[]
if Configure.xAxis=='dataRatio':
    for _ratio in Configure.dataRatio:
        if Configure.mulThread > 0:
            argList=[]
            argList.append(Configure.allDeviceNum)
            argList.append(_ratio)
            argList.append(Configure.errorDistribution)
            argList.append(Configure.errorParameter)
            argList.append(Configure.dataDistribution)
            argList.append(Configure.dataParameter)
            argList.append(id)
            allArgList.append((argList,None))
        else:
            experiment(Configure.allDeviceNum,_ratio,Configure.errorDistribution,
                       Configure.errorParameter,Configure.dataDistribution,Configure.dataParameter,id)
        id+=1
elif Configure.xAxis=='deviceNum':
    for _allDeviceNum in Configure.allDeviceNum:
        if Configure.mulThread>0:
            argList = []
            argList.append(_allDeviceNum)
            argList.append(Configure.dataRatio)
            argList.append(Configure.errorDistribution)
            argList.append(Configure.errorParameter)
            argList.append(Configure.dataDistribution)
            argList.append(Configure.dataParameter)
            argList.append(id)
            allArgList.append((argList, None))
        else:
            experiment(_allDeviceNum, Configure.dataRatio, Configure.errorDistribution,
                       Configure.errorParameter, Configure.dataDistribution, Configure.dataParameter, id)
        id+=1
elif Configure.xAxis=='distribution':
    for _dataParameter in Configure.dataParameter:
        for _errorParameter in Configure.errorParameter:
            if Configure.mulThread>0:
                argList = []
                argList.append(Configure.allDeviceNum)
                argList.append(Configure.dataRatio)
                argList.append(Configure.errorDistribution)
                argList.append(_errorParameter)
                argList.append(Configure.dataDistribution)
                argList.append(_dataParameter)
                argList.append(id)
                allArgList.append((argList, None))
            else:
                # _thread.start_new_thread(experiment,(Configure.allDeviceNum, Configure.dataRatio,
                #                                      Configure.errorDistribution,_errorParameter,
                #                                      Configure.dataDistribution, _dataParameter, id,))
                while (True):
                    thread=experimentThread(Configure.allDeviceNum, Configure.dataRatio,
                                                         Configure.errorDistribution,_errorParameter,
                                                         Configure.dataDistribution, _dataParameter, id)

                    thread.start()
                    thread.join(64)
                    if thread.success:
                        break
                    else:
                        if thread.is_alive():
                            ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(thread.ident))
                # experiment(Configure.allDeviceNum, Configure.dataRatio, Configure.errorDistribution,
                #            _errorParameter, Configure.dataDistribution, _dataParameter, id)
            id+=1
        if Configure.mulThread<=0:
            temp=np.zeros((len(Configure.errorParameter),3))
            temp[:, 0] = np.array(errorRateInformationConvex[id - len(Configure.errorParameter):id])[:,1]
            temp[:, 1] = np.array(errorRateInformationGreedy[id - len(Configure.errorParameter):id])[:,1]
            temp[:, 2] = np.array(errorRateInformationLinear[id - len(Configure.errorParameter):id])[:,1]
            viz.bar(
                X=temp,
                opts=dict(
                    stacked=False,
                    legend=['SLSQP','Greedy','Ours'],
                    rownames=[str(i) for i in Configure.errorParameter],
                )
            )
            viz.text(str(_dataParameter))

if Configure.mulThread>0:
    requestList=threadpool.makeRequests(experiment,allArgList)
    [pool.putRequest(req) for req in requestList]
    pool.wait()

# sort
if Configure.mulThread>0:
    sorted(errorRateInformationLinear,key=lambda x:x[0])
    sorted(errorRateInformationGreedy,key=lambda x:x[0])
    sorted(errorRateInformationConvex,key=lambda x:x[0])
    sorted(timeLinear,key=lambda x:x[0])
    sorted(timeGreedy,key=lambda x:x[0])
    sorted(timeConvex,key=lambda x:x[0])
    sorted(improveConvex,key=lambda x:x[0])
    sorted(improveGreedy,key=lambda x:x[0])
errorRateInformationConvex=[i[1] for i in errorRateInformationConvex]
errorRateInformationGreedy=[i[1] for i in errorRateInformationGreedy]
errorRateInformationLinear=[i[1] for i in errorRateInformationLinear]
improveConvex=[i[1] for i in improveConvex]
improveGreedy=[i[1] for i in improveGreedy]
timeConvex=[i[1] for i in timeConvex]
timeGreedy=[i[1] for i in timeGreedy]
timeLinear=[i[1] for i in timeLinear]

# Computing the improvement
totalErrorRateLinear=np.mean(errorRateInformationLinear)
totalErrorRateGreedy=np.mean(errorRateInformationGreedy)
totalErrorRateConvex=np.mean(errorRateInformationConvex)
improveGreedyTotal=(totalErrorRateGreedy-totalErrorRateLinear)*100/totalErrorRateGreedy
improveConvexTotal=(totalErrorRateConvex-totalErrorRateLinear)*100/totalErrorRateConvex

plotData=open("./result/"+name+".txt","w")
print(errorRateInformationLinear,file=plotData)
print(errorRateInformationGreedy,file=plotData)
print(errorRateInformationConvex,file=plotData)
plotData.close()

xAxis=None
if Configure.plot=='scatter':
    if Configure.xAxis=='deviceNum':
        xAxis=Configure.allDeviceNum
    elif Configure.xAxis=='dataRatio':
        xAxis=Configure.dataRatio
    plt.scatter(xAxis,
                errorRateInformationLinear,
                color='r',
                label='Ours',
                marker='o')
    plt.scatter(xAxis,
                errorRateInformationGreedy,
                color='g',
                label='Greedy (decreased by %.2f%%)' % improveGreedyTotal,
                marker='x')
    plt.scatter(xAxis,
                errorRateInformationConvex,
                color='b',
                label='SLSQP (decreased by %.2f%%)' % improveConvexTotal,
                marker='*')
    plt.legend()
    plt.savefig('result/'+name+'.png')
    plt.show()
    plt.close()

    timeData=open("./result/"+name+"_time.txt","w")
    print(timeLinear,file=timeData)
    print(timeGreedy,file=timeData)
    print(timeConvex,file=timeData)
    plt.plot(xAxis,timeLinear,color='r',label='Ours')
    plt.plot(xAxis,timeGreedy,color='g',label='Greedy')
    plt.plot(xAxis,timeConvex,color='b',label='SLSQP')
    plt.legend()
    plt.savefig("./result/"+name+"_time.png")
    plt.show()
    plt.close()
elif Configure.plot=='bar':
    xtext=[]
    for i in Configure.dataParameter:
        for j in Configure.errorParameter:
            xtext.append(str(i)+"-"+str(j))
    xposition=np.arange(len(Configure.dataParameter)*len(Configure.errorParameter))
    bar_width=0.25
    plt.bar(x=xposition,
            height=errorRateInformationConvex,
            label='SLSQP Algorithm',
            color='b',
            align='center',
            width=bar_width)
    plt.bar(x=xposition+bar_width,
            height=errorRateInformationGreedy,
            label='Greedy',
            color='g',
            align='center',
            width=bar_width)
    plt.bar(x=xposition+2*bar_width,
            height=errorRateInformationLinear,
            label='Ours',
            color='r',
            align='center',
            width=bar_width)
    plt.xticks(xposition+bar_width,xtext,rotation=60)
    plt.legend()
    plt.savefig("./result/"+name+".png")
    plt.show()
    plt.close()
elif Configure.plot=='hot':
    fig=plt.figure()
    ax=fig.add_subplot(121)
    ax.set_title('Compare to Convex')
    ax.set_xticks(range(len(Configure.errorParameter)))
    ax.set_xticklabels([str(i) for i in Configure.errorParameter],rotation=60)
    ax.set_yticks(range(len(Configure.dataParameter)))
    ax.set_yticklabels([str(i) for i in Configure.dataParameter])
    improveConvex=np.array(improveConvex)
    improveConvex=improveConvex.reshape(len(Configure.dataParameter),len(Configure.errorParameter))
    im=ax.imshow(improveConvex,cmap=plt.cm.hot_r)
    plt.colorbar(im)
    im.set_clim(vmin=-0.1, vmax=0.5)
    ax = fig.add_subplot(122)
    ax.set_title('Compare to Greedy')
    ax.set_xticks(range(len(Configure.errorParameter)))
    ax.set_xticklabels([str(i) for i in Configure.errorParameter],rotation=60)
    ax.set_yticks(range(len(Configure.dataParameter)))
    ax.set_yticklabels([str(i) for i in Configure.dataParameter])
    improveGreedy = np.array(improveGreedy)
    improveGreedy=improveGreedy.reshape(len(Configure.dataParameter), len(Configure.errorParameter))
    im = ax.imshow(improveGreedy,cmap=plt.cm.hot_r)
    im.set_clim(vmin=-0.1,vmax=0.5)
    plt.colorbar(im)
    plt.savefig("./result/" + name + "_hot.png")
    viz.matplot(plt)
    plt.show()
    plt.close()