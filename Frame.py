import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import linprog
import copy
import Configure

edgeNum=2
name = Configure.xAxis + \
       "_" + str(Configure.allDeviceNum) + \
       "_" + str(Configure.dataRatio) + \
       "_" + str(Configure.errorDistribution) + \
       "_" + str(Configure.errorParameter) + \
       "_" + str(Configure.dataDistribution) + \
       "_" + str(Configure.dataParameter)

if Configure.standard=='max':
    name+="_max"
if Configure.cases!=None:
    name+=Configure.cases
if Configure.relate:
    name+="_relate"

fi=open("./result/select_devices/"+name+'.txt',"w")
id=1

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
        global id
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
            y=y/sum(y)
            plt.scatter(x,y,color='gray',label='unselected',marker='o')
            plt.scatter(x[finalDecision.flatten().astype(bool)],y[finalDecision.flatten().astype(bool)],
                        color='r',label='selected',marker='o')
            plt.savefig('./result/select_devices/'+name+'_figure'+str(id)+'.png')
            id+=1
            plt.show()
            plt.close()
            print(x,file=fi)
            print(y,file=fi)
            print(finalDecision.flatten(),file=fi)

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