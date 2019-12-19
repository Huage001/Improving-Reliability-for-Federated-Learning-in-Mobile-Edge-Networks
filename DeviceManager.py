import copy

import numpy as np
from scipy.stats import f

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
            for i in range(self.deviceNum):
                if res[i]<0:
                    res[i]=0.001
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