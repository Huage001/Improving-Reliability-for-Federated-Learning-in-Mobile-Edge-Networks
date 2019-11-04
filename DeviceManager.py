import numpy as np
import copy

class DeviceManager:

    def getComputeDelayDistribution(self):
        '''
        Read the compute abilities of all devices from input file
        But now, we only use Gaussian distribution
        :return: A ndarray of float numbers, with shape (self.deviceNum,)
        '''
        return np.random.rand(self.deviceNum)
        #return np.array([1,1,1])

    def getDataSizeDistribution(self):
        '''
        Similar to the function above
        :return: A ndarray of float numbers, with shape (self.deviceNum,)
        '''
        return np.random.rand(self.deviceNum)
        #return np.array([1,1,1])

    def getIsAliveDistribution(self):
        '''
        Similar to the function above
        :return: A ndarray of {0,1}, with shape (self.deviceNum,)
        '''
        res=np.random.randint(0,2,self.deviceNum)
        return res
        # res=[]
        # for i in range(self.deviceNum):
        #     res.append(1)
        # return np.array(res)

    def __init__(self,deviceNum):
        self.deviceNum=deviceNum
        self.allComputeDelay=self.getComputeDelayDistribution()
        self.isAlive=None
        self.dataSize=None
        self.allDataSize=None
        self.computeDelay=None
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
        self.computeDelay=self.allComputeDelay[self.isAlive.astype(bool)]
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

    def getComputeDelay(self):
        '''
        :return: a list of the ComputeDelay of each device
        '''
        return self.computeDelay
