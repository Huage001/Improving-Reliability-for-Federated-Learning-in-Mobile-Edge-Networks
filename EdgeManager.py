import numpy as np

class EdgeManager:

    def getTransmitDelayDistribution(self):
        '''
        Read the transmitDelay of all edges from input file
        But now, we only use Gaussian distribution
        Note that the last edge is so-called virtual edge
        so its transmit delay should be 0
        :return: A ndarray of float numbers, with shape (self.edgeNum,)
        '''
        res=np.random.rand(self.edgeNum)
        res=np.append(res,0)
        # res=np.array([1,1,0])
        return res

    def getUserCapacityDistribution(self):
        '''
        Similar to the function above
        Note that the last edge is so-called virtual edge
        so its user capacity should be inf
        :return: A ndarray of float numbers, with shape (self.edgeNum,)
        '''
        res=np.random.rand(self.edgeNum)*20
        res=np.append(res,0)
        # res=np.array([1,2,0])
        return res

    def __init__(self,edgeNum):
        self.edgeNum=edgeNum
        self.transmitDelay=self.getTransmitDelayDistribution()
        self.userCapacity=self.getUserCapacityDistribution()

    def getAllUserCapacity(self):
        '''
        :return: A list of the userCapacity of each edge
        '''
        return self.userCapacity

    def getTransmitDelay(self):
        '''
        :return: A list of the transmitDelay of each edge
        '''
        return self.transmitDelay
