from DeviceManager import DeviceManager
from Frame import Frame
import Configure

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

if Configure.xAxis=='single':
    worker = Frame()
    worker.devices = DeviceManager(Configure.allDeviceNum,Configure.dataParameter,Configure.errorParameter,
                            Configure.dataDistribution,Configure.errorDistribution,Configure.relate)
    worker.allDeviceNum = Configure.allDeviceNum
    worker.devices.setDataSizeDistribution()
    worker.deviceNum = int(worker.devices.setIsAliveDistribution())
    worker.isAvail = worker.devices.getIsAlive()
    worker.ratio = Configure.dataRatio
    print(worker.convexOptimizer())
    print(worker.linearOptimizer())
    print(worker.greedy())