import numpy as np

times=30
standard='ave' # ave or max
mulThread=0
relate=False


#-------------Configuration 1: x axis is selected data ratio-------------#

# xAxis='dataRatio'
# plot='scatter'
#
# minRatio=0.1
# maxRatio=0.9
# cases=20
# dataRatio=np.linspace(minRatio,maxRatio,cases).tolist()
#
# allDeviceNum=25
#
# errorDistribution='normal'
# errorParameter=[0.2,0.1]
# dataDistribution='zipf'
# dataParameter=3

#-----------------Configuration 2: x axis is device number---------------#

xAxis='deviceNum'
plot='scatter'

minDeviceNum=3
maxDeviceNum=40
stride=1
allDeviceNum=range(minDeviceNum,maxDeviceNum+1,stride)

dataRatio=0.5

errorDistribution='zipf'# normal, f, or zipf
errorParameter=2
dataDistribution='zipf'
dataParameter=2

#------Configuration 3: x axis is different distribution parameters------#

# xAxis='distribution'
# # plot='bar'
# plot='hot'
#
# allDeviceNum=25
#
# dataRatio=0.5
#
# dataDistribution='zipf'
# errorDistribution='normal'
#
# # dataParameter=[[1,1],[2,1],[3,1]]
# dataParameter=[]
# for i in np.linspace(1.1,2.5,15):
#     dataParameter.append(i)
#
# # errorParameter=[[0.03,0.1],[0.05,0.1],[0.07,0.1]]
# errorParameter=[]
# for i in np.linspace(0.01,0.15,15):
#     errorParameter.append([i,0.1])

#------Configuration 4: we only do a single experiment------#

# xAxis='single'
# plot='scatter'
# allDeviceNum=100
# dataRatio=0.8
# dataDistribution='normal'
# errorDistribution='normal'
# dataParameter=[1,1]
# errorParameter=[0.05,0.01]