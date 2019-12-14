# README

## 记录所有输入的数据分布

这里可以先不看，直接看第二部分参数就可

| 文件名     | 日志名     | 数据条数均值 | 数据分布 | 参数    | 文件数量 | 核数，单个map/reduce内存 |
| ---------- | ---------- | ------------ | -------- | ------- | -------- | ------------------------ |
| input2     | logs2      | 300000       | uniform  | 0.5-1.5 | /        | /核数不够，有两段折线    |
| input3     | logs3      | 300000       | uniform  | 0.8-1.2 | /        | /核数不够，有两段折线    |
| *input2    | p_logs2    | 300000       | uniform  | 0.5-1.5 | 32       | 1024                     |
| input4     | *newlogs4* | 500000       | uniform  | 1       | 20       |                          |
| *newinput4 | p_logs4    | 500000       | uniform  | 0.5-1.5 | 32       |                          |
| *input5    | p_logs5    | 1000000      | uniform  | 0.5-1.5 | 32       |                          |
| *input6    | p_logs6    | 2000000      | uniform  | 0.5-1.5 | 32       |                          |

蓝线为开始，红线为结束

## 参数

yarn.nodemanager.resource.cpu-vcores       最大可用核数：38

yarn.nodemanager.resource.memory-mb    节点最大可用内存：50GB（虚拟内存率：4）

yarn.scheduler.maximum-allocation-vcores 单个AM容器可申请的最大核数：38

yarn.scheduler.maximum-allocation-mb       单个AM容器可申请的最大内存：50GB

mapreduce.map.memory.mb		 map container可用内存：1024MB

mapreduce.map.cpu.vcores		    map container可用核数：1

mapreduce.reduce.memory.mb     reduce container可用内存：1024MB

mapreduce.reduce.cpu.vcores        reduce container可用核数：1

平均输入文件大小：2/4/8/16MB

文件大小的数据分布：uniform(0.5,1.5)

并行数（文件个数）：32

迭代轮数：20

聚类个数： k=5

