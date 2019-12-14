import os
import re
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from matplotlib.backends.backend_pdf import PdfPages

#fpath = '../newlogs4'

appnum = 20         #一共迭代几轮
inputfilenum = 32   #输入文件的个数
select_appnum = 20   #选前几轮画图

def cal_map_interval(fpath, appnum=appnum, inputfilenum=inputfilenum, select_appnum=select_appnum):
    file_lst = []
    map_start_time_lst = []
    reduce_start_time_lst = []
    map_end_time_lst = []
    reduce_end_time_lst = []
    
    for root, dirs, files in os.walk(fpath, topdown=False):
        if 'syslog' in files:
            matchObj = re.match(".*container.*_0000(.*)",root)
            num = int(matchObj.group(1))
            if num != 1 and num <= inputfilenum+1:
                file_lst.append(root+'/syslog')
        if 'stdout' in files:
            file_lst.append(root+'/stdout')
    for filepath in file_lst:
        f = open(filepath, 'r', encoding = 'utf-8')
        lines = f.readlines()
        f.close()
        try:
            for line in lines:
                matchObj = re.match("(.*) (.*),(.*) INFO \[main\] org.apache.hadoop.security.SecurityUtil: Updating Configuration", line)
                matchObj2 = re.match("Time: (.*) (.*) (.*) endMap ", line)
                if matchObj:
                    date = matchObj.group(1).split('-')
                    time = matchObj.group(2).split(':')
                    ms = matchObj.group(3)
                    timestamp = int(date[2])*24*3600 + int(time[0])*3600 + int(time[1])*60 + int(time[2]) + int(ms)*0.001
                    map_start_time_lst.append(timestamp)
                    #if(timestamp > 845697): 
                    #    print(filepath)
                if matchObj2:
                    date = matchObj2.group(1).split('-')
                    time = matchObj2.group(2).split(':')
                    ms = matchObj2.group(3)
                    timestamp = int(date[2])*24*3600 + int(time[0])*3600 + int(time[1])*60 + int(time[2]) + int(ms)*0.001
                    map_end_time_lst.append(timestamp)        
        except ValueError as Argument:
            print(Argument, "filepath:", filepath)
        except AttributeError:
            print('line:',line)
                    
    map_start_time_lst.sort()
    map_end_time_lst.sort()
    reduce_start_time_lst.sort()
    reduce_end_time_lst.sort()
    
    map_start_time_lst = map_start_time_lst[:select_appnum*inputfilenum]
    map_end_time_lst = map_end_time_lst[:select_appnum*inputfilenum]
    
    starttime = map_start_time_lst[0]    
    map_start_time_lst = [i-starttime for i in map_start_time_lst]
    map_end_time_lst = [i-starttime for i in map_end_time_lst]
    reduce_start_time_lst = [i-starttime for i in reduce_start_time_lst]
    reduce_end_time_lst = [i-starttime for i in reduce_end_time_lst]
    
    map_start_interval = [map_start_time_lst[i+inputfilenum-1]-map_start_time_lst[i] for i in range(0, len(map_start_time_lst), inputfilenum)]
    map_end_interval = [map_end_time_lst[i+inputfilenum-1]-map_end_time_lst[i] for i in range(0, len(map_end_time_lst), inputfilenum)]
    return np.array(map_start_interval), np.array(map_end_interval)
    
fpath_lst = ['../p_logs2', '../p_logs4','../p_logs5', '../p_logs6']

map_start_interval_lst, map_end_interval_lst = [], []
for fpath in fpath_lst:
    s_itv, e_itv = cal_map_interval(fpath)
    map_start_interval_lst.append(s_itv)
    map_end_interval_lst.append(e_itv)

xposition1 = np.arange(1.2,5)
xposition2 = np.arange(1.5,5.5)
barwidth = 0.25

for i in range(len(map_start_interval_lst)):
    if i == 0:
        plt.bar(xposition1[i], map_start_interval_lst[i].mean(), yerr = sqrt(map_start_interval_lst[i].var()), color='b', label='Triggering StartUps', width=barwidth)
    else:
        plt.bar(xposition1[i], map_start_interval_lst[i].mean(), yerr = sqrt(map_start_interval_lst[i].var()), color='b', width=barwidth)
for i in range(len(map_end_interval_lst)):
    if i == 0:
        plt.bar(xposition2[i], map_end_interval_lst[i].mean(), yerr = sqrt(map_end_interval_lst[i].var()), color='r', label='Aggregating Updates', width=barwidth)
    else:
        plt.bar(xposition2[i], map_end_interval_lst[i].mean(), yerr = sqrt(map_end_interval_lst[i].var()), color='r', width=barwidth)

fontx={'size':21, 'weight':'normal'}
fonty={'size':23, 'weight':'normal'}
plt.xlabel("Average Size of KMeans Inputs",fontx)
plt.ylabel('Durations (s)',fonty)
plt.xticks(xposition1+barwidth/2+0.02,labels=(['2M','4M','8M','16M']),fontsize=20)
plt.yticks(fontsize=22)
plt.legend(fontsize = 16)
plt.tight_layout()


pdf = PdfPages('../time interval5.pdf')
pdf.savefig()
pdf.close()

plt.show()
plt.clf()
















