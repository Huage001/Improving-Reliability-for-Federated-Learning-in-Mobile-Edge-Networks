
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

fpath = '../p_logs6'
savepath = '../newlogs2pictures/'
appnum = 20         #一共迭代几轮
inputfilenum = 32   #输入文件的个数
select_appnum = 6   #选前几轮画图

for timespace in range(8, 9):   #要选的统计时间间隔的范围（调参

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
    mapendtime0 = map_end_time_lst[0]
    
    map_start_time_lst = [i-starttime for i in map_start_time_lst]
    map_end_time_lst = [i-starttime for i in map_end_time_lst]
    reduce_start_time_lst = [i-starttime for i in reduce_start_time_lst]
    reduce_end_time_lst = [i-starttime for i in reduce_end_time_lst]
    
    space_num = int(max(map_end_time_lst))//timespace + 1
    map_start_count_lst = [0] * (space_num + 1)
    map_end_count_lst = [0] * (space_num + 1)
    i = 1
    for item in map_start_time_lst:
        while i <= space_num:
            if item < timespace * i:
                map_start_count_lst[i] += 1
                break
            else:
                i += 1
    i = 1
    for item in map_end_time_lst:
        while i <= space_num:
            if item < timespace * i:
                map_end_count_lst[i] += 1
                break
            else:
                i += 1
    x = [i*timespace for i in range(0, space_num+1)]
    
    del0_start_lst = [i for i in zip(x, map_start_count_lst) if i[1] != 0]
    del0_end_lst = [i for i in zip(x, map_end_count_lst) if i[1] != 0]

    fontx={'size':21, 'weight':'normal'}
    fonty={'size':18, 'weight':'normal'}

    plt.xlabel("Time(s)", fontx)
    plt.ylabel("Concurrent Transmissions", fonty)
    plt.plot(x, map_start_count_lst, c='b', label = 'Iterative Starts')
    plt.scatter([i[0] for i in del0_start_lst], [i[1] for i in del0_start_lst], c='b', marker = 'x')
    plt.xticks(fontsize = 18)
    plt.yticks(np.arange(0,40,5),fontsize = 22)
    plt.plot(x, map_end_count_lst, c='red', label = 'Iterative Updates')
    plt.scatter([i[0] for i in del0_end_lst], [i[1] for i in del0_end_lst], c='red', marker = '^')
    plt.tight_layout()
    
    #保存png
    #plt.title("file:{} timespace:{}".format(fpath[3:], timespace))
    #plt.savefig(savepath+fpath[3:]+',space '+str(timespace))
    
    #保存pdf
    #plt.legend(bbox_to_anchor=(-0.02,0.98),ncol = 2, loc = 'lower left', prop={'size':14, 'weight':'normal'})
    #pdf = PdfPages('logs2 timespace4.pdf')
    #pdf.savefig()
    #pdf.close()
    
    #plt.show()
    #plt.clf()



    #plt.legend(bbox_to_anchor=(0., 0.85, 1.01, .35),ncol = 2, loc = 'upper right', prop={'size':14, 'weight':'normal'})
    '''
    del0_start_lst = []
    del0_end_lst = []
    for item in zip(x, map_start_count_lst):
        if item[1] != 0:
            del0_start_lst.append(item)
    for item in zip(x, map_end_count_lst):
        if item[1] != 0:
            del0_end_lst.append(item)
    '''


'''
    for filepath in file_lst:
        f = open(filepath, 'r', encoding = 'utf-8')
        lines = f.readlines()
        f.close()
        try:
            for line in lines:
                matchObj = re.match("Time: (.*) (.*) (.*) (.*) ", line)
                if matchObj:
                    date = matchObj.group(1).split('-')
                    time = matchObj.group(2).split(':')
                    ms = matchObj.group(3)
                    startend = matchObj.group(4)
                    timestamp = int(date[2])*24*3600 + int(time[0])*3600 + int(time[1])*60 + int(time[2]) + int(ms)*0.001
                    if startend == 'startMap':
                        map_start_time_lst.append(timestamp)
                    elif startend == 'endMap':
                        map_end_time_lst.append(timestamp)
                    elif startend == 'startReduce':
                        reduce_start_time_lst.append(timestamp)
                    elif startend == 'endReduce':
                        reduce_end_time_lst.append(timestamp)
        except ValueError as Argument:
            print(Argument, "filepath:", filepath)
'''







