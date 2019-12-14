import os
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

#fpath = '../newlogs4'

appnum = 20         #一共迭代几轮
inputfilenum = 32   #输入文件的个数
select_appnum = 20   #选前几轮画图

def cal_map_trans_timespace(fpath, appnum=appnum, inputfilenum=inputfilenum, select_appnum=select_appnum):
    file_lst = []
    map_start_time_lst = []
    map_end_time_lst = []
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
    map_start_time_lst = map_start_time_lst[:select_appnum*inputfilenum]
    map_end_time_lst = map_end_time_lst[:select_appnum*inputfilenum]
    
    starttime = map_start_time_lst[0]    
    map_start_time_lst = [i-starttime for i in map_start_time_lst]
    map_end_time_lst = [i-starttime for i in map_end_time_lst]

    map_start_trans_timespace = [map_start_time_lst[i]-map_start_time_lst[i-1] for i in range(1, len(map_start_time_lst)) if i % inputfilenum != 0]
    map_end_trans_timespace = [map_end_time_lst[i+1]-map_end_time_lst[i] for i in range(0, len(map_end_time_lst)-1)]
    
    return np.array(map_start_trans_timespace), np.array(map_end_trans_timespace)

fpath_lst = ['../p_logs2','../p_logs4','../p_logs5','../p_logs6']
label_lst = ['2M','4M','8M','16M']

map_start_trans_timespace, map_end_trans_timespace = [], []

for i in range(len(fpath_lst)):
    s_itv, e_itv = cal_map_trans_timespace(fpath_lst[i])
    map_start_trans_timespace.append(s_itv)
    map_end_trans_timespace.append(e_itv)


plt.xlim([0,0.6])
plt.ylim([0.6,1.02])
for i in range(len(fpath_lst)):
    #plt.hist(map_start_trans_timespace[i], bins=10000, cumulative=True, normed=True, histtype = 'step',label=fpath_lst[i])
    hist, bin_edge = np.histogram(map_start_trans_timespace[i], bins=10000, density=True)
    for j in range(1, len(hist)):
        hist[j] += hist[j-1]
    plt.plot(bin_edge[:-1], hist/hist[-1], label='Avg. Size of KMeans Inputs:'+label_lst[i])
plt.xticks(np.arange(0,0.7,0.1),fontsize=20)
plt.yticks(np.arange(0.6,1.02,0.1),fontsize=20)
fontx={'size':20, 'weight':'normal'}
fonty={'size':23, 'weight':'normal'}
plt.xlabel('Lag between Concecutive StartUps (s)',fontx)
plt.ylabel('CDF',fonty)
plt.annotate("Over 92% of lags\n are less than 0.1s.", (0.1,0.925), xycoords='data',xytext = (0.17,0.85), fontsize = 17, arrowprops=dict(arrowstyle='->'))
plt.legend(fontsize=14,loc = 'lower right')
#plt.grid()

plt.tight_layout()
plt.savefig('../1.png')


pdf = PdfPages('../cdf4.pdf')
pdf.savefig()
pdf.close()

plt.show()
plt.clf()












