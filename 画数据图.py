# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 18:56:57 2019

@author: lenovo
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from scipy.optimize import curve_fit
from matplotlib.backends.backend_pdf import PdfPages

#df = pd.read_csv('全数据1.csv', engine = 'python')
protocol_lst = ['tcp', 'udp', 'icmp']
service_lst = list(set(df['service']))

def relation(df, protocol_type:str, serv:str, count_type:str, error_type:str):
    df = df[(df['protocol_type'] == protocol_type) & (df['service'] == serv)]
    result = {}
    for i in range(len(df)):
        count0 = df.iloc[i][count_type]
        error_rate = df.iloc[i][error_type]
        if count0 not in result.keys():
            result[count0] = [error_rate]
        else:
            result[count0].append(error_rate)
    ave_error_rate = []
    for item in result:
        ave_error_rate.append(sum(list(result[item]))/len(result[item]))
    r = np.array(list(result.values()))
    var = [np.std(item)**2 for item in r]
    if len(result) != 0:
        plt.title(protocol_type+" "+serv+" "+count_type+" "+error_type)
        #plt.scatter(result.keys(), ave_error_rate)
        #plt.bar(result.keys(), ave_error_rate, yerr = var)
        #plt.savefig("全数据误差图/"+protocol_type+" "+serv+" "+count_type+" "+error_type)
        #plt.show()
        plt.clf()
    return list(result.keys()), ave_error_rate,var

def relation_minmax_ave(df, protocol_type:str, serv:str, count_type:str, error_type:str):
    df = df[(df['protocol_type'] == protocol_type) & (df['service'] == serv)]
    result = {}
    for i in range(len(df)):
        count0 = df.iloc[i][count_type]
        error_rate = df.iloc[i][error_type]
        if count0 not in result.keys():
            result[count0] = [error_rate]
        else:
            result[count0].append(error_rate)
    ave_error_rate = []
    for item in result:
        ave_error_rate.append((max(list(result[item])) + min(list(result[item])))/ 2)
    if len(result) != 0:
        plt.title(protocol_type+" "+serv+" "+count_type+" "+error_type)
        plt.scatter(result.keys(), ave_error_rate)
        plt.savefig("最大最小平均数据图/"+protocol_type+" "+serv+" "+count_type+" "+error_type)
        #plt.show()
        plt.clf()
    return

def relation_box(df, protocol_type:str, serv:str, count_type:str, error_type:str):
    df = df[(df['protocol_type'] == protocol_type) & (df['service'] == serv)]
    result = {}
    for i in range(len(df)):
        count0 = df.iloc[i][count_type]
        error_rate = df.iloc[i][error_type]
        if count0 not in result.keys():
            result[count0] = [error_rate]
        else:
            result[count0].append(error_rate)
    if len(result) != 0:
        #把连接数补全，未出现的连接数补全为0
        for i in range(max(result.keys())):
            if i not in result.keys():
                result[i] = []
        boxdata = [x[1] for x in sorted(result.items())]
        #print(boxdata)
        plt.title(protocol_type+" "+serv+" "+count_type+" "+error_type)
        #plt.boxplot(boxdata)
        
        plt.savefig("全数据box图/"+protocol_type+" "+serv+" "+count_type+" "+error_type)
        #plt.show()
        plt.clf()
    return

"""
#创建全数据dataframe
df1 = pd.read_csv('unsw-nb15/UNSW-NB15_1.csv')
df2 = pd.read_csv('unsw-nb15/UNSW-NB15_2.csv')
df3 = pd.read_csv('unsw-nb15/UNSW-NB15_3.csv')
df4 = pd.read_csv('unsw-nb15/UNSW-NB15_4.csv')
ori = list(df2.columns.values)
target = list(df1.columns.values)
df2 = df2.rename(columns = dict(zip(ori,target)))
ori = list(df3.columns.values)
df3 = df3.rename(columns = dict(zip(ori,target)))
ori = list(df4.columns.values)
df4 = df4.rename(columns = dict(zip(ori,target)))
df = pd.concat([df1,df2,df3,df4])
"""

#unsw数据集相关分析
"""
proto_lst1 = list(df.proto)
l = len(proto_lst1)
for i in range(l):
    if proto_lst1[i] == 'a/n':
        proto_lst1[i] = 'a-n'
    if proto_lst1[i] == 'ax.25':
        proto_lst1[i] = 'ax-25'
df.proto = proto_lst1
proto_lst = sorted(list(set(df['proto'])))
serv_lst = sorted(list(set(df['service'])))
count_type_lst = ['ct_srv_src','ct_srv_dst','ct_dst_ltm','ct_src_ ltm','ct_src_dport_ltm','ct_dst_sport_ltm','ct_dst_src_ltm']
"""
def unsw_relation(df, protocol_type:str, serv:str, count_type:str, error_type:str):
    df = df[(df['proto'] == protocol_type) & (df['service'] == serv)]
    #df = df[(df['proto'] == protocol_type) & (df['service'] == serv) & (df['is_ftp_login'] == 1)]
    result = {}
    """ for i in range(len(df)):
        count0 = df.iloc[i][count_type]
        error_rate = df.iloc[i][error_type]
        if count0 not in result.keys():
            result[count0] = [error_rate]
        else:
            result[count0].append(error_rate)"""
    delay_lst = list(df[error_type])
    count_lst = list(df[count_type])
    for i in range(len(count_lst)):
        if count_lst[i] not in result.keys():
            result[count_lst[i]] = [delay_lst[i]]
        else:
            result[count_lst[i]].append(delay_lst[i])
    ave_error_rate = []
    for item in result:
        ave_error_rate.append(sum(list(result[item]))/len(result[item]))
    r = np.array(list(result.values()))
    var = [np.std(item)**2 for item in r]
    if len(result) != 0:
        plt.title(protocol_type+" "+serv+" "+count_type+" "+error_type)
        #plt.scatter(result.keys(), ave_error_rate)
        plt.bar(result.keys(), ave_error_rate, yerr = var)
        plt.savefig("unsw误差图/"+protocol_type+" "+serv+" "+count_type+" "+error_type)
        #plt.show()
        plt.clf()
    return

def unsw_relation_pkg(df, protocol_type:str, serv:str, count_type:str, loss_pkg:str, total_pkg:str):
    df = df[(df['proto'] == protocol_type) & (df['service'] == serv) & (df[total_pkg] != 0)]
    result = {}
    """for i in range(len(df)):
        count0 = df.iloc[i][count_type]
        error_rate = df.iloc[i][loss_pkg]/df.iloc[i]
        if count0 not in result.keys():
            result[count0] = [error_rate]
        else:
            result[count0].append(error_rate)"""
    loss_pkg_lst = np.array(df[loss_pkg])
    total_pkg_lst = np.array(df[total_pkg])
    
    count_lst = list(df[count_type])
    error_rate_lst = loss_pkg_lst / total_pkg_lst
    for i in range(len(count_lst)):
        if count_lst[i] not in result.keys():
            result[count_lst[i]] = [error_rate_lst[i]]
        else:
            result[count_lst[i]].append(error_rate_lst[i])
    ave_error_rate = []
    for item in result:
        ave_error_rate.append(sum(list(result[item]))/len(result[item]))
    r = np.array(list(result.values()))
    var = [np.std(item)**2 for item in r]
    if len(result) != 0:
        plt.title(protocol_type+" "+serv+" "+count_type+" "+ loss_pkg)
        #plt.scatter(result.keys(), ave_error_rate)
        plt.bar(result.keys(), ave_error_rate, yerr = var)
        plt.savefig("unsw误差图pkg/"+protocol_type+" "+serv+" "+count_type+" "+ loss_pkg)
        #plt.show()
        plt.clf()
    return


"""---------------------------------画论文里的图-----------------------------------"""

"""-----------smtp srv_count srv_serror_rate--------------"""

x, y, var = relation(ori_df,'tcp','telnet','srv_count','srv_serror_rate')
points = sorted(list(zip(x,y,var)))
x = np.array([item[0] for item in points])
y = np.array([item[1] for item in points])
var = np.array([item[2] for item in points])


"""-----------bar图-----------"""
'''
pdf = PdfPages('telnet-bar.pdf')
#plt.figure(figsize = (7,5))
fontx = {'size':21, 'weight':'normal'}
fonty = {'size':23, 'weight':'normal'}
plt.xlabel("Number of Concurrent Transmissions",fontx)
plt.ylabel('Error Rate',fonty)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.bar(x, y, yerr = var, label = "Error rate")
#plt.plot([0],[0],color = 'black',label = "Var of error rate")
#plt.legend(fontsize = 12)
plt.tight_layout()
pdf.savefig()
#plt.savefig('111')
pdf.close()
'''

"""----------散点图-----------"""
"""pdf = PdfPages('smtp-scatter1.pdf')
#plt.figure(figsize = (7,5))
font = {'size':18, 'weight':'normal'}
plt.xlabel("Connection numbers",font)
#plt.xlabel("number of connections to the same service as the current connection in the past two seconds")
#plt.ylabel('Percentage of connections that have "SYN" errors',font)
#plt.ylabel('% of connections that have "SYN" errors',font)
plt.ylabel('"SYN" error rate',font)
#plt.ylabel('percentage of "SYN" errors',font)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.bar(x, y, yerr = var, label = "Error rate")
#plt.plot([0],[0],color = 'black',label = "Var of error rate")
#plt.legend(fontsize = 12)
#plt.tight_layout()
#pdf.savefig()
#plt.savefig('111')
pdf.close()"""

"""-----------xxx srv_count srv_serror_rate--------------"""
"""x, y, var = relation(ori_df,'tcp','pop_3','srv_count','srv_serror_rate')
points = sorted(list(zip(x,y,var)))
x = np.array([item[0] for item in points])
y = np.array([item[1] for item in points])
var = np.array([item[2] for item in points])

pdf = PdfPages('pop_3-bar.pdf')
#plt.figure(figsize = (7,5))
fontx = {'size':21, 'weight':'normal'}
fonty = {'size':23, 'weight':'normal'}
plt.xlabel("Number of Concurrent Connections",fontx)
plt.ylabel('Error Rate',fonty)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.bar(x, y, yerr = var, label = "Error rate")
#plt.plot([0],[0],color = 'black',label = "Var of error rate")
#plt.legend(fontsize = 12)
plt.tight_layout()
pdf.savefig()
#plt.savefig('222')
pdf.close()
"""


"""-----------------------------------拟合曲线------------------------------------"""

x, y, var = relation(ori_df,'tcp','smtp','srv_count','srv_serror_rate')
def func(x, a, b, c):
    return a / (b + np.exp(c-x))
def func2(x,a,b,c,d):
    return (a*d) / (1+(x/c)**b) + d
def func3(x,a,b,c):
    return (1-a*np.exp(-x)) / (b + c * np.exp(-x))
points = sorted(list(zip(x,y,var)))
x = np.array([item[0] for item in points])
y = np.array([item[1] for item in points])
var = np.array([item[2] for item in points])

#非线性最小二乘法拟合
popt, pcov = curve_fit(func3, x, y)
#获取popt里面是拟合系数
print(popt)
a = popt[0]
b = popt[1]
c = popt[2]
#d = popt[3]
yvals = func3(x,a,b,c) #拟合y值
print('popt:', popt)
print('系数a:', a)
print('系数b:', b)
print('系数c:', c)
print('系数pcov:', pcov)
print('系数yvals:', yvals)
#绘图
pdf = PdfPages('smtp-fitting curve.pdf')
#plt.figure(figsize = (7,5))
fontx = {'size':21, 'weight':'normal'}
fonty = {'size':23, 'weight':'normal'}
plt.xlabel("Number of Concurrent Transmissions",fontx)
plt.ylabel('Error Rate',fonty)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plot1 = plt.plot(x, y, 's',label='Original values')
plot2 = plt.plot(x, yvals, 'r',label='Fitting curve')
plt.legend(fontsize = 14, loc=4) #指定legend的位置右下角
#plt.title('curve_fit')
#plt.show()
plt.tight_layout()
pdf.savefig()
pdf.close()



"""
fontx = {'size':20, 'weight':'normal'}
fonty = {'size':23, 'weight':'normal'}
plt.xlabel("Number of Concurrent Connections",fontx)
plt.ylabel('Error Rate',fonty)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
"""