# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 19:30:40 2019

@author: lenovo
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


fname = '1.txt'

file=open(fname, 'r')
lines=list(file.readlines())
temp_file=open('temp_file.txt','w')
for line in lines:
    new_line=line[1:len(line)-2]
    print(new_line,file=temp_file)
temp_file.close()
data=np.loadtxt('temp_file.txt',delimiter=',')

print(fname)

ours = data[0]
greedy = data[1]
SLSQP = data[2]

length = len(ours)
g_part_ave_up = 0
s_part_ave_up = 0

g_max_up = 0
g_min_up = 1
s_max_up = 0
s_min_up = 1
g_ave_up = 0
s_ave_up = 0

size = ["small","medium","large"]
param = [1,2,3]
for i in range(length//3):
    g_part_ave_up = (greedy[i] - ours[i]) / greedy[i]
    s_part_ave_up = (SLSQP[i] - ours[i]) / SLSQP[i]
    print("参数:{},".format(param[0])+size[i%3]+",相对于greedy提升:", g_part_ave_up)
    print("参数:{},".format(param[0])+size[i%3]+",相对于SLSQP提升:", s_part_ave_up)
    
for i in range(length//3,length//3*2):
    g_part_ave_up = (greedy[i] - ours[i]) / greedy[i]
    s_part_ave_up = (SLSQP[i] - ours[i]) / SLSQP[i]
    print("参数:{},".format(param[1])+size[i%3]+",相对于greedy提升:", g_part_ave_up)
    print("参数:{},".format(param[1])+size[i%3]+",相对于SLSQP提升:", s_part_ave_up)
    
for i in range(length//3 * 2, length):
    g_part_ave_up = (greedy[i] - ours[i]) / greedy[i]
    s_part_ave_up = (SLSQP[i] - ours[i]) / SLSQP[i]
    print("参数:{},".format(param[2])+size[i%3]+",相对于greedy提升:", g_part_ave_up)
    print("参数:{},".format(param[2])+size[i%3]+",相对于SLSQP提升:", s_part_ave_up)



'''
#这是旧的
for i in range(length//3):
    g_part_ave_up = (greedy[i] - ours[i]) / greedy[i]
    s_part_ave_up = (SLSQP[i] - ours[i]) / SLSQP[i]
    print("uniform分布,"+size[i%3]+",相对于greedy提升:", g_part_ave_up)
    print("uniform分布,"+size[i%3]+",相对于SLSQP提升:", s_part_ave_up)
    
for i in range(length//3,length//3*2):
    g_part_ave_up = (greedy[i] - ours[i]) / greedy[i]
    s_part_ave_up = (SLSQP[i] - ours[i]) / SLSQP[i]
    print("f分布,"+size[i%3]+",相对于greedy提升:", g_part_ave_up)
    print("f分布,"+size[i%3]+",相对于SLSQP提升:", s_part_ave_up)
    
for i in range(length//3 * 2, length):
    g_part_ave_up = (greedy[i] - ours[i]) / greedy[i]
    s_part_ave_up = (SLSQP[i] - ours[i]) / SLSQP[i]
    print("zip分布,"+size[i%3]+",相对于greedy提升:", g_part_ave_up)
    print("zip分布,"+size[i%3]+",相对于SLSQP提升:", s_part_ave_up)
'''
'''
for i in range(length):
    g_tmp_up = (greedy[i] - ours[i]) / greedy[i]
    s_tmp_up = (SLSQP[i] - ours[i]) / greedy[i]
    if(g_tmp_up > g_max_up): 
        g_max_up = g_tmp_up
    if(g_tmp_up < g_min_up):
        g_min_up = g_tmp_up
    if(s_tmp_up > s_max_up): 
        s_max_up = s_tmp_up
    if(s_tmp_up < s_min_up):
        s_min_up = s_tmp_up
    g_ave_up += g_tmp_up
    s_ave_up += s_tmp_up
g_ave_up /= length
s_ave_up /= length

print("全局,相对于greedy最小提升：",g_min_up)
print("全局,相对于greedy最大提升：",g_max_up)
print("全局,相对于greedy平均提升：",g_ave_up)
print("全局,相对于SLSQP最小提升：",s_min_up)
print("全局,相对于SLSQP最大提升：",s_max_up)
print("全局,相对于SLSQP平均提升：",s_ave_up)
'''
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    