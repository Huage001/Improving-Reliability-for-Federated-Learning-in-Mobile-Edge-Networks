# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 09:09:07 2019

@author: -
"""

import random
import numpy as np

foldername = 'input5'
filenum = 40



for num in range(21, filenum+1):
    f = open('../'+foldername+'/Instance{}.txt'.format(num),'w')
    print('正在生成第{}个文件...'.format(num))
    for it in range(int(1000000*random.uniform(0.5,1.5))):
        lst = [random.randint(0,1000) for i in range(2)]
        s = str(lst).lstrip("[").replace(" ", '').replace(']', '\n')
        f.write(s)
    f.close()
print('文件生成结束!')