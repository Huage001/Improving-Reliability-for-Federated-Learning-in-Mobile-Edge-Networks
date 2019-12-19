import numpy as np
import matplotlib.pyplot as plt
import copy
from matplotlib.backends.backend_pdf import PdfPages

xposition1 = np.arange(3)
xposition2 = np.arange(4,7)
xposition3 = np.arange(8,11)
bar_width=0.25
#y_axis_range = np.arange(0.0,0.8,0.2)

fname = './result/figure6.txt'

file=open(fname, 'r')
lines=list(file.readlines())
temp_file=open('temp_file.txt','w')
for line in lines:
    new_line=line[1:len(line)-2]
    print(new_line,file=temp_file)
temp_file.close()
data1=np.loadtxt('temp_file.txt',delimiter=',')

data=copy.copy(data1)
data[:,0:3]=data1[:,6:9]
data[:,6:9]=data1[:,0:3]

print(fname)

plt.bar(x=xposition2, height=data[2][3:6], label='SLSQP', color='b',align='center',width=bar_width)
plt.bar(x=xposition2+bar_width, height=data[1][3:6], label='GDS', color='g',align='center',width=bar_width)
plt.bar(x=xposition2+2*bar_width, height=data[0][3:6], label='ranRFL', color='r',align='center',width=bar_width)

plt.bar(x=xposition3, height=data[2][6:9], color='b',align='center',width=bar_width)
plt.bar(x=xposition3+bar_width, height=data[1][6:9],color='g',align='center',width=bar_width)
plt.bar(x=xposition3+2*bar_width, height=data[0][6:9],color='r',align='center',width=bar_width)

plt.bar(x=xposition1, height=data[2][0:3], color='b',align='center',width=bar_width)
plt.bar(x=xposition1+bar_width, height=data[1][0:3],color='g',align='center',width=bar_width)
plt.bar(x=xposition1+2*bar_width, height=data[0][0:3],color='r',align='center',width=bar_width)

fontx={'size':22, 'weight':'normal'}
fonty={'size':23, 'weight':'normal'}
plt.ylabel('Average Error Rate',fonty)
# plt.xticks(list(xposition1+bar_width)+list(xposition2+bar_width)+list(xposition3+bar_width),labels=(
#     '$[Z_S,E_S]$','$[Z_S,E_M]$','$[Z_S,E_L]$','$[Z_M,E_S]$','$[Z_M,E_M]$','$[Z_M,E_L]$',
#     '$[Z_L,E_S]$','$[Z_L,E_M]$','$[Z_L,E_L]$'),fontsize=10,rotation=30)
plt.xticks(list(xposition1+bar_width)+list(xposition2+bar_width)+list(xposition3+bar_width),labels=(
    '$[F_S,E_S]$','$[F_S,E_M]$','$[F_S,E_L]$','$[F_M,E_S]$','$[F_M,E_M]$','$[F_M,E_L]$',
    '$[F_L,E_S]$','$[F_L,E_M]$','$[F_L,E_L]$'),fontsize=10,rotation=30)
plt.xlabel('[F Skewness, Error Rate of Devices]',fontx)
plt.yticks(fontsize=22)
plt.legend(fontsize=16)
plt.tight_layout()
#plt.gca().axes.get_yaxis().set_visible(False)    #隐藏y轴

# pdf=PdfPages('./result/pdf/graph_bar_zipf.pdf')
# pdf.savefig()
# pdf.close()

plt.savefig('./result/pdf/graph_bar_f1.pdf',bbox_inches='tight')

plt.show()