import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

fname='plotData_data_distribution_new.txt'

file=open(fname, 'r')
lines=file.readlines()
temp_file=open('temp_file.txt','w')
for line in lines:
    new_line=line[1:len(line)-2]
    print(new_line,file=temp_file)
temp_file.close()
data=np.loadtxt('temp_file.txt',delimiter=',')
print(data[2][0:3])
# plt.scatter(np.linspace(0.1,0.7,20),data[0],color='r',label='ranRFL',marker='o',s=100)
# plt.scatter(np.linspace(0.1,0.7,20),data[1],color='g',label='GDS (%.2f%% higher than ours)'%29.58,marker='x',s=100)
# plt.scatter(np.linspace(0.1,0.7,20),data[2],color='b',label='SLSQP (%.2f%% higher than ours)'%22.11,marker='*',s=100)
#
# fontx={'size':22, 'weight':'normal'}
# fonty={'size':23, 'weight':'normal'}
# plt.xlabel("Proportion of selected data",fontx)
# plt.ylabel('Average Error Rate',fonty)
# plt.xticks(fontsize=22)
# plt.yticks(fontsize=22)
# plt.legend()
# plt.tight_layout()

xposition=np.arange(3)
bar_width=0.25
plt.bar(x=xposition, height=data[2][0:3], label='SLSQP', color='b',align='center',width=bar_width)
plt.bar(x=xposition+bar_width, height=data[1][0:3], label='GDS', color='g',align='center',width=bar_width)
plt.bar(x=xposition+2*bar_width, height=data[0][0:3], label='ranRFL', color='r',align='center',width=bar_width)
fontx={'size':22, 'weight':'normal'}
fonty={'size':23, 'weight':'normal'}
#plt.xlabel("Different Normal Distribution",fontx)
#plt.ylabel('Error Rate',fonty)
#plt.xticks([1.5],labels=('Normal',),fontsize=22)
#plt.yticks(fontsize=22)
plt.legend()
plt.tight_layout()
plt.subplot(1,3,1)
plt.bar(x=xposition, height=data[2][3:6], label='SLSQP', color='b',align='center',width=bar_width)
plt.bar(x=xposition+bar_width, height=data[1][3:6], label='GDS', color='g',align='center',width=bar_width)
plt.bar(x=xposition+2*bar_width, height=data[0][3:6], label='ranRFL', color='r',align='center',width=bar_width)
fontx={'size':22, 'weight':'normal'}
fonty={'size':23, 'weight':'normal'}
# plt.xlabel("Different F Distribution",fontx)
plt.ylabel('Error Rate',fonty)
plt.xticks([1.5],labels=('F',),fontsize=22)
plt.yticks(fontsize=22)
plt.legend()
plt.tight_layout()
plt.subplot(1,3,2)
plt.bar(x=xposition, height=data[2][6:9], label='SLSQP', color='b',align='center',width=bar_width)
plt.bar(x=xposition+bar_width, height=data[1][6:9], label='GDS', color='g',align='center',width=bar_width)
plt.bar(x=xposition+2*bar_width, height=data[0][6:9], label='ranRFL', color='r',align='center',width=bar_width)
fontx={'size':22, 'weight':'normal'}
fonty={'size':23, 'weight':'normal'}
# plt.xlabel("Different Zipf Distribution",fontx)
# plt.ylabel('Error Rate',fonty)
plt.xticks([1.5],labels=('Zipf',),fontsize=22)
plt.yticks(fontsize=22)
plt.legend()
plt.tight_layout()
plt.subplot(1,3,3)
plt.bar(x=xposition, height=data[2][0:3], label='SLSQP', color='b',align='center',width=bar_width)
plt.bar(x=xposition+bar_width, height=data[1][0:3], label='GDS', color='g',align='center',width=bar_width)
plt.bar(x=xposition+2*bar_width, height=data[0][0:3], label='ranRFL', color='r',align='center',width=bar_width)
fontx={'size':22, 'weight':'normal'}
fonty={'size':23, 'weight':'normal'}
plt.xticks([1.5],labels=('Normal',),fontsize=22)
plt.yticks(fontsize=22)
plt.legend()
plt.tight_layout()
plt.subplot(1,3,3)

pdf=PdfPages('delicate picture/figure7.pdf')
pdf.savefig()
pdf.close()

plt.show()
