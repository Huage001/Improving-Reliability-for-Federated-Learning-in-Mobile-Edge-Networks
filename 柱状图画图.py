import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

xposition1 = np.arange(3)
xposition2 = np.arange(4,7)
xposition3 = np.arange(8,11)
bar_width=0.25
#y_axis_range = np.arange(0.0,0.8,0.2)

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
# plt.xlabel("Different F Distribution",fontx)
plt.ylabel('Average Error Rate',fonty)
plt.xticks(list(xposition1+bar_width)+list(xposition2+bar_width)+list(xposition3+bar_width),labels=('$S_1$','$M_1$','$L_1$','$S_2$','$M_2$','$L_2$','$S_3$','$M_3$','$L_3$'),fontsize=21)
plt.yticks(fontsize=22)
plt.legend(fontsize = 16)
plt.tight_layout()
#plt.gca().axes.get_yaxis().set_visible(False)    #隐藏y轴

pdf=PdfPages(fname+'.pdf')
pdf.savefig()
pdf.close()

plt.show()