import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

total=100
name='./result/select_devices/single_100_0.8_normal_[0.05, 0.01]_normal_[1, 1].txt'
with open(name,"r") as F:
    read_data=F.read()
data=read_data.split()
x_data=[float(i) for i in data[0:total]]
y_data=[float(i) for i in data[total:total*2]]
select_convex=[float(i) for i in data[total*2:total*3]]
select_linear=[float(i) for i in data[total*5:total*6]]
select_greedy=[float(i) for i in data[total*8:total*9]]

fontx={'size':22, 'weight':'normal'}
fonty={'size':23, 'weight':'normal'}

plt.xticks(fontsize=18)
plt.yticks(fontsize=22)
# plt.ylim(np.array([0.02,0.07))
plt.ylim(0.015,0.075)
plt.xlabel("Device Error Rate",fontx)
plt.ylabel('Data Proportion',fonty)
plt.tight_layout()
plt.scatter(np.array(x_data)[~np.array(select_convex).astype(bool)],
            np.array(y_data)[~np.array(select_convex).astype(bool)],color='gray',label='Unselected Devices',marker='x')
plt.scatter(np.array(x_data)[np.array(select_convex).astype(bool)],
            np.array(y_data)[np.array(select_convex).astype(bool)],
            color='b',label='Selected by SLSQP',marker='o')
plt.legend(loc=3,fontsize=12,ncol=2)
# pdf=PdfPages('./result/pdf/figure_select_convex.pdf')
# pdf.savefig()
# pdf.close()
plt.savefig('./result/pdf/figure_select_convex.pdf',bbox_inches='tight')
plt.show()

plt.xticks(fontsize=18)
plt.yticks(fontsize=22)
plt.xlabel("Device Error Rate",fontx)
plt.ylabel('Data Proportion',fonty)
plt.ylim(0.015,0.075)
plt.tight_layout()
plt.scatter(np.array(x_data)[~np.array(select_linear).astype(bool)],
            np.array(y_data)[~np.array(select_linear).astype(bool)],color='gray',label='Unselected Devices',marker='x')
plt.scatter(np.array(x_data)[np.array(select_linear).astype(bool)],
            np.array(y_data)[np.array(select_linear).astype(bool)],
            color='r',label='Selected by ranRFL',marker='o')
plt.legend(fontsize=12,ncol=2)
# pdf=PdfPages('./result/pdf/figure_select_linear.pdf')
# pdf.savefig()
# pdf.close()
plt.savefig('./result/pdf/figure_select_linear.pdf',bbox_inches='tight')
plt.show()

plt.xticks(fontsize=18)
plt.yticks(fontsize=22)
plt.xlabel("Device Error Rate",fontx)
plt.ylabel('Data Proportion',fonty)
plt.ylim(0.015,0.075)
plt.tight_layout()
plt.scatter(np.array(x_data)[~np.array(select_greedy).astype(bool)],
            np.array(y_data)[~np.array(select_greedy).astype(bool)],color='gray',label='Unselected Devices',marker='x')
plt.scatter(np.array(x_data)[np.array(select_greedy).astype(bool)],
            np.array(y_data)[np.array(select_greedy).astype(bool)],
            color='g',label='Selected by GDS',marker='o')
plt.legend(fontsize=12,ncol=2)
# pdf=PdfPages('./result/pdf/figure_select_greedy.pdf')
# pdf.savefig()
# pdf.close()
plt.savefig('./result/pdf/figure_select_greedy.pdf',bbox_inches='tight')
plt.show()
