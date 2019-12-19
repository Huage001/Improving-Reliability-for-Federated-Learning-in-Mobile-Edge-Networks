import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

fname = './result/distribution_25_0.5_normal_[0.01, 0.1]_to_[0.15, 0.1]_f_[3, 1]_to_[11, 11]_3_relate'
errorParameterLen=15

file=open(fname+'.txt', 'r')
lines=list(file.readlines())
temp_file=open('temp_file.txt','w')
for line in lines:
    new_line=line[1:len(line)-2]
    print(new_line,file=temp_file)
temp_file.close()
data=np.loadtxt('temp_file.txt',delimiter=',')
dataParameterLen=data.shape[1]/errorParameterLen

linear=data[0,:].reshape(int(dataParameterLen), int(errorParameterLen))
greedy=data[1,:].reshape(int(dataParameterLen), int(errorParameterLen))
convex=data[2,:].reshape(int(dataParameterLen), int(errorParameterLen))

improveConvex=(convex-linear)/convex
improveGreedy=(greedy-linear)/greedy

# improveConvex=np.reshape(improveConvex,[3,5,15])
# improveGreedy=np.reshape(improveGreedy,[3,5,15])
# improveConvex=np.transpose(improveConvex,[1,0,2])
# improveGreedy=np.transpose(improveGreedy,[1,0,2])
# improveConvex=np.reshape(improveConvex,[15,15])
# improveGreedy=np.reshape(improveGreedy,[15,15])


plt.figure(figsize=(12.8,9.6))
fontx={'size':30, 'weight':'normal'}
fonty={'size':30, 'weight':'normal'}
#plt.legend(fontsize=18)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.tight_layout()


plt.xticks([0,4,9,14],[0.01,0.05,0.10,0.15])
plt.xlabel('Average Device Error Rate',fontx)
ylable=[]
for i in range(1,12,5):
    for j in range(3,12,2):
        ylable.append([j,i])
# ylable=[1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5]
plt.yticks(range(0,15),ylable)
# plt.ylabel('Zip Parameter',fonty)
plt.ylabel('[F Parameter 1, F parameter 2]',fonty)
im=plt.imshow(improveConvex*100,cmap=plt.cm.hot_r)
color=plt.colorbar(im)
color.ax.tick_params(labelsize=30)
color.set_label('Average Error Rate Decreased (%)',fontdict=fontx)
im.set_clim(vmin=-10,vmax=50)
#pdf=PdfPages('./result/pdf/figure_color_convex.pdf')
plt.tight_layout()
plt.savefig('./result/pdf/plt_figure_color_convex_f.pdf',bbox_inches='tight')
plt.show()
plt.close()


plt.figure(figsize=(12.8,9.6))
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.tight_layout()
plt.xticks([0,4,9,14],[0.01,0.05,0.10,0.15])
plt.xlabel('Average Device Error Rate',fontx)
plt.yticks(range(0,15),ylable)
# plt.ylabel('Zip Parameter',fonty)
plt.ylabel('[F Parameter 1, F parameter 2]',fonty)
im=plt.imshow(improveGreedy*100,cmap=plt.cm.hot_r)
color=plt.colorbar(im)
color.ax.tick_params(labelsize=30)
color.set_label('Average Error Rate Decreased (%)',fontdict=fontx)
im.set_clim(vmin=-10,vmax=50)
#pdf=PdfPages('./result/pdf/figure_color_greedy.pdf')
plt.tight_layout()
plt.savefig('./result/pdf/plt_figure_color_greedy_f.pdf',bbox_inches='tight')
plt.show()
plt.close()

print(np.mean(improveConvex))
print(np.mean(improveGreedy))

print(np.max(improveConvex))
print(np.max(improveGreedy))