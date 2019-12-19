import matplotlib.pyplot as plt
import numpy as np

fname="./result/deviceNum_40_0.5_zipf_2_zipf_2_time"
file=open(fname+".txt", 'r')
lines=list(file.readlines())
temp_file=open('temp_file.txt','w')
for line in lines:
    new_line=line[1:len(line)-2]
    print(new_line,file=temp_file)
temp_file.close()
data=np.loadtxt('temp_file.txt',delimiter=',')

fontx={'size':22, 'weight':'normal'}
fonty={'size':23, 'weight':'normal'}
plt.ylabel('Time Consumed(s)',fonty)
plt.xlabel('Number of Devices',fontx)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

plt.plot(range(3,41),data[0],color='r',label='ranRFL')
plt.plot(range(3,41),data[1],color='g',label='GDS')
plt.plot(range(3,41),data[2],color='b',label='SLSQP')
plt.legend(fontsize=18)
plt.savefig("./result/pdf/graph_time.pdf",bbox_inches='tight')
plt.show()
plt.close()