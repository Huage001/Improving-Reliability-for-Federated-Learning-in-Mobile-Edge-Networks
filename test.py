import numpy as np
from scipy.stats import f
import matplotlib.pyplot as plt
a=np.random.rand(5)
b=f.pdf(np.random.uniform(0,4,100), 100, 100)
# b=np.random.zipf(1.8,40)
c=b/sum(b)
print(c)
group=np.linspace(0,1,100)
plt.hist(c,group)
plt.show()
