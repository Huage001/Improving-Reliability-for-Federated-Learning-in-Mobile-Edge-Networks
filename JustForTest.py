from scipy.optimize import minimize
from scipy.optimize import fsolve
import numpy as np
import matplotlib.pyplot as plt
from DeviceManager import DeviceManager
from scipy.stats import f

a=np.random.uniform(0,4,10)
b=f.pdf(np.random.uniform(0,4,20), 1, 1)
print(b)

