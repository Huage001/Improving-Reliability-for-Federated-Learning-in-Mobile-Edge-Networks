from scipy.optimize import minimize
from scipy.optimize import fsolve
import numpy as np
import matplotlib.pyplot as plt

def func(x):
    return x*(3**x)-64
r=fsolve(func,6)
print(r)
print(3%2)
