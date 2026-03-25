import numpy as np

def mol():
    return

def twolevel():
    b1 = np.array([1,0])
    b2 = np.array([0,1])
    return b1, b2

def Gauss(k, sigma, x):
    return np.exp(1j*k*x)*np.exp(-(x/sigma)**2)

