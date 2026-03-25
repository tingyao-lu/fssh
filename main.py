import hamiltonian as ham
import dynamics as dyn
import basis as bas
import models as pes
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import scipy.ndimage

# parameters
M = 2000 # au
nsteps = 100000
x = np.linspace(-10, 10, nsteps)
dx = x[1] - x[0]
rho = np.zeros((nsteps, 2, 2), dtype=complex) 
active = 0

# basis set for a 2-level system
# b0,b1 = bas.twolevel()
# c0, c1 = 1,0
# bmol=c0*b0+c1*b1


# hamiltonian
n=2
h0 = ham.H0(n)
tully1 = pes.SAC(0.01, 1.6, 0.005, 1,x) #matrix of nsteps*2*2
tully2 = pes.DAC(0.1, 0.28, 0.015, 0.06, 0.05, x)
tully3 = pes.ECwR(0.0006, 0.1, 0.9, x)

models = [tully1, tully2, tully3]
names = ['1', '2', '3']

model = models[0]

htot = ham.Htot(h0, model, n)
