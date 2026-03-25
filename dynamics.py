import numpy as np
#h_bar = 1.054*10**(-34) # Planck's constant
h_bar = 1 # atomic units

def rho(cij):
    return cij @ cij.conj().T

def dij(b, blast, dx):
    for j in range(2):
        if np.real(np.vdot(blast[:, j], b[:, j])) < 0:
            b[:, j] *= -1
    d = np.zeros((2, 2), dtype=complex)
    db0_dx = (b[:, 0] - blast[:, 0]) / dx
    db1_dx = (b[:, 1] - blast[:, 1]) / dx

    d[0, 1] = np.vdot(b[:, 0], db1_dx)
    d[1, 0] = np.vdot(b[:, 1], db0_dx)
    return d

def vij(n, b, htot):
    vij = np.zeros((n, n))
    for i in range(len(b)):
        for j in range(len(b)):
            vij[i][j] = b[:,i].T @ htot @ b[:,j]
    return vij  

def bkl(rho, h1, Rdot, dij, dt):
    term1 = 2/h_bar*((np.conj(rho)*h1)).imag
    term2 = 2*(np.conj(rho)*Rdot*dij).real
    return term1 - term2

def akk(bkl, k):
    return np.sum([bkl[k, l] for l in range(len(bkl)) if l != k])

def VV_integrator(R, v, F, M, dx, dt):
    R_new = R + v*dt + 0.5*(F/M)*dt**2
    F_new =  -np.gradient(R_new, dx)
    v_new = v + 0.5*(F + F_new)/M * dt
    return R_new, v_new, F_new


