import hamiltonian as ham
import dynamics as dyn
import basis as bas
import models as pes
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import scipy.ndimage
from main import htot, x, h0, model, n, nsteps, dx, M
import multiprocessing

def run_trajectory(P_val):
    # initial condition
    R = -10.0
    V = P_val / M  # velocity in au
    P = P_val
    dt = 0.1
    # initial adiabatic coefficients
    c = np.array([1+0j, 0j])  # start on lower state
    active = 1

    #list to store results
    d_list = []
    v_list = []
    e0_list = []
    e1_list = []
    active_pes = []
    pop0_list = []
    pop1_list = []
    rho = np.zeros((nsteps, 2, 2), dtype=complex)

    i = 0

    while R >= -10 and R <= 10 and i < nsteps:
        # find current grid index
        idx = int(round((R + 10) / dx))
        idx = min(max(idx, 0), nsteps-1)

        # est calculation to obtain basis sets
        if i != 0:
            blast = b
            e0last, e1last = e0, e1
        else: blast = np.zeros((n, n))
        
        eigenvalues, eigenvectors = np.linalg.eigh(htot[idx,:,:])
        e0, e1 = eigenvalues[0], eigenvalues[1] #adiabatic energies
        e0_list.append(e0)
        e1_list.append(e1)
        b0, b1 = eigenvectors[:, 0], eigenvectors[:, 1] #adiabatic vectors
        b = np.array([b0, b1]).T # matrix of adiabatic vectors as columns

        rho[i,:,:] = np.outer(c, np.conj(c)) # akj = ck(cj*)
        pop0_list.append(np.abs(c[0])**2)
        pop1_list.append(np.abs(c[1])**2)
        if i == 0:
            dij = np.zeros((n, n), dtype=complex)
        else:
            dij = dyn.dij(b, blast, dx)
        d_list.append(dij[0,1])
        vij = dyn.vij(n, b, htot[idx,:,:])

        # dynamics
        k = active
        j = 1-active
        active_pes.append(e0 if active == 0 else e1)

        bkl = dyn.bkl(rho[i,:,:], vij, V, dij, dt)
        akk = dyn.akk(bkl, k)
        if not np.isfinite(akk) or abs(akk) < 1e-14:
            g_kj = 0.0
        else:
            g_kj = np.real(dt * bkl[k, j] / akk)
            g_kj = max(0.0, min(1.0, g_kj))

        #hopping option
        if np.random.rand() < g_kj:
            # hop from state k to state j
            # momentum rescaling to conserve energy after hop
            E_kinold = 0.5 * M * V**2
            E_potold = e0 if active == 0 else e1
            E_potnew = e0 if j == 0 else e1
            E_kinnew = E_kinold + E_potold - E_potnew
            if E_kinnew >= 0:
                V = np.sign(V) * np.sqrt(2 * E_kinnew / M)
                active = j
            # else: frustrated hop, stay on current

        # compute force on active surface
        if i > 0:
            if active == 0:
                F = -(e0 - e0last) / dx
            else:
                F = -(e1 - e1last) / dx
        else:
            F = 0.0  # initial force

        # propagate electronic coefficients
        E_diag = np.array([e0, e1])
        nac_term = dij * V
        dc_dt = -1j * (E_diag * c + nac_term @ c)
        c += dc_dt * dt
        # normalize
        c /= np.linalg.norm(c)

        # update nuclear position and momentum
        R += V * dt
        V += F / M * dt
        P = M * V  # update P

        i += 1

    #print(f"Trajectory P={P_val}, final R={R:.2f}, active={active}, pop0={pop0_list[-1]:.3f}, pop1={pop1_list[-1]:.3f}")
    return pop0_list[-1], pop1_list[-1], R, active  # return final populations, position, and active state

def run_simulations():
    results_dict = {}
    ntraj = 10
    for P_val in range(5, 30):  # smaller range
        with multiprocessing.Pool() as pool:
            results = pool.map(run_trajectory, [P_val] * ntraj)
        avg_pop0 = np.mean([r[0] for r in results])
        avg_pop1 = np.mean([r[1] for r in results])
        
        # Compute probabilities
        transmitted_lower = sum(1 for r in results if r[2] > 0 and r[3] == 0)
        reflected_lower = sum(1 for r in results if r[2] <= 0 and r[3] == 0)
        transmitted_upper = sum(1 for r in results if r[2] > 0 and r[3] == 1)
        
        prob_trans_lower = transmitted_lower / ntraj
        prob_refl_lower = reflected_lower / ntraj
        prob_trans_upper = transmitted_upper / ntraj
        
        results_dict[P_val] = (avg_pop0, avg_pop1, prob_trans_lower, prob_refl_lower, prob_trans_upper)
        print(f"P={P_val}, avg_pop0={avg_pop0:.4f}, avg_pop1={avg_pop1:.4f}, prob_trans_lower={prob_trans_lower:.4f}, prob_refl_lower={prob_refl_lower:.4f}, prob_trans_upper={prob_trans_upper:.4f}")
    return results_dict

if __name__ == '__main__':
    run_simulations()
    print("All simulations completed.")
