import models as pes
import numpy as np
import matplotlib.pyplot as plt

omega0 = 0.1
mu     = 1.0
E0     = 0.05
omega  = 0.1

t = np.linspace(0, 100, 1000)

vij = pes.CW(mu=mu, E0=E0, omega=omega, t=t)  # [nstep, 2, 2]
#vij = pes.LaserPulse(omega0=omega0, mu=mu, E0=E0, omega=omega, t0=50, sigma=10, t=t)

e0_list, e1_list = [], []
for i in range(len(t)):
    evals, _ = np.linalg.eigh(vij[i, :, :])
    e0_list.append(evals[0])
    e1_list.append(evals[1])

field    = E0 * np.cos(omega * t)          # E(t)
coupling = vij[:, 0, 1]                    # off-diagonal element

fig, axes = plt.subplots(3, 1, figsize=(7, 8), sharex=True)

axes[0].plot(t, field, color="#A67BB5", linewidth=2)
axes[0].set_ylabel("E(t) (a.u.)")
axes[0].set_title("Laser Pulse")

axes[1].plot(t, e0_list, color="#A0C4FF", label="E0", linewidth=2)
axes[1].plot(t, e1_list, color="#FFB3BA", label="E1", linewidth=2)
axes[1].set_ylabel("Adiabatic energy (a.u.)")
axes[1].set_title("Adiabatic PES")
axes[1].legend(loc="upper right")

axes[2].plot(t, coupling, color="#B5EAD7", linewidth=2)
axes[2].set_ylabel("V₀₁(t) (a.u.)")
axes[2].set_xlabel("Time (a.u.)")
axes[2].set_title("Off-diagonal coupling")

fig.tight_layout()
fig.savefig('plotLaser.png', dpi=300)
plt.show()


