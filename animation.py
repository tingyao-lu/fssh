import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from main import x, nsteps
from hoppingalg import active_pes, e0_list, e1_list

# PES
e0 = np.array(e0_list)
e1 = np.array(e1_list)


# build figure
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(x, e0, color = "#A0C4FF", label="E0", linewidth=3)
ax.plot(x, e1, color = "#FFB3BA", label="E1", linewidth=3)

dot, = ax.plot(x, active_pes, color = "#A67BB5", marker='o', markersize=8)
# trail line is omitted to avoid drawing the path history

ax.set_xlim(x.min(), x.max())
ax.set_ylim(min(e0.min(), e1.min()) - 0.001, max(e0.max(), e1.max()) + 0.001)
ax.set_xlabel("R (a.u.)")
ax.set_ylabel("Energy (a.u.)")
ax.set_title("Tully I - nuclear motion on PES")
ax.legend() 
fig.tight_layout() 

def energy_on_surface(R, surf):
    if surf == 0:
        return np.interp(R, x, e0)
    else:
        return np.interp(R, x, e1)

def init():
    dot.set_data([], [])
    return (dot,)

def update(nsteps):
    R_now = x[nsteps]
    surf = active_pes[nsteps]
    E = energy_on_surface(R_now, surf)

    dot.set_data([R_now], [E])

    ax.set_title(f"Tully model III | step = {nsteps}")
    return (dot,)

ani = FuncAnimation(
    fig,
    update,
    frames=range(0,len(x),10),
    init_func=init,
    blit=True,
    interval=100
)

ani.save("TullyIII.gif", writer=PillowWriter(fps=300))
plt.show()
