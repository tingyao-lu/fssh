from main import htot, x
from hoppingalg import d_list, e0_list, e1_list
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(6,5)) 
ax.plot(x, e0_list, color = "#A0C4FF", label="E0", linewidth=3) 
ax.plot(x, e1_list, color = "#FFB3BA",  label="E1", linewidth=3) 
ax.plot(x, np.array(d_list), color = "#A67BB5", label=r"$NAC$", linestyle="--", linewidth=3) 
ax.set_xlabel(r"$R$ (a.u.)") 
ax.set_ylabel("Energy (a.u.)") # Adiabatic PES
ax.set_title("Tully Model III") 
ax.legend(loc="upper right") 
fig.tight_layout() 
fig.savefig('tully3.png', dpi=300) 
plt.show()
