import numpy as np
from main import htot, dx
import dynamics as dyn

print("=== NAC symmetry check ===")
for i in [1000, 5000, 50000]:
    _, ep = np.linalg.eigh(htot[i-1])
    _, ec = np.linalg.eigh(htot[i])
    _, en = np.linalg.eigh(htot[i+1])
    bl  = np.array([ep[:,0], ep[:,1]]).T
    bc  = np.array([ec[:,0], ec[:,1]]).T
    bpl = np.array([en[:,0], en[:,1]]).T
    d = dyn.dij(bc, bl, bpl, dx)
    print(f"i={i}:  d01={d[0,1]:.8f}  d10={d[1,0]:.8f}")
    print(f"       d10 + conj(d01) = {d[1,0] + d[0,1].conj():.2e}")
    print(f"       d00={d[0,0]:.2e}  d11={d[1,1]:.2e}")
