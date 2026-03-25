import numpy as np


def SAC(A, B, C, D, x):
    """Simple avoided crossing potential matrix.

    The result always has a trailing ``(2,2)`` block structure.  For a scalar
    ``x`` a plain ``(2,2)`` array is returned; for an array the returned shape
    is ``x.shape + (2,2)``.
    """
    # compute sign factor (+1 for x>0, -1 otherwise)
    shape = np.shape(x)
    v11 = np.where(
        x > 0,
        A * (1 - np.exp(-B * x)),
        -A * (1 - np.exp(B * x))
    )
    v22 = -v11
    v12 = C * np.exp(-D * x**2)
    v21 = v12
    out = np.empty(shape + (2, 2))
    out[..., 0, 0] = v11
    out[..., 0, 1] = v12
    out[..., 1, 0] = v21
    out[..., 1, 1] = v22
    return out


def DAC(A, B, C, D, E0, x):
    """Double avoided crossing potential matrix.

    Like :func:`SAC`, this is vectorized over ``x``.
    """
    v11 = np.zeros_like(x)
    v22 = -A * np.exp(-B * x**2) + E0
    v12 = C * np.exp(-D * x**2)
    v21 = v12
    out = np.empty(np.shape(x) + (2, 2))
    out[..., 0, 0] = v11
    out[..., 0, 1] = v12
    out[..., 1, 0] = v21
    out[..., 1, 1] = v22
    return out


def ECwR(A, B, C, x):
    """Extended coupling with reflection potential.

    Handles array-valued ``x`` identically to the other potentials.
    """
    shape = np.shape(x)
    v11 = np.full(shape, A)
    v22 = np.full(shape, -A)
    v12 = np.where(x < 0,
                   B * np.exp(C * x),
                   B * (2 - np.exp(-C * x)))
    v21 = v12
    out = np.empty(shape + (2, 2))
    out[..., 0, 0] = v11
    out[..., 0, 1] = v12
    out[..., 1, 0] = v21
    out[..., 1, 1] = v22
    return out


def LZ(A, B, x):
    """Landau-Zener potential matrix.

    This is a special case of :func:`SAC` with ``B = D`` and ``C = A``.
    """
    return SAC(A, B, A, B, x)

def LZpop(V12, dV11, dV22, dx):
    """Landau-Zener population transfer probability."""
    return np.exp(-2*np.pi * V12**2 / ((dV11/dx - dV22/dx)*dx))
