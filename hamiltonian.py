import numpy as np


def Htot(H0, H1, n=None):
    """Total Hamiltonian constructed from two pieces.

    ``H0`` and ``H1`` may be either callables accepting ``n`` or already
    evaluated arrays.  The previous version assumed both were functions; this
    made the call site in :mod:`main` awkward.  The new implementation handles
    both styles gracefully.
    """
    if callable(H0):
        h0 = H0(n)
    else:
        h0 = H0
    if callable(H1):
        h1 = H1(n)
    else:
        h1 = H1

    return h0 + h1


def H0(n):
    return np.zeros((n, n))

def H1(n):
    return np.zeros((n, n))

