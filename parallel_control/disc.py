"""
Linear time invariant (LTI) system discretization routines (using numpy).

@author: Simo Särkkä
"""

from scipy import linalg
import numpy as np


def lti_disc(F, L, Qc=None, dt=1.0):
    """ Discretize a system of the form
          dx/dt = F x + L w,
        where w is a white noise with spectral density Qc to
          x[k+1] = A x[k] + w_k, w_k ~ N(0,Q)
        """
    n = F.shape[0]

    if Qc is None:
        Qc = np.eye(L.shape[1], dtype=F.dtype)
    if np.isscalar(Qc):
        Qc = np.array([[Qc]])

    Phi = np.zeros((2*n,2*n), dtype=F.dtype)
    Phi[ 0:n,  0:n] = F
    Phi[ 0:n,  n: ] = L @ Qc @ L.T
    Phi[n:2*n, n: ] = -F.T

    EPhi = linalg.expm(Phi * dt)
    A = EPhi[0:n, 0:n]
    Q = EPhi[0:n, n:] @ A.T

    return A, Q

def lti_disc_u(F, L, G=None, Qc=None, dt=1.0):
    """ Discretize a system of the form (note that G is the input gain!)
          dx/dt = F x + G u + L w,
        where w is a white noise with spectral density Qc to
          x[k+1] = A x[k] + B u[k] + w_k, w_k ~ N(0,Q)
        """
    A, Q = lti_disc(F, L, Qc, dt)

    n = F.shape[0]

    if G is None:
        G = np.eye(n, dtype=F.dtype)

    m = G.shape[1]

    Psi = np.zeros((n+m,n+m), dtype=F.dtype)
    Psi[0:n, 0:n] = F
    Psi[0:n, n: ] = G

    EPsi = linalg.expm(Psi * dt)
    B = EPsi[0:n, n:]

    return A, B, Q

