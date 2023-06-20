"""
Numpy-based Linear Quadratic Regulator (LQR) and Tracker (LQT) routines, both sequential and parallel
(though the parallel ones are not really implemented in parallel but simulated). The aim of these routines
is to provide a simple and fast implementation of the parallel LQR and LQT algorithms which can be used as
a reference implementation for more complex parallel LQR/LQT implementations (such as in TensorFlow).

@author: Simo Särkkä
"""

from scipy import linalg
import numpy as np
from parallel_control.my_assoc_scan import my_assoc_scan


##############################################################################
# Combination functions for the parallel LQR and LQT as well as the
# generic associative scan calls for LQR/LQT
##############################################################################

def combine_abcej(Aij, bij, Cij, etaij, Jij, Ajk, bjk, Cjk, etajk, Jjk):
    """ Combine two conditional value functions in backward pass of parallel LQT.

    Parameters:
        Aij, bij, Cij, etaij, Jij: parameters of conditional value function V_{i->j}(x_i, x_j)
        Ajk, bjk, Cjk, etajk, Jjk: parameters of conditional value function V_{j->k}(x_j, x_k)

    Returns:
        Aik, bik, Cik, etaik, Jik: parameters of conditional value function V_{i->k}(x_i, x_k)
    """
    I = np.eye(Aij.shape[0], dtype=Aij.dtype)
    LU, piv = linalg.lu_factor(I + Cij @ Jjk)
    Aik = Ajk @ linalg.lu_solve((LU, piv), Aij)
    bik = Ajk @ linalg.lu_solve((LU, piv), bij + Cij @ etajk) + bjk
    Cik = Ajk @ linalg.lu_solve((LU, piv), Cij @ Ajk.T) + Cjk
    LU, piv = linalg.lu_factor(I + Jjk @ Cij)
    etaik = Aij.T @ linalg.lu_solve((LU, piv), etajk - Jjk @ bij) + etaij
    Jik = Aij.T @ linalg.lu_solve((LU, piv), Jjk @ Aij) + Jij
    return Aik, bik, Cik, etaik, Jik


def combine_fc(Fij, cij, Fjk, cjk):
    """ Combine two functions in forward pass of parallel LQT.

    Parameters:
        Fij, cij: parameters of function f_{i->j}(x_i)
        Fjk, cjk: parameters of function f_{j->k}(x_j)

    Returns:
        Fik, cik: parameters of function f_{i->k}(x_i)
    """
    Fik = Fjk @ Fij
    cik = Fjk @ cij + cjk
    return Fik, cik


def par_backward_pass_scan(elems):
    """ Perform LQT backward associative scan to the backward pass elements.

    Parameters:
        elems: list of tuples (A, b, C, eta, J)

    Returns:
        Reverse prefix sums as a list of tuples (A, b, C, eta, J)
    """
    return my_assoc_scan(lambda x, y: combine_abcej(*x, *y), elems, reverse=True)


def par_forward_pass_scan(elems):
    """ Perform LQT forward associative scan to the backward pass elements.

    Parameters:
        elems: list of tuples (F, c)

    Returns:
        Forward prefix sums as a list of tuples (F, c)
    """
    return my_assoc_scan(lambda x, y: combine_fc(*x, *y), elems, reverse=False)


def par_fwdbwd_pass_scan(elems):
    """ Perform LQT forward associative scan type 2 to the forward pass elements.

    Parameters:
        elems: list of tuples (A, b, C, eta, J)

    Returns:
        Forward prefix sums as a list of tuples (A, b, C, eta, J)
    """
    return my_assoc_scan(lambda x, y: combine_abcej(*x, *y), elems, reverse=False)


##############################################################################
#
# Sequential / parallel LQR -- please see more general LQT below
#
##############################################################################

class LQR:
    """
    Class containing basic LQR model parameters and methods for computing
    all kinds of useful things from it. The model is
    
    x[k+1] = F x[k] + L u[k]
      J(u) = E{ 1/2 x[T].T XT x[T]
        + sum_{k=0}^{T-1} 1/2 (x[k].T X x[k] + 1/2 u[k].T U u[k] }
             
    """

    def __init__(self, F, L, X, U, XT):
        """ Initialize the LQR model with the parameters.

        Parameters:
            F: State transition matrix
            L: Control matrix
            X: State cost matrix
            U: Control cost matrix
            XT: Terminal state cost matrix
        """
        self.F = F
        self.L = L
        self.X = X
        self.U = U
        self.XT = XT

    ###########################################################################
    # Sequential computation of gains, value functions, states, and controls
    ###########################################################################

    def seqBackwardPass(self, T):
        """ Solve the LQR Riccati equation backwards starting from k=T. Returns
            the list of control gains for 0:T-1 and list of matrices
            such that V(x_k) = 1/2 x_k.T S_k x_k for k=0:T

        Parameters:
            T: Length of the horizon

        Returns:
            Kx_list: List of control gains for k=0:T-1
            S_list: List of matrices such that V(x_k) = 1/2 x_k.T S_k x_k for k=0:T
        """
        S = self.XT

        S_list = [S]
        Kx_list = []

        for _ in reversed(range(T)):
            Kx = linalg.solve(self.U + self.L.T @ S @ self.L,
                              self.L.T @ S @ self.F, assume_a='pos')
            S = self.F.T @ S @ (self.F - self.L @ Kx) + self.X
            S_list.append(S)
            Kx_list.append(Kx)

        S_list.reverse()
        Kx_list.reverse()

        return Kx_list, S_list

    def seqForwardPass(self, x0, Kx_list):
        """ Simulate the LQR system forward starting from x0 using the gains.

        Parameters:
            x0: Initial state
            Kx_list: List of control gains for k=0:T-1

        Returns:
            u_list: List of controls for k=0:T-1
            x_list: List of states for k=0:T
        """
        u_list = []
        x_list = [x0]

        x = x0
        for k in range(len(Kx_list)):
            u = -Kx_list[k] @ x
            x = self.F @ x + self.L @ u
            u_list.append(u)
            x_list.append(x)

        return u_list, x_list

    ###########################################################################
    # Parallel (simulated parallel) computation of the gains and value functions
    ###########################################################################

    def parBackwardPass_init(self, T):
        """ Initialize the parallel backward pass of length T.

        Parameters:
            T: Length of the horizon

        Returns:
            elems: List of tuples (A, b, C, eta, J) for k=0:T
        """
        A = self.F
        b = np.zeros_like(self.F[:, 0])
        C = self.L @ linalg.solve(self.U, self.L.T, assume_a='pos')
        eta = np.zeros_like(self.F[:, 0])
        J = self.X
        elems = T * [(A, b, C, eta, J)]

        A = np.zeros_like(self.F)
        b = np.zeros_like(self.F[:, 0])
        C = np.zeros_like(self.F)
        eta = np.zeros_like(self.F[:, 0])
        J = self.XT
        elem = (A, b, C, eta, J)
        elems.append(elem)

        return elems

    def parBackwardPass_extract(self, elems):
        """ Extract the results of backward pass from element list.

        Parameters:
            elems: List of tuples (A, b, C, eta, J) for k=0:T

        Returns:
            Kx_list: List of control gains for k=0:T-1
            S_list: List of matrices such that V(x_k) = 1/2 x_k.T S_k x_k for k=0:T
        """
        (A, b, C, eta, J) = elems[0]
        S_list = [J]
        Kx_list = []
        for k in range(len(elems) - 1):
            (A, b, C, eta, J) = elems[k + 1]
            S = J
            Kx = linalg.solve(self.U + self.L.T @ S @ self.L,
                              self.L.T @ S @ self.F, assume_a='pos')
            S_list.append(S)
            Kx_list.append(Kx)

        return Kx_list, S_list

    def parBackwardPass(self, T):
        """ Perform the whole parallel backward pass of length T.

        Parameters:
            T: Length of the horizon

        Returns:
            Kx_list: List of control gains for k=0:T-1
            S_list: List of matrices such that V(x_k) = 1/2 x_k.T S_k x_k for k=0:T
        """

        # Initialize
        elems = self.parBackwardPass_init(T)

        # Call the associative scan
        elems = par_backward_pass_scan(elems)

        # Extract the results
        return self.parBackwardPass_extract(elems)

    ###########################################################################
    # Parallel (simulated parallel) computation of the states and controls
    ###########################################################################

    def parForwardPass_init(self, x0, Kx_list):
        """ Initialize the parallel forward pass elements starting from x0
            and having the given control gains for 0:T-1.

        Parameters:
            x0: Initial state
            Kx_list: List of control gains for k=0:T-1

        Returns:
            elems: List of tuples (F, c) for k=0:T-1
        """
        elems = []
        for k in range(len(Kx_list)):
            if k == 0:
                tF = np.zeros_like(self.F)
                tc = (self.F - self.L @ Kx_list[k]) @ x0
            else:
                tF = self.F - self.L @ Kx_list[k]
                tc = np.zeros_like(self.F[:, 0])
            elems.append((tF, tc))

        return elems

    def parForwardPass_extract(self, x0, Kx_list, elems):
        """ Extract the results of forward parallel pass starting from x0
            and having the given elements. Returns controls and states.

        Parameters:
            x0: Initial state
            Kx_list: List of control gains for k=0:T-1
            elems: List of tuples (F, c) for k=0:T-1

        Returns:
            u_list: List of controls for k=0:T-1
            x_list: List of states for k=0:T
        """
        x_list = [x0]
        u_list = []

        for k in range(len(Kx_list)):
            if k == 0:
                u = -Kx_list[k] @ x0
            else:
                tF, tc = elems[k - 1]
                u = -Kx_list[k] @ tc

            tF, tc = elems[k]
            x = tc
            x_list.append(x)
            u_list.append(u)

        return u_list, x_list

    def parForwardPass(self, x0, Kx_list):
        """ Perform the whole parallel forward pass starting from x0.

        Parameters:
            x0: Initial state
            Kx_list: List of control gains for k=0:T-1

        Returns:
            u_list: List of controls for k=0:T-1
            x_list: List of states for k=0:T
        """
        # Initialize
        elems = self.parForwardPass_init(x0, Kx_list)

        # Call the associative scan
        elems = par_forward_pass_scan(elems)

        # Extract the results
        return self.parForwardPass_extract(x0, Kx_list, elems)

    ###########################################################################
    # Solving of discrete algebraic Riccati equations for stationary LQR
    ###########################################################################

    def lqrDare(self):
        """ Solve the discrete algebraic Riccati equation for stationary LQR using ARE solver.

        Returns:
            K: Control gain
            S: Value function matrix
        """
        S = linalg.solve_discrete_are(self.F, self.L, self.X, self.U)
        K = linalg.solve(self.U + self.L.T @ S @ self.L,
                         self.L.T @ S @ self.F, assume_a='pos')
        return K, S

    def lqrIter(self, n):
        """ Solve the discrete algebraic Riccati equation for stationary LQR using iterative method.

        Parameters:
            n: Number of iterations

        Returns:
            K: Control gain
            S: Value function matrix
        """
        S = self.X
        K = None
        for _ in range(n):
            K = linalg.solve(self.U + self.L.T @ S @ self.L,
                             self.L.T @ S @ self.F, assume_a='pos')
            S = self.F.T @ S @ (self.F - self.L @ K) + self.X

        return K, S

    def lqrDouble(self, n):
        """ Solve the discrete algebraic Riccati equation for stationary LQR using double iteration method.

        Parameters:
            n: Number of iterations

        Returns:
            K: Control gain
            S: Value function matrix
        """
        A = self.F
        b = np.zeros_like(self.F[:, 0])
        C = self.L @ linalg.solve(self.U, self.L.T)
        eta = np.zeros_like(self.F[:, 0])
        J = self.X

        for _ in range(n):
            A, b, C, eta, J = combine_abcej(A, b, C, eta, J, A, b, C, eta, J)

        S = J
        K = linalg.solve(self.U + self.L.T @ S @ self.L,
                         self.L.T @ S @ self.F, assume_a='pos')
        return K, S


##############################################################################
#
# Sequential / parallel LQT
#
##############################################################################

class LQT:
    """
    A class containing general LQT model parameters and methods for doing useful
    this with it. The model is
    
    x[k+1] = F[k] x[k] + c[k] + L[k] u[k]
      J(u) = E{ 1/2 (H[T] x[T] - r[T)].T X[T] (H[T] x[T] - r[T])
        + sum_{k=0}^{T-1} 1/2 (H[k] x[k] - r[k]).T X[k] (H[k] x[k] - r[k])
                        + 1/2 (Z[k] u[k] - s[k]).T U[k] (Z[k] u[k] - s[k])
                            + (H[k] x[k] - r[k]).T M[k] (Z[k] u[k] - s[k]) }
             
    """

    def __init__(self, F, L, X, U, XT, c, H, r, HT, rT, Z, s, M):
        """ Create LQT from given matrices. No defaults are applied. See checkAndExpand for more flexible interface.

        Parameters:
            F: State transition matrices
            L: Control matrices
            X: State cost matrices
            U: Control cost matrices
            XT: Terminal state cost matrix
            c: State offsets
            H: State cost output matrices
            r: State cost output offsets (i.e., the reference trajectory)
            HT: Terminal state cost output matrix
            rT: Terminal state cost output offset
            Z: Control cost output matrices
            s: Control cost output offsets
            M: Cross term matrices
        """
        self.F = F
        self.L = L
        self.X = X
        self.U = U
        self.XT = XT

        self.c = c

        self.H = H
        self.r = r
        self.HT = HT
        self.rT = rT

        self.Z = Z
        self.s = s
        self.M = M

    @classmethod
    def checkAndExpand(cls, F, L, X, U, XT=None, c=None, H=None, r=None, HT=None, rT=None, Z=None, s=None, M=None,
                       T=None):
        """
        Create LQT from given matrices. Also apply defaults as needed.
        Check that all dimension match and then convert all the indexed
        parameters into lists of length T by replication if they are not
        already.

        Parameters:
            F: State transition matrices
            L: Control matrices
            X: State cost matrices
            U: Control cost matrices
            XT: Terminal state cost matrix (default: eye())
            c: State offsets (default: zeros())
            H: State cost output matrices (default: eye())
            r: State cost output offsets (i.e., the reference trajectory) (default: zeros())
            HT: Terminal state cost output matrix (default: eye())
            rT: Terminal state cost output offset (default: zeros())
            Z: Control cost output matrices (default: eye())
            s: Control cost output offsets (default: zeros())
            M: Cross term matrices (default: zeros())
            T: Number of time steps (default: deduced from F,L,X,U)

        Returns:
            LQT: The LQT object
        """
        # Figure out T
        if isinstance(F, list):
            T = len(F)
        elif isinstance(L, list):
            T = len(L)
        elif isinstance(X, list):
            T = len(X)
        elif isinstance(U, list):
            T = len(U)
        else:
            if T is None:
                raise ValueError("Parameter T cannot be None when F,L,X,U are matrices.")

        # Figure out sizes
        if isinstance(F, list):
            xdim = F[0].shape[0]
        else:
            xdim = F.shape[0]

        if isinstance(L, list):
            udim = L[0].shape[1]
        else:
            udim = L.shape[1]

        if isinstance(X, list):
            rdim = X[0].shape[0]
        else:
            rdim = X.shape[0]

        # Apply defaults
        if XT is None:
            XT = np.eye(rdim)
        if c is None:
            c = np.zeros(xdim)
        if H is None:
            H = np.eye(rdim, xdim)
        if r is None:
            r = np.zeros(rdim)
        if HT is None:
            HT = np.eye(rdim, xdim)
        if rT is None:
            rT = np.zeros(rdim)
        if Z is None:
            Z = np.eye(udim)
        if s is None:
            s = np.zeros(udim)
        if M is None:
            M = np.zeros((rdim, udim))

        # This looks a bit clumsy indeed:
        if not isinstance(F, list):
            F = T * [F]
        elif len(F) != T:
            raise ValueError(f"F has wrong length ({len(F)}), should be T={T}")
        if not isinstance(L, list):
            L = T * [L]
        elif len(L) != T:
            raise ValueError(f"L has wrong length ({len(L)}), should be T={T}")
        if not isinstance(X, list):
            X = T * [X]
        elif len(X) != T:
            raise ValueError(f"X has wrong length ({len(X)}), should be T={T}")
        if not isinstance(U, list):
            U = T * [U]
        elif len(U) != T:
            raise ValueError(f"U has wrong length ({len(U)}), should be T={T}")
        if not isinstance(c, list):
            c = T * [c]
        elif len(c) != T:
            raise ValueError(f"c has wrong length ({len(c)}), should be T={T}")
        if not isinstance(H, list):
            H = T * [H]
        elif len(H) != T:
            raise ValueError(f"H has wrong length ({len(H)}), should be T={T}")
        if not isinstance(r, list):
            r = T * [r]
        elif len(r) != T:
            raise ValueError(f"r has wrong length ({len(r)}), should be T={T}")
        if not isinstance(Z, list):
            Z = T * [Z]
        elif len(Z) != T:
            raise ValueError(f"Z has wrong length ({len(Z)}), should be T={T}")
        if not isinstance(s, list):
            s = T * [s]
        elif len(s) != T:
            raise ValueError(f"s has wrong length ({len(s)}), should be T={T}")
        if not isinstance(M, list):
            M = T * [M]
        elif len(M) != T:
            raise ValueError(f"F has wrong length ({len(M)}), should be T={T}")

        # Check that the sizes match
        rdim = XT.shape[0]  # This can vary
        if XT.shape != (rdim, rdim):
            raise ValueError(f"XT has wrong shape {XT.shape}, should be {(rdim, rdim)}")
        if HT.shape != (rdim, xdim):
            raise ValueError(f"HT has wrong shape {HT.shape}, should be {(rdim, xdim)}")
        if rT.shape != (rdim,):
            raise ValueError(f"rT has wrong shape {rT.shape}, should be {(rdim,)}")

        for k in range(T):
            # We don't let state and input to be variable dimensional
            xdim = F[0].shape[0]
            udim = L[0].shape[1]

            if F[k].shape != (xdim, xdim):
                raise ValueError(f"F[{k}] has wrong shape {F[k].shape}, should be {(xdim, xdim)}")
            if L[k].shape != (xdim, udim):
                raise ValueError(f"L[{k}] has wrong shape {L[k].shape}, should be {(xdim, udim)}")

            # r can be variable dimensional
            rdim = X[k].shape[0]

            if X[k].shape != (rdim, rdim):
                raise ValueError(f"X[{k}] has wrong shape {X[k].shape}, should be {(rdim, rdim)}")
            if U[k].shape != (udim, udim):
                raise ValueError(f"U[{k}] has wrong shape {U[k].shape}, should be {(udim, udim)}")
            if c[k].shape != (xdim,):
                raise ValueError(f"c[{k}] has wrong shape {c[k].shape}, should be {(xdim,)}")
            if H[k].shape != (rdim, xdim):
                raise ValueError(f"H[{k}] has wrong shape {H[k].shape}, should be {(rdim, xdim)}")
            if r[k].shape != (rdim,):
                raise ValueError(f"r[{k}] has wrong shape {r[k].shape}, should be {(rdim,)}")
            if Z[k].shape != (udim, udim):
                raise ValueError(f"Z[{k}] has wrong shape {Z[k].shape}, should be {(udim, udim)}")
            if s[k].shape != (udim,):
                raise ValueError(f"s[{k}] has wrong shape {s[k].shape}, should be {(udim,)}")
            if M[k].shape != (rdim, udim):
                raise ValueError(f"M[{k}] has wrong shape {M[k].shape}, should be {(rdim, udim)}")

        return cls(F, L, X, U, XT, c, H, r, HT, rT, Z, s, M)

    ###########################################################################
    # Sequential computation of gains, value functions, states, and controls
    ###########################################################################

    def seqBackwardPass(self):
        """Sequential backward pass to compute control laws and value functions.

        Returns:
            Kx_list: List of control gains for 0:T-1.
            d_list: List of control offsets for 0:T-1.
            S_list: List of value function matrices for 0:T.
            v_list: List of value function offsets for 0:T.
        """
        T = len(self.F)

        S = self.HT.T @ self.XT @ self.HT
        v = self.HT.T @ self.XT @ self.rT

        Kx_list = []
        d_list = []
        S_list = [S]
        v_list = [v]

        for k in reversed(range(T)):
            CF, low = linalg.cho_factor(self.Z[k].T @ self.U[k] @ self.Z[k]
                                        + self.L[k].T @ S @ self.L[k])
            Kx = linalg.cho_solve((CF, low),
                                  self.Z[k].T @ self.M[k].T @ self.H[k]
                                  + self.L[k].T @ S @ self.F[k])
            d = linalg.cho_solve((CF, low),
                                 self.Z[k].T @ self.U[k] @ self.s[k]
                                 + self.Z[k].T @ self.M[k].T @ self.r[k]
                                 - self.L[k].T @ S @ self.c[k]
                                 + self.L[k].T @ v)
            v = self.H[k].T @ self.X[k] @ self.r[k] \
                - Kx.T @ self.Z[k].T @ self.U[k] @ self.s[k] \
                + self.H[k].T @ self.M[k] @ self.s[k] \
                - Kx.T @ self.Z[k].T @ self.M[k].T @ self.r[k] \
                + (self.F[k] - self.L[k] @ Kx).T \
                @ (v - S @ self.c[k])
            S = self.H[k].T @ self.X[k] @ self.H[k] \
                - self.H[k].T @ self.M[k] @ self.Z[k] @ Kx \
                + self.F[k].T @ S @ (self.F[k] - self.L[k] @ Kx)
            Kx_list.append(Kx)
            d_list.append(d)
            S_list.append(S)
            v_list.append(v)

        Kx_list.reverse()
        d_list.reverse()
        S_list.reverse()
        v_list.reverse()

        return Kx_list, d_list, S_list, v_list

    def seqForwardPass(self, x0, Kx_list, d_list):
        """ Sequential forward pass to compute states and controls.

        Parameters:
            x0: Initial state.
            Kx_list: List of control gains for 0:T-1.
            d_list: List of control offsets for 0:T-1.

        Returns:
            u_list: List of controls for 0:T-1.
            x_list: List of states for 0:T.
        """
        u_list = []
        x_list = [x0]

        x = x0
        for k in range(len(Kx_list)):
            u = -Kx_list[k] @ x + d_list[k]
            x = self.F[k] @ x + self.c[k] + self.L[k] @ u
            u_list.append(u)
            x_list.append(x)

        return u_list, x_list

    def seqSimulation(self, x0, u_list):
        """ Sequential simulation of the system.

        Parameters:
            x0: Initial state.
            u_list: List of controls for 0:T-1.

        Returns:
            x_list: List of states for 0:T.
        """
        x_list = [x0]

        x = x0
        for k in range(len(u_list)):
            u = u_list[k]
            x = self.F[k] @ x + self.c[k] + self.L[k] @ u
            x_list.append(x)

        return x_list

    ###########################################################################
    # Parallel (simulated parallel) computation of the gains and value functions
    ###########################################################################

    def parBackwardPass_init(self):
        """ Parallel LQT backward pass initialization.

        Returns:
            elems: List of tuples (A, b, C, eta, J) for 0:T.
        """
        T = len(self.F)

        elems = []
        for k in range(T):
            LU, piv = linalg.lu_factor(self.U[k] @ self.Z[k])
            A = self.F[k] - self.L[k] @ linalg.lu_solve((LU, piv),
                                                        self.M[k].T @ self.H[k])
            b = self.c[k] + self.L[k] @ linalg.lu_solve((LU, piv),
                                                        self.M[k].T @ self.r[k]) \
                + self.L[k] @ linalg.solve(self.Z[k], self.s[k])
            C = self.L[k] @ linalg.solve(self.Z[k].T @ self.U[k] @ self.Z[k], self.L[k].T,
                                         assume_a='pos')
            Y = self.X[k] - self.M[k] @ linalg.solve(self.U[k], self.M[k].T)
            eta = self.H[k].T @ Y @ self.r[k]
            J = self.H[k].T @ Y @ self.H[k]
            elem = (A, b, C, eta, J)
            elems.append(elem)

        A = np.zeros_like(self.F[0])
        b = np.zeros_like(self.F[0][:, 0])
        C = np.zeros_like(self.F[0])
        eta = self.HT.T @ self.XT @ self.rT
        J = self.HT.T @ self.XT @ self.HT
        elem = (A, b, C, eta, J)
        elems.append(elem)

        return elems

    def parBackwardPass_extract(self, elems):
        """ Parallel LQT backward pass result extraction.

        Parameters:
            elems: List of tuples (A, b, C, eta, J) for 0:T.

        Returns:
            Kx_list: List of control gains for 0:T-1.
            d_list: List of control offsets for 0:T-1.
            S_list: List of value function matrices for 0:T.
            v_list: List of value function offsets for 0:T.
        """
        (A, b, C, eta, J) = elems[0]

        Kx_list = []
        d_list = []
        S_list = [J]
        v_list = [eta]
        for k in range(len(elems) - 1):
            (A, b, C, eta, J) = elems[k + 1]
            S = J
            v = eta
            CF, low = linalg.cho_factor(self.Z[k].T @ self.U[k] @ self.Z[k]
                                        + self.L[k].T @ S @ self.L[k])
            Kx = linalg.cho_solve((CF, low),
                                  self.Z[k].T @ self.M[k].T @ self.H[k]
                                  + self.L[k].T @ S @ self.F[k])
            d = linalg.cho_solve((CF, low),
                                 self.Z[k].T @ self.U[k] @ self.s[k]
                                 + self.Z[k].T @ self.M[k].T @ self.r[k]
                                 - self.L[k].T @ S @ self.c[k]
                                 + self.L[k].T @ v)

            Kx_list.append(Kx)
            d_list.append(d)
            S_list.append(S)
            v_list.append(v)

        return Kx_list, d_list, S_list, v_list

    def parBackwardPass(self):
        """ Parallel LQT backward pass.

        Returns:
            Kx_list: List of control gains for 0:T-1.
            d_list: List of control offsets for 0:T-1.
            S_list: List of value function matrices for 0:T.
            v_list: List of value function offsets for 0:T.
        """

        # Initialize
        elems = self.parBackwardPass_init()

        # Call the associative scan
        elems = par_backward_pass_scan(elems)

        # Extract the results
        return self.parBackwardPass_extract(elems)

    ###########################################################################
    # Parallel (simulated parallel) computation of the states and controls,
    # this version is with composition of functions formula
    ###########################################################################

    def parForwardPass_init(self, x0, Kx_list, d_list):
        """ Parallel LQT forward pass initialization.

        Parameters:
            x0: Initial state.
            Kx_list: List of control gains for 0:T-1.
            d_list: List of control offsets for 0:T-1.

        Returns:
            elems: List of tuples (F, c) for 0:T.
        """

        elems = []
        for k in range(len(Kx_list)):
            if k == 0:
                tF = np.zeros_like(self.F[k])
                tc = (self.F[k] - self.L[k] @ Kx_list[k]) @ x0 \
                     + self.c[k] + self.L[k] @ d_list[k]
            else:
                tF = self.F[k] - self.L[k] @ Kx_list[k]
                tc = self.c[k] + self.L[k] @ d_list[k]
            elems.append((tF, tc))

        return elems

    def parForwardPass_extract(self, x0, Kx_list, d_list, elems):
        """ Parallel LQT forward pass result extraction.

        Parameters:
            x0: Initial state.
            Kx_list: List of control gains for 0:T-1.
            d_list: List of control offsets for 0:T-1.
            elems: List of tuples (F, c) for 0:T-1.

        Returns:
            u_list: List of controls for 0:T-1.
            x_list: List of states for 0:T.
        """
        x_list = [x0]
        u_list = []

        for k in range(len(Kx_list)):
            if k == 0:
                u = -Kx_list[k] @ x0 + d_list[k]
            else:
                tF, tc = elems[k - 1]
                u = -Kx_list[k] @ tc + d_list[k]

            tF, tc = elems[k]
            x = tc
            x_list.append(x)
            u_list.append(u)

        return u_list, x_list

    def parForwardPass(self, x0, Kx_list, d_list):
        """ Parallel LQT forward pass.

        Parameters:
            x0: Initial state.
            Kx_list: List of control gains for 0:T-1.
            d_list: List of control offsets for 0:T-1.

        Returns:
            u_list: List of controls for 0:T-1.
            x_list: List of states for 0:T.
        """
        # Initialize
        elems = self.parForwardPass_init(x0, Kx_list, d_list)

        # Call the associative scan
        elems = par_forward_pass_scan(elems)

        # Extract the results
        return self.parForwardPass_extract(x0, Kx_list, d_list, elems)

    ###########################################################################
    # Parallel (simulated parallel) computation of the states and controls,
    # this version is with forward-backward value function combination
    # with auxiliary initialization for x0
    ###########################################################################

    def parFwdBwdPass_init(self, x0):
        """ Parallel LQT forward-backward pass initialization.

        Parameters:
            x0: Initial state.

        Returns:
            elems: List of tuples (A, b, C, eta, J) for 0:T.
        """
        T = len(self.F)

        elems = []
        A = np.zeros_like(self.F[0])
        b = x0
        C = np.zeros_like(self.F[0])
        eta = np.zeros_like(b)
        J = np.zeros_like(C)
        elem = (A, b, C, eta, J)
        elems.append(elem)

        # Note that here we have some copy-paste code from parBackwardPass_init()
        for k in range(T):
            LU, piv = linalg.lu_factor(self.U[k] @ self.Z[k])
            A = self.F[k] - self.L[k] @ linalg.lu_solve((LU, piv),
                                                        self.M[k].T @ self.H[k])
            b = self.c[k] + self.L[k] @ linalg.lu_solve((LU, piv),
                                                        self.M[k].T @ self.r[k]) \
                + self.L[k] @ linalg.solve(self.Z[k], self.s[k])
            C = self.L[k] @ linalg.solve(self.Z[k].T @ self.U[k] @ self.Z[k], self.L[k].T,
                                         assume_a='pos')
            Y = self.X[k] - self.M[k] @ linalg.solve(self.U[k], self.M[k].T)
            eta = self.H[k].T @ Y @ self.r[k]
            J = self.H[k].T @ Y @ self.H[k]
            elem = (A, b, C, eta, J)
            elems.append(elem)

        return elems

    def parFwdBwdPass_extract(self, Kx_list, d_list, S_list, v_list, elems):
        """ Parallel LQT forward-backward pass result extraction.

        Parameters:
            Kx_list: List of control gains for 0:T-1.
            d_list: List of control offsets for 0:T-1.
            S_list: List of backward value function matrices for 0:T.
            v_list: List of backward value function offsets for 0:T.
            elems: List of forward tuples (A, b, C, eta, J) for 0:T.

        Returns:
            u_list: List of controls for 0:T-1.
            x_list: List of states for 0:T.
        """
        u_list = []
        x_list = []

        for k in range(len(elems)):
            (A, b, C, eta, J) = elems[k]

            x = linalg.solve(np.eye(C.shape[0]) + C @ S_list[k], b + C @ v_list[k])
            x_list.append(x)
            if k < len(elems) - 1:
                u = -Kx_list[k] @ x + d_list[k]
                u_list.append(u)

        return u_list, x_list

    def parFwdBwdPass(self, x0, Kx_list, d_list, S_list, v_list):
        """ Parallel LQT forward-backward pass.

        Parameters:
            x0: Initial state.
            Kx_list: List of control gains for 0:T-1.
            d_list: List of control offsets for 0:T-1.
            S_list: List of backward value function matrices for 0:T.
            v_list: List of backward value function offsets for 0:T.

        Returns:
            u_list: List of controls for 0:T-1.
            x_list: List of states for 0:T.
        """

        # Initialize
        elems = self.parFwdBwdPass_init(x0)

        # Call the associative scan
        elems = par_fwdbwd_pass_scan(elems)

        # Extract the results
        return self.parFwdBwdPass_extract(Kx_list, d_list, S_list, v_list, elems)

    ###########################################################################
    # Parallel (simulated parallel) computation of the states and controls,
    # this version is with forward-backward value function combination
    # with evaluation at x0 in the end.
    ###########################################################################

    def parFwdBwdPass2_init(self):
        """ Parallel LQT forward-backward pass initialization version 2.

        Returns:
            elems: List of tuples (A, b, C, eta, J) for 0:T.
        """
        T = len(self.F)

        elems = []

        # Note that here we have some copy-paste code from parBackwardPass_init()
        for k in range(T):
            LU, piv = linalg.lu_factor(self.U[k] @ self.Z[k])
            A = self.F[k] - self.L[k] @ linalg.lu_solve((LU, piv),
                                                        self.M[k].T @ self.H[k])
            b = self.c[k] + self.L[k] @ linalg.lu_solve((LU, piv),
                                                        self.M[k].T @ self.r[k]) \
                + self.L[k] @ linalg.solve(self.Z[k], self.s[k])
            C = self.L[k] @ linalg.solve(self.Z[k].T @ self.U[k] @ self.Z[k], self.L[k].T,
                                         assume_a='pos')
            Y = self.X[k] - self.M[k] @ linalg.solve(self.U[k], self.M[k].T)
            eta = self.H[k].T @ Y @ self.r[k]
            J = self.H[k].T @ Y @ self.H[k]
            elem = (A, b, C, eta, J)
            elems.append(elem)

        return elems

    def parFwdBwdPass2_extract(self, x0, Kx_list, d_list, S_list, v_list, elems):
        """ Parallel LQT forward-backward pass result extraction version 2.

        Parameters:
            x0: Initial state.
            Kx_list: List of control gains for 0:T-1.
            d_list: List of control offsets for 0:T-1.
            S_list: List of backward value function matrices for 0:T.
            v_list: List of backward value function offsets for 0:T.
            elems: List of forward tuples (A, b, C, eta, J) for 0:T.

        Returns:
            u_list: List of controls for 0:T-1.
            x_list: List of states for 0:T.
        """

        u_list = []
        x_list = []

        for k in range(len(elems) + 1):
            if k == 0:
                x = x0
            else:
                (A, b, C, eta, J) = elems[k - 1]
                x = linalg.solve(np.eye(C.shape[0]) + C @ S_list[k], A @ x0 + b + C @ v_list[k])
            x_list.append(x)

            if k < len(elems):
                u = -Kx_list[k] @ x + d_list[k]
                u_list.append(u)

        return u_list, x_list

    def parFwdBwdPass2(self, x0, Kx_list, d_list, S_list, v_list):
        """ Parallel LQT forward-backward pass version 2.

        Parameters:
            x0: Initial state.
            Kx_list: List of control gains for 0:T-1.
            d_list: List of control offsets for 0:T-1.
            S_list: List of backward value function matrices for 0:T.
            v_list: List of backward value function offsets for 0:T.

        Returns:
            u_list: List of controls for 0:T-1.
            x_list: List of states for 0:T.
        """

        # Initialize
        elems = self.parFwdBwdPass2_init()

        # Call the associative scan
        elems = par_fwdbwd_pass_scan(elems)

        # Extract the results
        return self.parFwdBwdPass2_extract(x0, Kx_list, d_list, S_list, v_list, elems)

    ###########################################################################
    # Batch solution to the general LQT problem. Note: does only allow
    # for constant sized r_k for k=0,...,T.
    ###########################################################################

    def batchSolution(self, x0):
        """ Batch solution to the general LQT problem.

        Parameters:
            x0: Initial state.

        Returns:
            u_list: List of controls for 0:T-1.
            x_list: List of states for 0:T.
        """

        # Initialize
        T = len(self.F)
        n = self.F[0].shape[0]  # x size
        m = self.L[0].shape[1]  # u size
        p = self.H[0].shape[0]  # output size

        A = np.zeros((n * T, n * T))
        d = np.zeros(n * T)
        B = np.zeros((n * T, m * T))
        tH1 = np.zeros((p * T, n * T))
        tr1 = np.zeros(p * T)
        tX = np.zeros((p * T, p * T))
        tH0 = np.zeros((p * T, n * T))
        tr0 = np.zeros(p * T)
        tZ = np.zeros((m * T, m * T))
        ts = np.zeros(m * T)
        tU = np.zeros((m * T, m * T))
        tM = np.zeros((p * T, m * T))

        # Fill in the matrices
        for i in range(T):
            A[n * i:n * (i + 1), n * i:n * (i + 1)] = np.eye(n)
            if i > 0:
                A[n * i:n * (i + 1), n * (i - 1):n * i] = -self.F[i]
                tH0[p * i:p * (i + 1), n * (i - 1):n * i] = self.H[i]
                tr0[p * i:p * (i + 1)] = self.r[i]
            else:
                tr0[p * i:p * (i + 1)] = self.r[i] - self.H[0] @ x0

            B[n * i:n * (i + 1), m * i:m * (i + 1)] = self.L[i]
            d[n * i:n * (i + 1)] = self.c[i]

            tZ[m * i:m * (i + 1), m * i:m * (i + 1)] = self.Z[i]
            ts[m * i:m * (i + 1)] = self.s[i]
            tU[m * i:m * (i + 1), m * i:m * (i + 1)] = self.U[i]
            tM[p * i:p * (i + 1), m * i:m * (i + 1)] = self.M[i]

            if i < T - 1:
                tH1[p * i:p * (i + 1), n * i:n * (i + 1)] = self.H[i + 1]
                tr1[p * i:p * (i + 1)] = self.r[i + 1]
                tX[p * i:p * (i + 1), p * i:p * (i + 1)] = self.X[i + 1]
            else:
                tH1[p * i:p * (i + 1), n * i:n * (i + 1)] = self.HT
                tr1[p * i:p * (i + 1)] = self.rT
                tX[p * i:p * (i + 1), p * i:p * (i + 1)] = self.XT

        d[0:n] = d[0:n] + self.F[0] @ x0

        tA1 = tH1 @ linalg.solve(A, B)
        tb1 = tr1 - tH1 @ linalg.solve(A, d)
        tA0 = tH0 @ linalg.solve(A, B)
        tb0 = tr0 - tH0 @ linalg.solve(A, d)

        # Solve the controls from a large linear system
        u = linalg.solve(tA1.T @ tX @ tA1 + tZ.T @ tU @ tZ + tA0.T @ tM @ tZ + tZ.T @ tM.T @ tA0,
                         tA1.T @ tX @ tb1 + tZ.T @ tU @ ts + tA0.T @ tM @ ts + tZ.T @ tM.T @ tb0)

        # Compute the states
        x = linalg.solve(A, B @ u + d)

        # Extract the results
        x_list = [x0]
        u_list = []
        for i in range(T):
            u_list.append(u[m * i:m * (i + 1)])
            x_list.append(x[n * i:n * (i + 1)])

        return u_list, x_list

    ###########################################################################
    # Processing with transformation from general cost to canonical form
    ###########################################################################

    def seqTransformedBackwardPass(self):
        """ Sequential backward pass for the transformed LQT problem.

        Returns:
            Kx_list: List of control gains for 0:T-1.
            d_list: List of control offsets for 0:T-1.
            S_list: List of value function matrices for 0:T.
            v_list: List of value function offsets for 0:T.
        """

        # Note: does only work with Z[k] = I
        T = len(self.F)
        n = self.F[0].shape[0]  # x size
        m = self.L[0].shape[1]  # u size
        p = self.H[0].shape[0]  # output size

        HT = self.HT
        rT = self.rT
        XT = self.XT
        F = []
        c = []
        L = []
        X = []
        H = []
        r = []
        U = []
        for k in range(T):
            UI = linalg.inv(self.U[k])
            F.append(self.F[k] - self.L[k] @ UI @ self.M[k].T @ self.H[k])
            c.append(self.c[k] + self.L[k] @ UI @ self.M[k].T @ self.r[k] + self.L[k] @ self.s[k])
            L.append(self.L[k])
            X.append(self.X[k] - self.M[k] @ UI @ self.M[k].T)
            H.append(self.H[k])
            r.append(self.r[k])
            U.append(self.U[k])

        trans_lqt = self.checkAndExpand(F, L, X, U, XT, c, H, r, HT, rT, T=T)

        Kx_list, d_list, S_list, v_list = trans_lqt.seqBackwardPass()

        for k in range(T):
            UI = linalg.inv(self.U[k])
            Kx_list[k] = Kx_list[k] + UI @ self.M[k].T @ self.H[k]
            d_list[k] = d_list[k] + UI @ self.M[k].T @ self.r[k] + self.s[k]

        return Kx_list, d_list, S_list, v_list

    ###########################################################################
    # Cost computation
    ###########################################################################

    def cost(self, u_list, x_list):
        """ Compute the cost of a trajectory.

        Parameters:
            u_list: List of control vectors for 0:T-1.
            x_list: List of state vectors for 0:T.

        Returns:
            res: Cost of the trajectory.
        """
        res = 0.5 * (self.HT @ x_list[-1] - self.rT).T @ self.XT @ (self.HT @ x_list[-1] - self.rT)

        for k in range(len(u_list)):
            res += 0.5 * (self.H[k] @ x_list[k] - self.r[k]).T @ self.X[k] @ (self.H[k] @ x_list[k] - self.r[k])
            res += 0.5 * (self.Z[k] @ u_list[k] - self.s[k]).T @ self.U[k] @ (self.Z[k] @ u_list[k] - self.s[k])
            res += (self.H[k] @ x_list[k] - self.r[k]).T @ self.M[k] @ (self.Z[k] @ u_list[k] - self.s[k])

        return res
