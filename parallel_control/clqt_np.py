"""
Continuous-time Numpy-based Linear Quadratic Regulator/Tracker routines, both sequential and parallel
(though the parallel ones are not really implemented in parallel but simulated). The aim of this is
to act as a reference implementation for more complex implementations (e.g. in TensorFlow).

@author: Simo Särkkä
"""

from scipy import linalg
import numpy as np
import parallel_control.lqt_np as lqt_np
import parallel_control.diffeq_np as diffeq_np
import parallel_control.disc as disc

##############################################################################
#
# Packing and unpacking routines for ODE solvers
#
##############################################################################

def pack_abcej(A,b,C,eta,J):
    """ Pack A,b,C,eta,J into a single vector x.

    Parameters:
        A, b, C, eta, J : Conditional value function parameters

    Returns:
        x : Packed vector
    """
    n = A.shape[0]
    x = np.zeros((n, 3 * n + 2))
    x[:, :n] = A
    x[:, n:(2 * n)] = C
    x[:, (2 * n):(3 * n)] = J
    x[:, 3 * n] = b
    x[:, 3 * n + 1] = eta
    return x

def unpack_abcej(x):
    """ Unpack A,b,C,eta,J from a packed vector x.

    Parameters:
        x : Packed vector

    Returns:
        A, b, C, eta, J : Conditional value function parameters
    """
    n = x.shape[0]
    A = x[:, :n]
    C = x[:, n:(2 * n)]
    J = x[:, (2 * n):(3 * n)]
    b = x[:, 3 * n]
    eta = x[:, 3 * n + 1]
    return A,b,C,eta,J

def pack_Psiphi(Psi,phi):
    """" Pack Psi,phi into a single vector x.

    Parameters:
        Psi, phi : Matrix and vector to be packed.

    Returns:
        x : Packed vector
    """
    n = Psi.shape[0]
    x = np.zeros((n, n+1))
    x[:, :n] = Psi
    x[:, n] = phi
    return x

def unpack_Psiphi(x):
    """ Unpack Psi,phi from a packed vector x.

    Parameters:
        x : Packed vector

    Returns:
        Psi, phi : Unpacked matrix and vector
    """
    n = x.shape[0]
    Psi = x[:, :n]
    phi = x[:, n]
    return Psi, phi

def pack_abc(A,b,C):
    """ Pack A,b,C into a single vector x.

    Parameters:
        A, b, C : Partial conditional value function parameters

    Returns:
        x : Packed vector
    """
    n = A.shape[0]
    x = np.zeros((n, 2 * n + 1))
    x[:, :n] = A
    x[:, n:(2 * n)] = C
    x[:, 2 * n] = b
    return x

def unpack_abc(x):
    """ Unpack A,b,C from a packed vector x.

    Parameters:
        x : Packed vector

    Returns:
        A, b, C : Partial conditional value function parameters
    """
    n = x.shape[0]
    A = x[:, :n]
    C = x[:, n:(2 * n)]
    b = x[:, 2 * n]
    return A,b,C

def pack_Sv(S,v):
    """ Pack value function parameters S,v into a single vector x.

    Parameters:
        S, v : Value function parameters

    Returns:
        x : Packed vector
    """
    n = S.shape[0]
    x = np.zeros((n, n + 1))
    x[:, :n] = S
    x[:, n] = v
    return x

def unpack_Sv(x):
    """ Unpack value function parameters S,v from a packed vector x.

    Parameters:
        x : Packed vector

    Returns:
        S, v : Value function parameters
    """
    n = x.shape[0]
    S = x[:, :n]
    v = x[:, n]
    return S,v


##############################################################################
#
# Continuous-time Sequential / parallel LQR -- LQT can be found below
#
##############################################################################

class CLQR:
    """
    Class containing basic LQR model parameters and methods for computing
    all kinds of useful things from it. The model is

    dx/dt = F x + L u
      J(u) = 1/2 x(T).T XT x(T)
         1/2 int_0^T [x(t).T X x(t) + 1/2 u(t).T U u(t)] dt

    """

    def __init__(self, F, L, X, U, XT, T):
        """ Initialize the CLQR model.

        Parameters:
            F: State feedback matrix
            L: Control feedback matrix
            X: State cost matrix
            U: Control cost matrix
            XT: Terminal state cost matrix
            T: Time horizon
        """
        self.F = F
        self.L = L
        self.X = X
        self.U = U
        self.XT = XT
        self.T = T

    ###########################################################################
    # Sequential computation of gains, value functions, states, and controls
    ###########################################################################

    def seqBackwardPass(self,steps,dt=None,S=None,method='rk4'):
        """
        Solve the Riccati equation and gains backwards from terminal condition.

        Parameters
        ----------
        steps : int
            Number of steps
        dt : float
            Time step, default dt=self.T/steps
        S : np.array
            The final step matrix, default self.XT
        method : str
          'rk4' uses Runge-Kutta (default); 'exact' uses matrix fractions to compute the exact solution.

        Returns
        -------
        Kx_list, S_list : Tuple(List, List)
            List of gains, list of Riccati solutions S (including initial and final)
        """

        if S is None:
            S = self.XT
        if dt is None:
            dt = self.T / steps

        S_list = [S]
        Kx_list = []

        ULT = linalg.solve(self.U, self.L.T, assume_a='pos')
        LULT = self.L @ ULT

        if method == 'rk4':
            # Use Runge-Kutta to solve the Riccati equation
            for _ in reversed(range(steps)):
                S = diffeq_np.rk4(lambda S: -self.F.T @ S - S @ self.F + S @ LULT @ S - self.X, -dt, S)
                S = 0.5 * (S + S.T)
                Kx = ULT @ S
                S_list.append(S)
                Kx_list.append(Kx)

        elif method == 'exact':
            # Use the method from Kalman (1960) Boletin de la Sociedad Matematica Mexicana. If we have
            #   H = [F  -LU^{-1}L^T;
            #        -X    -F^T]
            # and consider d[C;D]/dt = H [C;D], then with S = D C^{-1} we get
            #   d[D C^{-1}] = dD C^{-1} - D C^{-1} dC C^{-1}
            #   = -X C C^{-1} - F^T D C^{-1} - D C^{-1} F C C^{-1} + D C^{-1} LU^{-1}L^T D C^{-1}
            #   = -X - F^T S - S F + S LU^{-1}L^T S
            # i.e. if Psi = expm((t-T) H), then
            #   S(t) = (Psi11 + Psi12 @ S(T)) (Psi21 + Psi22)^{-1}
            n = self.F.shape[0]
            H = np.block([[self.F, -LULT], [-self.X, -self.F.T]])
            Psi = linalg.expm(-dt*H)
            CD = np.block([[np.eye(n)],[S]])
            for _ in reversed(range(steps)):
                CD = Psi @ CD
                S  = linalg.solve(CD[0:n,:], CD[n:,:].T, transposed=True)
                S = 0.5 * (S + S.T)
                Kx = ULT @ S
                S_list.append(S)
                Kx_list.append(Kx)

        else:
            raise ValueError(f"Unknown method {method}")

        S_list.reverse()
        Kx_list.reverse()

        return Kx_list, S_list


    def seqForwardPass(self, x0, Kx_list, dt=None, method='rk4', u_zoh=False):
        """
          Solve the forward trajectory using ZOH for Kx and optionally for u. Note that when u_zoh=False, then x is more
          accurate, but we cannot exactly get x when controlling it with the returned u_list.

        Parameters
        ----------
        x0 : np.array
          Initial state
        Kx_list : List
          List of gains
        dt : float
            Time step, default dt=self.T/steps
        method : str
          'rk4' uses Runge-Kutta (default); 'exact' uses matrix exponentials to compute the exact solution.
        u_zoh : bool
          use ZOH also for u when computing x, default False

        Returns
        -------
        u_list, x_list : Tuple(List, List)
            List of inputs, list of states (initial and final)
        """

        steps = len(Kx_list)
        if dt is None:
            dt = self.T / steps
        u_list = []
        x_list = [x0]

        if method == 'rk4':
            x = x0
            for k in range(steps):
                u = - Kx_list[k] @ x
                if u_zoh:
                    f = lambda x: self.F @ x + self.L @ u
                else:
                    f = lambda x: (self.F - self.L @ Kx_list[k]) @ x
                x = diffeq_np.rk4(f, dt, x)
                u_list.append(u)
                x_list.append(x)

        elif method == 'exact':
            x = x0
            A, B, _ = disc.lti_disc_u(self.F, self.L, self.L, dt=dt)
            for k in range(steps):
                u = - Kx_list[k] @ x
                if u_zoh:
                    x = A @ x + B @ u
                else:
                    x = linalg.expm((self.F - self.L @ Kx_list[k]) * dt) @ x
                u_list.append(u)
                x_list.append(x)

        else:
            raise ValueError(f"Unknown method {method}")

        return u_list, x_list

    def fwdbwdpass_ode_f(self, x):
        """" ODE function for forward integration of A, b, C.

        Parameters:
             x: Current packed vector x of (A,b,C)

        Returns:
            dx: Time derivative of packed x of (A,b,C)
        """
        ULT = linalg.solve(self.U, self.L.T, assume_a='pos')
        LULT = self.L @ ULT

        A, b, C = unpack_abc(x)

        dA = self.F @ A - C @ self.X @ A
        db = self.F @ b - C @ self.X @ b
        dC = self.F @ C - C @ self.X @ C + C @ self.F.T + LULT

        dx = pack_abc(dA,db,dC)

        return dx

    def seqFwdBwdPass(self, x0, steps, dt=None, A=None, b=None, C=None, method='rk4'):
        """" Solve forward value function parameters A, b, C forward.

        Parameters:
            x0: Initial state.
            steps: Number of steps.
            dt: Time step, default dt=self.T/steps
            A: Initial A, default A=zeros
            b: Initial b, default b=x0
            C: Initial C, default C=zeros
            method: 'rk4' uses Runge-Kutta (default and only supported);

        Returns:
            A_list, b_list, C_list: Lists of A, b, C
        """
        if dt is None:
            dt = self.T / steps
        if A is None:
            A = np.zeros_like(self.F)
        if b is None:
            b = x0
        if C is None:
            C = np.zeros_like(self.F)

        A_list = [A]
        b_list = [b]
        C_list = [C]

        if method == 'rk4':
            for k in range(steps):
                x = pack_abc(A, b, C)
                x = diffeq_np.rk4(self.fwdbwdpass_ode_f, dt, x)
                A, b, C = unpack_abc(x)
                C = 0.5 * (C + C.T)
                A_list.append(A)
                b_list.append(b)
                C_list.append(C)

        elif method == 'exact':
            raise ValueError(f"Not yet implemented: {method}")
        else:
            raise ValueError(f"Unknown method: {method}")

        return A_list, b_list, C_list


    def combineForwardBackward(self, Kx_list, S_list, A_list, b_list, C_list):
        """ Combine forward and backward pass to get u and x.

        Parameters:
            Kx_list: List of gains
            S_list: List of S matrices
            A_list: List of A matrices
            b_list: List of b vectors
            C_list: List of C matrices

        Returns:
            u_list, x_list: List of inputs, list of states
        """
        u_list = []
        x_list = []

        for k in range(len(S_list)):
            S = S_list[k]
            A = A_list[k] # These should actually be zero
            b = b_list[k]
            C = C_list[k]
            x = linalg.solve(np.eye(C.shape[0]) + C @ S, b)
            x_list.append(x)
            if k < len(Kx_list):
                u = -Kx_list[k] @ x
                u_list.append(u)

        return u_list, x_list


    ###########################################################################
    # Parallel computation of gains and value functions backwards
    ###########################################################################

    def bwpass_bw_ode_f(self, x):
        """" ODE function for backward integration of A, b, C, eta, J.

         Parameters:
                x: Current packed vector x of (A,b,C,eta,J)

        Returns:
            dx: Time derivative of packed x of (A,b,C,eta,J)
        """
        ULT = linalg.solve(self.U, self.L.T, assume_a='pos')
        LULT = self.L @ ULT

        A, b, C, eta, J = unpack_abcej(x)

        dA = A @ LULT @ J - A @ self.F
        db = -A @ LULT @ eta
        dC = -A @ LULT @ A.T
        deta = J @ LULT @ eta - self.F.T @ eta
        dJ = -J @ self.F + J @ LULT @ J - self.F.T @ J - self.X

        dx = pack_abcej(dA,db,dC,deta,dJ)

        return dx

    def bwpass_fw_ode_f(self, x):
        """" ODE function for forward integration of A, b, C, eta, J.

        Parameters:
            x: Current packed vector x of (A,b,C,eta,J)

        Returns:
            dx: Time derivative of packed x of (A,b,C,eta,J)
        """
        ULT = linalg.solve(self.U, self.L.T, assume_a='pos')
        LULT = self.L @ ULT

        A, b, C, eta, J = unpack_abcej(x)

        dA = self.F @ A - C @ self.X @ A
        db = self.F @ b - C @ self.X @ b
        dC = self.F @ C - C @ self.X @ C + C @ self.F.T + LULT
        deta = -A.T @ self.X @ b
        dJ = A.T @ self.X @ A

        dx = pack_abcej(dA,db,dC,deta,dJ)

        return dx

    def parBackwardPass_init(self, blocks, steps, method='rk4', forward=False):
        """
         Initialize the parallel backward pass with "blocks" blocks using "steps" backward integration
            steps. Returns list of initial elements.
        
        Parameters
        ----------
        blocks : int
          Number of blocks to split
        steps : int
          Number of steps per block
        method : str
          'rk4' uses Runge-Kutta (default); 'exact' uses matrix fractions to compute the exact solution (TODO).
        forward : bool
          Use forward (as opposed to backward) versions of the differential equations

        Returns
        -------
        elems : List
          List of element tuples (A,b,C,eta,J)
        """

        A = np.eye(self.F.shape[0])
        b = np.zeros_like(self.F[:,0])
        C = np.zeros_like(self.F)
        eta = np.zeros_like(self.F[:,0])
        J = np.zeros_like(self.F)

        if method == 'rk4':
            dt = self.T / steps / blocks

            if forward:
                for _ in range(steps):
                    x = pack_abcej(A, b, C, eta, J)
                    x = diffeq_np.rk4(self.bwpass_fw_ode_f, dt, x)
                    A, b, C, eta, J = unpack_abcej(x)
                    C = 0.5 * (C + C.T)
                    J = 0.5 * (J + J.T)
            else:
                for _ in reversed(range(steps)):
                    x = pack_abcej(A, b, C, eta, J)
                    x = diffeq_np.rk4(self.bwpass_bw_ode_f, -dt, x)
                    A, b, C, eta, J = unpack_abcej(x)
                    C = 0.5 * (C + C.T)
                    J = 0.5 * (J + J.T)

        elif method == 'exact':
            raise ValueError(f"Not yet implemented method {method}")
        else:
            raise ValueError(f"Unknown method {method}")

        elems = blocks * [(A,b,C,eta,J)]

        A = np.zeros_like(self.F)
        b = np.zeros_like(self.F[:,0])
        C = np.zeros_like(self.F)
        eta = np.zeros_like(self.F[:,0])
        J = self.XT
        elem = (A,b,C,eta,J)
        elems.append(elem)

        return elems

    def parBackwardPass_extract(self, elems, steps, method='rk4'):
        """
         Extract parallel backward pass results, backward integrate the intermediate values, and compute gains.

        Parameters
        ----------
        elems : List
          List of element tuples (A,b,C,eta,J)
        steps : int
          Number of steps per block
        method : str
          'rk4' uses Runge-Kutta (default); 'exact' uses matrix fractions to compute the exact solution.

        Returns
        -------
        Kx_list, S_list : List
          List of gains, list of Riccati solutions
        """

        dt = self.T / steps / (len(elems)-1)

        (A, b, C, eta, J) = elems[0]

        S_list = [J]
        Kx_list = []
        for k in range(len(elems)-1):
            (A,b,C,eta,J) = elems[k+1]
            bKx_list, bS_list = self.seqBackwardPass(steps, dt=dt, S=J, method=method)
            for i in range(len(bKx_list)):
                S_list.append(bS_list[i+1])
                Kx_list.append(bKx_list[i])

        return Kx_list, S_list

    def parBackwardPass(self, blocks, steps, init_method='rk4', extract_method='rk4'):
        """
         Perform the parallel backward pass with "blocks" blocks using "steps" backward integration steps.

        Parameters
        ----------
        blocks : int
          Number of blocks to split
        steps : int
          Number of steps per block
        init_method : str
          'rk4' uses Runge-Kutta (default); 'exact' uses matrix fractions to compute the exact solution for
          initialization of the blocks (TODO).
        extract_method : str
          'rk4' uses Runge-Kutta (default); 'exact' uses matrix fractions to compute the exact solution for
          extraction of the final result.

        Returns
        -------
        Kx_list, S_list : Tuple(List, List)
          List of gains, list of Riccati solutions
        """

        # Initialize
        elems = self.parBackwardPass_init(blocks, steps, method=init_method)

        # Call the associative scan
        elems = lqt_np.par_backward_pass_scan(elems)

        # Extract the results
        return self.parBackwardPass_extract(elems, steps, method=extract_method)


    ###########################################################################
    # Parallel computation of states and controls forward
    ###########################################################################

    def fwpass_fw_ode_f(self,x,Kx):
        """ Forward pass forward ODE.

        Parameters:
            x: Packed vector of Psi and phi.

        Returns:
            dx: Time derivative of x.
        """
        Psi, phi = unpack_Psiphi(x)
        tF = self.F - self.L @ Kx
        dPsi = tF @ Psi
        dphi = tF @ phi
        dx = pack_Psiphi(dPsi, dphi)
        return dx

    def fwpass_bw_ode_f(self,x,Kx):
        """ Forward pass backward ODE.

        Parameters:
            x: Packed vector of Psi and phi.

        Returns:
            dx: Time derivative of x.
        """
        Psi, phi = unpack_Psiphi(x)
        tF = self.F - self.L @ Kx
        dPsi = -Psi @ tF
        dphi = np.zeros_like(phi)
        dx = pack_Psiphi(dPsi, dphi)
        return dx

    def parForwardPass_init(self, x0, Kx_list, blocks, steps, method='rk4', forward=True):
        """
        Initialize the parallel forward pass.

        Parameters
        ----------
        x0 : np.array
          Initial state
        Kx_list : List
          List of gains (of length blocks * steps)
        blocks : int
          Number of blocks
        steps : int
          Number of steps per block
        method : str
          'rk4' uses Runge-Kutta (default); 'exact' uses matrix fractions to compute the exact solution for
          initialization of the blocks (TODO).
        forward : bool
          If the integrations should be done forward (as opposed to backwards), default True

        Returns
        -------
        elems : List
          List of element tuples (Psi,phi)
        """

        if blocks * steps != len(Kx_list):
            raise ValueError(f"Invalid arguments: blocks * steps != len(Kx_list)")

        elems = []
#        Psi = np.zeros_like(self.F) # This would work as well
        Psi = np.eye(self.F.shape[0])
        phi = x0
        elems.append((Psi,phi))

        dt = self.T / steps / blocks

        for k in range(blocks):
            Psi = np.eye(self.F.shape[0])
            phi = np.zeros_like(x0)

            if method == 'rk4':
                if forward:
                    for i in range(steps):
                        x = pack_Psiphi(Psi, phi)
                        x = diffeq_np.rk4(self.fwpass_fw_ode_f, dt, x, param=Kx_list[k * steps + i])
                        Psi, phi = unpack_Psiphi(x)
                else:
                    for i in reversed(range(steps)):
                        x = pack_Psiphi(Psi, phi)
                        x = diffeq_np.rk4(self.fwpass_bw_ode_f, -dt, x, param=Kx_list[k * steps + i])
                        Psi, phi = unpack_Psiphi(x)

            elif method == 'exact':
                raise ValueError(f"Not yet implemented method {method}")
            else:
                raise ValueError(f"Unknown method {method}")

            elems.append((Psi, phi))

        return elems

    def parForwardPass_extract(self, Kx_list, elems, steps, method='rk4', u_zoh=False):
        """
         Extract parallel forward pass results, forward integrate the intermediate values, and compute controls.

        Parameters
        ----------
        Kx_list : List
          List of gains (of length blocks * steps)
        elems : List
          List of element tuples (Psi,phi)
        steps : int
          Number of steps per block
        method : str
          'rk4' uses Runge-Kutta (default); 'exact' uses matrix fractions to compute the exact solution.
        u_zoh : bool
          use ZOH also for u when computing x, default False

        Returns
        -------
        u_list, x_list : Tuple(List, List)
          List of controls, list of states
        """

        blocks = len(elems)-1
        if blocks * steps != len(Kx_list):
            raise ValueError(f"Invalid arguments: blocks * steps != len(Kx_list)")

        dt = self.T / steps / blocks

        u_list = []
        x_list = []
        for k in range(len(elems)-1):
            (Psi, phi) = elems[k]
            bu_list, bx_list = self.seqForwardPass(phi, Kx_list[k*steps:(k+1)*steps], dt=dt, method=method, u_zoh=u_zoh)
            for i in range(len(bu_list)):
                u_list.append(bu_list[i])
                x_list.append(bx_list[i])
        (Psi, phi) = elems[-1]
        x_list.append(phi)

        return u_list, x_list

    def parForwardPass(self, x0, Kx_list, blocks, steps, init_method='rk4', extract_method='rk4', u_zoh=False):
        """
        Perform parallel forward pass.

        Parameters
        ----------
        x0 : np.array
          Initial state
        Kx_list : List
          List of gains (of length blocks * steps)
        blocks : int
          Number of blocks
        steps : int
          Number of steps per block
        init_method : str
          'rk4' uses Runge-Kutta (default); 'exact' uses matrix fractions to compute the exact solution for
          initialization of the blocks (TODO).
        extract_method : str
          'rk4' uses Runge-Kutta (default); 'exact' uses matrix fractions to compute the exact solution
          at the extraction phase.
        u_zoh : bool
          use ZOH also for u when computing x, default False

        Returns
        -------
        u_list, x_list : Tuple(List, List)
          List of controls, list of states
        """

        elems = self.parForwardPass_init(x0, Kx_list, blocks, steps, method=init_method)
        elems = lqt_np.par_forward_pass_scan(elems)
        return self.parForwardPass_extract(Kx_list, elems, steps, method=extract_method, u_zoh=u_zoh)

    ###########################################################################
    # Parallel computation of (backward and) forward value functions
    ###########################################################################

    def parFwdBwdPass_init(self, x0, blocks, steps, method='rk4', forward=True):
        """
         Initialize the parallel value function forwaed pass with "blocks" blocks using "steps" backward integration
            steps. Returns list of initial elements.

        Parameters
        ----------
        x0 : np.array
          Initial state
        blocks : int
          Number of blocks to split
        steps : int
          Number of steps per block
        method : str
          'rk4' uses Runge-Kutta (default); 'exact' uses matrix fractions to compute the exact solution (TODO).
        forward : bool
          Use forward (as opposed to backward) versions of the differential equations

        Returns
        -------
        elems : List
          List of element tuples (A,b,C,eta,J)
        """

        elems = []

        A = np.zeros_like(self.F)
        b = x0
        C = np.zeros_like(self.F)
        eta = np.zeros_like(self.F[:, 0])
        J = np.zeros_like(self.F)
        elem = (A, b, C, eta, J)
        elems.append(elem)

        A = np.eye(self.F.shape[0])
        b = np.zeros_like(self.F[:, 0])
        C = np.zeros_like(self.F)
        eta = np.zeros_like(self.F[:, 0])
        J = np.zeros_like(self.F)

        # Here we have copy-paste from Backward_init:
        if method == 'rk4':
            dt = self.T / steps / blocks

            if forward:
                for _ in range(steps):
                    x = pack_abcej(A, b, C, eta, J)
                    x = diffeq_np.rk4(self.bwpass_fw_ode_f, dt, x)
                    A, b, C, eta, J = unpack_abcej(x)
                    C = 0.5 * (C + C.T)
                    J = 0.5 * (J + J.T)
            else:
                for _ in reversed(range(steps)):
                    x = pack_abcej(A, b, C, eta, J)
                    x = diffeq_np.rk4(self.bwpass_bw_ode_f, -dt, x)
                    A, b, C, eta, J = unpack_abcej(x)
                    C = 0.5 * (C + C.T)
                    J = 0.5 * (J + J.T)

        elif method == 'exact':
            raise ValueError(f"Not yet implemented method {method}")
        else:
            raise ValueError(f"Unknown method {method}")

        for i in range(blocks):
            elem = (A, b, C, eta, J)
            elems.append(elem)

        return elems

    def parFwdBwdPass_extract(self, Kx_list, S_list, elems, steps, method='rk4'):
        """ Extract the controls and states from the parallel forward-backward pass results.

        Parameters:
            Kx_list: List of gains.
            S_list: List of value function matrices.
            elems: List of (A, b, C, eta, J)
            steps: Number of steps per block.
            method: Numerical integration method 'rk4' or 'exact'.

        Returns:
            u_list: List of controls.
            x_list: List of states.
        """

        blocks = len(elems)-1
        if blocks * steps != len(Kx_list):
            raise ValueError(f"Invalid arguments: blocks * steps != len(Kx_list)")
        if blocks * steps != len(S_list)-1:
            raise ValueError(f"Invalid arguments: blocks * steps != len(S_list)-1")

        dt = self.T / steps / blocks

        u_list = []
        x_list = []
        for k in range(blocks):
            (A, b, C, eta, J) = elems[k]
            A_list, b_list, C_list = self.seqFwdBwdPass(None, steps, dt=dt, A=A, b=b, C=C, method=method)
            bu_list, bx_list = self.combineForwardBackward(Kx_list[k*steps:(k+1)*steps], S_list[k*steps:(k+1)*steps+1],
                                                           A_list, b_list, C_list)
            for i in range(len(bu_list)):
                u_list.append(bu_list[i])
                x_list.append(bx_list[i])
        x_list.append(bx_list[-1])

        return u_list, x_list

    def parFwdBwdPass(self, x0, Kx_list, S_list, blocks, steps, init_method='rk4', extract_method='rk4'):
        """ Parallel computation of forward and backward value functions.

        Parameters:
            x0: Initial state.
            Kx_list: List of gains.
            S_list: List of value function matrices.
            blocks: Number of blocks to split.
            steps: Number of steps per block.
            init_method: Numerical integration method for initialization 'rk4' or 'exact'.
            extract_method: Numerical integration method for extraction 'rk4' or 'exact'.

        Returns:
            u_list: List of controls.
            x_list: List of states.
        """

        elems = self.parFwdBwdPass_init(x0, blocks, steps, method=init_method, forward=True)
        elems = lqt_np.par_fwdbwd_pass_scan(elems)
        u_list, x_list = self.parFwdBwdPass_extract(Kx_list, S_list, elems, steps, method=extract_method)

        return u_list, x_list

##############################################################################
#
# Continuous-time Sequential / parallel LQT
#
##############################################################################

class CLQT:
    """
    Class containing basic LQT model parameters and methods for computing
    all kinds of useful things from it. The model is

    dx/dt = F(t) x + c(t) + L(t) u
      J(u) = 1/2 (HT x(T) - rT).T XT (HT x(T) - rT)
         1/2 int_0^T (H(t) x(t) - r(t)).T X (H(t) x(t) - r(t)) + 1/2 u(t).T U(t) u(t)] dt

    we can also shift the starting point to t0
    """

    def __init__(self, F, L, X, U, XT, c, H, r, HT, rT, T):
        """ Initialize the model.

        Parameters:
            F: Function t -> F(t) returning the state feedback matrix.
            L: Function t -> L(t) returning the control matrix.
            X: Function t -> X(t) returning the state cost matrix.
            U: Function t -> U(t) returning the control cost matrix.
            XT: The final state cost matrix.
            c: Function t -> c(t) returning the offset vector.
            H: Function t -> H(t) returning the state output matrix.
            r: Function t -> r(t) returning the reference trajectory.
            HT: The final state output matrix.
            rT: The final reference trajectory.
            T: The final time.
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
        self.T = T

    ###########################################################################
    # Sequential computation of gains, value functions, states, and controls
    ###########################################################################

    def riccati_ode_f(self, x, t):
        """" ODE function for backward integration of S, v.

        Parameters:
            x: Packed vector of (S, v).
            t: Time.

        Returns:
            dx: Time derivative of x (i.e., dS/dt, dv/dt).
        """

        S,v = unpack_Sv(x)

        U = self.U(t)
        L = self.L(t)
        ULT = linalg.solve(U, L.T, assume_a='pos')
        LULT = L @ ULT

        F = self.F(t)
        X = self.X(t)
        H = self.H(t)
        r = self.r(t)
        c = self.c(t)

        dS = -F.T @ S - S @ F + S @ LULT @ S - H.T @ X @ H
        dv = -H.T @ X @ r + S @ c - F.T @ v + S @ LULT @ v

        dx = pack_Sv(dS,dv)

        return dx


    def seqBackwardPass(self, steps, dt=None, t0=None, S=None, v=None):
        """
        Solve the Riccati equation and gains backwards from terminal condition.

        Parameters
        ----------
        steps : int
            Number of steps
        dt : float
            Time step, default dt=self.T/steps
        t0 : float
            Initial time (default 0)
        S : np.array
            The final step matrix, default self.HT.T @ self.XT @ self.HT
        v : np.array
            The final step vector, default self.HT.T @ self.XT @ self.rT

        Returns
        -------
        Kx_list, d_list, S_list, v_list : Tuple(List, List)
            List of gains, biases, Riccati solutions S, and vectors v (including initial and final)
        """

        if t0 is None:
            t0 = 0.0
        if S is None:
            S = self.HT.T @ self.XT @ self.HT
        if v is None:
            v = self.HT.T @ self.XT @ self.rT
        if dt is None:
            dt = self.T / steps

        S_list = [S]
        v_list = [v]
        Kx_list = []
        d_list = []

        for k in reversed(range(steps)):
            t = t0 + (k+1) * dt
            x = pack_Sv(S,v)
            x = diffeq_np.rk4(self.riccati_ode_f, -dt, x, t)
            S,v = unpack_Sv(x)
            S = 0.5 * (S + S.T)

            t = t0 + k * dt
            U = self.U(t)
            L = self.L(t)
            ULT = linalg.solve(U, L.T, assume_a='pos')

            Kx = ULT @ S
            d = ULT @ v
            S_list.append(S)
            v_list.append(v)
            Kx_list.append(Kx)
            d_list.append(d)

        Kx_list.reverse()
        d_list.reverse()
        S_list.reverse()
        v_list.reverse()

        return Kx_list, d_list, S_list, v_list

    def seqForwardPass(self, x0, Kx_list, d_list, dt=None, t0=None, u_zoh=False):
        """
          Solve the forward trajectory using ZOH for Kx and optionally for u. Note that when u_zoh=False, then x is more
          accurate, but we cannot exactly get x when controlling it with the returned u_list.

        Parameters
        ----------
        x0 : np.array
          Initial state
        Kx_list : List
          List of gains
        d_list : List
          List of control offsets
        dt : float
            Time step, default dt=self.T/steps
        t0 : float
          Initial time (default 0)
        u_zoh : bool
          use ZOH also for u when computing x, default False

        Returns
        -------
        u_list, x_list : Tuple(List, List)
            List of inputs, list of states (initial and final)
        """

        steps = len(Kx_list)
        if t0 is None:
            t0 = 0.0
        if dt is None:
            dt = self.T / steps
        u_list = []
        x_list = [x0]

        x = x0
        for k in range(steps):
            t = t0 + k * dt
            u = - Kx_list[k] @ x + d_list[k]
            if u_zoh:
                f = lambda x, t: self.F(t) @ x + self.L(t) @ u + self.c(t)
            else:
                f = lambda x, t: self.F(t) @ x + self.L(t) @ (-Kx_list[k] @ x + d_list[k]) + self.c(t)
            x = diffeq_np.rk4(f, dt, x, t)
            u_list.append(u)
            x_list.append(x)

        return u_list, x_list

    def fwdbwdpass_ode_f(self, x, t):
        """" ODE function for forward integration of A, b, C.

        Parameters:
            x: Packed vector of (A, b, C).
            t: Time.

        Returns:
            dx: Time derivative of x.
        """

        U = self.U(t)
        L = self.L(t)
        ULT = linalg.solve(U, L.T, assume_a='pos')
        LULT = L @ ULT

        A, b, C = unpack_abc(x)

        F = self.F(t)
        X = self.X(t)
        H = self.H(t)
        r = self.r(t)
        c = self.c(t)

        dA = F @ A - C @ H.T @ X @ H @ A
        db = F @ b + C @ H.T @ X @ r - C @ H.T @ X @ H @ b + c
        dC = F @ C - C @ H.T @ X @ H @ C + C @ F.T + LULT

        dx = pack_abc(dA,db,dC)

        return dx

    def seqFwdBwdPass(self, x0, steps, dt=None, t0=None, A=None, b=None, C=None):
        """" Solve forward value function parameters A, b, C forward.

        Parameters:
            x0: Initial state.
            steps: Number of steps.
            dt: Time step.
            t0: Initial time.
            A: Initial A.
            b: Initial b.
            C: Initial C.

        Returns:
            A_list, b_list, C_list: Lists of A, b, C.
        """

        if t0 is None:
            t0 = 0.0
        if dt is None:
            dt = self.T / steps
        if A is None:
            A = np.zeros_like(self.F(0))
        if b is None:
            b = x0
        if C is None:
            C = np.zeros_like(self.F(0))

        A_list = [A]
        b_list = [b]
        C_list = [C]

        for k in range(steps):
            t = t0 + k * dt
            x = pack_abc(A, b, C)
            x = diffeq_np.rk4(self.fwdbwdpass_ode_f, dt, x, t)
            A, b, C = unpack_abc(x)
            C = 0.5 * (C + C.T)
            A_list.append(A)
            b_list.append(b)
            C_list.append(C)

        return A_list, b_list, C_list


    def combineForwardBackward(self, Kx_list, d_list, S_list, v_list, A_list, b_list, C_list):
        """ Combine forward and backward pass to get u and x.

        Parameters:
            Kx_list: List of control gains.
            d_list: List of control offsets.
            S_list: List of backward value function matrices S.
            v_list: List of backward value function vectors v.
            A_list: List of forward value function matrices A.
            b_list: List of forward value function vectors b.
            C_list: List of forward value function matrices C.

        Returns:
            u_list, x_list: List of inputs, list of states.
        """

        u_list = []
        x_list = []

        for k in range(len(S_list)):
            S = S_list[k]
            v = v_list[k]
            A = A_list[k] # These should actually be zero
            b = b_list[k]
            C = C_list[k]
            x = linalg.solve(np.eye(C.shape[0]) + C @ S, b + C @ v)
            x_list.append(x)
            if k < len(Kx_list):
                u = -Kx_list[k] @ x + d_list[k]
                u_list.append(u)

        return u_list, x_list

    ###########################################################################
    # Parallel computation of gains and value functions backwards
    ###########################################################################

    def bwpass_bw_ode_f(self, x, t):
        """" ODE function for backward integration of A, b, C, eta, J.

        Parameters:
            x: Packed vector of (A, b, C, eta, J).
            t: Time.

        Returns:
            dx: Time derivative of x.
        """
        U = self.U(t)
        L = self.L(t)
        ULT = linalg.solve(U, L.T, assume_a='pos')
        LULT = L @ ULT

        A, b, C, eta, J = unpack_abcej(x)

        F = self.F(t)
        X = self.X(t)
        H = self.H(t)
        r = self.r(t)
        c = self.c(t)

        dA = A @ LULT @ J - A @ F
        db = -A @ LULT @ eta - A @ c
        dC = -A @ LULT @ A.T
        deta = -H.T @ X @ r + J @ LULT @ eta - F.T @ eta + J @ c
        dJ = - H.T @ X @ H + J @ LULT @ J - J @ F - F.T @ J

        dx = pack_abcej(dA,db,dC,deta,dJ)

        return dx

    def bwpass_fw_ode_f(self, x, t):
        """" ODE function for forward integration of A, b, C, eta, J.

        Parameters:
            x: Packed vector of (A, b, C, eta, J).
            t: Time.

        Returns:
            dx: Time derivative of x.
        """

        U = self.U(t)
        L = self.L(t)
        ULT = linalg.solve(U, L.T, assume_a='pos')
        LULT = L @ ULT

        A, b, C, eta, J = unpack_abcej(x)

        F = self.F(t)
        X = self.X(t)
        H = self.H(t)
        r = self.r(t)
        c = self.c(t)

        dA = F @ A - C @ H.T @ X @ H @ A
        db = F @ b + C @ H.T @ X @ r - C @ H.T @ X @ H @ b + c
        dC = F @ C - C @ H.T @ X @ H @ C + C @ F.T + LULT

        deta = A.T @ H.T @ X @ r - A.T @ H.T @ X @ H @ b
        dJ = A.T @ H.T @ X @ H @ A

        dx = pack_abcej(dA,db,dC,deta,dJ)

        return dx

    def parBackwardPass_init(self, blocks, steps, dt=None, t0=None, forward=False):
        """
         Initialize the parallel backward pass with "blocks" blocks using "steps" backward integration
            steps. Returns list of initial elements.

        Parameters
        ----------
        blocks : int
          Number of blocks to split
        steps : int
          Number of steps per block
        dt : float
            Time step, default dt=self.T/steps/blocks
        t0 : float
          Initial time (default 0)
        forward : bool
          Use forward (as opposed to backward) versions of the differential equations

        Returns
        -------
        elems : List
          List of element tuples (A,b,C,eta,J)
        """
        if t0 is None:
            t0 = 0.0
        if dt is None:
            dt = self.T / steps / blocks

        elems = []

        for k in range(blocks):
            A = np.eye((self.F(0)).shape[0])
            b = np.zeros_like(A[:, 0])
            C = np.zeros_like(A)
            eta = np.zeros_like(A[:, 0])
            J = np.zeros_like(A)

            if forward:
                for i in range(steps):
                    t = t0 + (k * steps + i) * dt
                    x = pack_abcej(A, b, C, eta, J)
                    x = diffeq_np.rk4(self.bwpass_fw_ode_f, dt, x, t)
                    A, b, C, eta, J = unpack_abcej(x)
                    C = 0.5 * (C + C.T)
                    J = 0.5 * (J + J.T)
            else:
                for i in reversed(range(steps)):
                    t = t0 + (k * steps + i + 1) * dt
                    x = pack_abcej(A, b, C, eta, J)
                    x = diffeq_np.rk4(self.bwpass_bw_ode_f, -dt, x, t)
                    A, b, C, eta, J = unpack_abcej(x)
                    C = 0.5 * (C + C.T)
                    J = 0.5 * (J + J.T)

            elem = (A, b, C, eta, J)
            elems.append(elem)

        A = np.zeros_like(self.F(0))
        b = np.zeros_like(A[:, 0])
        C = np.zeros_like(A)
        eta = self.HT.T @ self.XT @ self.rT
        J = self.HT.T @ self.XT @ self.HT
        elem = (A, b, C, eta, J)
        elems.append(elem)

        return elems

    def parBackwardPass_extract(self, elems, steps, dt=None, t0=None):
        """
         Extract parallel backward pass results, backward integrate the intermediate values, and compute gains.

        Parameters
        ----------
        elems : List
          List of element tuples (A,b,C,eta,J)
        steps : int
          Number of steps per block
        dt : float
            Time step, default dt=self.T / steps / (len(elems) - 1)
        t0 : float
          Initial time (default 0)

        Returns
        -------
        Kx_list, d_list, S_list, v_list : List
          List of gains, list of Riccati solutions, and the offsets/biases
        """
        blocks = len(elems) - 1
        if t0 is None:
            t0 = 0.0
        if dt is None:
            dt = self.T / steps / blocks

        (A, b, C, eta, J) = elems[0]

        S_list = [J]
        v_list = [eta]
        Kx_list = []
        d_list = []

        for k in range(blocks):
            t1 = t0 + k * dt * steps

            (A, b, C, eta, J) = elems[k + 1]
            bKx_list, bd_list, bS_list, bv_list = self.seqBackwardPass(steps, dt=dt, t0=t1, S=J, v=eta)
            for i in range(len(bKx_list)):
                S_list.append(bS_list[i + 1])
                v_list.append(bv_list[i + 1])
                Kx_list.append(bKx_list[i])
                d_list.append(bd_list[i])

        return Kx_list, d_list, S_list, v_list

    def parBackwardPass(self, blocks, steps, dt=None, t0=None, forward=False):
        """
         Perform the parallel backward pass with "blocks" blocks using "steps" backward integration steps.

        Parameters
        ----------
        blocks : int
          Number of blocks to split
        steps : int
          Number of steps per block
        dt : float
            Time step, default dt=self.T/steps/blocks
        t0 : float
          Initial time (default 0)
        forward : bool
          Use forward (as opposed to backward) versions of the differential equations

        Returns
        -------
        Kx_list, d_list, S_list, v_list : Tuple(List, List)
          List of gains, list of Riccati solutions, etc.
        """
        if t0 is None:
            t0 = 0.0
        if dt is None:
            dt = self.T / steps / blocks

        # Initialize
        elems = self.parBackwardPass_init(blocks, steps, dt=dt, t0=t0, forward=forward)

        # Call the associative scan
        elems = lqt_np.par_backward_pass_scan(elems)

        # Extract the results
        return self.parBackwardPass_extract(elems, steps, dt=dt, t0=t0)

    ###########################################################################
    # Parallel computation of states and controls forward
    ###########################################################################

    def fwpass_fw_ode_f(self,x,t,Kx_d):
        """ Forward pass forward ODE function.

        Parameters:
            x: Packed vector of Psi and phi
            t: Time
            Kx_d: Tuple of Kx and d

        Returns:
            dx: Tie derivative of x
        """
        Kx, d = Kx_d

        Psi, phi = unpack_Psiphi(x)
        L = self.L(t)
        tF = self.F(t) - L @ Kx
        tc = self.c(t) + L @ d
        dPsi = tF @ Psi
        dphi = tF @ phi + tc
        dx = pack_Psiphi(dPsi, dphi)
        return dx

    def fwpass_bw_ode_f(self,x,t,Kx_d):
        """ Forward pass backward ODE function.

        Parameters:
            x: Packed vector of Psi and phi
            t: Time
            Kx_d: Tuple of Kx and d

        Returns:
            dx: Time derivative of x
        """

        Kx, d = Kx_d

        Psi, phi = unpack_Psiphi(x)
        L = self.L(t)
        tF = self.F(t) - L @ Kx
        tc = self.c(t) + L @ d
        dPsi = -Psi @ tF
        dphi = -Psi @ tc
        dx = pack_Psiphi(dPsi, dphi)
        return dx

    def parForwardPass_init(self, x0, Kx_list, d_list, blocks, steps, dt=None, t0=None, forward=True):
        """
        Initialize the parallel forward pass.

        Parameters
        ----------
        x0 : np.array
          Initial state
        Kx_list : List
          List of gains (of length blocks * steps)
        d_list : List
          List of control biases (of length blocks * steps)
        blocks : int
          Number of blocks
        steps : int
          Number of steps per block
        dt : float
            Time step, default dt=self.T / steps / blocks
        t0 : float
          Initial time (default 0)
        forward : bool
          If the integrations should be done forward (as opposed to backwards), default True

        Returns
        -------
        elems : List
          List of element tuples (Psi,phi)
        """
        if t0 is None:
            t0 = 0.0
        if dt is None:
            dt = self.T / steps / blocks

        if blocks * steps != len(Kx_list):
            raise ValueError(f"Invalid arguments: blocks * steps != len(Kx_list)")

        elems = []
        Psi = np.eye((self.F(0)).shape[0])
        phi = x0
        elems.append((Psi,phi))

        for k in range(blocks):
            Psi = np.eye((self.F(0)).shape[0])
            phi = np.zeros_like(x0)

            if forward:
                for i in range(steps):
                    t = t0 + (k * steps + i) * dt
                    x = pack_Psiphi(Psi, phi)
                    x = diffeq_np.rk4(self.fwpass_fw_ode_f, dt, x, t, param=(Kx_list[k * steps + i],d_list[k * steps + i]))
                    Psi, phi = unpack_Psiphi(x)
            else:
                for i in reversed(range(steps)):
                    t = t0 + (k * steps + i + 1) * dt
                    x = pack_Psiphi(Psi, phi)
                    x = diffeq_np.rk4(self.fwpass_bw_ode_f, -dt, x, t, param=(Kx_list[k * steps + i],d_list[k * steps + i]))
                    Psi, phi = unpack_Psiphi(x)

            elems.append((Psi, phi))

        return elems

    def parForwardPass_extract(self, Kx_list, d_list, elems, steps, dt=None, t0=None, u_zoh=False):
        """
         Extract parallel forward pass results, forward integrate the intermediate values, and compute controls.

        Parameters
        ----------
        Kx_list : List
          List of gains (of length blocks * steps)
        d_list : List
          Control biases
        elems : List
          List of element tuples (Psi,phi)
        steps : int
          Number of steps per block
        dt : float
            Time step, default dt=self.T / steps / blocks
        t0 : float
          Initial time (default 0)
        u_zoh : bool
          use ZOH also for u when computing x, default False

        Returns
        -------
        u_list, x_list : Tuple(List, List)
          List of controls, list of states
        """

        blocks = len(elems)-1
        if blocks * steps != len(Kx_list):
            raise ValueError(f"Invalid arguments: blocks * steps != len(Kx_list)")

        if t0 is None:
            t0 = 0.0
        if dt is None:
            dt = self.T / steps / blocks

        u_list = []
        x_list = []
        for k in range(blocks):
            (Psi, phi) = elems[k]
            t1 = t0 + k * steps * dt
            bu_list, bx_list = self.seqForwardPass(phi, Kx_list[k*steps:(k+1)*steps], d_list[k*steps:(k+1)*steps], dt=dt, t0=t1, u_zoh=u_zoh)
            for i in range(len(bu_list)):
                u_list.append(bu_list[i])
                x_list.append(bx_list[i])
        (Psi, phi) = elems[-1]
        x_list.append(phi)

        return u_list, x_list

    def parForwardPass(self, x0, Kx_list, d_list, blocks, steps, dt=None, t0=None, forward=True, u_zoh=False):
        """
        Perform parallel forward pass.

        Parameters
        ----------
        x0 : np.array
          Initial state
        Kx_list : List
          List of gains (of length blocks * steps)
        d_list : List
          List of control biases (of length blocks * steps)
        blocks : int
          Number of blocks
        steps : int
          Number of steps per block
        dt : float
            Time step, default dt=self.T / steps / blocks
        t0 : float
          Initial time (default 0)
        u_zoh : bool
          use ZOH also for u when computing x, default False

        Returns
        -------
        u_list, x_list : Tuple(List, List)
          List of controls, list of states
        """
        if t0 is None:
            t0 = 0.0
        if dt is None:
            dt = self.T / steps / blocks

        elems = self.parForwardPass_init(x0, Kx_list, d_list, blocks, steps, dt=dt, t0=t0, forward=forward)
        elems = lqt_np.par_forward_pass_scan(elems)
        return self.parForwardPass_extract(Kx_list, d_list, elems, steps, dt=dt, t0=t0, u_zoh=u_zoh)

    ###########################################################################
    # Parallel computation of (backward and) forward value functions
    ###########################################################################

    def parFwdBwdPass_init(self, x0, blocks, steps, dt=None, t0=None, forward=True):
        """
         Initialize the parallel value function forward pass with "blocks" blocks using "steps" backward integration
            steps. Returns list of initial elements.

        Parameters
        ----------
        x0 : np.array
          Initial state
        blocks : int
          Number of blocks to split
        steps : int
          Number of steps per block
        dt : float
            Time step, default dt=self.T / steps / blocks
        t0 : float
          Initial time (default 0)
        forward : bool
          Use forward (as opposed to backward) versions of the differential equations

        Returns
        -------
        elems : List
          List of element tuples (A,b,C,eta,J)
        """
        if t0 is None:
            t0 = 0.0
        if dt is None:
            dt = self.T / steps / blocks

        elems = []

        A = np.zeros_like(self.F(0))
        b = x0
        C = np.zeros_like(A)
        eta = np.zeros_like(A[:, 0])
        J = np.zeros_like(A)
        elem = (A, b, C, eta, J)
        elems.append(elem)

        # Here we have copy-paste from Backward_init, it would also be
        # possible to call that function and just modify start/end elements
        for k in range(blocks):
            A = np.eye((self.F(0)).shape[0])
            b = np.zeros_like(A[:, 0])
            C = np.zeros_like(A)
            eta = np.zeros_like(A[:, 0])
            J = np.zeros_like(A)

            if forward:
                for i in range(steps):
                    t = t0 + (k * steps + i) * dt
                    x = pack_abcej(A, b, C, eta, J)
                    x = diffeq_np.rk4(self.bwpass_fw_ode_f, dt, x, t)
                    A, b, C, eta, J = unpack_abcej(x)
                    C = 0.5 * (C + C.T)
                    J = 0.5 * (J + J.T)
            else:
                for i in reversed(range(steps)):
                    t = t0 + (k * steps + i + 1) * dt
                    x = pack_abcej(A, b, C, eta, J)
                    x = diffeq_np.rk4(self.bwpass_bw_ode_f, -dt, x, t)
                    A, b, C, eta, J = unpack_abcej(x)
                    C = 0.5 * (C + C.T)
                    J = 0.5 * (J + J.T)

            elem = (A, b, C, eta, J)
            elems.append(elem)

        return elems

    def parFwdBwdPass_extract(self, Kx_list, d_list, S_list, v_list, elems, steps, dt=None, t0=None):
        """ Extract the control and state trajectories from the parallel forward-backward pass results.

        Parameters:
            Kx_list: List of control gains.
            d_list: List of control biases.
            S_list: List of backward value function matrices.
            v_list: List of backward value function vectors.
            elems: List of tuples (A,b,C,eta,J) from forward pass.
            steps: Number of steps per block.
            dt: Time step.
            t0: Initial time.

        Returns:
            u_list: List of control trajectories.

        """

        blocks = len(elems)-1
        if blocks * steps != len(Kx_list):
            raise ValueError(f"Invalid arguments: blocks * steps != len(Kx_list)")
        if blocks * steps != len(S_list)-1:
            raise ValueError(f"Invalid arguments: blocks * steps != len(S_list)-1")

        if t0 is None:
            t0 = 0.0
        if dt is None:
            dt = self.T / steps / blocks

        u_list = []
        x_list = []
        for k in range(blocks):
            (A, b, C, eta, J) = elems[k]
            t1 = t0 + k * steps * dt
            A_list, b_list, C_list = self.seqFwdBwdPass(None, steps, dt=dt, t0=t1, A=A, b=b, C=C)
            bu_list, bx_list = self.combineForwardBackward(Kx_list[k*steps:(k+1)*steps], d_list[k*steps:(k+1)*steps],
                                                           S_list[k*steps:(k+1)*steps+1], v_list[k*steps:(k+1)*steps+1],
                                                           A_list, b_list, C_list)
            for i in range(len(bu_list)):
                u_list.append(bu_list[i])
                x_list.append(bx_list[i])
        x_list.append(bx_list[-1])

        return u_list, x_list

    def parFwdBwdPass(self, x0, Kx_list, d_list, S_list, v_list, blocks, steps, dt=None, t0=None, forward=True):
        """ Parallel computation of control and state trajectories using forward-backward pass.

        Parameters:
            x0: Initial state.
            Kx_list: List of control gains.
            d_list: List of control biases.
            S_list: List of backward value function matrices.
            v_list: List of backward value function vectors.
            blocks: Number of blocks to split.
            steps: Number of steps per block.
            dt: Time step.
            t0: Initial time.
            forward: Use forward (as opposed to backward) versions of the differential equations.

        Returns:
            u_list: List of control trajectories.
            x_list: List of state trajectories.
        """
        if t0 is None:
            t0 = 0.0
        if dt is None:
            dt = self.T / steps / blocks

        elems = self.parFwdBwdPass_init(x0, blocks, steps, dt=dt, t0=t0, forward=forward)
        elems = lqt_np.par_fwdbwd_pass_scan(elems)
        u_list, x_list = self.parFwdBwdPass_extract(Kx_list, d_list, S_list, v_list, elems, steps, dt=dt, t0=t0)

        return u_list, x_list

