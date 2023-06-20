"""
Numpy/Jax versions of routines for solving continuous-time nonlinear 1d problems.

@author: Simo Särkkä
"""

from scipy import linalg
import numpy as np
import jax
import jax.numpy as jnp
import parallel_control.diffeq_np as diffeq_np
#import parallel_control.fsc_np as fsc_np
import parallel_control.sqp_np as sqp_np
from parallel_control.my_assoc_scan import my_assoc_scan

##############################################################################
# Misc. constants and utilities
##############################################################################

_HJB_GRID_NP_INFTY = 1e20 # This is used as "infinity" in computations


##############################################################################
#
# Evaluation of conditional value functions
#
##############################################################################

# This version of the cost function uses approximation
#
#         x[0] = x0
#         C[0] = 0
#       u_i(t) = a_i + b_i (t - t_i)
#       x[i+1] = rk4(f(x, u_i(t)), dt, x[i], t_i) # Jointly with one below
#       C[i+1] = rk4(L(x, u_i(t)), dt, x[i], t_i)
#
# with constraint x[n] = xf
#
def fun_and_con_1(g, L, T, x0, xf, a_vec, b_vec):
    """ Evaluate cost and constraint for given control parameters.

    Parameters:
        g: Model function g(x,u)
        L: Model function L(x,u)
        T: Time horizon
        x0: Initial state
        xf: Final state
        a_vec: Control parameters a_i
        b_vec: Control parameters b_i

    Returns:
        cT: Cost at time T
        c_vec: Constraint violation vector
    """
    dt = T / a_vec.shape[0]
    t_vec = jax.numpy.arange(0, T, dt)
    def body(carry, inp):
        a, b, ti = inp
        def f(xc, t):
            u = a + b * (t - ti)
            return jnp.array([g(xc[0],u), L(xc[0],u)])

        xc = carry
        xc = diffeq_np.rk4(f, dt, xc, ti)
        return xc, xc

    xc_final, xc_list = jax.lax.scan(body, jnp.array([x0,0]), (a_vec, b_vec, t_vec))
    xT = xc_final[0]
    cT = xc_final[1]
    return cT, jnp.array([xT - xf])

def V_eval_1(g, L, T, steps, x0_grid, xf_grid, max_iter):
    """ Evaluate conditional value function on a grid using SQP.

    Parameters:
        g: Model function g(x,u)
        L: Model function L(x,u)
        T: Time horizon
        steps: Number of steps
        x0_grid: Grid for initial state
        xf_grid: Grid for final state
        max_iter: Maximum number of iterations for SQP

    Returns:
        V: Conditional value function on grid
    """
    xf_mesh, x0_mesh = jnp.meshgrid(xf_grid, x0_grid)
    x0_vec = jnp.reshape(x0_mesh, (x0_mesh.shape[0] * x0_mesh.shape[1], ))
    xf_vec = jnp.reshape(xf_mesh, (xf_mesh.shape[0] * xf_mesh.shape[1], ))

    def V_eval_one(x0, xf):
        theta0 = jnp.zeros((2 * steps,), dtype=jnp.float64)
        theta, iter, crit = sqp_np.local_eq_fast(lambda theta: fun_and_con_1(g, L, T, x0, xf, theta[0::2], theta[1::2])[0],
                                                 lambda theta: fun_and_con_1(g, L, T, x0, xf, theta[0::2], theta[1::2])[1],
                                                 theta0, max_iter, quiet=True)

        return fun_and_con_1(g, L, T, x0, xf, theta[0::2], theta[1::2])[0]

    V = jax.vmap(V_eval_one)(x0_vec, xf_vec)

    return jnp.reshape(V, x0_mesh.shape)

# This version of the cost function uses approximation
#
#         x[0] = x0
#         C[0] = 0
#       u_0(t) = a_0 + b_0 t
#       u_i(t) = u_{i-1}(dt) + b_i (t - t_i)
#       x[i+1] = rk4(f(x, u_i(t)), dt, x[i], t_i) # Jointly with one below
#       C[i+1] = rk4(L(x, u_i(t)), dt, x[i], t_i)
#
# with constraint x[n] = xf
def fun_and_con_2(g, L, T, x0, xf, theta):
    """ Evaluate cost and constraint for given control parameters.

    Parameters:
        g: Model function g(x,u)
        L: Model function L(x,u)
        T: Time horizon
        x0: Initial state
        xf: Final state
        theta: Control parameters theta = (a_0, b_0, b_1, ..., b_{n-1})

    Returns:
        cT: Cost at time T
        c_vec: Constraint violation vector
    """
    dt = T / (theta.shape[0]-1)
    t_vec = jax.numpy.arange(0, T, dt)
    def body(carry, inp):
        _, prev_u = carry
        b, ti = inp
        def f(xc, t):
            u = prev_u + b * (t - ti)
            return jnp.array([g(xc[0],u), L(xc[0],u)])

        xc, _ = carry
        xc = diffeq_np.rk4(f, dt, xc, ti)
        prev_u = prev_u + b * dt
        return (xc, prev_u), (xc, prev_u)

    res_final, res_list = jax.lax.scan(body, (jnp.array([x0,0.0]), theta[0]), (theta[1:], t_vec))
    xc_final, _ = res_final
    xc_list, _  = res_list
    xT = xc_final[0]
    cT = xc_final[1]
    return cT, jnp.array([xT - xf])

def V_eval_2(g, L, T, steps, x0_grid, xf_grid, max_iter):
    """ Evaluate conditional value function on a grid using SQP.

    Parameters:
        g: Model function g(x,u)
        L: Model function L(x,u)
        T: Time horizon
        steps: Number of steps
        x0_grid: Grid for initial state
        xf_grid: Grid for final state
        max_iter: Maximum number of iterations for SQP

    Returns:
        V: Conditional value function on grid
    """
    xf_mesh, x0_mesh = jnp.meshgrid(xf_grid, x0_grid)
    x0_vec = jnp.reshape(x0_mesh, (x0_mesh.shape[0] * x0_mesh.shape[1], ))
    xf_vec = jnp.reshape(xf_mesh, (xf_mesh.shape[0] * xf_mesh.shape[1], ))

    def V_eval_one(x0, xf):
        theta0 = jnp.zeros((steps + 1,), dtype=jnp.float64)
        theta, iter, crit = sqp_np.local_eq_fast(lambda theta: fun_and_con_2(g, L, T, x0, xf, theta)[0],
                                                 lambda theta: fun_and_con_2(g, L, T, x0, xf, theta)[1],
                                                 theta0, max_iter, quiet=True)

        return fun_and_con_2(g, L, T, x0, xf, theta)[0]

    V = jax.vmap(V_eval_one)(x0_vec, xf_vec)

    return jnp.reshape(V, x0_mesh.shape)

##############################################################################
#
# Combination
#
##############################################################################

# Direct combination
def combine_V(Vij, Vjk):
    """ Combine two value functions.

    Parameters:
        Vij: Value function for i -> j
        Vjk: Value function for j -> k

    Returns:
        Vik: Value function for i -> k
    """

#    Vik = np.zeros_like(Vij)
#    for i in range(Vij.shape[0]):
#        for k in range(Vjk.shape[1]):
#            Vik[i,k] = (Vij[i,:] + Vjk[:,k]).min()

    Vik = jnp.min(jnp.expand_dims(Vij,-1) + jnp.expand_dims(Vjk,0), axis=1)
    return Vik

def combine_V2(Vij, Vjk):
    """ Combine two value functions with direct combination with argmin (for testing)

    Parameters:
        Vij: Value function for i -> j
        Vjk: Value function for j -> k

    Returns:
        Vik: Value function for i -> k
    """
    Vsum = jnp.expand_dims(Vij,-1) + jnp.expand_dims(Vjk,0)
    j = jnp.argmin(Vsum, axis=1, keepdims=True)
    Vik = jnp.take_along_axis(Vsum, j, axis=1)
    return Vik[:,0,:]

def combine_V_interp(Vij, Vjk):
    """ Combine two value functions with interpolation.

    Parameters:
        Vij: Value function for i -> j
        Vjk: Value function for j -> k

    Returns:
        Vik: Value function for i -> k
    """

#    lam = 0.1
#    lam = 1e-6
    lam = 0.0
    Vsum = jnp.expand_dims(Vij,-1) + jnp.expand_dims(Vjk,0)
    j = jnp.argmin(Vsum, axis=1, keepdims=True)
    jm1 = jnp.abs(j - 1)  # Converts -1 to 1
    nm1 = Vij.shape[0] - 1
    jp1 = nm1 - jnp.abs(nm1 - j - 1)  # Converts n to n-2

    fi = jnp.take_along_axis(Vsum, j, axis=1)
    fm1 = jnp.take_along_axis(Vsum, jm1, axis=1)
    fp1 = jnp.take_along_axis(Vsum, jp1, axis=1)

    Vik = fi - (fp1 - fm1)**2 / (fp1 + fm1 - 2.0 * fi + lam) / 8.0
#    Vik = fi - (fp1 - fm1) ** 2 / (jnp.abs(fp1 + fm1 - 2.0 * fi) + lam) / 8.0

    return Vik[:,0,:]

def combine_V_interp2(Vij, Vjk):
    """ Combine two value functions with interpolation with loop (for testing only).

    Parameters:
        Vij: Value function for i -> j
        Vjk: Value function for j -> k

    Returns:
        Vik: Value function for i -> k
    """
#    lam = 1e-6
    lam = 0.0
    Vik = np.zeros_like(Vij)
    for i in range(Vij.shape[0]):
        for k in range(Vjk.shape[1]):
            f = Vij[i,:] + Vjk[:,k]
            j = jnp.argmin(f)
            jm1 = jnp.abs(j-1) # Converts -1 to 1
            nm1 = f.shape[0]-1
            jp1 = nm1 - jnp.abs(nm1-j-1)  # Converts n to n-2
            Vik[i,k] = f[j] - (f[jp1] - f[jm1])**2 / (f[jp1] + f[jm1] - 2.0 * f[j] + lam) / 8.0
#            if j == 0 or j == f.shape[0]-1:
#                Vik[i,k] = f[j]
#            else:
#                Vik[i,k] = f[j] - (f[j+1] - f[j-1])**2 / (f[j+1] + f[j-1] - 2.0 * f[j] + 1e-6) / 8.0

    return Vik

def par_backward_pass_scan(elems):
    """ Parallel backward pass using scan without interpolation."""
    return my_assoc_scan(lambda x, y: combine_V(x, y), elems, reverse=True)

def par_backward_pass_scan_interp(elems):
    """ Parallel backward pass using scan with interpolation."""
    return my_assoc_scan(lambda x, y: combine_V_interp(x, y), elems, reverse=True)

##############################################################################
#
# 1d continuous-time nonlinear control solvers
#
##############################################################################

class HJB_Grid_1d:
    """
    Grid based sequential and parallel solvers for 1d continuous-time problems of the form

    dx/dt = f(x,u)
    C[u] = LT(x(T)) + int_{t_0}^{t_f} L(x(t),u(t)) dt

    """

    def __init__(self, f, L, LT, T, blocks, steps, fd_x_grid):
        """ Initialize the solver.

        Parameters:
            f: Dynamic model function f(x,u)
            L: Running cost function L(x,u)
            LT: Terminal cost function LT(x)
            T: Time horizon
            blocks: Number of blocks to divide the time horizon into
            steps: Number of steps per block
            fd_x_grid: Grid for finite difference approximation of x
        """
        self.f = f
        self.L = L
        self.LT = LT
        self.T = T
        self.blocks = blocks
        self.steps = steps
        self.fd_x_grid = fd_x_grid

    def directShootingValueFunction(self, method=2, max_iter=10):
        """ Solve the conditional value function using direct shooting.

        Parameters:
            method: 1 for first method, 2 for second
            max_iter: Maximum number of iterations for SQP

        Returns:
            V: Conditional value function matrix
        """
        print('Evaluating conditional value function on grid...')
        if method == 1:
            V = V_eval_1(self.f, self.L, self.T / self.blocks, self.steps, self.fd_x_grid, self.fd_x_grid, max_iter)
        elif method == 2:
            V = V_eval_2(self.f, self.L, self.T / self.blocks, self.steps, self.fd_x_grid, self.fd_x_grid, max_iter)
        else:
            print(f'Unknown shooting method {method}, defaulting to 1')
            V = V_eval_1(self.f, self.L, self.T / self.blocks, self.steps, self.fd_x_grid, self.fd_x_grid, max_iter)

        print('Done.')
        return V

    def seqBackwardPass_symfd(self, fd_u_grid):
        """ Sequential backward pass using finite differences.
        
        Parameters:
            fd_u_grid: Grid for finite difference approximation on u
        
        Returns:
            V_list: List of value functions for each block
        """
        V_list = []

        fd_x_mesh, fd_u_mesh = jnp.meshgrid(self.fd_x_grid, fd_u_grid)
        dx = self.fd_x_grid[1] - self.fd_x_grid[0]  # We assume uniform discretization
        dt = self.T / self.blocks / self.steps

        V_fd = self.LT(self.fd_x_grid)
        V_list.append(V_fd)

        for i in range(self.blocks):
            for j in range(self.steps):
                DV_fd_0 = (V_fd[1:2] - V_fd[0:1]) / dx
                DV_fd_1 = (V_fd[2:] - V_fd[:-2]) / (2.0 * dx)
                DV_fd_2 = (V_fd[-1:] - V_fd[-2:-1]) / dx
                DV_fd = jnp.concatenate((DV_fd_0, DV_fd_1, DV_fd_2))
                Q_fd = self.L(fd_x_mesh, fd_u_mesh) + self.f(fd_x_mesh, fd_u_mesh) * DV_fd

                ui = jnp.argmin(Q_fd, axis=0)
                u = fd_u_grid[ui]
                Q_fd = self.L(self.fd_x_grid, u) + self.f(self.fd_x_grid, u) * DV_fd
                V_fd = V_fd + Q_fd * dt

            V_list.append(V_fd)

        V_list.reverse()

        return V_list

    def seqBackwardPass_upwind(self, fd_u_grid):
        """ Sequential backward pass using upwind finite differences.

        Parameters:
            fd_u_grid: Grid for finite difference approximation on u

        Returns:
            V_list: List of value functions for each block
        """
        V_list = []

        fd_x_mesh, fd_u_mesh = jnp.meshgrid(self.fd_x_grid, fd_u_grid)
        dx = self.fd_x_grid[1] - self.fd_x_grid[0]  # We assume uniform discretization
        dt = self.T / self.blocks / self.steps

        V_fd = self.LT(self.fd_x_grid)
        V_list.append(V_fd)

        for i in range(self.blocks):
            for j in range(self.steps):
                DV_fd_0 = (V_fd[1:2] - V_fd[0:1]) / dx
                DV_fd_1f = (V_fd[2:] - V_fd[1:-1]) / dx
                DV_fd_1b = (V_fd[1:-1] - V_fd[0:-2]) / dx
                DV_fd_2 = (V_fd[-1:] - V_fd[-2:-1]) / dx
                DV_fdf = jnp.concatenate((DV_fd_0, DV_fd_1f, DV_fd_2))
                DV_fdb = jnp.concatenate((DV_fd_0, DV_fd_1b, DV_fd_2))

                s = jnp.sign(self.f(fd_x_mesh, fd_u_mesh))
                Q_fd = self.L(fd_x_mesh, fd_u_mesh) \
                       + (0.5 + 0.5 * s) * self.f(fd_x_mesh, fd_u_mesh) * DV_fdf \
                       + (0.5 - 0.5 * s) * self.f(fd_x_mesh, fd_u_mesh) * DV_fdb
                V_fd = V_fd + jnp.min(Q_fd, 0) * dt

            V_list.append(V_fd)

        V_list.reverse()

        return V_list

    def seqBackwardPass_assoc(self, block_V):
        """ Sequential backward pass using associative combination function.

        Parameters:
            block_V: Value function matrix for a block

        Returns:
            V_list: List of value functions for each block
        """
        xdim = block_V.shape[0]

        V_list = []

        VT = np.full((xdim, xdim), _HJB_GRID_NP_INFTY)
        for x in range(xdim):
            for xp in range(xdim):
                VT[x, xp] = self.LT(self.fd_x_grid[x])

        V_list.append(VT[:, 0])
        V = VT

        for k in range(self.blocks):
#            V = combine_V(block_V, V)
            V = combine_V_interp(block_V, V)
            V_list.append(V[:, 0])

        V_list.reverse()

        return V_list


    def parBackwardPass_init(self, block_V):
        """ Parallel backward pass initialization.

        Parameters:
            block_V: Value function matrix for a block

        Returns:
            elems: List of elements for parallel backward pass
        """
        elems = []

        for k in range(self.blocks):
            elems.append(block_V)

        xdim = block_V.shape[0]

        VT = np.full((xdim, xdim), _HJB_GRID_NP_INFTY)
        for x in range(xdim):
            for xp in range(xdim):
                VT[x, xp] = self.LT(self.fd_x_grid[x])
        elems.append(VT)

        return elems

    def parBackwardPass_extract(self, elems):
        """ Parallel backward pass result extraction.

        Parameters:
            elems: List of elements from parallel backward pass

        Returns:
            V_list: List of value functions for each block
        """
        V_list = [elems[0][:, 0]]
        for k in range(len(elems) - 1):
            V = elems[k + 1][:, 0]
            V_list.append(V)

        return V_list

    def parBackwardPass(self, block_V):
        """ Parallel backward pass.

        Parameters:
            block_V: Value function matrix for a block

        Returns:
            V_list: List of value functions for each block
        """
        elems = self.parBackwardPass_init(block_V)
#        elems = par_backward_pass_scan(elems)
        elems = par_backward_pass_scan_interp(elems)
        V_list = self.parBackwardPass_extract(elems)
        return V_list


