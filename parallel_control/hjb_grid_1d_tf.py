"""
TensorFlow versions of routines for solving continuous-time nonlinear 1d problems.

@author: Simo Särkkä
"""

import tensorflow as tf
import tensorflow_probability as tfp
import parallel_control.diffeq_tf as diffeq_tf
import parallel_control.sqp_tf as sqp_tf

import math

##############################################################################
# Misc. constants and utilities
##############################################################################

_HJB_GRID_TF_INFTY = 1e20 # This is used as "infinity" in computations

# Abbreviations for linear algebra
mm = tf.linalg.matmul
mv = tf.linalg.matvec
top = tf.linalg.matrix_transpose

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
@tf.function
def fun_and_con_1(g, L, T, x0, xf, a_vec, b_vec):
    """ Computes the cost and constraint for given control parameters.

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
    dt = T / a_vec.shape[-1]
    t_vec = tf.range(tf.constant(0.0, dtype=T.dtype), T, dt)

    def body(carry, inp):
        a, b, ti = inp
        def f(xc, t, p):
            u = a + b * (t - ti)
            return tf.stack([g(xc[..., 0],u), L(xc[..., 0],u)], axis=-1)

        x, c = carry
        xc = diffeq_tf.rk4(f, dt, tf.stack([x,c], axis=-1), ti)

        return xc[..., 0], xc[..., 1]

    z0 = tf.zeros_like(x0)
    xs, cs = tf.scan(body, (tf.transpose(a_vec), tf.transpose(b_vec), t_vec),
                     initializer=(x0, z0),
                     reverse=False)

    xT = xs[-1, :]
    cT = cs[-1, :]
    return cT, tf.stack([xT - xf], axis=-1)

@tf.function
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
    xf_mesh, x0_mesh = tf.meshgrid(xf_grid, x0_grid)
    x0_vec = tf.reshape(x0_mesh, (x0_mesh.shape[0] * x0_mesh.shape[1], ))
    xf_vec = tf.reshape(xf_mesh, (xf_mesh.shape[0] * xf_mesh.shape[1], ))

    @tf.function
    def fun(theta, par):
        x0 = par[..., 0]
        xf = par[..., 1]
        a_vec = theta[..., 0::2]
        b_vec = theta[..., 1::2]
        fun_v, _ = fun_and_con_1(g, L, T, x0, xf, a_vec, b_vec)
        return fun_v

    @tf.function
    def con(theta, par):
        x0 = par[..., 0]
        xf = par[..., 1]
        a_vec = theta[..., 0::2]
        b_vec = theta[..., 1::2]
        _, con_v = fun_and_con_1(g, L, T, x0, xf, a_vec, b_vec)
        return con_v

    theta0 = tf.zeros((x0_vec.shape[0], 2 * steps), dtype=tf.float64)
    par = tf.stack([x0_vec, xf_vec], axis=-1)
    thetas, lams, crits = sqp_tf.local_eq_fast(fun, con, theta0, par, max_iter, batch=True)
    theta = thetas[-1, ...]
    V = fun(theta, par)

    return tf.reshape(V, x0_mesh.shape)

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

@tf.function
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
    dt = T / (theta.shape[-1] - 1)
    t_vec = tf.range(tf.constant(0.0, dtype=T.dtype), T, dt)

    def body(carry, inp):
        _, _, prev_u = carry
        b, ti = inp
        def f(xc, t, p):
            u = prev_u + b * (t - ti)
            return tf.stack([g(xc[..., 0],u), L(xc[..., 0],u)], axis=-1)

        x, c, _ = carry
        xc = diffeq_tf.rk4(f, dt, tf.stack([x,c], axis=-1), ti)
        prev_u = prev_u + b * dt

        return xc[..., 0], xc[..., 1], prev_u

    z0 = tf.zeros_like(x0)
    u0 = theta[..., 0]
    xs, cs, us = tf.scan(body, (tf.transpose(theta[..., 1:]), t_vec),
                         initializer=(x0, z0, u0),
                         reverse=False)

    xT = xs[-1, :]
    cT = cs[-1, :]
    return cT, tf.stack([xT - xf], axis=-1)

@tf.function
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
    xf_mesh, x0_mesh = tf.meshgrid(xf_grid, x0_grid)
    x0_vec = tf.reshape(x0_mesh, (x0_mesh.shape[0] * x0_mesh.shape[1], ))
    xf_vec = tf.reshape(xf_mesh, (xf_mesh.shape[0] * xf_mesh.shape[1], ))

    @tf.function
    def fun(theta, par):
        x0 = par[..., 0]
        xf = par[..., 1]
        fun_v, _ = fun_and_con_2(g, L, T, x0, xf, theta)
        return fun_v

    @tf.function
    def con(theta, par):
        x0 = par[..., 0]
        xf = par[..., 1]
        _, con_v = fun_and_con_2(g, L, T, x0, xf, theta)
        return con_v

    theta0 = tf.zeros((x0_vec.shape[0], 1 + steps), dtype=tf.float64)
    par = tf.stack([x0_vec, xf_vec], axis=-1)
    thetas, lams, crits = sqp_tf.local_eq_fast(fun, con, theta0, par, max_iter, batch=True)
    theta = thetas[-1, ...]
    V = fun(theta, par)

    return tf.reshape(V, x0_mesh.shape)

##############################################################################
#
# Combination
#
##############################################################################

def combine_V(Vij, Vjk):
    """ Combine two value functions.

    Parameters:
        Vij: Value function for i -> j
        Vjk: Value function for j -> k

    Returns:
        Vik: Value function for i -> k
    """
    Vik = tf.reduce_min(tf.expand_dims(Vij,-1) + tf.expand_dims(Vjk,1), axis=2)
    return Vik

def combine_V2(Vij, Vjk):
    """ Combine two value functions with direct combination with argmin (for testing)

    Parameters:
        Vij: Value function for i -> j
        Vjk: Value function for j -> k

    Returns:
        Vik: Value function for i -> k
    """
    Vsum = tf.expand_dims(Vij,-1) + tf.expand_dims(Vjk,1)
    j = tf.expand_dims(tf.math.argmin(Vsum, axis=2), 2)
    jT = tf.transpose(j, perm=[0,1,3,2])
    VsumT = tf.transpose(Vsum, perm=[0,1,3,2])
    return tf.gather_nd(VsumT, jT, batch_dims=3)

def combine_V_rev(Vjk, Vij):
    """ Combine two value functions with reversed order.

    Parameters:
        Vjk: Value function for j -> k
        Vij: Value function for i -> j

    Returns:
        Vik: Value function for i -> k
    """
    return combine_V(Vij, Vjk)

def combine_V_interp(Vij, Vjk):
    """ Combine two value functions with interpolation.

    Parameters:
        Vij: Value function for i -> j
        Vjk: Value function for j -> k

    Returns:
        Vik: Value function for i -> k
    """
    lam = tf.constant(0.0, dtype=Vij.dtype)

    Vsum = tf.expand_dims(Vij,-1) + tf.expand_dims(Vjk,1)
    j = tf.expand_dims(tf.math.argmin(Vsum, axis=2), 2)
    jT = tf.transpose(j, perm=[0,1,3,2])

    jTm1 = tf.math.abs(jT - 1)  # Converts -1 to 1
    nm1 = Vij.shape[1] - 1
    jTp1 = nm1 - tf.math.abs(nm1 - jT - 1)  # Converts n to n-2

    VsumT = tf.transpose(Vsum, perm=[0,1,3,2])

    fi = tf.gather_nd(VsumT, jT, batch_dims=3)
    fm1 = tf.gather_nd(VsumT, jTm1, batch_dims=3)
    fp1 = tf.gather_nd(VsumT, jTp1, batch_dims=3)

    return fi - (fp1 - fm1)**2 / (fp1 + fm1 - tf.constant(2.0, dtype=Vij.dtype) * fi + lam) / tf.constant(8.0, dtype=Vij.dtype)

def combine_V_interp_rev(Vjk, Vij):
    """ Combine two value functions with interpolation in reversed order.

    Parameters:
        Vjk: Value function for j -> k
        Vij: Value function for i -> j

    Returns:
        Vik: Value function for i -> k
    """
    return combine_V_interp(Vij, Vjk)


##############################################################################
#
# The actual HJB solvers
#
##############################################################################

def seq_bw_pass_symfd(f, L, LT, T, blocks, steps, fd_x_grid, fd_u_grid):
    """ Sequential backward pass with symmetric finite differences.

    Parameters:
        f: Dynamic model function
        L: Running cost function
        LT: Terminal cost function
        T: Time horizon
        blocks: Number of blocks
        steps: Number of steps per block
        fd_x_grid: Grid for x
        fd_u_grid: Grid for u

    Returns:
        V_fds: Value functions
    """

    fd_x_mesh, fd_u_mesh = tf.meshgrid(fd_x_grid, fd_u_grid)

    dx = fd_x_grid[1] - fd_x_grid[0]  # We assume uniform discretization
    dt = T / blocks / steps

    L_mesh = L(fd_x_mesh, fd_u_mesh)
    f_mesh = f(fd_x_mesh, fd_u_mesh)

    def body(carry, inp):
        V_fd = carry

        DV_fd_0 = (V_fd[1:2] - V_fd[0:1]) / dx
        DV_fd_1 = (V_fd[2:] - V_fd[:-2]) / (2.0 * dx)
        DV_fd_2 = (V_fd[-1:] - V_fd[-2:-1]) / dx

        DV_fd = tf.concat((DV_fd_0, DV_fd_1, DV_fd_2), axis=0)
        Q_fd = L_mesh + f_mesh * DV_fd
        V_fd = V_fd + tf.reduce_min(Q_fd, axis=0) * dt

#        ui = tf.math.argmin(Q_fd, axis=0)
#        u = tf.gather(fd_u_grid, ui, axis=-1, batch_dims=0)
#        Q_fd = L(fd_x_grid, u) + f(fd_x_grid, u) * DV_fd
#        V_fd = V_fd + Q_fd * dt

        return V_fd

    V_fd = LT(fd_x_grid)

    V_fds = tf.scan(body, tf.range(0, blocks * steps, dtype=tf.int32),
                    initializer=V_fd,
                    reverse=True)

    V_fds = tf.concat((V_fds, tf.expand_dims(V_fd, 0)), axis=0)
    V_fds = V_fds[0::steps]

    return V_fds

def seq_bw_pass_upwind(f, L, LT, T, blocks, steps, fd_x_grid, fd_u_grid):
    """ Sequential backward pass with upwind finite differences.

    Parameters:
        f: Dynamic model function
        L: Running cost function
        LT: Terminal cost function
        T: Time horizon
        blocks: Number of blocks
        steps: Number of steps per block
        fd_x_grid: Grid for x
        fd_u_grid: Grid for u

    Returns:
        V_fds: Value functions
    """
    fd_x_mesh, fd_u_mesh = tf.meshgrid(fd_x_grid, fd_u_grid)

    dx = fd_x_grid[1] - fd_x_grid[0]  # We assume uniform discretization
    dt = T / blocks / steps

    L_mesh = L(fd_x_mesh, fd_u_mesh)
    f_mesh = f(fd_x_mesh, fd_u_mesh)
    s_mesh = tf.math.sign(f_mesh)

    def body(carry, inp):
        V_fd = carry

        DV_fd_0 = (V_fd[1:2] - V_fd[0:1]) / dx
        DV_fd_1f = (V_fd[2:] - V_fd[1:-1]) / dx
        DV_fd_1b = (V_fd[1:-1] - V_fd[0:-2]) / dx
        DV_fd_2 = (V_fd[-1:] - V_fd[-2:-1]) / dx
        DV_fdf = tf.concat((DV_fd_0, DV_fd_1f, DV_fd_2), axis=0)
        DV_fdb = tf.concat((DV_fd_0, DV_fd_1b, DV_fd_2), axis=0)

        Q_fd = L_mesh \
               + (0.5 + 0.5 * s_mesh) * f_mesh * DV_fdf \
               + (0.5 - 0.5 * s_mesh) * f_mesh * DV_fdb
        V_fd = V_fd + tf.reduce_min(Q_fd, axis=0) * dt

        return V_fd

    V_fd = LT(fd_x_grid)

    V_fds = tf.scan(body, tf.range(0, blocks * steps, dtype=tf.int32),
                    initializer=V_fd,
                    reverse=True)

    V_fds = tf.concat((V_fds, tf.expand_dims(V_fd, 0)), axis=0)
    V_fds = V_fds[0::steps]

    return V_fds

@tf.function
def direct_shooting_block_V(g, L, block_T, steps, fd_x_grid, max_iter=10):
    """ Direct shooting based conditional value function for a single block."""
    return V_eval_2(g, L, block_T, steps, fd_x_grid, fd_x_grid, max_iter)

def seq_bw_pass_assoc(LT, blocks, fd_x_grid, block_V):
    """ Sequential backward pass over the blocks with associative combination.

    Parameters:
        LT: Terminal value function
        blocks: Number of blocks
        fd_x_grid: Grid for x
        block_V: Value function for the block

    Returns:
        Vs: Value functions
    """
    block_V = tf.expand_dims(block_V, 0)

    def body(carry, inp):
        V_prev = carry
        V_prev = tf.expand_dims(V_prev, 0)
        return combine_V_interp(block_V, V_prev)[0, ...]

    _, x_mesh = tf.meshgrid(fd_x_grid, fd_x_grid)
    VT = LT(x_mesh)
    Vs = tf.scan(body, tf.range(0, blocks, dtype=tf.int32),
                 initializer=VT,
                 reverse=True)

    VT = VT[:,0]
    Vs = Vs[:,:,0]

    return tf.concat((Vs, tf.expand_dims(VT, 0)), axis=0)

def par_bw_pass(LT, blocks, fd_x_grid, block_V, max_parallel=10000):
    """ Parallel backward pass over the blocks.

    Parameters:
        LT: Terminal value function
        blocks: Number of blocks
        fd_x_grid: Grid for x
        block_V: Value function for the block
        max_parallel: Maximum number of parallel elements in the associative scan

    Returns:
        Vs: Value functions
    """

    # Initialize elements
    elems = tf.broadcast_to(tf.expand_dims(block_V, 0), (blocks,) + block_V.shape)
    _, x_mesh = tf.meshgrid(fd_x_grid, fd_x_grid)
    VT = LT(x_mesh)
    elems = tf.concat((elems, tf.expand_dims(VT,0)), axis=0)

    # Do associative scan and extract result
    rev_elems = tf.reverse(elems, axis=[0])

    rev_elems = tfp.math.scan_associative(combine_V_interp_rev,
                                          rev_elems,
                                          max_num_levels=math.ceil(math.log2(max_parallel)))

    Vs = tf.reverse(rev_elems, axis=[0])[..., 0]
    return Vs

