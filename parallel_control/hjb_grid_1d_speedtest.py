"""
HJB 1d speedtest routines.

@author: Simo Särkkä
"""

import tensorflow as tf

import parallel_control.hjb_grid_1d_tf as hjb_grid_1d_tf
import parallel_control.cnonlin_models_tf as cnonlin_models_tf

import time

def get_model(x_grid_steps = 100, u_grid_steps = 100):
    """ Get a HJB controller for an example model.

    Parameters:
        x_grid_steps: Number of grid points in the state space
        u_grid_steps: Number of grid points in the control space

    Returns:
        f: Dynamic model function
        L: Running cost function
        LT: Terminal cost function
        T: Time horizon
        x_grid: Grid points in the state space
        u_grid: Grid points in the control space
    """
    f, L, LT, T, x_grid, u_grid = cnonlin_models_tf.velocity_model(x_grid_steps, u_grid_steps)
    return f, L, LT, T, x_grid, u_grid

@tf.function(reduce_retracing=True)
def seq_upwind(f, L, LT, block_T, blocks, steps, x_grid, u_grid):
    """ Sequential upwind solver for HJB equation.

    Parameters:
        f: Dynamic model function
        L: Running cost function
        LT: Terminal cost function
        block_T: Block time
        blocks: Number of blocks
        steps: Number of steps
        x_grid: Grid points in the state space
        u_grid: Grid points in the control space

    Returns:
        Vs: Value functions
    """
    T = block_T * blocks
    Vs = hjb_grid_1d_tf.seq_bw_pass_upwind(f, L, LT, T, blocks, steps, x_grid, u_grid)
    return Vs

def seq_upwind_speedtest(model, blocks, steps=10, block_T=0.1, n_iter=10, device='/CPU:0'):
    """ Speedtest for sequential upwind solver.

    Parameters:
        model: Model tuple (f, L, LT, T, x_grid, u_grid)
        blocks: Number of blocks
        steps: Number of steps
        block_T: Block time
        n_iter: Number of iterations
        device: Device to run on

    Returns:
        elapsed: Elapsed time
        err1: Error 1
        err2: Error 2
    """

    print('Running seq_upwind_speedtest on device %s' % device)
    f, L, LT, T, x_grid, u_grid = model
    block_T = tf.constant(block_T, dtype=tf.float64)
    T = block_T * blocks

    with tf.device(device):
        _ = seq_upwind(f, L, LT, block_T, blocks, steps, x_grid, u_grid) # Compilation run
        tic = time.time()
        for i in range(n_iter):
            Vs = seq_upwind(f, L, LT, block_T, blocks, steps, x_grid, u_grid)
        toc = time.time()
        elapsed = (toc - tic) / n_iter

    Vs0 = hjb_grid_1d_tf.seq_bw_pass_upwind(f, L, LT, T, blocks, steps, x_grid, u_grid)

    err1 = tf.reduce_max(tf.math.abs(Vs - Vs0))
    n = x_grid.shape[0]
    err2 = tf.reduce_max(tf.math.abs(Vs[:, (n // 3):(2 * n // 3)] - Vs0[:, (n // 3):(2 * n // 3)]))

    print('blocks=%d, steps=%d, grid=%d, err1=%f, err2=%f took %f ms.' % (blocks, steps, n, err1.numpy(), err2.numpy(), 1000.0 * elapsed))
    return elapsed, err1, err2


@tf.function(reduce_retracing=True)
def seq_assoc(f, L, LT, block_T, blocks, steps, x_grid):
    """ Sequential associative combination solver for HJB equation.

    Parameters:
        f: Dynamic model function
        L: Running cost function
        LT: Terminal cost function
        block_T: Block time
        blocks: Number of blocks
        steps: Number of steps
        x_grid: Grid points in the state space

    Returns:
        Vs: Value functions
    """
    block_V = hjb_grid_1d_tf.direct_shooting_block_V(f, L, block_T, steps, x_grid, max_iter=10)
    Vs = hjb_grid_1d_tf.seq_bw_pass_assoc(LT, blocks, x_grid, block_V)
    return Vs

def seq_assoc_speedtest(model, blocks, steps=10, block_T=0.1, n_iter=10, device='/CPU:0'):
    """ Speedtest for sequential associative combination solver.

    Parameters:
        model: Model tuple (f, L, LT, T, x_grid, u_grid)
        blocks: Number of blocks
        steps: Number of steps
        block_T: Block time
        n_iter: Number of iterations
        device: Device to run on

    Returns:
        elapsed: Elapsed time
        err1: Error 1
        err2: Error 2
    """
    print('Running seq_assoc_speedtest on device %s' % device)
    f, L, LT, T, x_grid, u_grid = model
    block_T = tf.constant(block_T, dtype=tf.float64)
    T = block_T * blocks

    with tf.device(device):
        _ = seq_assoc(f, L, LT, block_T, blocks, steps, x_grid) # Compilation run
        tic = time.time()
        for i in range(n_iter):
            Vs = seq_assoc(f, L, LT, block_T, blocks, steps, x_grid)
        toc = time.time()
        elapsed = (toc - tic) / n_iter

    Vs0 = hjb_grid_1d_tf.seq_bw_pass_upwind(f, L, LT, T, blocks, steps, x_grid, u_grid)

    err1 = tf.reduce_max(tf.math.abs(Vs - Vs0))
    n = x_grid.shape[0]
    err2 = tf.reduce_max(tf.math.abs(Vs[:, (n // 3):(2 * n // 3)] - Vs0[:, (n // 3):(2 * n // 3)]))

    print('blocks=%d, steps=%d, grid=%d, err1=%f, err2=%f took %f ms.' % (blocks, steps, n, err1.numpy(), err2.numpy(), 1000.0 * elapsed))
    return elapsed, err1, err2


@tf.function(reduce_retracing=True)
def par_assoc(f, L, LT, block_T, blocks, steps, x_grid):
    """ Parallel solver for HJB equation.

    Parameters:
        f: Dynamic model function
        L: Running cost function
        LT: Terminal cost function
        block_T: Block time
        blocks: Number of blocks
        steps: Number of steps
        x_grid: Grid points in the state space

    Returns:
        Vs: Value functions
    """
    block_V = hjb_grid_1d_tf.direct_shooting_block_V(f, L, block_T, steps, x_grid, max_iter=10)
    Vs = hjb_grid_1d_tf.par_bw_pass(LT, blocks, x_grid, block_V, max_parallel=blocks)
    return Vs

def par_assoc_speedtest(model, blocks, steps=10, block_T=0.1, n_iter=10, device='/CPU:0'):
    """ Speedtest for parallel solver.

    Parameters:
        model: Model tuple (f, L, LT, T, x_grid, u_grid)
        blocks: Number of blocks
        steps: Number of steps
        block_T: Block time
        n_iter: Number of iterations
        device: Device to run on

    Returns:
        elapsed: Elapsed time
        err1: Error 1
        err2: Error 2
    """
    print('Running par_assoc_speedtest on device %s' % device)
    f, L, LT, T, x_grid, u_grid = model
    block_T = tf.constant(block_T, dtype=tf.float64)
    T = block_T * blocks

    with tf.device(device):
        _ = par_assoc(f, L, LT, block_T, blocks, steps, x_grid) # Compilation run
        tic = time.time()
        for i in range(n_iter):
            Vs = par_assoc(f, L, LT, block_T, blocks, steps, x_grid)
        toc = time.time()
        elapsed = (toc - tic) / n_iter

    Vs0 = hjb_grid_1d_tf.seq_bw_pass_upwind(f, L, LT, T, blocks, steps, x_grid, u_grid)

    err1 = tf.reduce_max(tf.math.abs(Vs - Vs0))
    n = x_grid.shape[0]
    err2 = tf.reduce_max(tf.math.abs(Vs[:, (n // 3):(2 * n // 3)] - Vs0[:, (n // 3):(2 * n // 3)]))

    print('blocks=%d, steps=%d, grid=%d, err1=%f, err2=%f took %f ms.' % (blocks, steps, n, err1.numpy(), err2.numpy(), 1000.0 * elapsed))
    return elapsed, err1, err2
