"""
Continuous-time linear speedtest routines (with TensorFlow).

@author: Simo Särkkä
"""

import numpy as np
import parallel_control.clqt_tf as clqt_tf
import parallel_control.parareal_clqt_tf as para_tf
import parallel_control.linear_model_tf as linear_model_tf
import tensorflow as tf
import time

# Abbreviations for TensorFlow functions
mm = tf.linalg.matmul
mv = tf.linalg.matvec

###############################################################################################
# Model(s)
###############################################################################################
def clqr_get_tracking_model(dtype=tf.float64):
    """ Get CLQT for a Wiener velocity linear model for speed testing.

    Parameters:
        dtype: Data type.

    Returns:
        x0: Initial state.
        XT: Terminal state.
        HT: Terminal cost matrix.
        rT: Terminal cost vector.
        F_f, L_f, X_f, U_f, c_f, H_f, r_f: CLQT model functions.
    """
    x0, T, XT, HT, rT, F_f, L_f, X_f, U_f, c_f, H_f, r_f = linear_model_tf.get_clqt(dtype=dtype)
    return x0, XT, HT, rT, T, F_f, L_f, X_f, U_f, c_f, H_f, r_f

###############################################################################################
# Generic linear model routines
###############################################################################################

def clqt_ref_sol_1(blocks, steps, x0, XT, HT, rT, T, F_f, L_f, X_f, U_f, c_f, H_f, r_f):
    """ Compute reference solution type 1 for CLQT for a linear model.

    Parameters:
        blocks: Number of blocks.
        steps: Number of steps per block.
        x0: Initial state.
        XT: Terminal state.
        HT: Terminal cost matrix.
        rT: Terminal cost vector.
        T: Time horizon.
        F_f, L_f, X_f, U_f, c_f, H_f, r_f: CLQT model functions.

    Returns:
        Ss1, vs1, Kxs1, ds1, xs1, us1: Reference solution.
    """

    dt, t0, ST, vT = clqt_tf.clqt_seq_backwardpass_defaults(blocks * steps, XT, HT, rT, T)
    Ss1, vs1, Kxs1, ds1 = clqt_tf.clqt_seq_backwardpass(blocks * steps, dt, t0, ST, vT, F_f, L_f, X_f, U_f, c_f, H_f, r_f)
    xs1, us1 = clqt_tf.clqt_seq_forwardpass(x0, Kxs1, ds1, dt, t0, F_f, L_f, c_f, u_zoh=False)
    return Ss1, vs1, Kxs1, ds1, xs1, us1

def clqt_ref_sol_2(blocks, steps, x0, XT, HT, rT, T, F_f, L_f, X_f, U_f, c_f, H_f, r_f):
    """ Compute reference solution type 2 for CLQT for a linear model.

    Parameters:
        blocks: Number of blocks.
        steps: Number of steps per block.
        x0: Initial state.
        XT: Terminal state.
        HT: Terminal cost matrix.
        rT: Terminal cost vector.
        T: Time horizon.
        F_f, L_f, X_f, U_f, c_f, H_f, r_f: CLQT model functions.

    Returns:
        Ss1, vs1, Kxs1, ds1, xs1, us1: Reference solution.
    """

    dt, t0, ST, vT = clqt_tf.clqt_seq_backwardpass_defaults(blocks * steps, XT, HT, rT, T)
    Ss1, vs1, Kxs1, ds1 = clqt_tf.clqt_seq_backwardpass(blocks * steps, dt, t0, ST, vT, F_f, L_f, X_f, U_f, c_f, H_f, r_f)
    dt, t0, A0, b0, C0 = clqt_tf.clqt_seq_fwdbwdpass_defaults(x0, blocks * steps, T)
    As1, bs1, Cs1 = clqt_tf.clqt_seq_fwdbwdpass(blocks * steps, dt, t0, A0, b0, C0, F_f, L_f, X_f, U_f, c_f, H_f, r_f)
    xs1, us1 = clqt_tf.clqt_combine_fwdbwd(Kxs1, ds1, Ss1, vs1, As1, bs1, Cs1)
    return Ss1, vs1, Kxs1, ds1, xs1, us1


@tf.function(reduce_retracing=True)
def clqt_seq_bw(blocks, steps, x0, XT, HT, rT, T, F_f, L_f, X_f, U_f, c_f, H_f, r_f):
    """ Run CLQT sequential backward pass.

    Parameters:
        blocks: Number of blocks.
        steps: Number of steps per block.
        x0: Initial state.
        XT: Terminal state.
        HT: Terminal cost matrix.
        rT: Terminal cost vector.
        T: Time horizon.
        F_f, L_f, X_f, U_f, c_f, H_f, r_f: CLQT model functions.

    Returns:
        Ss1, vs1, Kxs1, ds1: CLQT solution.
    """
    dt, t0, ST, vT = clqt_tf.clqt_seq_backwardpass_defaults(blocks * steps, XT, HT, rT, T)
    Ss1, vs1, Kxs1, ds1 = clqt_tf.clqt_seq_backwardpass(blocks * steps, dt, t0, ST, vT, F_f, L_f, X_f, U_f, c_f, H_f, r_f)
    return Ss1, vs1, Kxs1, ds1

def clqt_seq_bw_speedtest(model, blocks, steps, n_iter=10, device='/CPU:0'):
    """ Run speedtest for CLQT sequential backward pass.

    Parameters:
        model: CLQT model to be used.
        blocks: Number of blocks.
        steps: Number of steps per block.
        n_iter: Number of iterations to run in the speed test.
        device: Device to run the speed test on.

    Returns:
        elapsed: Average elapsed time per iteration.
        err: Maximum error between reference and the present CLQT solution.
    """

    print('Running clqt_seq_bw_speedtest on device %s' % device)
    with tf.device(device):
        _ = clqt_seq_bw(blocks, steps, *model)  # Compilation run
        tic = time.time()
        for i in range(n_iter):
            Ss1, vs1, Kxs1, ds1 = clqt_seq_bw(blocks, steps, *model)
        toc = time.time()
        elapsed = (toc - tic) / n_iter

    Ss2, vs2, Kxs2, ds2, xs2, us2 = clqt_ref_sol_1(blocks, steps, *model)
    err = tf.reduce_max(tf.math.abs(Ss1 - Ss2))

    print('blocks=%d, steps=%d took %f ms.' % (blocks, steps, 1000.0 * elapsed))
    return elapsed, err


@tf.function(reduce_retracing=True)
def clqt_seq_bw_fw(blocks, steps, x0, XT, HT, rT, T, F_f, L_f, X_f, U_f, c_f, H_f, r_f):
    """ Run CLQT sequential backward pass followed by forward pass.

    Parameters:
        blocks: Number of blocks.
        steps: Number of steps per block.
        x0: Initial state.
        XT: Terminal state.
        HT: Terminal cost matrix.
        rT: Terminal cost vector.
        T: Time horizon.
        F_f, L_f, X_f, U_f, c_f, H_f, r_f: CLQT model functions.

    Returns:
        Ss1, vs1, Kxs1, ds1, xs1, us1: CLQT solution.
    """
    dt, t0, ST, vT = clqt_tf.clqt_seq_backwardpass_defaults(blocks * steps, XT, HT, rT, T)
    Ss1, vs1, Kxs1, ds1 = clqt_tf.clqt_seq_backwardpass(blocks * steps, dt, t0, ST, vT, F_f, L_f, X_f, U_f, c_f, H_f, r_f)
    xs1, us1 = clqt_tf.clqt_seq_forwardpass(x0, Kxs1, ds1, dt, t0, F_f, L_f, c_f, u_zoh=False)
    return Ss1, vs1, Kxs1, ds1, xs1, us1

def clqt_seq_bw_fw_speedtest(model, blocks, steps, n_iter=10, device='/CPU:0'):
    """ Run speedtest for CLQT sequential backward pass followed by forward pass.

    Parameters:
        model: CLQT model to be used.
        blocks: Number of blocks.
        steps: Number of steps per block.
        n_iter: Number of iterations to run in the speed test.
        device: Device to run the speed test on.

    Returns:
        elapsed: Average elapsed time per iteration.
        err: Maximum error between reference and the present CLQT solution.
    """
    print('Running clqt_seq_bw_fw_speedtest on device %s' % device)
    with tf.device(device):
        _ = clqt_seq_bw_fw(blocks, steps, *model)  # Compilation run
        tic = time.time()
        for i in range(n_iter):
            Ss1, vs1, Kxs1, ds1, xs1, us1 = clqt_seq_bw_fw(blocks, steps, *model)
        toc = time.time()
        elapsed = (toc - tic) / n_iter

    Ss2, vs2, Kxs2, ds2, xs2, us2 = clqt_ref_sol_1(blocks, steps, *model)
    err = tf.reduce_max(tf.math.abs(xs1 - xs2))

    print('blocks=%d, steps=%d took %f ms.' % (blocks, steps, 1000.0 * elapsed))
    return elapsed, err


@tf.function(reduce_retracing=True)
def clqt_par_bw(blocks, steps, x0, XT, HT, rT, T, F_f, L_f, X_f, U_f, c_f, H_f, r_f):
    """ Run CLQT parallel backward pass.

    Parameters:
        blocks: Number of blocks.
        steps: Number of steps per block.
        x0: Initial state.
        XT: Terminal state.
        HT: Terminal cost matrix.
        rT: Terminal cost vector.
        T: Time horizon.
        F_f, L_f, X_f, U_f, c_f, H_f, r_f: CLQT model functions.

    Returns:
        Ss3, vs3, Kxs3, ds3: CLQT solution.
    """
    dt, t0, ST, vT = clqt_tf.par_backwardpass_defaults(blocks, steps, XT, HT, rT, T)
    Ss3, vs3, Kxs3, ds3 = clqt_tf.clqt_par_backwardpass(blocks, steps, dt, t0, ST, vT, F_f, L_f, X_f, U_f, c_f, H_f, r_f, max_parallel=blocks)
    return Ss3, vs3, Kxs3, ds3

def clqt_par_bw_speedtest(model, blocks, steps, n_iter=10, device='/CPU:0'):
    """ Run speedtest for CLQT parallel backward pass.

    Parameters:
        model: CLQT model to be used.
        blocks: Number of blocks.
        steps: Number of steps per block.
        n_iter: Number of iterations to run in the speed test.
        device: Device to run the speed test on.

    Returns:
        elapsed: Average elapsed time per iteration.
        err: Maximum error between reference and the present CLQT solution.
    """
    print('Running clqt_par_bw_speedtest on device %s' % device)
    with tf.device(device):
        _ = clqt_par_bw(blocks, steps, *model)  # Compilation run
        tic = time.time()
        for i in range(n_iter):
            Ss1, vs1, Kxs1, ds1 = clqt_par_bw(blocks, steps, *model)
        toc = time.time()
        elapsed = (toc - tic) / n_iter

    Ss2, vs2, Kxs2, ds2, xs2, us2 = clqt_ref_sol_1(blocks, steps, *model)
    err = tf.reduce_max(tf.math.abs(Ss1 - Ss2))

    print('blocks=%d, steps=%d took %f ms.' % (blocks, steps, 1000.0 * elapsed))
    return elapsed, err


@tf.function(reduce_retracing=True)
def clqt_par_bw_fw(blocks, steps, x0, XT, HT, rT, T, F_f, L_f, X_f, U_f, c_f, H_f, r_f):
    """ Run CLQT parallel backward pass followed by forward pass.

    Parameters:
        blocks: Number of blocks.
        steps: Number of steps per block.
        x0: Initial state.
        XT: Terminal state.
        HT: Terminal cost matrix.
        rT: Terminal cost vector.
        T: Time horizon.
        F_f, L_f, X_f, U_f, c_f, H_f, r_f: CLQT model functions.

    Returns:
        Ss3, vs3, Kxs3, ds3, xs3, us3: CLQT solution.
    """
    dt, t0, ST, vT = clqt_tf.par_backwardpass_defaults(blocks, steps, XT, HT, rT, T)
    Ss3, vs3, Kxs3, ds3 = clqt_tf.clqt_par_backwardpass(blocks, steps, dt, t0, ST, vT, F_f, L_f, X_f, U_f, c_f, H_f, r_f, max_parallel=blocks)
    xs3, us3 = clqt_tf.clqt_par_forwardpass(blocks, steps, x0, Kxs3, ds3, dt, t0, F_f, L_f, c_f, forward=True, u_zoh=False, max_parallel=blocks)
    return Ss3, vs3, Kxs3, ds3, xs3, us3

def clqt_par_bw_fw_speedtest(model, blocks, steps, n_iter=10, device='/CPU:0'):
    """ Run speedtest for CLQT parallel backward pass followed by forward pass.

    Parameters:
        model: CLQT model to be used.
        blocks: Number of blocks.
        steps: Number of steps per block.
        n_iter: Number of iterations to run in the speed test.
        device: Device to run the speed test on.

    Returns:
        elapsed: Average elapsed time per iteration.
        err: Maximum error between reference and the present CLQT solution.
    """
    print('Running clqt_par_bw_fw_speedtest on device %s' % device)
    with tf.device(device):
        _ = clqt_par_bw_fw(blocks, steps, *model)  # Compilation run
        tic = time.time()
        for i in range(n_iter):
            Ss1, vs1, Kxs1, ds1, xs1, us1 = clqt_par_bw_fw(blocks, steps, *model)
        toc = time.time()
        elapsed = (toc - tic) / n_iter

    Ss2, vs2, Kxs2, ds2, xs2, us2 = clqt_ref_sol_1(blocks, steps, *model)
    err = tf.reduce_max(tf.math.abs(xs1 - xs2))

    print('blocks=%d, steps=%d took %f ms.' % (blocks, steps, 1000.0 * elapsed))
    return elapsed, err


@tf.function(reduce_retracing=True)
def clqt_par_bw_fwbw(blocks, steps, x0, XT, HT, rT, T, F_f, L_f, X_f, U_f, c_f, H_f, r_f):
    """ Run CLQT parallel backward pass followed by forward value function pass.

    Parameters:
        blocks: Number of blocks.
        steps: Number of steps per block.
        x0: Initial state.
        XT: Terminal state.
        HT: Terminal cost matrix.
        rT: Terminal cost vector.
        T: Time horizon.
        F_f, L_f, X_f, U_f, c_f, H_f, r_f: CLQT model functions.

    Returns:
        Ss3, vs3, Kxs3, ds3, xs3, us3: CLQT solution.
    """

    dt, t0, ST, vT = clqt_tf.par_backwardpass_defaults(blocks, steps, XT, HT, rT, T)
    Ss3, vs3, Kxs3, ds3 = clqt_tf.clqt_par_backwardpass(blocks, steps, dt, t0, ST, vT, F_f, L_f, X_f, U_f, c_f, H_f, r_f, max_parallel=blocks)
    xs3, us3 = clqt_tf.clqt_par_fwdbwdpass(blocks, steps, x0, Ss3, vs3, Kxs3, ds3, dt, t0, F_f, L_f, X_f, U_f, c_f, H_f, r_f, forward=True, max_parallel=blocks)
    return Ss3, vs3, Kxs3, ds3, xs3, us3

def clqt_par_bw_fwbw_speedtest(model, blocks, steps, n_iter=10, device='/CPU:0'):
    """ Run speedtest for CLQT parallel backward pass followed by forward value function pass.

    Parameters:
        model: CLQT model to be used.
        blocks: Number of blocks.
        steps: Number of steps per block.
        n_iter: Number of iterations to run in the speed test.
        device: Device to run the speed test on.

    Returns:
        elapsed: Average elapsed time per iteration.
        err: Maximum error between reference and the present CLQT solution.
    """

    print('Running clqt_par_bw_fwbw_speedtest on device %s' % device)
    with tf.device(device):
        _ = clqt_par_bw_fwbw(blocks, steps, *model)  # Compilation run
        tic = time.time()
        for i in range(n_iter):
            Ss1, vs1, Kxs1, ds1, xs1, us1 = clqt_par_bw_fwbw(blocks, steps, *model)
        toc = time.time()
        elapsed = (toc - tic) / n_iter

    Ss2, vs2, Kxs2, ds2, xs2, us2 = clqt_ref_sol_2(blocks, steps, *model)
    err = tf.reduce_max(tf.math.abs(xs1 - xs2))

    print('blocks=%d, steps=%d took %f ms.' % (blocks, steps, 1000.0 * elapsed))
    return elapsed, err


@tf.function(reduce_retracing=True)
def clqt_parareal_bw(blocks, steps, niter, x0, XT, HT, rT, T, F_f, L_f, X_f, U_f, c_f, H_f, r_f):
    """ Run Parareal for solving the backward pass of CLQT problem.

    Parameters:
        blocks: Number of blocks.
        steps: Number of steps per block.
        niter: Number of Parareal iterations.
        x0: Initial state.
        XT: Terminal state.
        HT: Terminal cost matrix.
        rT: Terminal cost vector.
        T: Time horizon.
        F_f, L_f, X_f, U_f, c_f, H_f, r_f: CLQT model functions.

    Returns:
        Ss2, vs2, Kxs2, ds2: CLQT solution.
    """
    Ss2, vs2, Kxs2, ds2 = para_tf.pclqt_backwardpass(blocks, steps, niter, XT, HT, rT, T, F_f, L_f, X_f, U_f, c_f, H_f, r_f)
    return Ss2, vs2, Kxs2, ds2

def clqt_parareal_bw_speedtest(model, blocks, steps, parareal_niter=2, n_iter=10, device='/CPU:0'):
    """ Run speedtest for Parareal for solving the backward pass of CLQT problem.

    Parameters:
        model: CLQT model to be used.
        blocks: Number of blocks.
        steps: Number of steps per block.
        parareal_niter: Number of Parareal iterations.
        n_iter: Number of iterations to run in the speed test.
        device: Device to run the speed test on.

    Returns:
        elapsed: Average elapsed time per iteration.
        err: Maximum error between reference and the present CLQT solution.
    """

    print('Running clqt_parareal_bw_speedtest(%d) on device %s' % (parareal_niter, device))
    with tf.device(device):
        _ = clqt_parareal_bw(blocks, steps, parareal_niter, *model)  # Compilation run
        tic = time.time()
        for i in range(n_iter):
            Ss1, vs1, Kxs1, ds1 = clqt_parareal_bw(blocks, steps, parareal_niter, *model)
        toc = time.time()
        elapsed = (toc - tic) / n_iter

    Ss2, vs2, Kxs2, ds2, xs2, us2 = clqt_ref_sol_1(blocks, steps, *model)
    err = tf.reduce_max(tf.math.abs(Ss1 - Ss2))

    print('blocks=%d, steps=%d took %f ms.' % (blocks, steps, 1000.0 * elapsed))
    return elapsed, err


@tf.function(reduce_retracing=True)
def clqt_parareal_bw_fw(blocks, steps, niter, x0, XT, HT, rT, T, F_f, L_f, X_f, U_f, c_f, H_f, r_f):
    """ Run Parareal for solving the backward and forward passes of CLQT problem.

    Parameters:
        blocks: Number of blocks.
        steps: Number of steps per block.
        niter: Number of Parareal iterations.
        x0: Initial state.
        XT: Terminal state.
        HT: Terminal cost matrix.
        rT: Terminal cost vector.
        T: Time horizon.
        F_f, L_f, X_f, U_f, c_f, H_f, r_f: CLQT model functions.

    Returns:
        Ss2, vs2, Kxs2, ds2, xs2, us2: CLQT solution.
    """
    Ss2, vs2, Kxs2, ds2 = para_tf.pclqt_backwardpass(blocks, steps, niter, XT, HT, rT, T, F_f, L_f, X_f, U_f, c_f, H_f, r_f)
    xs2, us2 = para_tf.pclqt_forwardpass(blocks, steps, niter, x0, T, Kxs2, ds2, F_f, L_f, c_f)
    return Ss2, vs2, Kxs2, ds2, xs2, us2

def clqt_parareal_bw_fw_speedtest(model, blocks, steps, parareal_niter=2, n_iter=10, device='/CPU:0'):
    """ Run speedtest for Parareal for solving the backward and forward passes of CLQT problem.

    Parameters:
        model: CLQT model to be used.
        blocks: Number of blocks.
        steps: Number of steps per block.
        parareal_niter: Number of Parareal iterations.
        n_iter: Number of iterations to run in the speed test.
        device: Device to run the speed test on.

    Returns:
        elapsed: Average elapsed time per iteration.
        err: Maximum error between reference and the present CLQT solution.
    """

    print('Running clqt_parareal_bw_fw_speedtest(%d) on device %s' % (parareal_niter, device))
    with tf.device(device):
        _ = clqt_parareal_bw_fw(blocks, steps, parareal_niter, *model)  # Compilation run
        tic = time.time()
        for i in range(n_iter):
            Ss1, vs1, Kxs1, ds1, xs1, us1 = clqt_parareal_bw_fw(blocks, steps, parareal_niter, *model)
        toc = time.time()
        elapsed = (toc - tic) / n_iter

    Ss2, vs2, Kxs2, ds2, xs2, us2 = clqt_ref_sol_1(blocks, steps, *model)
    err = tf.reduce_max(tf.math.abs(xs1 - xs2))

    print('blocks=%d, steps=%d took %f ms.' % (blocks, steps, 1000.0 * elapsed))
    return elapsed, err


@tf.function(reduce_retracing=True)
def clqt_parareal_bw_fwbw(blocks, steps, niter, x0, XT, HT, rT, T, F_f, L_f, X_f, U_f, c_f, H_f, r_f):
    """ Run Parareal for solving the backward and forward value function passes of CLQT problem.

    Parameters:
        blocks: Number of blocks.
        steps: Number of steps per block.
        niter: Number of Parareal iterations.
        x0: Initial state.
        XT: Terminal state.
        HT: Terminal cost matrix.
        rT: Terminal cost vector.
        T: Time horizon.
        F_f, L_f, X_f, U_f, c_f, H_f, r_f: CLQT model functions.

    Returns:
        Ss2, vs2, Kxs2, ds2, xs2, us2: CLQT solution.
    """
    Ss2, vs2, Kxs2, ds2 = para_tf.pclqt_backwardpass(blocks, steps, niter, XT, HT, rT, T, F_f, L_f, X_f, U_f, c_f, H_f, r_f)
    As2, bs2, Cs2 = para_tf.pclqt_fwdbwdpass(blocks, steps, niter, x0, T, F_f, L_f, X_f, U_f, c_f, H_f, r_f)
    xs2, us2 = clqt_tf.clqt_combine_fwdbwd(Kxs2, ds2, Ss2, vs2, As2, bs2, Cs2)
    return Ss2, vs2, Kxs2, ds2, xs2, us2

def clqt_parareal_bw_fwbw_speedtest(model, blocks, steps, parareal_niter=2, n_iter=10, device='/CPU:0'):
    """ Run speedtest for Parareal for solving the backward and forward value function passes of CLQT problem.

    Parameters:
        model: CLQT model to be used.
        blocks: Number of blocks.
        steps: Number of steps per block.
        parareal_niter: Number of Parareal iterations.
        n_iter: Number of iterations to run in the speed test.
        device: Device to run the speed test on.

    Returns:
        elapsed: Average elapsed time per iteration.
        err: Maximum error between reference and the present CLQT solution.
    """
    print('Running clqt_parareal_bw_fwbw_speedtest(%d) on device %s' % (parareal_niter, device))
    with tf.device(device):
        _ = clqt_parareal_bw_fwbw(blocks, steps, parareal_niter, *model)  # Compilation run
        tic = time.time()
        for i in range(n_iter):
            Ss1, vs1, Kxs1, ds1, xs1, us1 = clqt_parareal_bw_fwbw(blocks, steps, parareal_niter, *model)
        toc = time.time()
        elapsed = (toc - tic) / n_iter

    Ss2, vs2, Kxs2, ds2, xs2, us2 = clqt_ref_sol_2(blocks, steps, *model)
    err = tf.reduce_max(tf.math.abs(xs1 - xs2))

    print('blocks=%d, steps=%d took %f ms.' % (blocks, steps, 1000.0 * elapsed))
    return elapsed, err

