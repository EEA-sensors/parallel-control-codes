"""
Nonlinear speedtest routines.

@author: Simo Särkkä
"""

import numpy as np
import parallel_control.nlqt_np as nlqt_np
import parallel_control.nonlinear_model_np as nonlinear_model_np
import tensorflow as tf
import parallel_control.lqt_tf as lqt_tf
import parallel_control.nlqt_tf as nlqt_tf
import parallel_control.nonlinear_model_tf as nonlinear_model_tf
import time

###############################################################################################
#
# General nonlinear speedtests
#
###############################################################################################


@tf.function
def nlqt_iter_seq(us, xs, f, Fx, Fu, Hs, HT, rs, rT, Xs, XT, Us):
    """ Perform sequential NLQT iterations.

    Parameters:
        us: Tensor of initial control inputs.
        xs: Tensor of initial states.
        f: Nonlinear state transition function.
        Fx: Jacobian of f w.r.t. x.
        Fu: Jacobian of f w.r.t. u.
        Hs: Tensor of state cost output matrices.
        HT: State cost output matrix at final time.
        rs: Tensor of state cost offsets.
        rT: State cost offset at final time.
        Xs: Tensor of state cost matrices.
        XT: State cost matrix at final time.
        Us: Tensor of control cost matrices.
    """
    for j in range(10):
       us, xs = nlqt_tf.nlqt_iterate_seq(us, xs, f, Fx, Fu, Hs, HT, rs, rT, Xs, XT, Us)
    return us, xs

def nlqt_iter_seq_speedtest(nlqt_gen, n_iter=10, device='/CPU:0'):
    """ Run speedtest for sequential NLQT iterations.

    Parameters:
        nlqt_gen: Generator of NLQT problems.
        n_iter: Number of iterations.
        device: Device to run on.

    Returns:
        T: List of time steps.
        D: List of state dimensions.
        run_times: List of run times.
    """
    print('Running nlqt_iter_seq_speedtest on device %s' % device)
    T = []
    D = []
    run_times = []
    for (us, xs, f, Fx, Fu, Hs, HT, rs, rT, Xs, XT, Us) in nlqt_gen:
        with tf.device(device):
            _ = nlqt_iter_seq(us, xs, f, Fx, Fu, Hs, HT, rs, rT, Xs, XT, Us)  # Compilation run
            tic = time.time()
            for i in range(n_iter):
                _ = nlqt_iter_seq(us, xs, f, Fx, Fu, Hs, HT, rs, rT, Xs, XT, Us)
            toc = time.time()
        T.append(us.shape[0])
        D.append(xs.shape[1])
        run_times.append((toc - tic) / n_iter)
        print('T=%d, D=%d took %f ms' % (us.shape[0], xs.shape[1], 1000.0 * run_times[-1]))
    return T, D, run_times


@tf.function
def nlqt_iter_par_1(us, xs, f, Fx, Fu, Hs, HT, rs, rT, Xs, XT, Us):
    """ Perform parallel NLQT iterations.

    Parameters:
        us: Tensor of initial control inputs.
        xs: Tensor of initial states.
        f: Nonlinear state transition function.
        Fx: Jacobian of f w.r.t. x.
        Fu: Jacobian of f w.r.t. u.
        Hs: Tensor of state cost output matrices.
        HT: State cost output matrix at final time.
        rs: Tensor of state cost offsets.
        rT: State cost offset at final time.
        Xs: Tensor of state cost matrices.
        XT: State cost matrix at final time.
        Us: Tensor of control cost matrices.

    Returns:
        us: Tensor of control inputs.
        xs: Tensor of states.
    """
    for j in range(10):
       us, xs = nlqt_tf.nlqt_iterate_par_1(us, xs, f, Fx, Fu, Hs, HT, rs, rT, Xs, XT, Us, max_parallel=rs.shape[0])
    return us, xs

def nlqt_iter_par_1_speedtest(nlqt_gen, n_iter=10, device='/CPU:0'):
    """ Run speedtest for parallel NLQT iterations.

    Parameters:
        nlqt_gen: Generator of NLQT problems.
        n_iter: Number of iterations.
        device: Device to run on.

    Returns:
        T: List of time steps.
        D: List of state dimensions.
        run_times: List of run times.
    """
    print('Running nlqt_iter_par_1_speedtest on device %s' % device)
    T = []
    D = []
    run_times = []
    for (us, xs, f, Fx, Fu, Hs, HT, rs, rT, Xs, XT, Us) in nlqt_gen:
        with tf.device(device):
            _ = nlqt_iter_par_1(us, xs, f, Fx, Fu, Hs, HT, rs, rT, Xs, XT, Us)  # Compilation run
            tic = time.time()
            for i in range(n_iter):
                _ = nlqt_iter_par_1(us, xs, f, Fx, Fu, Hs, HT, rs, rT, Xs, XT, Us)
            toc = time.time()
        T.append(us.shape[0])
        D.append(xs.shape[1])
        run_times.append((toc - tic) / n_iter)
        print('T=%d, D=%d took %f ms' % (us.shape[0], xs.shape[1], 1000.0 * run_times[-1]))
    return T, D, run_times


@tf.function
def nlqt_iter_par_2(us, xs, f, Fx, Fu, Hs, HT, rs, rT, Xs, XT, Us):
    """" Perform parallel NLQT iterations (fw/bw version).

    Parameters:
        us: Tensor of initial control inputs.
        xs: Tensor of initial states.
        f: Nonlinear state transition function.
        Fx: Jacobian of f w.r.t. x.
        Fu: Jacobian of f w.r.t. u.
        Hs: Tensor of state cost output matrices.
        HT: State cost output matrix at final time.
        rs: Tensor of state cost offsets.
        rT: State cost offset at final time.
        Xs: Tensor of state cost matrices.
        XT: State cost matrix at final time.
        Us: Tensor of control cost matrices.

    Returns:
        us: Tensor of control inputs.
        xs: Tensor of states.
    """
    for j in range(10):
        us, xs = nlqt_tf.nlqt_iterate_par_2(us, xs, f, Fx, Fu, Hs, HT, rs, rT, Xs, XT, Us, max_parallel=rs.shape[0])
    return us, xs

def nlqt_iter_par_2_speedtest(nlqt_gen, n_iter=10, device='/CPU:0'):
    """ Run speedtest for parallel NLQT iterations (fw/bw version).

    Parameters:
        nlqt_gen: Generator of NLQT problems.
        n_iter: Number of iterations.
        device: Device to run on.

    Returns:
        T: List of time steps.
        D: List of state dimensions.
        run_times: List of run times.
    """
    print('Running nlqt_iter_par_2_speedtest on device %s' % device)
    T = []
    D = []
    run_times = []
    for (us, xs, f, Fx, Fu, Hs, HT, rs, rT, Xs, XT, Us) in nlqt_gen:
        with tf.device(device):
            _ = nlqt_iter_par_2(us, xs, f, Fx, Fu, Hs, HT, rs, rT, Xs, XT, Us)  # Compilation run
            tic = time.time()
            for i in range(n_iter):
                _ = nlqt_iter_par_2(us, xs, f, Fx, Fu, Hs, HT, rs, rT, Xs, XT, Us)
            toc = time.time()
        T.append(us.shape[0])
        D.append(xs.shape[1])
        run_times.append((toc - tic) / n_iter)
        print('T=%d, D=%d took %f ms' % (us.shape[0], xs.shape[1], 1000.0 * run_times[-1]))
    return T, D, run_times


###############################################################################################
# Nonlinear tracking model
###############################################################################################

def nonlinear_generator(start=2, stop=5, num=20, dtype=tf.float64):
    """ Generate nonlinear tracking problems.

    Parameters:
        start: Start of logspace data lengths.
        stop: End of logspace data lengths.
        num: Number of data lengths.
        dtype: Data type.

    Returns:
        Generator of NLQT problems yielding (us, xs, f, Fx, Fu, Hs, HT, rs, rT, Xs, XT, Us).
    """
    T_list = np.logspace(start, stop, num=num, base=10).astype(int)
    print('Nonlinear T_list = %s' % str(T_list))

    for k in range(T_list.shape[0]):
        model = nonlinear_model_np.NonlinearModel()
        xyt, xyt_dense = model.genData(T_list[k])
        lqt, x0 = model.getLQT(xyt)
        nlqt = nlqt_np.NLQT(lqt, model)
        u, x = model.initialGuess(lqt, x0, init_to_zero=True)

        us = tf.convert_to_tensor(u, dtype=dtype)
        xs = tf.convert_to_tensor(x, dtype=dtype)
        Fs, cs, Ls, Hs, HT, rs, rT, Xs, XT, Us = lqt_tf.lqt_np_to_tf(nlqt.lqt, dtype=dtype)

        f = nonlinear_model_tf.nonlinear_model_f
        Fx = nonlinear_model_tf.nonlinear_model_Fx
        Fu = nonlinear_model_tf.nonlinear_model_Fu

        yield (us, xs, f, Fx, Fu, Hs, HT, rs, rT, Xs, XT, Us)
