"""
Discrete-time linear speedtest routines.

@author: Simo Särkkä
"""

import numpy as np
import parallel_control.lqt_tf as lqt_tf
import parallel_control.condensing_tf as condensing_tf
import parallel_control.linear_model_np as linear_model_np
import parallel_control.mass_model_np as mass_model_np
import tensorflow as tf
import time

# Abbreviations for linear algebra routines
mm = tf.linalg.matmul
mv = tf.linalg.matvec

###############################################################################################
# Generic linear model routines
###############################################################################################

def lqt_reference(x0, lqt):
    """ Reference LQT to check that the result is correct.

    Parameters:
        x0: Initial state.
        lqt: LQT object.

    Returns:
        Kx_list1: List of control gains.
        d_list1: List of control offsets.
        S_list1: List of value function matrices.
        v_list1: List of value function vectors.
        x_list1: List of states.
        u_list1: List of controls.
    """
    Kx_list1, d_list1, S_list1, v_list1 = lqt.seqBackwardPass()
    u_list1, x_list1 = lqt.seqForwardPass(x0, Kx_list1, d_list1)
    return Kx_list1, d_list1, S_list1, v_list1, x_list1, u_list1

# Sequential backward pass
@tf.function
def lqt_sequential_bw(Fs, cs, Ls, Hs, HT, rs, rT, Xs, XT, Us):
    """ Sequential backward pass for LQT.

    Parameters:
        Fs, cs, Ls, Hs, HT, rs, rT, Xs, XT, Us: LQT parameters.

    Returns:
        Ss, vs, Kxs, ds: Value function matrices, vectors, control gains, and offsets.
    """
    Ss, vs, Kxs, ds = lqt_tf.lqt_seq_backwardpass(Fs, cs, Ls, Hs, HT, rs, rT, Xs, XT, Us)
    return Ss, vs, Kxs, ds

def lqt_sequential_bw_speedtest(lqt_gen, n_iter=10, device='/CPU:0'):
    """ Speedtest for sequential backward pass.

    Parameters:
        lqt_gen: LQT generator.
        n_iter: Number of iterations.
        device: Device to use.

    Returns:
        T: List of time steps.
        D: List of state dimensions.
        run_times: List of run times.
    """
    print('Running lqt_sequential_bw_speedtest on device %s' % device)
    T = []
    D = []
    run_times = []
    for (x0_tf, Fs, cs, Ls, Hs, HT, rs, rT, Xs, XT, Us) in lqt_gen:
        with tf.device(device):
            _ = lqt_sequential_bw(Fs, cs, Ls, Hs, HT, rs, rT, Xs, XT, Us)  # Compilation run
            tic = time.time()
            for i in range(n_iter):
                Ss, vs, Kxs, ds = lqt_sequential_bw(Fs, cs, Ls, Hs, HT, rs, rT, Xs, XT, Us)
            toc = time.time()
        T.append(Fs.shape[0])
        D.append(Fs.shape[1])
        run_times.append((toc - tic) / n_iter)
        print('T=%d, D=%d took %f ms' % (Fs.shape[0], Fs.shape[1], 1000.0 * run_times[-1]))
    return T, D, run_times

# Parallel backward pass
@tf.function
def lqt_parallel_bw(Fs, cs, Ls, Hs, HT, rs, rT, Xs, XT, Us):
    """ Parallel backward pass for LQT.

    Parameters:
        Fs, cs, Ls, Hs, HT, rs, rT, Xs, XT, Us: LQT parameters.

    Returns:
        Ss, vs, Kxs, ds: Value function matrices, vectors, control gains, and offsets.
    """
    Ss, vs, Kxs, ds = lqt_tf.lqt_par_backwardpass(Fs, cs, Ls, Hs, HT, rs, rT, Xs, XT, Us, max_parallel=Fs.shape[0])
    return Ss, vs, Kxs, ds

def lqt_parallel_bw_speedtest(lqt_gen, n_iter=10, device='/CPU:0'):
    """ Speedtest for parallel backward pass.

    Parameters:
        lqt_gen: LQT generator.
        n_iter: Number of iterations.
        device: Device to use.

    Returns:
        T: List of time steps.
        D: List of state dimensions.
        run_times: List of run times.
    """
    print('Running lqt_parallel_bw_speedtest on device %s' % device)
    T = []
    D = []
    run_times = []
    for (x0_tf, Fs, cs, Ls, Hs, HT, rs, rT, Xs, XT, Us) in lqt_gen:
        with tf.device(device):
            _ = lqt_parallel_bw(Fs, cs, Ls, Hs, HT, rs, rT, Xs, XT, Us) # Compilation run
            tic = time.time()
            for i in range(n_iter):
                _ = lqt_parallel_bw(Fs, cs, Ls, Hs, HT, rs, rT, Xs, XT, Us)
            toc = time.time()
        T.append(Fs.shape[0])
        D.append(Fs.shape[1])
        run_times.append((toc - tic) / n_iter)
        print('T=%d, D=%d took %f ms' % (Fs.shape[0], Fs.shape[1], 1000.0 * run_times[-1]))
    return T, D, run_times

# Pure sequential backward-forward
@tf.function
def lqt_sequential_bwfw(x0_tf, Fs, cs, Ls, Hs, HT, rs, rT, Xs, XT, Us):
    """ Sequential backward and forward pass for LQT.

    Parameters:
        x0_tf: Initial state.
        Fs, cs, Ls, Hs, HT, rs, rT, Xs, XT, Us: LQT parameters.

    Returns:
        xs_tf_seq, us_tf_seq: State and control trajectories.
    """
    Ss, vs, Kxs, ds = lqt_tf.lqt_seq_backwardpass(Fs, cs, Ls, Hs, HT, rs, rT, Xs, XT, Us)
    xs_tf_seq, us_tf_seq = lqt_tf.lqt_seq_forwardpass(x0_tf, Fs, cs, Ls, Kxs, ds)
    return xs_tf_seq, us_tf_seq

def lqt_sequential_bwfw_speedtest(lqt_gen, n_iter=10, device='/CPU:0'):
    """ Speedtest for sequential backward and forward pass.

    Parameters:
        lqt_gen: LQT generator.
        n_iter: Number of iterations.
        device: Device to use.

    Returns:
        T: List of time steps.
        D: List of state dimensions.
        run_times: List of run times.
    """

    print('Running lqt_sequential_bwfw_speedtest on device %s' % device)
    T = []
    D = []
    run_times = []
    for (x0_tf, Fs, cs, Ls, Hs, HT, rs, rT, Xs, XT, Us) in lqt_gen:
        with tf.device(device):
            _ = lqt_sequential_bwfw(x0_tf, Fs, cs, Ls, Hs, HT, rs, rT, Xs, XT, Us) # Compilation run
            tic = time.time()
            for i in range(n_iter):
                _ = lqt_sequential_bwfw(x0_tf, Fs, cs, Ls, Hs, HT, rs, rT, Xs, XT, Us)
            toc = time.time()
        T.append(Fs.shape[0])
        D.append(Fs.shape[1])
        run_times.append((toc - tic) / n_iter)
        print('T=%d, D=%d took %f ms' % (Fs.shape[0], Fs.shape[1], 1000.0 * run_times[-1]))
    return T, D, run_times

# Parallel with with backward-forward pass type 1
@tf.function
def lqt_parallel_bwfw_1(x0_tf, Fs, cs, Ls, Hs, HT, rs, rT, Xs, XT, Us):
    """" Parallel backward and forward pass for LQT.

    Parameters:
        x0_tf: Initial state.
        Fs, cs, Ls, Hs, HT, rs, rT, Xs, XT, Us: LQT parameters.

    Returns:
        xs_tf_par1, us_tf_par1: State and control trajectories.
    """
    Ss, vs, Kxs, ds = lqt_tf.lqt_par_backwardpass(Fs, cs, Ls, Hs, HT, rs, rT, Xs, XT, Us, max_parallel=Fs.shape[0])
    xs_tf_par1, us_tf_par1 = lqt_tf.lqt_par_forwardpass(x0_tf, Fs, cs, Ls, Kxs, ds, max_parallel=Fs.shape[0])
    return xs_tf_par1, us_tf_par1

def lqt_parallel_bwfw_1_speedtest(lqt_gen, n_iter=10, device='/CPU:0'):
    """ Speedtest for parallel backward and forward pass.

    Parameters:
        lqt_gen: LQT generator.
        n_iter: Number of iterations.
        device: Device to use.

    Returns:
        T: List of time steps.
        D: List of state dimensions.
        run_times: List of run times.
    """
    print('Running lqt_parallel_bwfw_1_speedtest on device %s' % device)
    T = []
    D = []
    run_times = []
    for (x0_tf, Fs, cs, Ls, Hs, HT, rs, rT, Xs, XT, Us) in lqt_gen:
        with tf.device(device):
            _ = lqt_parallel_bwfw_1(x0_tf, Fs, cs, Ls, Hs, HT, rs, rT, Xs, XT, Us) # Compilation run
            tic = time.time()
            for i in range(n_iter):
                _ = lqt_parallel_bwfw_1(x0_tf, Fs, cs, Ls, Hs, HT, rs, rT, Xs, XT, Us)
            toc = time.time()
        T.append(Fs.shape[0])
        D.append(Fs.shape[1])
        run_times.append((toc - tic) / n_iter)
        print('T=%d, D=%d took %f ms' % (Fs.shape[0], Fs.shape[1], 1000.0 * run_times[-1]))
    return T, D, run_times

# Parallel with with backward-forward pass type 2
@tf.function
def lqt_parallel_bwfw_2(x0_tf, Fs, cs, Ls, Hs, HT, rs, rT, Xs, XT, Us):
    """ Parallel backward and forward (forward-backward type) pass for LQT.

    Parameters:
        x0_tf: Initial state.
        Fs, cs, Ls, Hs, HT, rs, rT, Xs, XT, Us: LQT parameters.

    Returns:
        xs_tf_par2, us_tf_par2: State and control trajectories.
    """
    Ss, vs, Kxs, ds = lqt_tf.lqt_par_backwardpass(Fs, cs, Ls, Hs, HT, rs, rT, Xs, XT, Us, max_parallel=Fs.shape[0])
    xs_tf_par2, us_tf_par2 = lqt_tf.lqt_par_fwdbwdpass(x0_tf, Fs, cs, Ls, Hs, rs, Xs, Us, Ss, vs, Kxs, ds, max_parallel=Fs.shape[0])
    return xs_tf_par2, us_tf_par2

def lqt_parallel_bwfw_2_speedtest(lqt_gen, n_iter=10, device='/CPU:0'):
    """ Speedtest for parallel backward and forward (forward-backward type) pass.

    Parameters:
        lqt_gen: LQT generator.
        n_iter: Number of iterations.
        device: Device to use.

    Returns:
        T: List of time steps.
        D: List of state dimensions.
        run_times: List of run times.
    """
    print('Running lqt_parallel_bwfw_2_speedtest on device %s' % device)
    T = []
    D = []
    run_times = []
    for (x0_tf, Fs, cs, Ls, Hs, HT, rs, rT, Xs, XT, Us) in lqt_gen:
        with tf.device(device):
            _ = lqt_parallel_bwfw_2(x0_tf, Fs, cs, Ls, Hs, HT, rs, rT, Xs, XT, Us) # Compilation run
            tic = time.time()
            for i in range(n_iter):
                _ = lqt_parallel_bwfw_2(x0_tf, Fs, cs, Ls, Hs, HT, rs, rT, Xs, XT, Us)
            toc = time.time()
        T.append(Fs.shape[0])
        D.append(Fs.shape[1])
        run_times.append((toc - tic) / n_iter)
        print('T=%d, D=%d took %f ms' % (Fs.shape[0], Fs.shape[1], 1000.0 * run_times[-1]))
    return T, D, run_times

# Sequential with partial condensing
@tf.function
def lqt_sequential_cond_bwfw(x0_tf, Fs, cs, Ls, Hs, HT, rs, rT, Xs, XT, Us, Nc):
    """ Sequential backward and forward pass with partial condensing for LQT.

    Parameters:
        x0_tf: Initial state.
        Fs, cs, Ls, Hs, HT, rs, rT, Xs, XT, Us: LQT parameters.
        Nc: Number of condensing steps.

    Returns:
        xs_tf_cond, us_tf_cond: State and control trajectories.
    """
    Fstar, cstar, Lstar, Hstar, rstar, Xstar, Ustar, Mstar, sstar, Lambda, cbar, Lbar \
        = condensing_tf.condense(Fs, cs, Ls, Hs, rs, Xs, Us, Nc)

    Fs_trans, cs_trans, Xs_trans \
        = lqt_tf.lqt_gen_to_canon(Fstar, cstar, Lstar, Hstar, rstar, Xstar, Ustar, Mstar, sstar)

    Ss, vs, Kxs_trans, ds_trans \
        = lqt_tf.lqt_seq_backwardpass(Fs_trans, cs_trans, Lstar, Hstar, HT, rstar, rT, Xs_trans, XT, Ustar)

    Kxs, ds = lqt_tf.lqt_canon_to_gen(Kxs_trans, ds_trans, Hstar, rstar, Ustar, Mstar, sstar)

    xs, us = lqt_tf.lqt_seq_forwardpass(x0_tf, Fstar, cstar, Lstar, Kxs, ds)

    us1, xs1 = condensing_tf.convertUX(us, xs, Lambda, cbar, Lbar, Fs.shape[0])

    return xs1, us1

def lqt_sequential_cond_bwfw_speedtest(lqt_gen, n_iter=10, device='/CPU:0', Nc=4):
    """ Speedtest for sequential backward and forward pass with partial condensing.

    Parameters:
        lqt_gen: LQT generator.
        n_iter: Number of iterations.
        device: Device to use.
        Nc: Number of condensing steps.

    Returns:
        T: List of time steps.
        D: List of state dimensions.
        run_times: List of run times.
    """
    print('Running lqt_sequential_cond_bwfw_speedtest(%d) on device %s' % (Nc,device))
    T = []
    D = []
    run_times = []
    for (x0_tf, Fs, cs, Ls, Hs, HT, rs, rT, Xs, XT, Us) in lqt_gen:
        with tf.device(device):
            _ = lqt_sequential_cond_bwfw(x0_tf, Fs, cs, Ls, Hs, HT, rs, rT, Xs, XT, Us, Nc) # Compilation run
            tic = time.time()
            for i in range(n_iter):
                _ = lqt_sequential_cond_bwfw(x0_tf, Fs, cs, Ls, Hs, HT, rs, rT, Xs, XT, Us, Nc)
            toc = time.time()
        T.append(Fs.shape[0])
        D.append(Fs.shape[1])
        run_times.append((toc - tic) / n_iter)
        print('T=%d, D=%d took %f ms' % (Fs.shape[0], Fs.shape[1], 1000.0 * run_times[-1]))
    return T, D, run_times

# Parallel 1 with partial condensing
@tf.function
def lqt_parallel_cond_bwfw_1(x0_tf, Fs, cs, Ls, Hs, HT, rs, rT, Xs, XT, Us, Nc):
    """ Parallel backward and forward pass with partial condensing for LQT.

    Parameters:
        x0_tf: Initial state.
        Fs, cs, Ls, Hs, HT, rs, rT, Xs, XT, Us: LQT parameters.
        Nc: Number of condensing steps.

    Returns:
        xs_tf_cond, us_tf_cond: State and control trajectories.
    """

    Fstar, cstar, Lstar, Hstar, rstar, Xstar, Ustar, Mstar, sstar, Lambda, cbar, Lbar \
        = condensing_tf.condense(Fs, cs, Ls, Hs, rs, Xs, Us, Nc)

    Fs_trans, cs_trans, Xs_trans \
        = lqt_tf.lqt_gen_to_canon(Fstar, cstar, Lstar, Hstar, rstar, Xstar, Ustar, Mstar, sstar)

    Ss, vs, Kxs_trans, ds_trans \
        = lqt_tf.lqt_par_backwardpass(Fs_trans, cs_trans, Lstar, Hstar, HT, rstar, rT, Xs_trans, XT, Ustar, max_parallel=Fs_trans.shape[0])

    Kxs, ds = lqt_tf.lqt_canon_to_gen(Kxs_trans, ds_trans, Hstar, rstar, Ustar, Mstar, sstar)

    xs, us = lqt_tf.lqt_par_forwardpass(x0_tf, Fstar, cstar, Lstar, Kxs, ds, max_parallel=Fs_trans.shape[0])

    us1, xs1 = condensing_tf.convertUX(us, xs, Lambda, cbar, Lbar, Fs.shape[0])

    return xs1, us1

def lqt_parallel_cond_bwfw_1_speedtest(lqt_gen, n_iter=10, device='/CPU:0', Nc=4):
    """ Speedtest for parallel backward and forward pass with partial condensing.

    Parameters:
        lqt_gen: LQT generator.
        n_iter: Number of iterations.
        device: Device to use.
        Nc: Number of condensing steps.

    Returns:
        T: List of time steps.
        D: List of state dimensions.
        run_times: List of run times.
    """

    print('Running lqt_parallel_cond_bwfw_1_speedtest(%d) on device %s' % (Nc,device))
    T = []
    D = []
    run_times = []
    for (x0_tf, Fs, cs, Ls, Hs, HT, rs, rT, Xs, XT, Us) in lqt_gen:
        with tf.device(device):
            _ = lqt_parallel_cond_bwfw_1(x0_tf, Fs, cs, Ls, Hs, HT, rs, rT, Xs, XT, Us, Nc) # Compilation run
            tic = time.time()
            for i in range(n_iter):
                _ = lqt_parallel_cond_bwfw_1(x0_tf, Fs, cs, Ls, Hs, HT, rs, rT, Xs, XT, Us, Nc)
            toc = time.time()
        T.append(Fs.shape[0])
        D.append(Fs.shape[1])
        run_times.append((toc - tic) / n_iter)
        print('T=%d, D=%d took %f ms' % (Fs.shape[0], Fs.shape[1], 1000.0 * run_times[-1]))
    return T, D, run_times

# Parallel 2 with partial condensing
@tf.function
def lqt_parallel_cond_bwfw_2(x0_tf, Fs, cs, Ls, Hs, HT, rs, rT, Xs, XT, Us, Nc):
    """ Parallel backward and forward pass (of forward-backward type) with partial condensing for LQT.

    Parameters:
        x0_tf: Initial state.
        Fs, cs, Ls, Hs, HT, rs, rT, Xs, XT, Us: LQT parameters.
        Nc: Number of condensing steps.

    Returns:
        xs_tf_cond, us_tf_cond: State and control trajectories.
    """
    Fstar, cstar, Lstar, Hstar, rstar, Xstar, Ustar, Mstar, sstar, Lambda, cbar, Lbar \
        = condensing_tf.condense(Fs, cs, Ls, Hs, rs, Xs, Us, Nc)

    Fs_trans, cs_trans, Xs_trans \
        = lqt_tf.lqt_gen_to_canon(Fstar, cstar, Lstar, Hstar, rstar, Xstar, Ustar, Mstar, sstar)

    Ss, vs, Kxs_trans, ds_trans \
        = lqt_tf.lqt_par_backwardpass(Fs_trans, cs_trans, Lstar, Hstar, HT, rstar, rT, Xs_trans, XT, Ustar, max_parallel=Fs_trans.shape[0])

    Kxs, ds = lqt_tf.lqt_canon_to_gen(Kxs_trans, ds_trans, Hstar, rstar, Ustar, Mstar, sstar)

    xs, us = lqt_tf.lqt_par_fwdbwdpass(x0_tf, Fs_trans, cs_trans, Lstar, Hstar, rstar, Xs_trans, Ustar, Ss, vs, Kxs, ds, max_parallel=Fs_trans.shape[0])

    us1, xs1 = condensing_tf.convertUX(us, xs, Lambda, cbar, Lbar, Fs.shape[0])

    return xs1, us1

def lqt_parallel_cond_bwfw_2_speedtest(lqt_gen, n_iter=10, device='/CPU:0', Nc=4):
    """ Speedtest for parallel backward and forward pass (of forward-backward type) with partial condensing.

    Parameters:
        lqt_gen: LQT generator.
        n_iter: Number of iterations.
        device: Device to use.
        Nc: Number of condensing steps.

    Returns:
        T: List of time steps.
        D: List of state dimensions.
        run_times: List of run times.
    """

    print('Running lqt_parallel_cond_bwfw_2_speedtest(%d) on device %s' % (Nc,device))
    T = []
    D = []
    run_times = []
    for (x0_tf, Fs, cs, Ls, Hs, HT, rs, rT, Xs, XT, Us) in lqt_gen:
        with tf.device(device):
            _ = lqt_parallel_cond_bwfw_2(x0_tf, Fs, cs, Ls, Hs, HT, rs, rT, Xs, XT, Us, Nc) # Compilation run
            tic = time.time()
            for i in range(n_iter):
                _ = lqt_parallel_cond_bwfw_2(x0_tf, Fs, cs, Ls, Hs, HT, rs, rT, Xs, XT, Us, Nc)
            toc = time.time()
        T.append(Fs.shape[0])
        D.append(Fs.shape[1])
        run_times.append((toc - tic) / n_iter)
        print('T=%d, D=%d took %f ms' % (Fs.shape[0], Fs.shape[1], 1000.0 * run_times[-1]))
    return T, D, run_times


###############################################################################################
# Tracking model
###############################################################################################

def tracking_generator(start=2, stop=5, num=20, dtype=tf.float64):
    """ Generator for tracking model over dataset size.

    Parameters:
        start: Start of logspace for data length.
        stop: End of logspace for data length.
        num: Number of data lengths.
        dtype: Data type.

    Returns:
        Generator for initial state and LQT parameters (x0_tf, Fs, cs, Ls, Hs, HT, rs, rT, Xs, XT, Us).
    """
    T_list = np.logspace(start, stop, num=num, base=10).astype(int)
    print('Tracking T_list = %s' % str(T_list))

    for k in range(T_list.shape[0]):
        model = linear_model_np.LinearModel()
        xy = model.genData(T_list[k] // 10)
        lqt, x0 = model.getLQT(xy)

        Fs, cs, Ls, Hs, HT, rs, rT, Xs, XT, Us = lqt_tf.lqt_np_to_tf(lqt, dtype=dtype)
        x0_tf  = tf.convert_to_tensor(x0, dtype=dtype)

        yield (x0_tf, Fs, cs, Ls, Hs, HT, rs, rT, Xs, XT, Us)

###############################################################################################
# Mass model
###############################################################################################

def mass_generator(N, start=2, stop=5, num=20, dtype=tf.float64):
    """ Generator for mass model over dataset size.

    Parameters:
        N: Number of masses.
        start: Start of logspace for data length.
        stop: End of logspace for data length.
        num: Number of data lengths.
        dtype: Data type.

    Returns:
        Generator for initial state and LQT parameters (x0_tf, Fs, cs, Ls, Hs, HT, rs, rT, Xs, XT, Us).
    """
    T_list = np.logspace(start, stop, num=num, base=10).astype(int)
    print('Mass(N=%d) T_list = %s' % (N,str(T_list)))

    for k in range(T_list.shape[0]):
        model = mass_model_np.MassModel(N)
        Tf = 10.0
        dt = Tf / T_list[k]
        lqt, x0 = model.getLQT(dt=dt, Tf=Tf)

        Fs, cs, Ls, Hs, HT, rs, rT, Xs, XT, Us = lqt_tf.lqt_np_to_tf(lqt, dtype=dtype)
        x0_tf  = tf.convert_to_tensor(x0, dtype=dtype)

        yield (x0_tf, Fs, cs, Ls, Hs, HT, rs, rT, Xs, XT, Us)

def mass_generator_2(T, nstart=2, nend=5, nstep=1, dtype=tf.float64):
    """ Generator for mass model over state size.

    Parameters:
        T: Number of time steps.
        nstart: Start of range for number of masses.
        nend: End of range for number of masses.
        nstep: Step size for number of masses.
        dtype: Data type.

    Returns:
        Generator for initial state and LQT parameters (x0_tf, Fs, cs, Ls, Hs, HT, rs, rT, Xs, XT, Us).
    """
    N_list = np.arange(nstart, nend, nstep, dtype=int)
    print('Mass(T=%d) N_list = %s' % (T,str(N_list)))

    for k in range(N_list.shape[0]):
        model = mass_model_np.MassModel(N_list[k])
        Tf = 10.0
        dt = Tf / T
        lqt, x0 = model.getLQT(dt=dt, Tf=Tf)

        Fs, cs, Ls, Hs, HT, rs, rT, Xs, XT, Us = lqt_tf.lqt_np_to_tf(lqt, dtype=dtype)
        x0_tf  = tf.convert_to_tensor(x0, dtype=dtype)

        yield (x0_tf, Fs, cs, Ls, Hs, HT, rs, rT, Xs, XT, Us)
