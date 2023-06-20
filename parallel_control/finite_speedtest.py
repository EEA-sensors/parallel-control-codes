"""
Finite model speedtest routines.

@author: Simo Särkkä
"""

import numpy as np
import math
import parallel_control.finite_model_np as finite_model_np
import matplotlib.pyplot as plt
import tensorflow as tf
import parallel_control.fsc_np as fsc_np
import parallel_control.fsc_tf as fsc_tf
import time

###############################################################################################
#
# General finite model speedtests
#
###############################################################################################

@tf.function
def fsc_seq_bw(fs, Ls, LT):
    """ Run sequential backward pass for a finite-state controller.
    """
    us, Vs = fsc_tf.fsc_seq_backwardpass(fs, Ls, LT)
    return us, Vs

def fsc_seq_bw_speedtest(fsc_gen, n_iter=10, device='/CPU:0'):
    print('Running fsc_seq_bw_speedtest on device %s' % device)
    T = []
    D = []
    run_times = []
    for (x0, fs, Ls, LT) in fsc_gen:
        with tf.device(device):
            _ = fsc_seq_bw(fs, Ls, LT) # Compilation run
            tic = time.time()
            for i in range(n_iter):
                _ = fsc_seq_bw(fs, Ls, LT)
            toc = time.time()
        T.append(fs.shape[0])
        D.append(LT.shape[0])
        run_times.append((toc - tic) / n_iter)
        print('T=%d, D=%d took %f ms' % (fs.shape[0], LT.shape[0], 1000.0 * run_times[-1]))
    return T, D, run_times


@tf.function
def fsc_par_bw(fs, Ls, LT):
    us, Vs = fsc_tf.fsc_par_backwardpass(fs, Ls, LT, max_parallel=fs.shape[0])
    return us, Vs

def fsc_par_bw_speedtest(fsc_gen, n_iter=10, device='/CPU:0'):
    print('Running fsc_par_bw_speedtest on device %s' % device)
    T = []
    D = []
    run_times = []
    for (x0, fs, Ls, LT) in fsc_gen:
        with tf.device(device):
            _ = fsc_par_bw(fs, Ls, LT) # Compilation run
            tic = time.time()
            for i in range(n_iter):
                _ = fsc_par_bw(fs, Ls, LT)
            toc = time.time()
        T.append(fs.shape[0])
        D.append(LT.shape[0])
        run_times.append((toc - tic) / n_iter)
        print('T=%d, D=%d took %f ms' % (fs.shape[0], LT.shape[0], 1000.0 * run_times[-1]))
    return T, D, run_times


@tf.function
def fsc_seq_bwfw(x0, fs, Ls, LT):
    us, Vs = fsc_tf.fsc_seq_backwardpass(fs, Ls, LT)
    min_us_tf_par, min_xs_tf_par = fsc_tf.fsc_seq_forwardpass(x0, fs, us)  # XXX: Outputs the wrong way
    return min_us_tf_par, min_xs_tf_par

def fsc_seq_bwfw_speedtest(fsc_gen, n_iter=10, device='/CPU:0'):
    print('Running fsc_seq_bwfw_speedtest on device %s' % device)
    T = []
    D = []
    run_times = []
    for (x0, fs, Ls, LT) in fsc_gen:
        with tf.device(device):
            _ = fsc_seq_bwfw(x0, fs, Ls, LT) # Compilation run
            tic = time.time()
            for i in range(n_iter):
                _ = fsc_seq_bwfw(x0, fs, Ls, LT)
            toc = time.time()
        T.append(fs.shape[0])
        D.append(LT.shape[0])
        run_times.append((toc - tic) / n_iter)
        print('T=%d, D=%d took %f ms' % (fs.shape[0], LT.shape[0], 1000.0 * run_times[-1]))
    return T, D, run_times


@tf.function
def fsc_par_bwfw_1(x0, fs, Ls, LT):
    us, Vs = fsc_tf.fsc_par_backwardpass(fs, Ls, LT, max_parallel=fs.shape[0])
    min_us_tf_par, min_xs_tf_par = fsc_tf.fsc_par_forwardpass(x0, fs, us, max_parallel=fs.shape[0])
    return min_us_tf_par, min_xs_tf_par

def fsc_par_bwfw_1_speedtest(fsc_gen, n_iter=10, device='/CPU:0'):
    print('Running fsc_par_bwfw_1_speedtest on device %s' % device)
    T = []
    D = []
    run_times = []
    for (x0, fs, Ls, LT) in fsc_gen:
        with tf.device(device):
            _ = fsc_par_bwfw_1(x0, fs, Ls, LT) # Compilation run
            tic = time.time()
            for i in range(n_iter):
                _ = fsc_par_bwfw_1(x0, fs, Ls, LT)
            toc = time.time()
        T.append(fs.shape[0])
        D.append(LT.shape[0])
        run_times.append((toc - tic) / n_iter)
        print('T=%d, D=%d took %f ms' % (fs.shape[0], LT.shape[0], 1000.0 * run_times[-1]))
    return T, D, run_times


@tf.function
def fsc_par_bwfw_2(x0, fs, Ls, LT):
    us, Vs = fsc_tf.fsc_par_backwardpass(fs, Ls, LT, max_parallel=fs.shape[0])
    min_us_tf_par, min_xs_tf_par = fsc_tf.fsc_par_fwdbwdpass(x0, fs, Ls, us, Vs, max_parallel=fs.shape[0])
    return min_us_tf_par, min_xs_tf_par

def fsc_par_bwfw_2_speedtest(fsc_gen, n_iter=10, device='/CPU:0'):
    print('Running fsc_par_bwfw_2_speedtest on device %s' % device)
    T = []
    D = []
    run_times = []
    for (x0, fs, Ls, LT) in fsc_gen:
        with tf.device(device):
            _ = fsc_par_bwfw_2(x0, fs, Ls, LT) # Compilation run
            tic = time.time()
            for i in range(n_iter):
                _ = fsc_par_bwfw_2(x0, fs, Ls, LT)
            toc = time.time()
        T.append(fs.shape[0])
        D.append(LT.shape[0])
        run_times.append((toc - tic) / n_iter)
        print('T=%d, D=%d took %f ms' % (fs.shape[0], LT.shape[0], 1000.0 * run_times[-1]))
    return T, D, run_times

###############################################################################################
# Finite model
###############################################################################################

def finite_generator(nx=21, start=2, stop=5, num=20, dtype=tf.float32):
    T_list = np.logspace(start, stop, num=num, base=10).astype(int)
    print('Finite T_list = %s' % str(T_list))

    for k in range(T_list.shape[0]):
        model = finite_model_np.FiniteModel()
        track, x0 = model.genData(T_list[k], nx=nx)
        fsc = model.getFSC(track)
        fs, Ls, LT = fsc_tf.fsc_np_to_tf(fsc, dtype=dtype)

        yield (x0, fs, Ls, LT)

