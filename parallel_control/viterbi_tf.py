"""
TensorFlow-based Viterbi algorithm implementation via its associated optimal control problem along with direct ones.

@author: Simo Särkkä
"""

import tensorflow as tf
import tensorflow_probability as tfp
import math

import parallel_control.viterbi_np as viterbi_np
import parallel_control.fsc_tf as fsc_tf

import time

# Abbreviations of tensorflow routines
mm = tf.linalg.matmul
mv = tf.linalg.matvec
top = tf.linalg.matrix_transpose

###########################################################################
# Utilities for creating a FSC and extracting its result
###########################################################################

def make_viterbi_fsc(model, y_list, dtype=tf.float64):
    """ Create a FSC for the Viterbi algorithm. This uses the numpy routines,
    don't speedtest this. Arguments should be in numpy format.

    Parameters:
        model: HMM model (such as GEModel).
        y_list: Observation sequence.
        dtype: Data type for the FSC.

    Returns:
        fs: Tensor of state transition matrices.
        Ls: Tensor of cost matrices.
        LT: Final cost vector.
    """

    viterbi = viterbi_np.Viterbi_np(model.prior, model.Pi, model.Po)
    fsc = viterbi.getFSC(y_list)

    fs, Ls, LT = fsc_tf.fsc_np_to_tf(fsc, dtype=dtype)

    return fs, Ls, LT

def get_viterbi_from_fsc(min_xs, Vs):
    """ Extract the Viterbi result from the FSC result. This can be called within a speed test.

    Parameters:
        min_xs: State sequence.
        Vs: Value functions from FSC.

    Returns:
        v_map: MAP state sequence.
        Vs_vit: Viterbi value functions.
    """
    v_map = tf.reverse(min_xs, axis=[0])
    Vs_vit = tf.reverse(Vs, axis=[0])

    return v_map, Vs_vit

###########################################################################
# The actual Viterbi routines. Call
#    fs, Ls, LT = make_viterbi_fsc()
# and then one of the following
###########################################################################

@tf.function
def viterbi_fsc_seq_bwfw(fs, Ls, LT):
    """ Viterbi algorithm using sequential FSC backward and forward passes.

    Parameters:
        fs: Tensor of state transition matrices.
        Ls: Tensor of cost matrices.
        LT: Final cost vector.

    Returns:
        v_map: MAP state sequence.
        Vs_vit: Viterbi value functions.
    """
    us, Vs = fsc_tf.fsc_seq_backwardpass(fs, Ls, LT)
    x0 = tf.argmin(Vs[0], axis=-1, output_type=fs.dtype)
    min_xs_tf_par, min_us_tf_par = fsc_tf.fsc_seq_forwardpass(x0, fs, us)
    return get_viterbi_from_fsc(min_xs_tf_par, Vs)

@tf.function
def viterbi_fsc_par_bwfw(fs, Ls, LT):
    """ Viterbi algorithm using parallel FSC backward and forward passes.

    Parameters:
        fs: Tensor of state transition matrices.
        Ls: Tensor of cost matrices.
        LT: Final cost vector.

    Returns:
        v_map: MAP state sequence.
        Vs_vit: Viterbi value functions.
    """
    us, Vs = fsc_tf.fsc_par_backwardpass(fs, Ls, LT, max_parallel=fs.shape[0])
    x0 = tf.argmin(Vs[0], axis=-1, output_type=fs.dtype)
    min_us_tf_par, min_xs_tf_par = fsc_tf.fsc_par_forwardpass(x0, fs, us, max_parallel=fs.shape[0])
    return get_viterbi_from_fsc(min_xs_tf_par, Vs)

@tf.function #(reduce_retracing=True)
def viterbi_fsc_par_bwfwbw(fs, Ls, LT):
    """ Viterbi algorithm using parallel FSC backward and forward-backward passes.

    Parameters:
        fs: Tensor of state transition matrices.
        Ls: Tensor of cost matrices.
        LT: Final cost vector.

    Returns:
        v_map: MAP state sequence.
        Vs_vit: Viterbi value functions.
    """

    us, Vs = fsc_tf.fsc_par_backwardpass(fs, Ls, LT, max_parallel=fs.shape[0])
    x0 = tf.argmin(Vs[0], axis=-1, output_type=fs.dtype)
    min_us_tf_par, min_xs_tf_par = fsc_tf.fsc_par_fwdbwdpass(x0, fs, Ls, us, Vs, max_parallel=fs.shape[0])
    return get_viterbi_from_fsc(min_xs_tf_par, Vs)

###########################################################################
# Speed tests for the above
###########################################################################

def viterbi_fsc_seq_bwfw_speedtest(model, steps, n_iter=10, device='/CPU:0'):
    """ Speedtest for Viterbi algorithm using sequential FSC backward and forward passes.

    Parameters:
        model: HMM model (such as GEModel).
        steps: Length of observation sequence.
        n_iter: Number of iterations to run.
        device: Device to run on.

    Returns:
        elapsed: Time per iteration.
        err: Error in the result.
    """

    x_list, y_list = model.genData(steps)
    fs, Ls, LT = make_viterbi_fsc(model, y_list)

    print('Running viterbi_fsc_seq_bwfw_speedtest on device %s' % device)
    with tf.device(device):
        v_map, Vs = viterbi_fsc_seq_bwfw(fs, Ls, LT)  # Compilation run
        tic = time.time()
        for i in range(n_iter):
            v_map, Vs = viterbi_fsc_seq_bwfw(fs, Ls, LT)
        toc = time.time()
        elapsed = (toc - tic) / n_iter

    viterbi = viterbi_np.Viterbi_np(model.prior, model.Pi, model.Po)
    ref_v_map, ref_V_list = viterbi.seqViterbi(y_list)
    err = tf.math.reduce_max(tf.math.abs(tf.convert_to_tensor(ref_V_list, dtype=Vs.dtype) - Vs))

    print('steps=%d took %f ms.' % (steps, 1000.0 * elapsed))
    return elapsed, err


def viterbi_fsc_par_bwfw_speedtest(model, steps, n_iter=10, device='/CPU:0'):
    """ Speedtest for Viterbi algorithm using parallel FSC backward and forward passes.

    Parameters:
        model: HMM model (such as GEModel).
        steps: Length of observation sequence.
        n_iter: Number of iterations to run.
        device: Device to run on.

    Returns:
        elapsed: Time per iteration.
        err: Error in the result.
    """

    x_list, y_list = model.genData(steps)
    fs, Ls, LT = make_viterbi_fsc(model, y_list)

    print('Running viterbi_fsc_par_bwfw_speedtest on device %s' % device)
    with tf.device(device):
        v_map, Vs = viterbi_fsc_par_bwfw(fs, Ls, LT)  # Compilation run
        tic = time.time()
        for i in range(n_iter):
            v_map, Vs = viterbi_fsc_par_bwfw(fs, Ls, LT)
        toc = time.time()
        elapsed = (toc - tic) / n_iter

    viterbi = viterbi_np.Viterbi_np(model.prior, model.Pi, model.Po)
    ref_v_map, ref_V_list = viterbi.parBwdFwdViterbi(y_list)
    err = tf.math.reduce_max(tf.math.abs(tf.convert_to_tensor(ref_V_list, dtype=Vs.dtype) - Vs))

    print('steps=%d took %f ms.' % (steps, 1000.0 * elapsed))
    return elapsed, err


def viterbi_fsc_par_bwfwbw_speedtest(model, steps, n_iter=10, device='/CPU:0'):
    """ Speedtest for Viterbi algorithm using parallel FSC backward and forward-backward passes.

    Parameters:
        model: HMM model (such as GEModel).
        steps: Length of observation sequence.
        n_iter: Number of iterations to run.
        device: Device to run on.

    Returns:
        elapsed: Time per iteration.
        err: Error in the result.
    """

    x_list, y_list = model.genData(steps)
    fs, Ls, LT = make_viterbi_fsc(model, y_list)

    print('Running viterbi_fsc_par_bwfwbw_speedtest on device %s' % device)
    with tf.device(device):
        v_map, Vs = viterbi_fsc_par_bwfwbw(fs, Ls, LT)  # Compilation run
        tic = time.time()
        for i in range(n_iter):
            v_map, Vs = viterbi_fsc_par_bwfwbw(fs, Ls, LT)
        toc = time.time()
        elapsed = (toc - tic) / n_iter

    viterbi = viterbi_np.Viterbi_np(model.prior, model.Pi, model.Po)
    ref_v_map, ref_V_list = viterbi.parBwdFwdBwdViterbi(y_list)
    err = tf.math.reduce_max(tf.math.abs(tf.convert_to_tensor(ref_V_list, dtype=Vs.dtype) - Vs))

    print('steps=%d took %f ms.' % (steps, 1000.0 * elapsed))
    return elapsed, err

###########################################################################
# Independent optimized implementation of the Viterbi
###########################################################################

def viterbi_seq_forwardpass(prior, Pi, Po, ys):
    """ Sequential forward pass of the Viterbi algorithm.

    Parameters:
        prior: Prior distribution over states.
        Pi: Transition matrix.
        Po: Emission matrix.
        ys: Observation sequence.

    Returns:
        us: Lookup tables for optimal state sequence.
        Vs: Viterbi value functions.
    """
    nl_prior = -tf.math.log(prior)
    nl_Pi = -tf.math.log(Pi)
    nl_Po = -tf.math.log(Po)

    def body(carry, inputs):
        y = inputs
        V, _ = carry

        psi = tf.expand_dims(nl_Po[:, y], 0) + nl_Pi
        temp = psi + tf.expand_dims(V, -1)

        u = tf.argmin(temp, axis=0, output_type=ys.dtype)
        V = tf.reduce_min(temp, axis=0)
        return V, u

    u0 = tf.zeros(Pi.shape[0], dtype=ys.dtype) # Dummy
    V0 = nl_prior

    Vs, us = tf.scan(body, ys,
                     initializer=(V0, u0),
                     reverse=False)

    Vs = tf.concat([tf.expand_dims(V0,0), Vs], axis=0)

    return us, Vs


def viterbi_seq_backwardpass(us, VT):
    """ Sequential backward pass of the Viterbi algorithm.

    Parameters:
        us: Lookup tables for optimal state sequence.
        VT: Final Viterbi value function.

    Returns:
        xs: Optimal state sequence.
    """
    def body(carry, inputs):
        u = inputs
        x = carry
        x = u[x]
        return x

    x = tf.argmin(VT, output_type=us.dtype)
    xs = tf.scan(body, us,
                 initializer=x,
                 reverse=True)

    return tf.concat([xs, tf.expand_dims(x,0)], axis=0)

@tf.function
def viterbi_seq_fwbw(prior, Pi, Po, ys):
    """ Perform sequential forward and backward passes of the Viterbi algorithm.

    Parameters:
        prior: Prior distribution over states.
        Pi: Transition matrix.
        Po: Emission matrix.
        ys: Observation sequence.

    Returns:
        xs: Optimal state sequence.
        Vs: Viterbi value functions.
    """
    us, Vs = viterbi_seq_forwardpass(prior, Pi, Po, ys)
    xs = viterbi_seq_backwardpass(us, Vs[-1, :])
    return xs, Vs


def viterbi_par_forwardpass_init(prior, Pi, Po, ys):
    """ Initialize forward pass of the parallel Viterbi algorithm.

    Parameters:
        prior: Prior distribution over states.
        Pi: Transition matrix.
        Po: Emission matrix.
        ys: Observation sequence.

    Returns:
        Vs: Conditional Viterbi value function elements.
    """
    V0 = tf.tile(tf.expand_dims(-tf.math.log(prior),-2), (prior.shape[0], 1))
    Vs = tf.expand_dims(-tf.math.log(Pi), 0) + tf.expand_dims(tf.gather(-tf.math.log(top(Po)), ys), -2)
    return tf.concat([tf.expand_dims(V0,0), Vs], axis=0)


def viterbi_par_comb_V(Vij, Vjk):
    """ Combine two conditional Viterbi value function elements.

    Parameters:
        Vij: Conditional Viterbi value function element for i -> j.
        Vjk: Conditional Viterbi value function element for j -> k.

    Returns:
        Vik: Conditional Viterbi value function element for i -> k.
    """
    Vik = tf.reduce_min(tf.expand_dims(Vij,-1) + tf.expand_dims(Vjk,1), axis=2)
    return Vik


def viterbi_par_comb_V_rev(Vjk, Vij):
    """ Combine two conditional Viterbi value function elements in reverse order.

    Parameters:
        Vjk: Conditional Viterbi value function element for j -> k.
        Vij: Conditional Viterbi value function element for i -> j.

    Returns:
        Vik: Conditional Viterbi value function element for i -> k.
    """
    return viterbi_par_comb_V(Vij, Vjk)


def viterbi_par_forwardpass(prior, Pi, Po, ys, max_parallel=10000):
    """ Perform forward pass of the parallel Viterbi algorithm.

    Parameters:
        prior: Prior distribution over states.
        Pi: Transition matrix.
        Po: Emission matrix.
        ys: Observation sequence.
        max_parallel: Maximum number of parallel operations for tfp.math.scan_associative.

    Returns:
        us: Lookup tables for optimal state sequence.
        Vs: Viterbi value functions.
    """
    elems1 = viterbi_par_forwardpass_init(prior, Pi, Po, ys)
    elems2 = tfp.math.scan_associative(viterbi_par_comb_V,
                                      elems1,
                                      max_num_levels=math.ceil(math.log2(max_parallel)))
    Ls = elems1[1:]
    Vs = elems2[:, 0, :]

    Vus = Ls + tf.expand_dims(Vs[:-1],-1)
    us = tf.argmin(Vus, axis=-2, output_type=ys.dtype)

    return us, Vs


def viterbi_par_backwardpass_init(us, VT):
    """ Initialize backward pass of the parallel Viterbi algorithm.

    Parameters:
        us: Lookup tables for optimal state sequence.
        VT: Final Viterbi value function.

    Returns:
        elems: Elements for parallel backward pass.
    """
    xT = tf.argmin(VT, output_type=us.dtype)
    return tf.concat([us, tf.expand_dims(tf.repeat(xT, VT.shape[0]),0)], axis=0)


def viterbi_par_comb_f(fij, fjk):
    """ Combine two backward functions in parallel Viterbi.

    Parameters:
        fij: Backward function for i -> j.
        fjk: Backward function for j -> k.

    Returns:
        fik: Backward function for i -> k
    """
    fik = tf.gather(fjk, fij, axis=-1, batch_dims=1)
    return fik


def viterbi_par_backwardpass(us, VT, max_parallel=10000):
    """ Perform backward pass of the parallel Viterbi algorithm.

    Parameters:
        us: Lookup tables for optimal state sequence.
        VT: Final Viterbi value function.
        max_parallel: Maximum number of parallel operations for tfp.math.scan_associative.

    Returns:
        xs: Optimal state sequence.
    """
    elems = viterbi_par_backwardpass_init(us, VT)

    rev_elems = tf.reverse(elems, axis=[0])

    rev_elems = tfp.math.scan_associative(viterbi_par_comb_f,
                                          rev_elems,
                                          max_num_levels=math.ceil(math.log2(max_parallel)))

    xs = tf.reverse(rev_elems, axis=[0])[..., 0]

    return xs


@tf.function
def viterbi_par_fwbw(prior, Pi, Po, ys):
    """ Perform forward and backward passes of the parallel Viterbi algorithm.

    Parameters:
        prior: Prior distribution over states.
        Pi: Transition matrix.
        Po: Emission matrix.
        ys: Observation sequence.

    Returns:
        xs: Optimal state sequence.
        Vs: Viterbi value functions.
    """
    us, Vs = viterbi_par_forwardpass(prior, Pi, Po, ys, max_parallel=ys.shape[0])
    xs = viterbi_par_backwardpass(us, Vs[-1, :], max_parallel=ys.shape[0])
    return xs, Vs


@tf.function
def viterbi_seqpar_fwbw(prior, Pi, Po, ys):
    """ Perform forward and backward passes of the sequential/parallel Viterbi algorithm.

    Parameters:
        prior: Prior distribution over states.
        Pi: Transition matrix.
        Po: Emission matrix.
        ys: Observation sequence.

    Returns:
        xs: Optimal state sequence.
        Vs: Viterbi value functions.
    """
    us, Vs = viterbi_seq_forwardpass(prior, Pi, Po, ys)
    xs = viterbi_par_backwardpass(us, Vs[-1, :], max_parallel=ys.shape[0])
    return xs, Vs


def viterbi_par_bwfwpass_init(prior, Pi, Po, ys):
    """ Initialize backward pass of backward-forward parallel Viterbi algorithm.

    Parameters:
        prior: Prior distribution over states.
        Pi: Transition matrix.
        Po: Emission matrix.
        ys: Observation sequence.

    Returns:
        elems: Elements for parallel backward pass.
    """
    VT = tf.zeros((prior.shape[0], prior.shape[0]), dtype=prior.dtype)
    Vs = tf.expand_dims(-tf.math.log(Pi), 0) + tf.expand_dims(tf.gather(-tf.math.log(top(Po)), ys), -2)
    return tf.concat([Vs, tf.expand_dims(VT,0)], axis=0)


def viterbi_par_bwfwpass(prior, Pi, Po, ys, Vs, max_parallel=10000):
    """ Perform backward pass of backward-forward parallel Viterbi algorithm.

    Parameters:
        prior: Prior distribution over states.
        Pi: Transition matrix.
        Po: Emission matrix.
        ys: Observation sequence.
        Vs: Forward Viterbi value functions.
        max_parallel: Maximum number of parallel operations for tfp.math.scan_associative.

    Returns:
        xs: Optimal state sequence.
        Vfs: Backward Viterbi value functions.
    """

    elems = viterbi_par_bwfwpass_init(prior, Pi, Po, ys)

    rev_elems = tf.reverse(elems, axis=[0])

    rev_elems = tfp.math.scan_associative(viterbi_par_comb_V_rev,
                                          rev_elems,
                                          max_num_levels=math.ceil(math.log2(max_parallel)))

    elems = tf.reverse(rev_elems, axis=[0])

    Vfs = elems[..., 0]
    xs = tf.argmin(Vfs + Vs, axis=-1, output_type=ys.dtype)

    return xs, Vfs


@tf.function
def viterbi_par_fwbwfw(prior, Pi, Po, ys):
    """ Perform forward and backward passes of the backward-forward parallel Viterbi algorithm.

    Parameters:
        prior: Prior distribution over states.
        Pi: Transition matrix.
        Po: Emission matrix.
        ys: Observation sequence.

    Returns:
        v_map: Optimal state sequence.
        Vs: Viterbi value functions.
    """
    us, Vs = viterbi_par_forwardpass(prior, Pi, Po, ys, max_parallel=ys.shape[0])
    v_map, Vfs = viterbi_par_bwfwpass(prior, Pi, Po, ys, Vs, max_parallel=ys.shape[0])
    return v_map, Vs


###########################################################################
# Speed tests for the above
###########################################################################

def viterbi_seq_fwbw_speedtest(model, steps, n_iter=10, device='/CPU:0'):
    """ Speed test for sequential Viterbi algorithm.

    Parameters:
        model: HMM model.
        steps: Number of steps to generate.
        n_iter: Number of iterations to run.
        device: Device to run on.

    Returns:
        elapsed: Elapsed time.
        err: Error in the result.
    """
    x_list, y_list = model.genData(steps)
    ys = tf.convert_to_tensor(y_list, dtype=tf.int32)
    prior = tf.convert_to_tensor(model.prior, dtype=tf.float64)
    Pi = tf.convert_to_tensor(model.Pi, dtype=tf.float64)
    Po = tf.convert_to_tensor(model.Po, dtype=tf.float64)

    print('Running viterbi_seq_fwbw_speedtest on device %s' % device)
    with tf.device(device):
        v_map, Vs = viterbi_seq_fwbw(prior, Pi, Po, ys)  # Compilation run
        tic = time.time()
        for i in range(n_iter):
            v_map, Vs = viterbi_seq_fwbw(prior, Pi, Po, ys)
        toc = time.time()
        elapsed = (toc - tic) / n_iter

    viterbi = viterbi_np.Viterbi_np(model.prior, model.Pi, model.Po)
    ref_v_map, ref_V_list = viterbi.seqViterbi(y_list)
    err = tf.math.reduce_max(tf.math.abs(tf.convert_to_tensor(ref_V_list, dtype=Vs.dtype) - Vs))

    print('steps=%d took %f ms (err = %f).' % (steps, 1000.0 * elapsed, err))
    return elapsed, err


def viterbi_par_fwbw_speedtest(model, steps, n_iter=10, device='/CPU:0'):
    """ Speed test for parallel Viterbi algorithm (with fw and bw passes).

    Parameters:
        model: HMM model.
        steps: Number of steps to generate.
        n_iter: Number of iterations to run.
        device: Device to run on.

    Returns:
        elapsed: Elapsed time.
        err: Error in the result.
    """

    x_list, y_list = model.genData(steps)
    ys = tf.convert_to_tensor(y_list, dtype=tf.int32)
    prior = tf.convert_to_tensor(model.prior, dtype=tf.float64)
    Pi = tf.convert_to_tensor(model.Pi, dtype=tf.float64)
    Po = tf.convert_to_tensor(model.Po, dtype=tf.float64)

    print('Running viterbi_par_fwbw_speedtest on device %s' % device)
    with tf.device(device):
        v_map, Vs = viterbi_par_fwbw(prior, Pi, Po, ys)  # Compilation run
        tic = time.time()
        for i in range(n_iter):
            v_map, Vs = viterbi_par_fwbw(prior, Pi, Po, ys)
        toc = time.time()
        elapsed = (toc - tic) / n_iter

    viterbi = viterbi_np.Viterbi_np(model.prior, model.Pi, model.Po)
    ref_v_map, ref_V_list = viterbi.seqViterbi(y_list)
    err = tf.math.reduce_max(tf.math.abs(tf.convert_to_tensor(ref_V_list, dtype=Vs.dtype) - Vs))

    print('steps=%d took %f ms (err = %f).' % (steps, 1000.0 * elapsed, err))
    return elapsed, err


def viterbi_seqpar_fwbw_speedtest(model, steps, n_iter=10, device='/CPU:0'):
    """ Speed test for sequential-parallel Viterbi algorithm.

    Parameters:
        model: HMM model.
        steps: Number of steps to generate.
        n_iter: Number of iterations to run.
        device: Device to run on.

    Returns:
        elapsed: Elapsed time.
        err: Error in the result.
    """
    x_list, y_list = model.genData(steps)
    ys = tf.convert_to_tensor(y_list, dtype=tf.int32)
    prior = tf.convert_to_tensor(model.prior, dtype=tf.float64)
    Pi = tf.convert_to_tensor(model.Pi, dtype=tf.float64)
    Po = tf.convert_to_tensor(model.Po, dtype=tf.float64)

    print('Running viterbi_seqpar_fwbw_speedtest on device %s' % device)
    with tf.device(device):
        v_map, Vs = viterbi_seqpar_fwbw(prior, Pi, Po, ys)  # Compilation run
        tic = time.time()
        for i in range(n_iter):
            v_map, Vs = viterbi_seqpar_fwbw(prior, Pi, Po, ys)
        toc = time.time()
        elapsed = (toc - tic) / n_iter

    viterbi = viterbi_np.Viterbi_np(model.prior, model.Pi, model.Po)
    ref_v_map, ref_V_list = viterbi.seqViterbi(y_list)
    err = tf.math.reduce_max(tf.math.abs(tf.convert_to_tensor(ref_V_list, dtype=Vs.dtype) - Vs))

    print('steps=%d took %f ms (err = %f).' % (steps, 1000.0 * elapsed, err))
    return elapsed, err


def viterbi_par_fwbwfw_speedtest(model, steps, n_iter=10, device='/CPU:0'):
    """ Speed test for parallel Viterbi algorithm (with fw, bw-fw passes).

    Parameters:
        model: HMM model.
        steps: Number of steps to generate.
        n_iter: Number of iterations to run.
        device: Device to run on.

    Returns:
        elapsed: Elapsed time.
        err: Error in the result.
    """
    x_list, y_list = model.genData(steps)
    ys = tf.convert_to_tensor(y_list, dtype=tf.int32)
    prior = tf.convert_to_tensor(model.prior, dtype=tf.float64)
    Pi = tf.convert_to_tensor(model.Pi, dtype=tf.float64)
    Po = tf.convert_to_tensor(model.Po, dtype=tf.float64)

    print('Running viterbi_par_fwbwfw_speedtest on device %s' % device)
    with tf.device(device):
        v_map, Vs = viterbi_par_fwbwfw(prior, Pi, Po, ys)  # Compilation run
        tic = time.time()
        for i in range(n_iter):
            v_map, Vs = viterbi_par_fwbwfw(prior, Pi, Po, ys)
        toc = time.time()
        elapsed = (toc - tic) / n_iter

    viterbi = viterbi_np.Viterbi_np(model.prior, model.Pi, model.Po)
    ref_v_map, ref_V_list = viterbi.seqViterbi(y_list)
    err = tf.math.reduce_max(tf.math.abs(tf.convert_to_tensor(ref_V_list, dtype=Vs.dtype) - Vs))

    print('steps=%d took %f ms (err = %f).' % (steps, 1000.0 * elapsed, err))
    return elapsed, err
