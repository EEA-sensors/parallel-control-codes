"""
Tensorflow-based Nonlinear (iterated) Linear Quadratic Tracker.

@author: Simo Särkkä
"""

import numpy as np
import tensorflow as tf

import parallel_control.lqt_np as lqt_np
import parallel_control.lqt_tf as lqt_tf
#import parallel_control.nlqt_np as nlqt_np

# Abbreviations for tensorflow
mm = tf.linalg.matmul
mv = tf.linalg.matvec

##############################################################################
#
# Nonlinear LQT
#
##############################################################################

def nlqt_linearize(us, xs, f, Fx, Fu):
    """ Linearize a nonlinear model around a trajectory.

    Parameters:
        us: Tensor of control inputs.
        xs: Tensor of states.
        f: Nonlinear state transition function.
        Fx: Jacobian of f w.r.t. x.
        Fu: Jacobian of f w.r.t. u.

    Returns:
        Fs: Tensor of linearized state transition matrices.
        cs: Tensor of linearized state offsets.
        Ls: Tensor of linearized control gain matrices.
    """
    fs  = f(xs, us)
    Fxs = Fx(xs, us)
    Fus = Fu(xs, us)

    # f(xp,up) = f(x,u) + Fx(x,u) (xp - x) + Fu(x,u) (up - u)
    Fs = Fxs
    Ls = Fus
    cs = fs - mv(Fxs, xs) - mv(Fus, us)

    return Fs, cs, Ls


def nlqt_simulate(x0, us, f):
    """ Simulate a nonlinear model. Note that this is not done in parallel.

    Parameters:
        x0: Initial state.
        us: Tensor of control inputs.
        f: Nonlinear state transition function.

    Returns:
        xs: Tensor of states.
    """
    def body(carry, inputs):
        x = carry
        u = inputs
        x = f(tf.expand_dims(x,0), tf.expand_dims(u,0))[0]
        return x

    xs = tf.scan(body, us, initializer=x0)
    xs = tf.concat([tf.expand_dims(x0, axis=0), xs], axis=0)

    return xs


def nlqt_cost(x0, us, f, Hs, HT, rs, rT, Xs, XT, Us):
    """ Compute the cost of a trajectory. Note that this is not done in parallel.

    Parameters:
        x0: Initial state.
        us: Tensor of control inputs.
        f: Nonlinear state transition function.
        Hs: Tensor of state cost output matrices.
        HT: State cost output matrix at final time.
        rs: Tensor of state cost offsets.
        rT: State cost offset at final time.
        Xs: Tensor of state cost matrices.
        XT: State cost matrix at final time.
        Us: Tensor of control cost matrices.

    Returns:
        J: Cost.
    """
    xs = nlqt_simulate(x0, us, f)
    return lqt_tf.lqt_cost(xs, us, Hs, HT, rs, rT, Xs, XT, Us)


@tf.function
def nlqt_iterate_seq(us, xs, f, Fx, Fu, Hs, HT, rs, rT, Xs, XT, Us):
    """ Perform one iteration of the nonlinear LQT algorithm using sequential LQT.

    Parameters:
        us: Tensor of control inputs.
        xs: Tensor of states.
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
        us: Tensor of updated control inputs.
        xs: Tensor of updated states.
    """
    Fs, cs, Ls = nlqt_linearize(us, xs[:-1], f, Fx, Fu)
    Ss, vs, Kxs, ds = lqt_tf.lqt_seq_backwardpass(Fs, cs, Ls, Hs, HT, rs, rT, Xs, XT, Us)
    xs, us = lqt_tf.lqt_seq_forwardpass(xs[0], Fs, cs, Ls, Kxs, ds)
    return us, xs

@tf.function
def nlqt_iterate_par_1(us, xs, f, Fx, Fu, Hs, HT, rs, rT, Xs, XT, Us, max_parallel=10000):
    """ Perform one iteration of the nonlinear LQT algorithm using parallel LQT.

    Parameters:
        us: Tensor of control inputs.
        xs: Tensor of states.
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
        max_parallel: Maximum number of parallel operations for associative scan.

    Returns:
        us: Tensor of updated control inputs.
        xs: Tensor of updated states.
    """
    Fs, cs, Ls = nlqt_linearize(us, xs[:-1], f, Fx, Fu)
    Ss, vs, Kxs, ds = lqt_tf.lqt_par_backwardpass(Fs, cs, Ls, Hs, HT, rs, rT, Xs, XT, Us, max_parallel=max_parallel)
    xs, us = lqt_tf.lqt_par_forwardpass(xs[0], Fs, cs, Ls, Kxs, ds, max_parallel=max_parallel)
    return us, xs

@tf.function
def nlqt_iterate_par_2(us, xs, f, Fx, Fu, Hs, HT, rs, rT, Xs, XT, Us, max_parallel=10000):
    """ Perform one iteration of the nonlinear LQT algorithm using fw/bw parallel LQT.

    Parameters:
        us: Tensor of control inputs.
        xs: Tensor of states.
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
        max_parallel: Maximum number of parallel operations for associative scan.

    Returns:
        us: Tensor of updated control inputs.
        xs: Tensor of updated states.
    """
    Fs, cs, Ls = nlqt_linearize(us, xs[:-1], f, Fx, Fu)
    Ss, vs, Kxs, ds = lqt_tf.lqt_par_backwardpass(Fs, cs, Ls, Hs, HT, rs, rT, Xs, XT, Us, max_parallel=max_parallel)
    xs, us = lqt_tf.lqt_par_fwdbwdpass(xs[0], Fs, cs, Ls, Hs, rs, Xs, Us, Ss, vs, Kxs, ds, max_parallel=max_parallel)
    return us, xs

