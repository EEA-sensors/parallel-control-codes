"""
Tensorflow-based Nonlinear (iterated) Linear Quadratic Tracker.

@author: Simo Särkkä
"""

import numpy as np
import tensorflow as tf

import parallel_control.lqt_np as lqt_np
import parallel_control.lqt_tf as lqt_tf
#import parallel_control.nlqt_np as nlqt_np


mm = tf.linalg.matmul
mv = tf.linalg.matvec

##############################################################################
#
# Nonlinear LQT
#
##############################################################################

def nlqt_linearize(us, xs, f, Fx, Fu):
    fs  = f(xs, us)
    Fxs = Fx(xs, us)
    Fus = Fu(xs, us)

    # f(xp,up) = f(x,u) + Fx(x,u) (xp - x) + Fu(x,u) (up - u)
    Fs = Fxs
    Ls = Fus
    cs = fs - mv(Fxs, xs) - mv(Fus, us)

    return Fs, cs, Ls


# Note that this is not done in parallel
def nlqt_simulate(x0, us, f):
    def body(carry, inputs):
        x = carry
        u = inputs
        x = f(tf.expand_dims(x,0), tf.expand_dims(u,0))[0]
        return x

    xs = tf.scan(body, us, initializer=x0)
    xs = tf.concat([tf.expand_dims(x0, axis=0), xs], axis=0)

    return xs


# Note that this is not done in parallel
def nlqt_cost(x0, us, f, Hs, HT, rs, rT, Xs, XT, Us):
    xs = nlqt_simulate(x0, us, f)
    return lqt_tf.lqt_cost(xs, us, Hs, HT, rs, rT, Xs, XT, Us)


@tf.function
def nlqt_iterate_seq(us, xs, f, Fx, Fu, Hs, HT, rs, rT, Xs, XT, Us):
    Fs, cs, Ls = nlqt_linearize(us, xs[:-1], f, Fx, Fu)
    Ss, vs, Kxs, ds = lqt_tf.lqt_seq_backwardpass(Fs, cs, Ls, Hs, HT, rs, rT, Xs, XT, Us)
    xs, us = lqt_tf.lqt_seq_forwardpass(xs[0], Fs, cs, Ls, Kxs, ds)
    return us, xs

@tf.function
def nlqt_iterate_par_1(us, xs, f, Fx, Fu, Hs, HT, rs, rT, Xs, XT, Us, max_parallel=10000):
    Fs, cs, Ls = nlqt_linearize(us, xs[:-1], f, Fx, Fu)
    Ss, vs, Kxs, ds = lqt_tf.lqt_par_backwardpass(Fs, cs, Ls, Hs, HT, rs, rT, Xs, XT, Us, max_parallel=max_parallel)
    xs, us = lqt_tf.lqt_par_forwardpass(xs[0], Fs, cs, Ls, Kxs, ds, max_parallel=max_parallel)
    return us, xs

@tf.function
def nlqt_iterate_par_2(us, xs, f, Fx, Fu, Hs, HT, rs, rT, Xs, XT, Us, max_parallel=10000):
    Fs, cs, Ls = nlqt_linearize(us, xs[:-1], f, Fx, Fu)
    Ss, vs, Kxs, ds = lqt_tf.lqt_par_backwardpass(Fs, cs, Ls, Hs, HT, rs, rT, Xs, XT, Us, max_parallel=max_parallel)
    xs, us = lqt_tf.lqt_par_fwdbwdpass(xs[0], Fs, cs, Ls, Hs, rs, Xs, Us, Ss, vs, Kxs, ds, max_parallel=max_parallel)
    return us, xs

