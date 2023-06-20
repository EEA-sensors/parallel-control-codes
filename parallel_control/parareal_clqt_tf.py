"""
Tensorflow version of Parareal for CLQT.

@author: Simo Särkkä
"""

from scipy import linalg
import numpy as np
import tensorflow as tf
import parallel_control.parareal_clqt_np as para_np
import parallel_control.clqt_tf as clqt_tf
import math
import unittest

import matplotlib.pyplot as plt

###########################################################################
#
# Backward pass
#
###########################################################################

@tf.function
def pclqt_init_backwardpass(blocks, XT, HT, rT, T, F_f, L_f, X_f, U_f, c_f, H_f, r_f):
    """ Initialize Parareal pass for CLQT backward pass.

    Parameters:
        blocks: Number of blocks.
        XT: State cost matrix at final time.
        HT: State cost output matrix at final time.
        rT: State cost offset at final time.
        T: Time horizon.
        F_f, L_f, X_f, U_f, c_f, H_f, r_f: CLQT parameter functions.

    Returns:
        Ss: Tensor of value function matrices.
        vs: Tensor of value function vectors.
    """
    dt, t0, ST, vT = clqt_tf.clqt_seq_backwardpass_defaults(blocks, XT, HT, rT, T)
    Ss, vs, Kxs, ds = clqt_tf.clqt_seq_backwardpass(blocks, dt, t0, ST, vT, F_f, L_f, X_f, U_f, c_f, H_f, r_f)
    return Ss, vs

@tf.function
def pclqt_dense_backwardpass(steps, T, Ss_curr, vs_curr, F_f, L_f, X_f, U_f, c_f, H_f, r_f):
    """ Perform dense Parareal pass for CLQT backward pass.

    Parameters:
        steps: Number of steps per block.
        T: Time horizon.
        Ss_curr: Tensor of current value function matrices.
        vs_curr: Tensor of current value function vectors.
        F_f, L_f, X_f, U_f, c_f, H_f, r_f: CLQT parameter functions.

    Returns:
        Ss_F: Tensor of dense value function matrices.
        vs_F: Tensor of dense value function vectors.
    """
    blocks = tf.shape(Ss_curr)[0] - 1
    dt = T / tf.cast(blocks * steps, dtype=Ss_curr.dtype)

    t0 = tf.range(0, blocks, dtype=Ss_curr.dtype) * steps * dt
    Ss_F, vs_F, Kxs, ds = clqt_tf.clqt_seq_backwardpass(steps, dt, t0, Ss_curr[1:], vs_curr[1:], F_f, L_f, X_f, U_f, c_f, H_f, r_f)

    Ss_F = Ss_F[:, 0, :, :]
    vs_F = vs_F[:, 0, :]

    Ss_F = tf.concat([Ss_F, tf.expand_dims(Ss_curr[-1], axis=0)], axis=0)
    vs_F = tf.concat([vs_F, tf.expand_dims(vs_curr[-1], axis=0)], axis=0)

    return Ss_F, vs_F

@tf.function
def pclqt_coarse_backwardpass(T, Ss_F, vs_F, Ss_G, vs_G, F_f, L_f, X_f, U_f, c_f, H_f, r_f):
    """ Perform coarse Parareal pass for CLQT backward pass.

    Parameters:
        T: Time horizon.
        Ss_F: Tensor of dense value function matrices.
        vs_F: Tensor of dense value function vectors.
        Ss_G: Tensor of coarse value function matrices.
        vs_G: Tensor of coarse value function vectors.
        F_f, L_f, X_f, U_f, c_f, H_f, r_f: CLQT parameter functions.

    Returns:
        Ss: Tensor of current value function matrices.
        vs: Tensor of current value function vectors.
        Ss_G_new: Tensor of coarse value function matrices.
        vs_G_new: Tensor of coarse value function vectors.
    """
    blocks = tf.shape(Ss_F)[0] - 1
    dt = T / tf.cast(blocks, dtype=Ss_F.dtype)

    def body(carry, inputs):
        t, S_F, v_F, S_G, v_G = inputs
        S, v, _, _ = carry

        Ss, vs, Kxs, ds = clqt_tf.clqt_seq_backwardpass(1, dt, t, S, v, F_f, L_f, X_f, U_f, c_f, H_f, r_f)

        S_G_new = Ss[0]
        v_G_new = vs[0]
        S = S_G_new + S_F - S_G
        v = v_G_new + v_F - v_G

        return S, v, S_G_new, v_G_new

    ts = tf.range(0, blocks, dtype=Ss_F.dtype) * dt

    Ss, vs, Ss_G_new, vs_G_new = tf.scan(body, (ts, Ss_F[:-1], vs_F[:-1], Ss_G[:-1], vs_G[:-1]),
                                         initializer=(Ss_F[-1], vs_F[-1], Ss_F[-1], vs_F[-1]),
                                         reverse=True)

    Ss = tf.concat([Ss, tf.expand_dims(Ss_F[-1], axis=0)], axis=0)
    vs = tf.concat([vs, tf.expand_dims(vs_F[-1], axis=0)], axis=0)
    Ss_G_new = tf.concat([Ss_G_new, tf.expand_dims(Ss_F[-1], axis=0)], axis=0)
    vs_G_new = tf.concat([vs_G_new, tf.expand_dims(vs_F[-1], axis=0)], axis=0)

    return Ss, vs, Ss_G_new, vs_G_new

@tf.function
def pclqt_final_backwardpass(steps, T, Ss_curr, vs_curr, F_f, L_f, X_f, U_f, c_f, H_f, r_f):
    """ Perform final Parareal pass for CLQT backward pass.

    Parameters:
        steps: Number of steps per block.
        T: Time horizon.
        Ss_curr: Tensor of current value function matrices.
        vs_curr: Tensor of current value function vectors.
        F_f, L_f, X_f, U_f, c_f, H_f, r_f: CLQT parameter functions.

    Returns:
        Ss: Tensor of final value function matrices.
        vs: Tensor of final value function vectors.
        Kxs: Control gains.
        ds: Control offsets.
    """
    blocks = tf.shape(Ss_curr)[0] - 1
    dt = T / tf.cast(blocks, dtype=Ss_curr.dtype) / steps

    t0 = tf.range(0, blocks, dtype=Ss_curr.dtype) * steps * dt
    Ss, vs, Kxs, ds = clqt_tf.clqt_seq_backwardpass(steps, dt, t0, Ss_curr[1:], vs_curr[1:], F_f, L_f, X_f, U_f, c_f, H_f, r_f)

    Ss = Ss[:, :-1, :, :]
    vs = vs[:, :-1, :]

    Ss_shape = tf.shape(Ss)
    Ss = tf.reshape(Ss, shape=(Ss_shape[0] * Ss_shape[1], Ss_shape[2], Ss_shape[3]))
    vs_shape = tf.shape(vs)
    vs = tf.reshape(vs, shape=(vs_shape[0] * vs_shape[1], vs_shape[2]))
    Kxs_shape = tf.shape(Kxs)
    Kxs = tf.reshape(Kxs, shape=(Kxs_shape[0] * Kxs_shape[1], Kxs_shape[2], Kxs_shape[3]))
    ds_shape = tf.shape(ds)
    ds = tf.reshape(ds, shape=(ds_shape[0] * ds_shape[1], ds_shape[2]))

    ST = Ss_curr[-1]
    vT = vs_curr[-1]

    Ss = tf.concat([Ss, tf.expand_dims(ST, axis=0)], axis=0)
    vs = tf.concat([vs, tf.expand_dims(vT, axis=0)], axis=0)

    return Ss, vs, Kxs, ds

@tf.function
def pclqt_backwardpass(blocks, steps, niter, XT, HT, rT, T, F_f, L_f, X_f, U_f, c_f, H_f, r_f):
    """ Perform Parareal CLQT backward pass.

    Parameters:
        blocks: Number of blocks.
        steps: Number of steps per block.
        niter: Number of Parareal iterations.
        XT: Terminal state.
        HT: Terminal cost.
        rT: Terminal constraint.
        T: Time horizon.
        F_f, L_f, X_f, U_f, c_f, H_f, r_f: CLQT parameter functions.

    Returns:
        Ss: Tensor of value function matrices.
        vs: Tensor of value function vectors.
        Kxs: Control gains.
        ds: Control offsets.
    """
    def body(carry, inputs):
        iter = inputs
        Ss_curr, vs_curr, Ss_G, vs_G = carry

        Ss_F, vs_F = pclqt_dense_backwardpass(steps, T, Ss_curr, vs_curr, F_f, L_f, X_f, U_f, c_f, H_f, r_f)
        Ss_curr, vs_curr, Ss_G, vs_G = pclqt_coarse_backwardpass(T, Ss_F, vs_F, Ss_G, vs_G, F_f, L_f, X_f, U_f, c_f, H_f, r_f)

        return Ss_curr, vs_curr, Ss_G, vs_G

    Ss_curr, vs_curr = pclqt_init_backwardpass(blocks, XT, HT, rT, T, F_f, L_f, X_f, U_f, c_f, H_f, r_f)
    Ss_G = Ss_curr
    vs_G = vs_curr

    Ss_curr_all, vs_curr_all, Ss_G_all, vs_G_all = tf.scan(body,
                                                           tf.range(0,niter),
                                                           initializer=(Ss_curr, vs_curr, Ss_G, vs_G),
                                                           reverse=False)

    Ss, vs, Kxs, ds = pclqt_final_backwardpass(steps, T, Ss_curr_all[-1], vs_curr_all[-1], F_f, L_f, X_f, U_f, c_f, H_f, r_f)

    return Ss, vs, Kxs, ds

###########################################################################
# Forward pass
###########################################################################

@tf.function
def pclqt_init_forwardpass(blocks, steps, x0, T, Kxs, ds, F_f, L_f, c_f, u_zoh=False):
    """ Initialize Parareal pass for CLQT forward pass.

    Parameters:
        blocks: Number of blocks.
        steps: Number of steps per block.
        x0: Initial state.
        T: Time horizon.
        Kxs: Control gains.
        ds: Control offsets.
        F_f, L_f, c_f: CLQT parameter functions.
        u_zoh: If True, use zero-order hold for control inputs.

    Returns:
        xs_curr: Tensor of current state trajectories.
    """

    dt = T / tf.cast(blocks, dtype=T.dtype)
    t0 = tf.constant(0.0, dtype=T.dtype)

    Kxs_shape = tf.shape(Kxs)
    Kxs = tf.reshape(Kxs, (blocks, steps, Kxs_shape[-2], Kxs_shape[-1]))
    ds_shape = tf.shape(ds)
    ds = tf.reshape(ds, (blocks, steps, ds_shape[-1]))

    xs_curr, _ = clqt_tf.clqt_seq_forwardpass(x0, Kxs[:, 0, :, :], ds[:, 0, :], dt, t0, F_f, L_f, c_f, u_zoh=u_zoh)
    return xs_curr

@tf.function
def pclqt_dense_forwardpass(steps, T, xs_curr, Kxs, ds, F_f, L_f, c_f, u_zoh=False):
    """ Perform dense Parareal pass for CLQT forward pass.

    Parameters:
        steps: Number of steps per block.
        T: Time horizon.
        xs_curr: Tensor of current state trajectories.
        Kxs: Control gains.
        ds: Control offsets.
        F_f, L_f, c_f: CLQT parameter functions.
        u_zoh: If True, use zero-order hold for control inputs.

    Returns:
        xs_F: Tensor of dense state trajectories.
    """
    blocks = tf.shape(xs_curr)[0] - 1
    dt = T / tf.cast(blocks * steps, dtype=xs_curr.dtype)

    t0 = tf.range(0, blocks, dtype=xs_curr.dtype) * tf.cast(steps, dtype=xs_curr.dtype) * dt

    Kxs_shape = tf.shape(Kxs)
    Kxs = tf.reshape(Kxs, (blocks, steps, Kxs_shape[-2], Kxs_shape[-1]))
    ds_shape = tf.shape(ds)
    ds = tf.reshape(ds, (blocks, steps, ds_shape[-1]))
    xs, us = clqt_tf.clqt_seq_forwardpass(xs_curr[0:-1], Kxs, ds, dt, t0, F_f, L_f, c_f, u_zoh=u_zoh)

    xs_F = xs[:, -1, :]
    xs_F = tf.concat([tf.expand_dims(xs_curr[0], axis=0), xs_F], axis=0)

    return xs_F

@tf.function
def pclqt_coarse_forwardpass(steps, T, xs_F, xs_G, Kxs, ds, F_f, L_f, c_f, u_zoh=False):
    """ Perform coarse Parareal pass for CLQT forward pass.

    Parameters:
        steps: Number of steps per block.
        T: Time horizon.
        xs_F: Tensor of dense state trajectories.
        xs_G: Tensor of coarse state trajectories.
        Kxs: Control gains.
        ds: Control offsets.
        F_f, L_f, c_f: CLQT parameter functions.
        u_zoh: If True, use zero-order hold for control inputs.

    Returns:
        xs: Tensor of current state trajectories.
        xs_G_new: Tensor of new coarse state trajectories.
    """
    blocks = tf.shape(xs_F)[0] - 1
    dt = T / tf.cast(blocks, dtype=xs_F.dtype)

    def body(carry, inputs):
        t, x_F, x_G, Kx, d = inputs
        x, _ = carry

        Kx = tf.expand_dims(Kx, axis=0)
        d = tf.expand_dims(d, axis=0)
        xs, us = clqt_tf.clqt_seq_forwardpass(x, Kx, d, dt, t, F_f, L_f, c_f, u_zoh=u_zoh)

        x_G_new = xs[-1]
        x = x_G_new + x_F - x_G

        return x, x_G_new

    ts = tf.range(0, blocks, dtype=xs_F.dtype) * dt

    xs, xs_G_new = tf.scan(body, (ts, xs_F[1:], xs_G[1:], Kxs[::steps], ds[::steps]),
                           initializer=(xs_F[0], xs_F[0]),
                           reverse=False)

    xs = tf.concat([tf.expand_dims(xs_F[0], axis=0), xs], axis=0)
    xs_G_new = tf.concat([tf.expand_dims(xs_F[0], axis=0), xs_G_new], axis=0)

    return xs, xs_G_new

@tf.function
def pclqt_final_forwardpass(steps, T, xs_curr, Kxs, ds, F_f, L_f, c_f, u_zoh=False):
    """ Perform final Parareal pass for CLQT forward pass.

    Parameters:
        steps: Number of steps per block.
        T: Time horizon.
        xs_curr: Tensor of current state trajectories.
        Kxs: Control gains.
        ds: Control offsets.
        F_f, L_f, c_f: CLQT parameter functions.
        u_zoh: If True, use zero-order hold for control inputs.

    Returns:
        xs: Tensor of final state trajectories.
        us: Tensor of final control trajectories.
    """

    blocks = tf.shape(xs_curr)[0] - 1
    dt = T / tf.cast(blocks * steps, dtype=xs_curr.dtype)

    t0 = tf.range(0, blocks, dtype=xs_curr.dtype) * tf.cast(steps, dtype=xs_curr.dtype) * dt

    Kxs_shape = tf.shape(Kxs)
    Kxs = tf.reshape(Kxs, (blocks, steps, Kxs_shape[-2], Kxs_shape[-1]))
    ds_shape = tf.shape(ds)
    ds = tf.reshape(ds, (blocks, steps, ds_shape[-1]))

    xs, us = clqt_tf.clqt_seq_forwardpass(xs_curr[:-1], Kxs, ds, dt, t0, F_f, L_f, c_f, u_zoh=u_zoh)

    xT = xs[-1, -1, :]
    xs = xs[:, :-1, :]
    xs_shape = tf.shape(xs)
    xs = tf.reshape(xs, shape=(xs_shape[0] * xs_shape[1], xs_shape[2]))
    xs = tf.concat([xs, tf.expand_dims(xT, axis=0)], axis=0)

    us_shape = tf.shape(us)
    us = tf.reshape(us, shape=(us_shape[0] * us_shape[1], us_shape[2]))

    return xs, us

@tf.function
def pclqt_forwardpass(blocks, steps, niter, x0, T, Kxs, ds, F_f, L_f, c_f, u_zoh=False):
    """ Perform CLQT forward pass using Parareal.

    Parameters:
        blocks: Number of blocks.
        steps: Number of steps per block.
        niter: Number of Parareal iterations.
        x0: Initial state.
        T: Time horizon.
        Kxs: Control gains.
        ds: Control offsets.
        F_f, L_f, c_f: CLQT parameter functions.
        u_zoh: If True, use zero-order hold for control inputs.

    Returns:
        xs: Tensor of state trajectories.
        us: Tensor of control trajectories.
    """

    def body(carry, inputs):
        iter = inputs
        xs_curr, xs_G = carry

        xs_F = pclqt_dense_forwardpass(steps, T, xs_curr, Kxs, ds, F_f, L_f, c_f, u_zoh=u_zoh)
        xs_curr, xs_G = pclqt_coarse_forwardpass(steps, T, xs_F, xs_G, Kxs, ds, F_f, L_f, c_f, u_zoh=u_zoh)

        return xs_curr, xs_G

    xs_curr = pclqt_init_forwardpass(blocks, steps, x0, T, Kxs, ds, F_f, L_f, c_f, u_zoh=u_zoh)
    xs_G = xs_curr

    xs_curr_all, xs_G_all = tf.scan(body,
                                    tf.range(0,niter),
                                    initializer=(xs_curr, xs_G),
                                    reverse=False)

    xs, us = pclqt_final_forwardpass(steps, T, xs_curr_all[-1], Kxs, ds, F_f, L_f, c_f, u_zoh=u_zoh)

    return xs, us

@tf.function
def pclqt_backwardforwardpass(blocks, steps, niter, x0_tf, XT, HT, rT, T, F_f, L_f, X_f, U_f, c_f, H_f, r_f):
    """ Perform CLQT backward and forward passes using Parareal.

    Parameters:
        blocks: Number of blocks.
        steps: Number of steps per block.
        niter: Number of Parareal iterations.
        x0_tf: Initial state.
        XT, HT, rT: Terminal cost parameters.
        T: Time horizon.
        F_f, L_f, X_f, U_f, c_f, H_f, r_f: CLQT parameter functions.

    Returns:
        xs2: Tensor of state trajectories.
        us2: Tensor of control trajectories.
    """
    Ss2, vs2, Kxs2, ds2 = pclqt_backwardpass(blocks, steps, niter, XT, HT, rT, T, F_f, L_f, X_f, U_f, c_f, H_f, r_f)
    xs2, us2 = pclqt_forwardpass(blocks, steps, niter, x0_tf, T, Kxs2, ds2, F_f, L_f, c_f)
    return xs2, us2

###########################################################################
# Forward-backward pass
###########################################################################

@tf.function
def pclqt_init_fwdbwdpass(blocks, x0, T, F_f, L_f, X_f, U_f, c_f, H_f, r_f):
    """ Initialize Parareal CLQT forward-backward pass.

    Parameters:
        blocks: Number of blocks.
        x0: Initial state.
        T: Time horizon.
        F_f, L_f, X_f, U_f, c_f, H_f, r_f: CLQT parameter functions.

    Returns:
        As: Tensor of current A matrices.
        bs: Tensor of current b vectors.
        Cs: Tensor of current C matrices.
    """
    dt, t0, A0, b0, C0 = clqt_tf.clqt_seq_fwdbwdpass_defaults(x0, blocks, T)
    As, bs, Cs = clqt_tf.clqt_seq_fwdbwdpass(blocks, dt, t0, A0, b0, C0, F_f, L_f, X_f, U_f, c_f, H_f, r_f)
    return As, bs, Cs

@tf.function
def pclqt_dense_fwdbwdpass(steps, T, As_curr, bs_curr, Cs_curr, F_f, L_f, X_f, U_f, c_f, H_f, r_f):
    """ Perform dense Parareal pass for CLQT forward-backward pass.

    Parameters:
        steps: Number of steps per block.
        T: Time horizon.
        As_curr: Tensor of current A matrices.
        bs_curr: Tensor of current b vectors.
        Cs_curr: Tensor of current C matrices.
        F_f, L_f, X_f, U_f, c_f, H_f, r_f: CLQT parameter functions.

    Returns:
        As_F: Tensor of dense A matrices.
        bs_F: Tensor of dense b vectors.
        Cs_F: Tensor of dense C matrices.
    """
    blocks = tf.shape(As_curr)[0] - 1
    dt = T / tf.cast(blocks * steps, dtype=As_curr.dtype)

    t0 = tf.range(0, blocks, dtype=As_curr.dtype) * steps * dt

    As_F, bs_F, Cs_F = clqt_tf.clqt_seq_fwdbwdpass(steps, dt, t0, As_curr[:-1], bs_curr[:-1], Cs_curr[:-1], F_f, L_f, X_f, U_f, c_f, H_f, r_f)

    As_F = As_F[:, -1, :, :]
    bs_F = bs_F[:, -1, :]
    Cs_F = Cs_F[:, -1, :, :]

    As_F = tf.concat([tf.expand_dims(As_curr[0], axis=0), As_F], axis=0)
    bs_F = tf.concat([tf.expand_dims(bs_curr[0], axis=0), bs_F], axis=0)
    Cs_F = tf.concat([tf.expand_dims(Cs_curr[0], axis=0), Cs_F], axis=0)

    return As_F, bs_F, Cs_F

@tf.function
def pclqt_coarse_fwdbwdpass(T, As_F, bs_F, Cs_F, As_G, bs_G, Cs_G, F_f, L_f, X_f, U_f, c_f, H_f, r_f):
    """ Perform coarse Parareal pass for CLQT forward-backward pass.

    Parameters:
        T: Time horizon.
        As_F: Tensor of dense A matrices.
        bs_F: Tensor of dense b vectors.
        Cs_F: Tensor of dense C matrices.
        As_G: Tensor of coarse A matrices.
        bs_G: Tensor of coarse b vectors.
        Cs_G: Tensor of coarse C matrices.
        F_f, L_f, X_f, U_f, c_f, H_f, r_f: CLQT parameter functions.

    Returns:
        As: Tensor of current A matrices.
        bs: Tensor of current b vectors.
        Cs: Tensor of current C matrices.
        As_G_new: Tensor of new coarse A matrices.
        bs_G_new: Tensor of new coarse b vectors.
        Cs_G_new: Tensor of new coarse C matrices.
    """
    blocks = tf.shape(As_F)[0] - 1
    dt = T / tf.cast(blocks, dtype=As_F.dtype)

    def body(carry, inputs):
        t, A_F, b_F, C_F, A_G, b_G, C_G = inputs
        A, b, C, _, _, _ = carry

        As, bs, Cs = clqt_tf.clqt_seq_fwdbwdpass(1, dt, t, A, b, C, F_f, L_f, X_f, U_f, c_f, H_f, r_f)

        A_G_new = As[-1]
        b_G_new = bs[-1]
        C_G_new = Cs[-1]
        A = A_G_new + A_F - A_G
        b = b_G_new + b_F - b_G
        C = C_G_new + C_F - C_G

        return A, b, C, A_G_new, b_G_new, C_G_new

    ts = tf.range(0, blocks, dtype=As_F.dtype) * dt

    As, bs, Cs, As_G_new, bs_G_new, Cs_G_new = tf.scan(body, (ts, As_F[1:], bs_F[1:], Cs_F[1:], As_G[1:], bs_G[1:], Cs_G[1:]),
                                                       initializer=(As_F[0], bs_F[0], Cs_F[0], As_F[0], bs_F[0], Cs_F[0]),
                                                       reverse=False)

    As = tf.concat([tf.expand_dims(As_F[0], axis=0), As], axis=0)
    bs = tf.concat([tf.expand_dims(bs_F[0], axis=0), bs], axis=0)
    Cs = tf.concat([tf.expand_dims(Cs_F[0], axis=0), Cs], axis=0)
    As_G_new = tf.concat([tf.expand_dims(As_F[0], axis=0), As_G_new], axis=0)
    bs_G_new = tf.concat([tf.expand_dims(bs_F[0], axis=0), bs_G_new], axis=0)
    Cs_G_new = tf.concat([tf.expand_dims(Cs_F[0], axis=0), Cs_G_new], axis=0)

    return As, bs, Cs, As_G_new, bs_G_new, Cs_G_new

@tf.function
def pclqt_final_fwdbwdpass(steps, T, As_curr, bs_curr, Cs_curr, F_f, L_f, X_f, U_f, c_f, H_f, r_f):
    """ Perform final Parareal pass for CLQT forward-backward pass.

    Parameters:
        steps: Number of steps in each block.
        T: Time horizon.
        As_curr: Tensor of current A matrices.
        bs_curr: Tensor of current b vectors.
        Cs_curr: Tensor of current C matrices.
        F_f, L_f, X_f, U_f, c_f, H_f, r_f: CLQT parameter functions.

    Returns:
        As: Tensor of final A matrices.
        bs: Tensor of final b vectors.
        Cs: Tensor of final C matrices.
    """
    blocks = tf.shape(As_curr)[0] - 1
    dt = T / tf.cast(blocks, dtype=As_curr.dtype) / steps

    t0 = tf.range(0, blocks, dtype=As_curr.dtype) * steps * dt
    As, bs, Cs = clqt_tf.clqt_seq_fwdbwdpass(steps, dt, t0, As_curr[:-1], bs_curr[:-1], Cs_curr[:-1], F_f, L_f, X_f, U_f, c_f, H_f, r_f)

    AT = As[-1, -1, :, :]
    bT = bs[-1, -1, :]
    CT = Cs[-1, -1, :, :]

    As = As[:, :-1, :, :]
    bs = bs[:, :-1, :]
    Cs = Cs[:, :-1, :, :]

    As_shape = tf.shape(As)
    As = tf.reshape(As, shape=(As_shape[0] * As_shape[1], As_shape[2], As_shape[3]))
    bs_shape = tf.shape(bs)
    bs = tf.reshape(bs, shape=(bs_shape[0] * bs_shape[1], bs_shape[2]))
    Cs_shape = tf.shape(Cs)
    Cs = tf.reshape(Cs, shape=(Cs_shape[0] * Cs_shape[1], Cs_shape[2], Cs_shape[3]))

    As = tf.concat([As, tf.expand_dims(AT, axis=0)], axis=0)
    bs = tf.concat([bs, tf.expand_dims(bT, axis=0)], axis=0)
    Cs = tf.concat([Cs, tf.expand_dims(CT, axis=0)], axis=0)

    return As, bs, Cs

@tf.function
def pclqt_fwdbwdpass(blocks, steps, niter, x0, T, F_f, L_f, X_f, U_f, c_f, H_f, r_f):
    """ Perform Parareal forward-backward pass for CLQT.

    Parameters:
        blocks: Number of blocks.
        steps: Number of steps in each block.
        niter: Number of Parareal iterations.
        x0: Initial state.
        T: Time horizon.
        F_f, L_f, X_f, U_f, c_f, H_f, r_f: CLQT parameter functions.

    Returns:
        As: Tensor of A matrices.
        bs: Tensor of b vectors.
        Cs: Tensor of C matrices.
    """
    def body(carry, inputs):
        iter = inputs
        As_curr, bs_curr, Cs_curr, As_G, bs_G, Cs_G = carry

        As_F, bs_F, Cs_F = pclqt_dense_fwdbwdpass(steps, T, As_curr, bs_curr, Cs_curr, F_f, L_f, X_f, U_f, c_f, H_f, r_f)
        As_curr, bs_curr, Cs_curr, As_G, bs_G, Cs_G = pclqt_coarse_fwdbwdpass(T, As_F, bs_F, Cs_F, As_G, bs_G, Cs_G, F_f, L_f, X_f, U_f, c_f, H_f, r_f)

        return As_curr, bs_curr, Cs_curr, As_G, bs_G, Cs_G

    As_curr, bs_curr, Cs_curr = pclqt_init_fwdbwdpass(blocks, x0, T, F_f, L_f, X_f, U_f, c_f, H_f, r_f)
    As_G = As_curr
    bs_G = bs_curr
    Cs_G = Cs_curr

    As_curr_all, bs_curr_all, Cs_curr_all, As_G_all, bs_G_all, Cs_G_all \
        = tf.scan(body,
                  tf.range(0,niter),
                  initializer=(As_curr, bs_curr, Cs_curr, As_G, bs_G, Cs_G),
                  reverse=False)

    As, bs, Cs = pclqt_final_fwdbwdpass(steps, T, As_curr_all[-1], bs_curr_all[-1], Cs_curr_all[-1], F_f, L_f, X_f, U_f, c_f, H_f, r_f)

    return As, bs, Cs

@tf.function
def pclqt_backwardbwdfwdpass(blocks, steps, niter, x0_tf, XT, HT, rT, T, F_f, L_f, X_f, U_f, c_f, H_f, r_f):
    """ Perform Parareal backward and backward-forward passes for CLQT.

    Parameters:
        blocks: Number of blocks.
        steps: Number of steps in each block.
        niter: Number of Parareal iterations.
        x0_tf: Initial state.
        XT: Terminal state.
        HT: Terminal cost.
        rT: Terminal constraint.
        T: Time horizon.
        F_f, L_f, X_f, U_f, c_f, H_f, r_f: CLQT parameter functions.

    Returns:
        xs2: Tensor of state trajectories.
        us2: Tensor of control trajectories.
    """
    Ss2, vs2, Kxs2, ds2 = pclqt_backwardpass(blocks, steps, niter, XT, HT, rT, T, F_f, L_f, X_f, U_f, c_f, H_f, r_f)
    As2, bs2, Cs2 = pclqt_fwdbwdpass(blocks, steps, niter, x0_tf, T, F_f, L_f, X_f, U_f, c_f, H_f, r_f)
    xs2, us2 = clqt_tf.clqt_combine_fwdbwd(Kxs2, ds2, Ss2, vs2, As2, bs2, Cs2)
    return xs2, us2

