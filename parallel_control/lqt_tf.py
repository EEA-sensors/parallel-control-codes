"""
TensorFlow-based Linear Quadratic Tracker (LQT) routines, both sequential and parallel.

@author: Simo Särkkä
"""

import tensorflow as tf
import tensorflow_probability as tfp
import math

mm = tf.linalg.matmul
mv = tf.linalg.matvec

###########################################################################
# Misc utilities
###########################################################################

def lqt_np_to_tf(lqt, dtype=tf.float64):
    Fs = tf.convert_to_tensor(lqt.F,  dtype=dtype)
    cs = tf.convert_to_tensor(lqt.c,  dtype=dtype)
    Ls = tf.convert_to_tensor(lqt.L,  dtype=dtype)
    Hs = tf.convert_to_tensor(lqt.H,  dtype=dtype) # We only support constant dimensional Hs, Xs, and rs
    HT = tf.convert_to_tensor(lqt.HT, dtype=dtype)
    rs = tf.convert_to_tensor(lqt.r,  dtype=dtype)
    rT = tf.convert_to_tensor(lqt.rT, dtype=dtype)
    Xs = tf.convert_to_tensor(lqt.X,  dtype=dtype)
    XT = tf.convert_to_tensor(lqt.XT, dtype=dtype)
    Us = tf.convert_to_tensor(lqt.U,  dtype=dtype)

    return Fs, cs, Ls, Hs, HT, rs, rT, Xs, XT, Us

def lqt_np_to_tf_gen(lqt, dtype=tf.float64):
    Fs = tf.convert_to_tensor(lqt.F,  dtype=dtype)
    cs = tf.convert_to_tensor(lqt.c,  dtype=dtype)
    Ls = tf.convert_to_tensor(lqt.L,  dtype=dtype)
    Hs = tf.convert_to_tensor(lqt.H,  dtype=dtype) # We only support constant dimensional Hs, Xs, and rs
    HT = tf.convert_to_tensor(lqt.HT, dtype=dtype)
    rs = tf.convert_to_tensor(lqt.r,  dtype=dtype)
    rT = tf.convert_to_tensor(lqt.rT, dtype=dtype)
    Xs = tf.convert_to_tensor(lqt.X,  dtype=dtype)
    XT = tf.convert_to_tensor(lqt.XT, dtype=dtype)
    Us = tf.convert_to_tensor(lqt.U,  dtype=dtype)
    Ms = tf.convert_to_tensor(lqt.M,  dtype=dtype)
    ss = tf.convert_to_tensor(lqt.s,  dtype=dtype)

    return Fs, cs, Ls, Hs, HT, rs, rT, Xs, XT, Us, Ms, ss

###########################################################################
# Sequential LQT
###########################################################################

@tf.function
def lqt_seq_backwardpass(Fs, cs, Ls, Hs, HT, rs, rT, Xs, XT, Us):
    def body(carry, inputs):
        F, c, L, H, r, X, U = inputs
        S, v, _, _ = carry

        CF = tf.linalg.cholesky(U + mm(mm(L, S, transpose_a=True), L))
        Kx = tf.linalg.cholesky_solve(CF, mm(mm(L, S, transpose_a=True), F))
        rh = mv(L, v - mv(S, c), transpose_a=True)
        d  = tf.linalg.cholesky_solve(CF, tf.expand_dims(rh,-1))[...,0]
        v  = mv(H, mv(X, r), transpose_a=True) + mv(F - mm(L,Kx), v - mv(S, c), transpose_a=True)
        S  = mm(F, mm(S, F - mm(L,Kx)), transpose_a=True) + mm(H, mm(X, H), transpose_a=True)

        return S, v, Kx, d

    KxT = tf.zeros((Ls.shape[-1],Ls.shape[-2]), dtype=Ls.dtype) # Dummy value
    dT  = tf.zeros(Ls.shape[-1], dtype=Ls.dtype)                # Dummy value
    vT  = mv(HT, mv(XT, rT), transpose_a=True)
    ST  = mm(HT, mm(XT, HT), transpose_a=True)
    Ss, vs, Kxs, ds = tf.scan(body, (Fs, cs, Ls, Hs, rs, Xs, Us),
                              initializer=(ST,vT,KxT,dT),
                              reverse=True)

    Ss = tf.concat([Ss, tf.expand_dims(ST, axis=0)], axis=0)
    vs = tf.concat([vs, tf.expand_dims(vT, axis=0)], axis=0)

    return Ss, vs, Kxs, ds

@tf.function
def lqt_seq_forwardpass(x0, Fs, cs, Ls, Kxs, ds):
    def body(carry, inputs):
        F, c, L, Kx, d = inputs
        x, _ = carry

        u = -mv(Kx, x) + d
        x = mv(F, x) + c + mv(L, u)

        return x, u

    u0 = tf.zeros_like(Ls[0,0,:])

    xs, us = tf.scan(body, (Fs, cs, Ls, Kxs, ds),
                     initializer=(x0, u0))

    xs = tf.concat([tf.expand_dims(x0, axis=0), xs], axis=0)

    return xs, us

###########################################################################
# Parallel LQT backward pass
###########################################################################

def lqt_par_backwardpass_init_most(Fs, cs, Ls, Hs, rs, Xs, Us):
    As = Fs
    bs = cs
    Cs = mm(Ls, tf.linalg.solve(Us, tf.transpose(Ls, perm=[0,2,1])))
    etas = mv(Hs, mv(Xs, rs), transpose_a=True)
    Js = mm(Hs, mm(Xs, Hs), transpose_a=True)
    return As, bs, Cs, etas, Js

def lqt_par_backwardpass_init_last(HT, rT, XT):
    nx = HT.shape[1]
    A = tf.zeros((nx,nx), dtype=HT.dtype) # TODO: Check the device
    b = tf.zeros(nx, dtype=HT.dtype)
    C = tf.zeros_like(A)
    eta = mv(HT, mv(XT, rT), transpose_a=True)
    J = mm(HT, mm(XT, HT), transpose_a=True)
    return A, b, C, eta, J

def lqt_par_comb_V(elemij, elemjk):
    Aij, bij, Cij, etaij, Jij = elemij
    Ajk, bjk, Cjk, etajk, Jjk = elemjk

#    I = tf.eye(Aij.shape[-1], dtype=Aij.dtype, batch_shape=(Aij.shape[0],))  # Original
    I = tf.expand_dims(tf.eye(Aij.shape[-2], dtype=Aij.dtype), 0)
    LU, p = tf.linalg.lu(I + mm(Cij, Jjk))
    Aik = mm(Ajk, tf.linalg.lu_solve(LU, p, Aij))
    bik = mv(Ajk, tf.linalg.lu_solve(LU, p, tf.expand_dims(bij + mv(Cij, etajk),-1))[...,0]) + bjk
    Cik = mm(Ajk, tf.linalg.lu_solve(LU, p, mm(Cij, Ajk, transpose_b=True))) + Cjk
    LU, p = tf.linalg.lu(I + mm(Jjk, Cij))   # Could be eliminated with clever transposing
    etaik = mv(Aij, tf.linalg.lu_solve(LU, p, tf.expand_dims(etajk - mv(Jjk, bij),-1))[...,0], transpose_a=True) + etaij
    Jik = mm(Aij, tf.linalg.lu_solve(LU, p, mm(Jjk, Aij)), transpose_a=True) + Jij

    return Aik, bik, Cik, etaik, Jik

def lqt_par_comb_V_rev(elemjk, elemij):
    return lqt_par_comb_V(elemij, elemjk)

@tf.function
def lqt_par_backwardpass(Fs, cs, Ls, Hs, HT, rs, rT, Xs, XT, Us, max_parallel=10000):
    elems_most = lqt_par_backwardpass_init_most(Fs, cs, Ls, Hs, rs, Xs, Us)
    elems_last = lqt_par_backwardpass_init_last(HT, rT, XT)

    elems = tuple(tf.concat([em, tf.expand_dims(el, 0)], axis=0)
                  for em, el in zip(elems_most, elems_last))

    rev_elems = tuple(tf.reverse(elem, axis=[0]) for elem in elems)

    rev_elems = tfp.math.scan_associative(lqt_par_comb_V_rev,
                                          rev_elems,
                                          max_num_levels=max(math.ceil(math.log2(max_parallel)),1))

    vs = tf.reverse(rev_elems[3], axis=[0])
    Ss = tf.reverse(rev_elems[4], axis=[0])

    CF  = tf.linalg.cholesky(Us + mm(mm(Ls, Ss[1:, ...], transpose_a=True), Ls))
    Kxs = tf.linalg.cholesky_solve(CF, mm(mm(Ls, Ss[1:, ...], transpose_a=True), Fs))
    rh  = mv(Ls, vs[1:, ...] - mv(Ss[1:, ...], cs), transpose_a=True)
    ds  = tf.linalg.cholesky_solve(CF, tf.expand_dims(rh, -1))[..., 0]

    return Ss, vs, Kxs, ds

###########################################################################
# Parallel LQT forward pass with decomposition of functions
###########################################################################

def lqt_par_forwardpass_init_first(x0, F, c, L, Kx, d):
    tF = tf.zeros_like(F)
    tc = mv(F - mm(L, Kx), x0) + c + mv(L, d)
    return tF, tc

def lqt_par_forwardpass_init_most(Fs, cs, Ls, Kxs, ds):
    tF = Fs - mm(Ls, Kxs)
    tc = cs + mv(Ls, ds)
    return tF, tc

def lqt_par_comb_f(elemij, elemjk):
    Fij, cij = elemij
    Fjk, cjk = elemjk
    Fik = mm(Fjk, Fij)
    cik = mv(Fjk, cij) + cjk
    return Fik, cik

@tf.function
def lqt_par_forwardpass(x0, Fs, cs, Ls, Kxs, ds, max_parallel=10000):
    elems_first = lqt_par_forwardpass_init_first(x0, Fs[0], cs[0], Ls[0], Kxs[0], ds[0])
    elems_most  = lqt_par_forwardpass_init_most(Fs[1:], cs[1:], Ls[1:], Kxs[1:], ds[1:])

    elems = tuple(tf.concat([tf.expand_dims(ef, 0), em], axis=0)
          for ef, em in zip(elems_first, elems_most))

    elems = tfp.math.scan_associative(lqt_par_comb_f,
                                      elems,
                                      max_num_levels=max(math.ceil(math.log2(max_parallel)),1))

    xs = elems[1]
    xs = tf.concat([tf.expand_dims(x0, axis=0), xs], axis=0)
    us = -mv(Kxs, xs[:-1,...]) + ds

    return xs, us

###########################################################################
# Parallel LQT forward pass with value function combination
###########################################################################
def lqt_par_fwdbwdpass_init_first(x0):
    A = tf.zeros((x0.shape[0],x0.shape[0]), dtype=x0.dtype) # TODO: Check device
    b = x0
    C = tf.zeros_like(A)
    eta = tf.zeros_like(x0)
    J = tf.zeros_like(A)
    return A, b, C, eta, J

def lqt_par_fwdbwdpass_init_most(Fs, cs, Ls, Hs, rs, Xs, Us):
    As = Fs
    bs = cs
    Cs = mm(Ls, tf.linalg.solve(Us, tf.transpose(Ls, perm=[0,2,1])))
    etas = mv(Hs, mv(Xs, rs), transpose_a=True)
    Js = mm(Hs, mm(Xs, Hs), transpose_a=True)
    return As, bs, Cs, etas, Js

@tf.function
def lqt_par_fwdbwdpass(x0, Fs, cs, Ls, Hs, rs, Xs, Us, Ss, vs, Kxs, ds, max_parallel=10000):
    elems_first = lqt_par_fwdbwdpass_init_first(x0)
    elems_most  = lqt_par_fwdbwdpass_init_most(Fs, cs, Ls, Hs, rs, Xs, Us)

    elems = tuple(tf.concat([tf.expand_dims(ef, 0), em], axis=0)
                  for ef, em in zip(elems_first, elems_most))

    elems = tfp.math.scan_associative(lqt_par_comb_V,
                                      elems,
                                      max_num_levels=max(math.ceil(math.log2(max_parallel)),1))

    Cs = elems[2]
    bs = elems[1]

    I  = tf.eye(Cs.shape[-1], dtype=Cs.dtype, batch_shape=(Cs.shape[0],))
    xs = tf.linalg.solve(I + mm(Cs, Ss), tf.expand_dims(bs + mv(Cs, vs),-1))[..., 0]
    us = -mv(Kxs, xs[:-1,...]) + ds

    return xs, us

###########################################################################
# Cost computation
###########################################################################
@tf.function
def lqt_cost(xs, us, Hs, HT, rs, rT, Xs, XT, Us):
    drT = mv(HT, xs[-1,...]) - rT
    drs = mv(Hs, xs[:-1,...]) - rs

    return 0.5 * tf.reduce_sum(mv(XT, drT) * drT) \
         + 0.5 * tf.reduce_sum(mv(Xs, drs) * drs) \
         + 0.5 * tf.reduce_sum(mv(Us, us) * us)

###########################################################################
# Conversion routines for general cost (with M,s)
###########################################################################

@tf.function
def lqt_gen_to_canon(Fs, cs, Ls, Hs, rs, Xs, Us, Ms, ss):
    CF = tf.linalg.cholesky(Us)
    Fs_new = Fs - mm(Ls, tf.linalg.cholesky_solve(CF, mm(Ms, Hs, transpose_a=True)))
    tmp = tf.linalg.cholesky_solve(CF, tf.expand_dims(mv(Ms, rs, transpose_a=True), -1))[..., 0]
    cs_new = cs + mv(Ls, tmp + ss)
    Xs_new = Xs - mm(Ms, tf.linalg.cholesky_solve(CF, tf.transpose(Ms, perm=[0,2,1])))
    return Fs_new, cs_new, Xs_new

@tf.function
def lqt_canon_to_gen(Kxs, ds, Hs, rs, Us, Ms, ss):
    CF = tf.linalg.cholesky(Us)
    Kxs_new = Kxs + tf.linalg.cholesky_solve(CF, mm(Ms, Hs, transpose_a=True))
    tmp = tf.linalg.cholesky_solve(CF, tf.expand_dims(mv(Ms, rs, transpose_a=True), -1))[..., 0]
    ds_new = ds + tmp + ss
    return Kxs_new, ds_new


