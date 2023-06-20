"""
TensorFlow-based Linear Quadratic Tracker (LQT) routines, both sequential and parallel.

@author: Simo Särkkä
"""

import tensorflow as tf
import tensorflow_probability as tfp
import math

# Abbreviations to make the code more readable
mm = tf.linalg.matmul
mv = tf.linalg.matvec

###########################################################################
#
# Recall that the LQT model is
#
#     x[k+1] = F[k] x[k] + c[k] + L[k] u[k]
#       J(u) = E{ 1/2 (H[T] x[T] - r[T)].T X[T] (H[T] x[T] - r[T])
#         + sum_{k=0}^{T-1} 1/2 (H[k] x[k] - r[k]).T X[k] (H[k] x[k] - r[k])
#                         + 1/2 (Z[k] u[k] - s[k]).T U[k] (Z[k] u[k] - s[k])
#                             + (H[k] x[k] - r[k]).T M[k] (Z[k] u[k] - s[k]) }
#
###########################################################################


###########################################################################
# Misc utilities
###########################################################################

def lqt_np_to_tf(lqt, dtype=tf.float64):
    """ Convert LQT object to TensorFlow tensors.

    Parameters:
        lqt: LQT object

    Returns
        Fs: Batched tensor of F matrices
        cs: Batched tensor of c vectors
        Ls: Batched tensor of L matrices
        Hs: Batched tensor of H matrices
        HT: The final H matrix
        rs: Batched tensor of r vectors
        rT: The final r vector
        Xs: Batched tensor of X matrices
        XT: The final X matrix
    """
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
    """ Convert a more general form LQT object to TensorFlow tensors.

    Parameters:
        lqt: LQT object

    Returns:
        Fs: Batched tensor of F matrices
        cs: Batched tensor of c vectors
        Ls: Batched tensor of L matrices
        Hs: Batched tensor of H matrices
        HT: The final H matrix
        rs: Batched tensor of r vectors
        rT: The final r vector
        Xs: Batched tensor of X matrices
        XT: The final X matrix
        Us: Batched tensor of U matrices
        Ms: Batched tensor of M matrices
        ss: Batched tensor of s vectors
    """

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
    """ Sequential backward pass of LQT.

    Parameters:
        Fs: Batched tensor of F matrices
        cs: Batched tensor of c vectors
        Ls: Batched tensor of L matrices
        Hs: Batched tensor of H matrices
        HT: The final H matrix
        rs: Batched tensor of r vectors
        rT: The final r vector
        Xs: Batched tensor of X matrices
        XT: The final X matrix
        Us: Batched tensor of U matrices
    """

    def body(carry, inputs):
        """ Body for scan for the sequential backward pass of LQT.

        Parameters:
            carry: Tuple of (S, v, X, U)
            inputs: Tuple of (F, c, L, H, r, X, U)

        Returns:
            carry: Tuple of (S, v, X, U)
        """
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
    """ Sequential forward pass of LQT.

    Parameters:
        x0: Initial state
        Fs: Batched tensor of F matrices
        cs: Batched tensor of c vectors
        Ls: Batched tensor of L matrices
        Kxs: Control gains
        ds: Control offsets

    Returns:
        xs: State trajectories
        us: Control trajectories
    """

    def body(carry, inputs):
        """ Body for scan for the sequential forward pass of LQT.

        Parameters:
            carry: Tuple of (x, u)
            inputs: Tuple of (F, c, L, Kx, d)

        Returns:
            carry: Tuple of (x, u)
        """

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
    """ Initialize the general elements of parallel backward pass of LQT.

    Parameters:
        Fs: Batched tensor of F matrices
        cs: Batched tensor of c vectors
        Ls: Batched tensor of L matrices
        Hs: Batched tensor of H matrices
        rs: Batched tensor of r vectors
        Xs: Batched tensor of X matrices
        Us: Batched tensor of U matrices

    Returns:
        As: Batched tensor of A matrices
        bs: Batched tensor of b vectors
        Cs: Batched tensor of C matrices
        etas: Batched tensor of eta vectors
        Js: Batched tensor of J matrices
    """

    As = Fs
    bs = cs
    Cs = mm(Ls, tf.linalg.solve(Us, tf.transpose(Ls, perm=[0,2,1])))
    etas = mv(Hs, mv(Xs, rs), transpose_a=True)
    Js = mm(Hs, mm(Xs, Hs), transpose_a=True)
    return As, bs, Cs, etas, Js

def lqt_par_backwardpass_init_last(HT, rT, XT):
    """ Initialize the last element of parallel backward pass of LQT.

    Parameters:
        HT: The final H matrix
        rT: The final r vector
        XT: The final X matrix

    Returns:
        A: The final A matrix
        b: The final b vector
        C: The final C matrix
        eta: The final eta vector
        J: The final J matrix
    """

    nx = HT.shape[1]
    A = tf.zeros((nx,nx), dtype=HT.dtype) # TODO: Check the device
    b = tf.zeros(nx, dtype=HT.dtype)
    C = tf.zeros_like(A)
    eta = mv(HT, mv(XT, rT), transpose_a=True)
    J = mm(HT, mm(XT, HT), transpose_a=True)
    return A, b, C, eta, J

def lqt_par_comb_V(elemij, elemjk):
    """ Combine two elements at the parallel backward pass of LQT.

    Parameters:
        elemij: Tuple of (Aij, bij, Cij, etaij, Jij)
        elemjk: Tuple of (Ajk, bjk, Cjk, etajk, Jjk)

    Returns:
        Aik, bik, Cik, etaik, Jik: Combined elements
    """
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
    """ Combine two elements at the parallel backward pass of LQT in reverse order.

    Parameters:
        elemjk: Tuple of (Ajk, bjk, Cjk, etajk, Jjk)
        elemij: Tuple of (Aij, bij, Cij, etaij, Jij)

    Returns:
        Aik, bik, Cik, etaik, Jik: Combined elements
    """
    return lqt_par_comb_V(elemij, elemjk)

@tf.function
def lqt_par_backwardpass(Fs, cs, Ls, Hs, HT, rs, rT, Xs, XT, Us, max_parallel=10000):
    """ Parallel backward pass of LQT.

    Parameters:
        Fs: Batched tensor of F matrices
        cs: Batched tensor of c vectors
        Ls: Batched tensor of L matrices
        Hs: Batched tensor of H matrices
        HT: The final H matrix
        rs: Batched tensor of r vectors
        rT: The final r vector
        Xs: Batched tensor of X matrices
        XT: The final X matrix
        Us: Batched tensor of U matrices
        max_parallel: Maximum number of parallel operations for tfp.math.scan_associative

    Returns:
        Ss: Batched tensor of value function matrices
        vs: Batched tensor of value function vectors
        Kxs: Batched tensor of control gain matrices
        ds: Batched tensor of control offsets
    """

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
    """ Initialize the first element of the parallel forward pass of LQT.

    Parameters:
        x0: Initial state
        F: F matrix
        c: c vector
        L: L matrix
        Kx: Control gain matrix
        d: Control offset vector

    Returns:
        tF: Initial tF matrix
        tc: Initial tc vector
    """
    tF = tf.zeros_like(F)
    tc = mv(F - mm(L, Kx), x0) + c + mv(L, d)
    return tF, tc

def lqt_par_forwardpass_init_most(Fs, cs, Ls, Kxs, ds):
    """ Initialize the general elements of the parallel forward pass of LQT.

    Parameters:
        Fs: Batched tensor of F matrices
        cs: Batched tensor of c vectors
        Ls: Batched tensor of L matrices
        Kxs: Batched tensor of control gain matrices
        ds: Batched tensor of control offset vectors

    Returns:
        tF: Batched tensor of initial tF matrices
        tc: Batched tensor of initial tc vectors
    """
    tF = Fs - mm(Ls, Kxs)
    tc = cs + mv(Ls, ds)
    return tF, tc

def lqt_par_comb_f(elemij, elemjk):
    """ Combine two elements at the parallel forward pass of LQT.

    Parameters:
        elemij: Tuple of (Fij, cij)
        elemjk: Tuple of (Fjk, cjk)

    Returns:
        Fik, cik: Combined elements
    """

    Fij, cij = elemij
    Fjk, cjk = elemjk
    Fik = mm(Fjk, Fij)
    cik = mv(Fjk, cij) + cjk
    return Fik, cik

@tf.function
def lqt_par_forwardpass(x0, Fs, cs, Ls, Kxs, ds, max_parallel=10000):
    """ Parallel forward pass of LQT.

    Parameters:
        x0: Initial state
        Fs: Batched tensor of F matrices
        cs: Batched tensor of c vectors
        Ls: Batched tensor of L matrices
        Kxs: Batched tensor of control gain matrices
        ds: Batched tensor of control offset vectors
        max_parallel: Maximum number of parallel operations for tfp.math.scan_associative

    Returns:
        xs: State trajectories
        us: Control trajectories
    """
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
    """ Initialize the first element of the parallel forward-backward pass of LQT.

    Parameters:
        x0: Initial state

    Returns:
        A: Initial A matrix
        b: Initial b vector
        C: Initial C matrix
        eta: Initial eta vector
        J: Initial J matrix
    """
    A = tf.zeros((x0.shape[0],x0.shape[0]), dtype=x0.dtype)
    b = x0
    C = tf.zeros_like(A)
    eta = tf.zeros_like(x0)
    J = tf.zeros_like(A)
    return A, b, C, eta, J

def lqt_par_fwdbwdpass_init_most(Fs, cs, Ls, Hs, rs, Xs, Us):
    """ Initialize the general elements of the parallel forward-backward pass of LQT.

    Parameters:
        Fs: Batched tensor of F matrices
        cs: Batched tensor of c vectors
        Ls: Batched tensor of L matrices
        Hs: Batched tensor of H matrices
        rs: Batched tensor of r vectors
        Xs: Batched tensor of X matrices
        Us: Batched tensor of U matrices

    Returns:
        As: Batched tensor of A matrices
        bs: Batched tensor of b vectors
        Cs: Batched tensor of C matrices
        etas: Batched tensor of eta vectors
        Js: Batched tensor of J matrices
    """
    As = Fs
    bs = cs
    Cs = mm(Ls, tf.linalg.solve(Us, tf.transpose(Ls, perm=[0,2,1])))
    etas = mv(Hs, mv(Xs, rs), transpose_a=True)
    Js = mm(Hs, mm(Xs, Hs), transpose_a=True)
    return As, bs, Cs, etas, Js

@tf.function
def lqt_par_fwdbwdpass(x0, Fs, cs, Ls, Hs, rs, Xs, Us, Ss, vs, Kxs, ds, max_parallel=10000):
    """ Parallel forward-backward pass of LQT.

    Parameters:
        x0: Initial state
        Fs: Batched tensor of F matrices
        cs: Batched tensor of c vectors
        Ls: Batched tensor of L matrices
        Hs: Batched tensor of H matrices
        rs: Batched tensor of r vectors
        Xs: Batched tensor of X matrices
        Us: Batched tensor of U matrices
        Ss: Batched tensor of value function matrices
        vs: Batched tensor of value function vectors
        Kxs: Batched tensor of control gain matrices
        ds: Batched tensor of control offset vectors
        max_parallel: Maximum number of parallel operations for tfp.math.scan_associative

    Returns:
        xs: State trajectories
        us: Control trajectories
    """

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
    """ Compute the cost of a LQT trajectory.

    Parameters:
        xs: State trajectory
        us: Control trajectory
        Hs: Batched tensor of H matrices
        HT: Terminal H matrix
        rs: Batched tensor of r vectors
        rT: Terminal r vector
        Xs: Batched tensor of X matrices
        XT: Terminal X matrix
        Us: Batched tensor of U matrices

    Returns:
        cost: Cost of the trajectory
    """
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
    """ Convert a general LQT cost to canonical form.

    Parameters:
        Fs: Batched tensor of F matrices
        cs: Batched tensor of c vectors
        Ls: Batched tensor of L matrices
        Hs: Batched tensor of H matrices
        rs: Batched tensor of r vectors
        Xs: Batched tensor of X matrices
        Us: Batched tensor of U matrices
        Ms: Batched tensor of M matrices
        ss: Batched tensor of s vectors

    Returns:
        Fs_new: Batched tensor of F matrices
        cs_new: Batched tensor of c vectors
        Xs_new: Batched tensor of X matrices
    """

    CF = tf.linalg.cholesky(Us)
    Fs_new = Fs - mm(Ls, tf.linalg.cholesky_solve(CF, mm(Ms, Hs, transpose_a=True)))
    tmp = tf.linalg.cholesky_solve(CF, tf.expand_dims(mv(Ms, rs, transpose_a=True), -1))[..., 0]
    cs_new = cs + mv(Ls, tmp + ss)
    Xs_new = Xs - mm(Ms, tf.linalg.cholesky_solve(CF, tf.transpose(Ms, perm=[0,2,1])))
    return Fs_new, cs_new, Xs_new

@tf.function
def lqt_canon_to_gen(Kxs, ds, Hs, rs, Us, Ms, ss):
    """ Convert a canonical LQT control law to general form.

    Parameters:
        Kxs: Control gain matrices
        ds: Control offset vectors
        Hs: Batched tensor of H matrices
        rs: Batched tensor of r vectors
        Us: Batched tensor of U matrices
        Ms: Batched tensor of M matrices
        ss: Batched tensor of s vectors

    Returns:
        Kxs_new: Control gain matrices
        ds_new: Control offset vectors
    """

    CF = tf.linalg.cholesky(Us)
    Kxs_new = Kxs + tf.linalg.cholesky_solve(CF, mm(Ms, Hs, transpose_a=True))
    tmp = tf.linalg.cholesky_solve(CF, tf.expand_dims(mv(Ms, rs, transpose_a=True), -1))[..., 0]
    ds_new = ds + tmp + ss
    return Kxs_new, ds_new


