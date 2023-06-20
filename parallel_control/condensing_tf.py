"""
Partial condensing in TensorFlow.

@author: Simo Särkkä (with some code from Adrien Corenflos)
"""

import numpy as np
import tensorflow as tf
import math

# Abbreviations for linear algebra functions
mm = tf.linalg.matmul
mv = tf.linalg.matvec

def _create_blkdiag(blocks):
    """ Create a block-diagonal matrix from given matrix blocks.

    Parameters:
        blocks: Matrix blocks.

    Returns:
        block_diag: Block diagonal matrix.
    """

    M, K, L = blocks.shape
    splitted_blocks = [b[0] for b in tf.split(blocks, M)]
    full_ops = [tf.linalg.LinearOperatorFullMatrix(b, is_square=False) for b in splitted_blocks]
    block_diag_ops = tf.linalg.LinearOperatorBlockDiag(full_ops, is_square=False)
    return block_diag_ops.to_dense()

def create_blkdiag(blocks):
    """ Form a block diagonal (batched) tensor from given (batch of) matrix blocks.
    The blocks should be N*M*K*L, where N is the batch index, M indices of blocks,
    and the blocks are K*L matrices. Returns N*(M*K)*(M*L) tensor.

    Parameters:
        blocks: (batch of) matrix blocks.

    Returns:
        block_diag: (batch of) block diagonal matrices.
    """
    return tf.vectorized_map(_create_blkdiag, blocks, fallback_to_while_loop=False)

def condense(Fs, cs, Ls, Hs, rs, Xs, Us, Nc):
    """ Partially condense a given LQT.

    Parameters:
        Fs, cs, Ls, Hs, rs, Xs, Us: LQT to be condensed.
        Nc: Number of condensed steps.

    Returns:
        Fstar, cstar, Lstar, Hstar, rstar, Xstar, Ustar, Mstar, sstar, Lambda, cbar, Lbar: Condensed LQT.
    """
    T = Fs.shape[0]

    # Append trivial dynamics if needed
    if (T % Nc) != 0:
        n = Nc - (T % Nc)
        pad = tf.eye(Fs.shape[1], batch_shape=(n,), dtype=Fs.dtype)
        Fs = tf.concat([Fs,pad], 0)
        pad = tf.zeros((n, cs.shape[1]), dtype=cs.dtype)
        cs = tf.concat([cs,pad], 0)
        pad = tf.zeros((n, Ls.shape[1], Ls.shape[2]), dtype=Ls.dtype)
        Ls = tf.concat([Ls,pad], 0)
        pad = tf.zeros((n, Hs.shape[1], Hs.shape[2]), dtype=Hs.dtype)
        Hs = tf.concat([Hs,pad], 0)
        pad = tf.zeros((n, rs.shape[1]), dtype=rs.dtype)
        rs = tf.concat([rs,pad], 0)
        pad = tf.zeros((n, Xs.shape[1], Xs.shape[2]), dtype=Xs.dtype)
        Xs = tf.concat([Xs,pad], 0)
        pad = tf.eye(Us.shape[1], batch_shape=(n,), dtype=Us.dtype)
        Us = tf.concat([Us,pad], 0)

    T = Fs.shape[0]
    Tc = T // Nc

    Xbar = create_blkdiag(tf.reshape(tf.expand_dims(Xs,0), (Tc,Nc,Xs.shape[1],Xs.shape[2])))
    Ubar = create_blkdiag(tf.reshape(tf.expand_dims(Us,0), (Tc,Nc,Us.shape[1],Us.shape[2])))
    Hbar = create_blkdiag(tf.reshape(tf.expand_dims(Hs,0), (Tc,Nc,Hs.shape[1],Hs.shape[2])))
    XUbar = tf.concat([tf.pad(Xbar, [[0,0], [0,0], [0,Ubar.shape[2]]]),
                       tf.pad(Ubar, [[0,0], [0,0], [Xbar.shape[2],0]])], axis=1)
    rsbar = tf.concat([tf.reshape(rs, shape=(Tc,Nc * rs.shape[1])),
                       tf.zeros((Tc,Nc*Us.shape[1]), dtype=rs.dtype)], axis=1)

    Fs_blocks = tf.reshape(tf.expand_dims(Fs, 0), (Tc, Nc, Fs.shape[1], Fs.shape[2]))
    Ls_blocks = tf.reshape(tf.expand_dims(Ls, 0), (Tc, Nc, Ls.shape[1], Ls.shape[2]))
    cs_blocks = tf.reshape(tf.expand_dims(cs, 0), (Tc, Nc, cs.shape[1]))

    # Form matrix Fstar = F[k+Nc-1]***F[k] as well as a matrix
    # Lambda = [I; F[k]; F[k+1] F[k]; ...; F[k+Nc-2] *** F[k]]
    Lambda_list = [tf.eye(Fs_blocks.shape[2], batch_shape=(Fs_blocks.shape[0],), dtype=Fs_blocks.dtype)]
    Fstar = Fs_blocks[:,0,:,:]
    for k in range(1,Fs_blocks.shape[1]):
        Lambda_list.append(Fstar)
        Fstar = mm(Fs_blocks[:,k,:,:], Fstar)
    Lambda = tf.concat(Lambda_list, 1)

    # Form matrixes Lbar, cbar, Lstar, and cstar
    Lbar_list = [tf.zeros((Tc, Ls_blocks.shape[2], Ls_blocks.shape[3] * Nc), dtype=Ls.dtype)]
    cbar_list = [tf.zeros((Tc, cs_blocks.shape[2]), dtype=Ls.dtype)]
    Lbar_row = Ls_blocks[:,0,:,:]
    cbar_row = cs_blocks[:,0,:]
    for k in range(Nc-1):
        Lbar_list.append(tf.pad(Lbar_row, [[0,0],[0,0],[0, Ls_blocks.shape[3] * Nc - Lbar_row.shape[2]]]))
        cbar_list.append(cbar_row)
        Lbar_row = tf.concat([mm(Fs_blocks[:,k+1,:,:], Lbar_row), Ls_blocks[:,k+1,:,:]], 2)
        cbar_row = mv(Fs_blocks[:,k+1,:,:], cbar_row) + cs_blocks[:,k+1,:]
    Lbar = tf.concat(Lbar_list, 1)
    cbar = tf.concat(cbar_list, 1)

    Lstar = tf.pad(Lbar_row, [[0,0],[0,0],[0, Ls_blocks.shape[3] * Nc - Lbar_row.shape[2]]])
    cstar = cbar_row

    # Transform cost function
    trans_r1 = tf.concat([mm(Hbar, Lambda), mm(Hbar, Lbar)], 2)
    trans_r2 = tf.concat([tf.zeros((Fs_blocks.shape[0],Lbar.shape[-1],Fs_blocks.shape[-1]), dtype=Fs_blocks.dtype),
                          tf.eye(Lbar.shape[-1], batch_shape=(Fs_blocks.shape[0],), dtype=Fs_blocks.dtype)], 2)
    trans = tf.concat([trans_r1, trans_r2], 1)

    XUMstar = mm(mm(trans, XUbar, transpose_a=True), trans)

    tmp = tf.concat([mv(Hbar, cbar), tf.zeros((Fs_blocks.shape[0], Lbar.shape[-1]), dtype=Fs_blocks.dtype)], 1)
    tmp = mv(trans, mv(XUbar, rsbar - tmp), transpose_a=True)
    rsstar = tf.linalg.solve(XUMstar, tf.expand_dims(tmp,-1))[..., 0]

    rstar = rsstar[:,0:Fs_blocks.shape[-1]]
    sstar = rsstar[:,Fs_blocks.shape[-1]:]

    Xstar = XUMstar[:,0:Fs_blocks.shape[-1],0:Fs_blocks.shape[-1]]
    Mstar = XUMstar[:,0:Fs_blocks.shape[-1],Fs_blocks.shape[-1]:]
    Ustar = XUMstar[:,Fs_blocks.shape[-1]:,Fs_blocks.shape[-1]:]
    Hstar = tf.eye(Fs_blocks.shape[-1], batch_shape=(Fs_blocks.shape[0],), dtype=Fs_blocks.dtype)

    return Fstar, cstar, Lstar, Hstar, rstar, Xstar, Ustar, Mstar, sstar, Lambda, cbar, Lbar

def convertUX(cond_us, cond_xs, Lambda, cbar, Lbar, T):
    """ Convert condensed controls and states to uncondensed controls and states.

    Parameters:
        cond_us: Condensed controls
        cond_xs: Condensed states
        Lambda: Lambda matrix
        cbar: cbar vector
        Lbar: Lbar matrix
        T: Total time steps

    Returns:
        us: Uncondensed controls
        xs: Uncondensed states
    """

    xs_blocks = mv(Lambda, cond_xs[:-1,:]) + cbar + mv(Lbar, cond_us)
    xdim = cond_xs.shape[1]
    Tc = xs_blocks.shape[0]
    Nc = xs_blocks.shape[1] // xdim
    udim = cond_us.shape[1] // Nc
    xs = tf.reshape(xs_blocks, (Nc * Tc, xdim))
    us = tf.reshape(cond_us, (Nc * Tc, udim))
    xs = tf.concat([xs[0:T,:], tf.expand_dims(cond_xs[-1,:],0)], 0)
    us = us[0:T,:]

    return us, xs

