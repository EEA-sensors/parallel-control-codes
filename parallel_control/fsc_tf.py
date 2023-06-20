"""
TensorFlow-based optimal finite state control (FSC) via dynamic programming.
Both sequential and parallel versions.

@author: Simo Särkkä
"""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import math

###########################################################################
# Misc utilities and constants
###########################################################################

def fsc_np_to_tf(fsc, dtype=tf.float64):
    """ Convert a finite-state controller from NumPy to TensorFlow format.

    Parameters:
        fsc: Finite-state controller object.
        dtype: Data type (default tf.float64).

    Returns:
        fs: Tensor of state transition matrices.
        Ls: Tensor of cost matrices.
        LT: Terminal cost vector.
    """
    fs = tf.convert_to_tensor(fsc.f, dtype=tf.int32)
    Ls = tf.convert_to_tensor(fsc.L, dtype=dtype)
    LT = tf.convert_to_tensor(fsc.LT, dtype=dtype)
    return fs, Ls, LT

###########################################################################
# Sequential FSC
###########################################################################

@tf.function
def fsc_seq_backwardpass(fs, Ls, LT):
    """ Run sequential backward pass for a finite-state controller.

    Parameters:
        fs: Tensor of state transition matrices.
        Ls: Tensor of cost matrices.
        LT: Terminal cost vector.

    Returns:
        us: Tensor of optimal control actions.
        Vs: Tensor of optimal cost-to-go matrices.
    """
    def body(carry, inputs):
        f, L = inputs
        V, _ = carry

        Vu = L + tf.gather_nd(V,tf.expand_dims(f,-1))
        u = tf.argmin(Vu, axis=-1, output_type=fs.dtype)
        V = tf.reduce_min(Vu, axis=-1)

        return V, u

    uT = tf.zeros(fs.shape[-2], dtype=fs.dtype) # Dummy
    VT = LT

    Vs, us = tf.scan(body, (fs, Ls),
                     initializer=(VT, uT),
                     reverse=True)

    Vs = tf.concat([Vs, tf.expand_dims(VT, axis=0)], axis=0)

    return us, Vs

@tf.function
def fsc_seq_forwardpass(x0, fs, us):
    """ Run sequential forward pass for a finite-state controller.

    Parameters:
        x0: Initial state.
        fs: Tensor of state transition matrices.
        us: Tensor of optimal control actions.

    Returns:
        min_xs: Tensor of optimal state trajectories.
        min_us: Tensor of optimal controls.
    """
    def body(carry, inputs):
        f, u = inputs
        min_x, _ = carry

        min_u = u[min_x]
        min_x = f[min_x,min_u]

        return min_x, min_u

    min_xs, min_us = tf.scan(body,  (fs, us),
                             initializer=(x0, 0))

    min_xs = tf.concat([tf.expand_dims(x0, axis=0), min_xs], axis=0)

    return min_xs, min_us


###########################################################################
# Parallel FSC backward pass
###########################################################################

def fsc_par_backwardpass_init_most(fs, Ls):
    """ Initialize general parallel backward pass elements for a finite-state controller.

    Parameters:
        fs: Tensor of state transition matrices.
        Ls: Tensor of cost matrices.

    Returns:
        Vs: Initialized value conditional value matrix elements.
    """
    nb = fs.shape[0]
    nx = fs.shape[1]
    nu = fs.shape[2]

    Vus = 1e20 * tf.ones((nb, nx, nu, nx), dtype=Ls.dtype) # TODO: Replace the magic constant with a constant

    #This is a bit of magic to do this:
    # Vu[x,u,f[x,u]] = L[x,u]
    # V[x,xp] = min_u Vu[x,u,xp]
    bind  = tf.expand_dims(tf.tile(tf.reshape(tf.range(nb), (nb, 1, 1)), (1, nx, nu)),-1) # TODO: Comment on dimensions
    xind  = tf.expand_dims(tf.tile(tf.reshape(tf.range(nx), (1, nx, 1)), (nb, 1, nu)),-1)
    uind  = tf.expand_dims(tf.tile(tf.reshape(tf.range(nu), (1, 1, nu)), (nb, nx, 1)),-1)
    xpind = tf.expand_dims(fs,-1)
    indices = tf.concat([bind,xind,uind,xpind],-1)
    Vus = tf.tensor_scatter_nd_update(Vus, indices, Ls)
    Vs  = tf.reduce_min(Vus, axis=-2)

    return Vs

def fsc_par_backwardpass_init_last(LT):
    """ Initialize terminal parallel backward pass element for a finite-state controller.

    Parameters:
        LT: Terminal costs.

    Returns:
        VT: Initialized terminal conditional value function element.
    """
    VT = tf.tile(tf.expand_dims(LT,-1), (1,LT.shape[0]))
    return VT

def fsc_par_comb_V(Vij, Vjk):
    """ Combine two conditional value function elements for a finite-state controller.

    Parameters:
        Vij: Value function element for i -> j.
        Vjk: Value function element for j -> k.

    Returns:
        Vik: Value function element for i -> k.
    """
    Vik = tf.reduce_min(tf.expand_dims(Vij,-1) + tf.expand_dims(Vjk,1), axis=2)
    return Vik

def fsc_par_comb_V_rev(Vjk, Vij):
    """ Combine two conditional value function elements for a finite-state controller in reverse order.

    Parameters:
        Vjk: Value function element for j -> k.
        Vij: Value function element for i -> j.

    Returns:
        Vik: Value function element for i -> k.
    """
    return fsc_par_comb_V(Vij, Vjk)


@tf.function
def fsc_par_backwardpass(fs, Ls, LT, max_parallel=10000):
    """ Run parallel backward pass for a finite-state controller.

    Parameters:
        fs: Tensor of state transition matrices.
        Ls: Tensor of cost matrices.
        LT: Terminal cost vector.
        max_parallel: Maximum number of parallel operations for tfp.math.scan_associative.

    Returns:
        us: Tensor of optimal control actions.
        Vs: Tensor of optimal cost-to-go matrices.
    """
    elems_most = fsc_par_backwardpass_init_most(fs, Ls)
    elems_last = fsc_par_backwardpass_init_last(LT)

    elems = tf.concat([elems_most, tf.expand_dims(elems_last, 0)], axis=0)
    rev_elems = tf.reverse(elems, axis=[0])

    rev_elems = tfp.math.scan_associative(fsc_par_comb_V_rev,
                                          rev_elems,
                                          max_num_levels=math.ceil(math.log2(max_parallel)))

    Vs = tf.reverse(rev_elems, axis=[0])[..., 0]

    Vus = Ls + tf.gather_nd(Vs[1:], tf.expand_dims(fs, -1), batch_dims=1)
    us  = tf.argmin(Vus, axis=-1, output_type=fs.dtype)

    return us, Vs

###########################################################################
# Parallel FSC forward pass with decomposition of functions
###########################################################################

def fsc_par_forwardpass_init_first(x0, nx):
    """ Initialize first parallel forward pass element for a finite-state controller.

    Parameters:
        x0: Initial state.
        nx: Number of states.

    Returns:
        e: Initialized first element.
    """
    e = tf.repeat(x0, nx)
    return e

def fsc_par_forwardpass_init_most(fs, us):
    """ Initialize general parallel forward pass elements for a finite-state controller.

    Parameters:
        fs: Tensor of state transition matrices.
        us: Tensor of optimal control actions.

    Returns:
        es: Initialized elements.
    """
    nb  = fs.shape[0]
    nx  = fs.shape[1]

    xis  = tf.expand_dims(tf.tile(tf.reshape(tf.range(nx), (1,nx)), (nb, 1)), -1)
    uis  = tf.expand_dims(tf.gather_nd(us, xis, batch_dims=1), -1)
    xuis = tf.concat([xis,uis], -1)
    es   = tf.gather_nd(fs, xuis, batch_dims=1)

    return es

def fsc_par_comb_f(fij, fjk):
    """ Combine two parallel forward pass elements for a finite-state controller.

    Parameters:
        fij: Forward pass element for i -> j.
        fjk: Forward pass element for j -> k.

    Returns:
        fik: Forward pass element for i -> k.
    """
#    fik = tf.gather_nd(fjk, tf.expand_dims(fij, -1), batch_dims=1)
    fik = tf.gather(fjk, fij, axis=-1, batch_dims=1)
    return fik


@tf.function
def fsc_par_forwardpass(x0, fs, us, max_parallel=10000):
    """ Run parallel forward pass for a finite-state controller.

    Parameters:
        x0: Initial state.
        fs: Tensor of state transition matrices.
        us: Tensor of optimal control actions.

    Returns:
        min_us: Tensor of optimal control actions.
        min_xs: Tensor of optimal states.
    """
    first_elem = fsc_par_forwardpass_init_first(x0, fs.shape[1])
    most_elems = fsc_par_forwardpass_init_most(fs, us)

    elems = tf.concat([tf.expand_dims(first_elem, 0), most_elems], axis=0)

    elems = tfp.math.scan_associative(fsc_par_comb_f,
                                      elems,
                                      max_num_levels=math.ceil(math.log2(max_parallel)))

    min_xs = elems[...,0]
#    min_us = tf.gather_nd(us, tf.expand_dims(min_xs[:-1],-1), batch_dims=1)
    min_us = tf.gather(us, min_xs[:-1], axis=-1, batch_dims=1)

    return min_us, min_xs

###########################################################################
# Parallel FSC forward pass with value function combination
###########################################################################

def fsc_par_fwdbwdpass_init_first(x0, L0):
    """ Initialize first parallel forward/backward pass elements for a finite-state controller.

    Parameters:
        x0: Initial state.
        L0: Initial cost matrix.

    Returns:
        e: Initialized first element.
    """

    nx = L0.shape[0]

    V = 1e20 * tf.ones((nx, nx), dtype=L0.dtype) # TODO: Replace the magic constant with a constant or use "inf"
    ind1 = tf.expand_dims(tf.range(nx), -1)
    ind2 = tf.expand_dims(tf.repeat(x0, nx), -1)
    indices = tf.concat([ind1,ind2], -1)
    V = tf.tensor_scatter_nd_update(V, indices, tf.zeros(nx, dtype=L0.dtype))
    return V

def fsc_par_fwdbwdpass_init_most(fs, Ls):
    """ Initialize general parallel forward/backward pass elements for a finite-state controller.

    Parameters:
        fs: Tensor of state transition matrices.
        Ls: Tensor of cost matrices.

    Returns:
        Vs: Initialized elements.
    """
    Vs = fsc_par_backwardpass_init_most(fs, Ls)
    return Vs


@tf.function
def fsc_par_fwdbwdpass(x0, fs, Ls, us, Vs, max_parallel=10000):
    """ Run parallel forward/backward pass for a finite-state controller.

    Parameters:
        x0: Initial state.
        fs: Tensor of state transition matrices.
        Ls: Tensor of cost matrices.
        us: Tensor of optimal control actions.
        Vs: Tensor of optimal cost-to-go matrices.
        max_parallel: Maximum number of parallel operations for tfp.math.scan_associative.

    Returns:
        min_us: Tensor of optimal control actions.
        min_xs: Tensor of optimal states.
    """
    first_elem = fsc_par_fwdbwdpass_init_first(x0, Ls[0, ...])
    most_elems = fsc_par_fwdbwdpass_init_most(fs, Ls)

    elems = tf.concat([tf.expand_dims(first_elem, 0), most_elems], axis=0)

    elems = tfp.math.scan_associative(fsc_par_comb_V,
                                      elems,
                                      max_num_levels=math.ceil(math.log2(max_parallel)))

    Vfs = elems[:,0,:]
    min_xs = tf.argmin(Vfs + Vs, axis=-1, output_type=fs.dtype)
#    min_us = tf.gather_nd(us, tf.expand_dims(min_xs[:-1],-1), batch_dims=1)
    min_us = tf.gather(us, min_xs[:-1], axis=-1, batch_dims=1)

    return min_us, min_xs


