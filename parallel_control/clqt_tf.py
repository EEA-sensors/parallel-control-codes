"""
TensorFlow-based continuous-time Linear Quadratic Tracker (LQT) routines, both sequential and parallel.

@author: Simo Särkkä
"""

import tensorflow as tf
import tensorflow_probability as tfp
import math

import parallel_control.diffeq_tf as diffeq_tf
import parallel_control.lqt_tf as lqt_tf

# Abbreviations to make code more readable
mm = tf.linalg.matmul
mv = tf.linalg.matvec
top = tf.linalg.matrix_transpose

##############################################################################
#
# Packing and unpacking routines for ODE solvers
#
##############################################################################

def pack_abcej(A,b,C,eta,J):
    """ Pack batched conditional value function parameters into a single tensor.

    Parameters:
        A, b, C, eta, J: Batched conditional value function parameters

    Returns:
        x: Batched packed parameters
    """
    b = tf.expand_dims(b, -1)
    eta = tf.expand_dims(eta, -1)
    return tf.concat((A, C, J, b, eta), axis=-1)

def unpack_abcej(x):
    """ Unpack packed conditional value function parameters from a single batched tensor.

    Parameters:
        x: Batched packed parameters

    Returns:
        A, b, C, eta, J: Batched conditional value function parameters
    """
    n = x.shape[-2]
    A = x[..., :n]
    C = x[..., n:(2 * n)]
    J = x[..., (2 * n):(3 * n)]
    b = x[..., 3 * n]
    eta = x[..., 3 * n + 1]
    return A, b, C, eta, J

def pack_Psiphi(Psi,phi):
    """ Pack batched trajectory function parameters into a single tensor.

    Parameters:
        Psi, phi: Batched trajectory function parameters

    Returns:
        x: Batched packed parameters
    """
    phi = tf.expand_dims(phi, -1)
    return tf.concat((Psi, phi), axis=-1)

def unpack_Psiphi(x):
    """ Unpack packed trajectory function parameters from a single batched tensor.

    Parameters:
        x: Batched packed parameters

    Returns:
        Psi, phi: Batched trajectory function parameters
    """

    n = x.shape[-2]
    Psi = x[..., :n]
    phi = x[..., n]
    return Psi, phi

def pack_abc(A,b,C):
    """ Pack batched conditional value function parameters into a single tensor.

    Parameters:
        A, b, C: Batched conditional value function parameters

    Returns:
        x: Batched packed parameters
    """
    b = tf.expand_dims(b, -1)
    return tf.concat((A, C, b), axis=-1)

def unpack_abc(x):
    """ Unpack packed conditional value function parameters from a single batched tensor.

    Parameters:
        x: Batched packed parameters

    Returns:
        A, b, C: Batched conditional value function parameters
    """

    n = x.shape[-2]
    A = x[..., :n]
    C = x[..., n:(2 * n)]
    b = x[..., 2 * n]
    return A, b, C

def pack_Sv(S,v):
    """ Pack batched value function parameters into a single tensor.

    Parameters:
        S, v: Batched value function parameters

    Returns:
        x: Batched packed parameters
    """
    v = tf.expand_dims(v,-1)
    return tf.concat((S, v), axis=-1)

def unpack_Sv(x):
    """ Unpack packed value function parameters from a single batched tensor.

    Parameters:
        x: Batched packed parameters

    Returns:
        S, v: Batched value function parameters
    """
    n = x.shape[-2]
    S = x[..., :n]
    v = x[..., n]
    return S, v

###########################################################################
# Sequential CLQT routines
###########################################################################

def clqt_seq_backwardpass_defaults(steps, XT, HT, rT, T):
    """ CLQT sequential backward pass default values.

    Parameters:
        steps: Number of steps
        XT: Final state cost matrix
        HT: Final state output matrix
        rT: Final state output vector (i.e., reference trajectory)
        T: Final time

    Returns:
        dt, t0, ST, vT: Default values for dt, t0, ST, vT
    """

    # Cannot have batching here
    dt = T / tf.cast(steps, dtype=T.dtype)
    t0 = tf.constant(0.0, dtype=XT.dtype)
    ST = mm(HT, mm(XT, HT), transpose_a=True)
    vT = mv(HT, mv(XT, rT), transpose_a=True)

    return dt, t0, ST, vT

@tf.function
def clqt_seq_backwardpass(steps, dt, t0, ST, vT, F_f, L_f, X_f, U_f, c_f, H_f, r_f):
    """ CLQT sequential backward pass

    When computing the full solution, use dt, t0, ST, vT from clqt_seq_defaults().
    Supports batch index in t0, ST, vT such that returns shape (batch)x(steps)x(...)

        Parameters
        ----------
        steps : int
            Number of steps
        dt : tf.tensor
            Time step
        t0 : tf.tensor
            Initial time(s), should be vector of size B if batched
        ST : tf.tensor
            The final step matrix, can have batch dimension B in front
        vT : tf.tensor
            The final step vector, can have batch dimension B in front
        F_f, L_f, X_f, U_f, c_f, H_f, r_f : Functions with time argument
            The LQT parameter matrix and vector functions

        Returns
        -------
        Ss, vs, Kxs, ds : Tensors
            Value function's Ss, v (including initial and final), gains Kxs, biases ds
    """

    def f(x, t, p):
        """" ODE function for backward integration of S, v.

        Parameters:
            x: Packed S, v

        Returns:
            dx: Packed dS/dt, dv/dt
        """
        S, v = unpack_Sv(x)

        U = U_f(t)
        L = L_f(t)
        F = F_f(t)
        X = X_f(t)
        H = H_f(t)
        r = r_f(t)
        c = c_f(t)

        LULT = mm(L, tf.linalg.solve(U, top(L)))
        dS = -mm(F, S, transpose_a=True) - mm(S, F) + mm(mm(S, LULT), S) - mm(mm(H, X, transpose_a=True), H)
        dv = -mv(H, mv(X, r), transpose_a=True) + mv(S, c) - mv(F, v, transpose_a=True) + mv(S, mv(LULT, v))

        dx = pack_Sv(dS, dv)

        return dx

    def body(carry, inputs):
        """ Body for scan computing the sequential backward pass.

        Parameters:
            carry: (S, v, Kx, d)
            inputs: t

        Returns:
            carry: (S, v, Kx, d)
        """
        t = inputs
        S, v, _, _ = carry

        x = pack_Sv(S, v)
        x = diffeq_tf.rk4(f, -dt, x, t + dt)
        S, v = unpack_Sv(x)
        S = 0.5 * (S + top(S))

        U = U_f(t)
        L = L_f(t)
        ULT = tf.linalg.solve(U, top(L))

        Kx = mm(ULT, S)
        d = mv(ULT, v)

        return S, v, Kx, d

    t = tf.constant(0.0, dtype=t0.dtype)
    L = L_f(t)  # Just to measure the size
    KxT = tf.zeros((L.shape[-1], L.shape[-2]), dtype=L.dtype)  # Dummy value
    dT  = tf.zeros(L.shape[-1], dtype=L.dtype)                 # Dummy value
    Ts  = tf.range(0, steps, dtype=ST.dtype) * dt

    def singlePass(elem):
        """ Single pass of the sequential backward pass when using vectorized map over (t0, ST, vT).

        Parameters:
            elem: (t0, ST, vT)

        Returns:
            Ss, vs, Kxs, ds: Value function matrices, value function vectors, gains, biases
        """
        _t0, _ST, _vT = elem

        Ss, vs, Kxs, ds = tf.scan(body, _t0 + Ts,
                                  initializer=(_ST,_vT,KxT,dT),
                                  reverse=True)

        Ss = tf.concat([Ss, tf.expand_dims(_ST, axis=0)], axis=0)
        vs = tf.concat([vs, tf.expand_dims(_vT, axis=0)], axis=0)

        # This is a hack to force a concrete batch shape:
#        Kxs = tf.concat([Kxs, tf.expand_dims(KxT, axis=0)], axis=0)[:-1]
#        ds  = tf.concat([ds, tf.expand_dims(dT, axis=0)], axis=0)[:-1]

        return Ss, vs, Kxs, ds

    if len(t0.shape) > 0:
        Ss_all, vs_all, Kxs_all, ds_all = tf.vectorized_map(singlePass, (t0,ST,vT), fallback_to_while_loop=False)
    else:
        Ss_all, vs_all, Kxs_all, ds_all = singlePass((t0,ST,vT))

    return Ss_all, vs_all, Kxs_all, ds_all

@tf.function
def clqt_seq_forwardpass(x0, Kxs, ds, dt, t0, F_f, L_f, c_f, u_zoh=False):
    """
      Solve the forward trajectory using ZOH for Kx and optionally for u. Note that when u_zoh=False, then x is more
      accurate, but we cannot exactly get x when controlling it with the returned us.
      Supports batch index in x0, Kxs, ds, t0 such that returns shape (batch)x(steps)x(...)

    Parameters
    ----------
    x0 : tf.tensor
        Initial time(s), should be of size BxD if batched
    Kxs : tf.tensor
        Gains for the whole time span with similar batching (if used)
    ds : tf.tensor
        Biases for the whole time span with similar batching (if used)
    dt : tf.tensor
        Time step
    t0 : tf.tensor
        Initial time(s), should be vector of size B if batched
    F_f, L_f, c_f : Functions of time
        Dynamic model matrix and vector functions
    u_zoh : bool
        use ZOH also for u when computing x, default False

    Returns
    -------
    us, xs : tf.tensor
        List of inputs, list of states (initial and final)
    """

    def f_zoh(x, t, p):
        """ ZOH forward dynamics function.

        Parameters:
            x: State
            t: Time
            p: Parameter containing the control input

        Returns:
            dx: State time derivative
        """
        u = p

        F = F_f(t)
        L = L_f(t)
        c = c_f(t)
        dx = mv(F, x) + mv(L, u) + c
        return dx

    def f_no_zoh(x, t, p):
        """ No-ZOH forward dynamics function.

        Parameters:
            x: State
            t: Time
            p: Parameters (Kx, d)

        Returns:
            dx: State time derivative
        """
        Kx, d = p

        F = F_f(t)
        L = L_f(t)
        c = c_f(t)
        u = -mv(Kx, x) + d
        dx = mv(F, x) + mv(L, u) + c
        return dx

    def body(carry, inputs):
        """ Body for scan computing the forward pass.

        Parameters:
            carry: (x, u)
            inputs: (t, Kx, d)

        Returns:
            carry: (x, u)
        """
        t, Kx, d = inputs
        x, _ = carry

        u = -mv(Kx, x) + d
        if u_zoh:
            x = diffeq_tf.rk4(f_zoh, dt, x, t, u)
        else:
            x = diffeq_tf.rk4(f_no_zoh, dt, x, t, (Kx, d))

        return x, u

    steps = ds.shape[-2]
#    t = tf.constant(0.0, dtype=t0.dtype)
#    L = L_f(t)  # Just to measure the size
#    u0 = tf.zeros_like(L[0,:])
    Ts = tf.range(0, steps, dtype=ds.dtype) * dt

    def singlePass(elem):
        """ Single pass of the sequential forward pass when using vectorized map over (t0, x0, Kxs, ds).

        Parameters:
            elem: (t0, x0, Kxs, ds)

        Returns:
            xs, us: State and input trajectories
        """
        _t0, _x0, _Kxs, _ds = elem

        u0 = _ds[0, :]
        xs, us = tf.scan(body, (_t0 + Ts, _Kxs, _ds),
                         initializer=(_x0, u0),
                         reverse=False)

        xs = tf.concat([tf.expand_dims(_x0, axis=0), xs], axis=0)

        # This is a hack to force a concrete batch shape:
#        us = tf.concat([us, tf.expand_dims(u0, axis=0)], axis=0)[:-1]

        return xs, us

    if len(t0.shape) > 0:
        xs_all, us_all = tf.vectorized_map(singlePass, (t0, x0, Kxs, ds), fallback_to_while_loop=False)
    else:
        xs_all, us_all = singlePass((t0, x0, Kxs, ds))

    return xs_all, us_all


def clqt_seq_fwdbwdpass_defaults(x0, steps, T):
    """ Default initial values for the forward-backward pass.

    Parameters:
        x0: Initial state
        steps: Number of steps
        T: Time span

    Returns:
        dt, t0, A0, b0, C0: Default values
    """

    # Cannot have batching here
    dt = T / tf.cast(steps, dtype=T.dtype)
    t0 = tf.constant(0.0, dtype=x0.dtype)
    A0 = tf.zeros((x0.shape[0], x0.shape[0]), dtype=x0.dtype)
    b0 = x0
    C0 = tf.zeros_like(A0)
    return dt, t0, A0, b0, C0

@tf.function
def clqt_seq_fwdbwdpass(steps, dt, t0, A0, b0, C0, F_f, L_f, X_f, U_f, c_f, H_f, r_f):
    """
      Solve forward value function parameters A, b, C forward

    Parameters
    ----------
    steps : int
        Number of steps
    dt : tf.tensor
        Time step
    t0 : tf.tensor
        Initial time, can be batched
    A0 : tf.tensor
        Initial A, can be batched
    b0 : tf.tensor
        Initial b, can be batched
    C0 : tf.tensor
        Initial C, can be batched
    F_f, L_f, X_f, U_f, c_f, H_f, r_f : Function of time
        LQT parameters

    Returns
    -------
    As, bs, Cs : tf.tensor
        Containing (steps+1)xDxD or (batch)x(steps+1)xDxD
    """

    def f(x, t, p):
        """" ODE function for forward integration of A, b, C.

        Parameters:
            x: Packed (A, b, C)
            t: Time
            p: Unused

        Returns:
            dx: Packed (dA/dt, db/dt, dC/dt)
        """

        A, b, C = unpack_abc(x)

        U = U_f(t)
        L = L_f(t)
        F = F_f(t)
        X = X_f(t)
        H = H_f(t)
        r = r_f(t)
        c = c_f(t)

        ULT = tf.linalg.solve(U, top(L))
        LULT = mm(L, ULT)

        dA = mm(F, A) - mm(mm(mm(mm(C, H, transpose_b=True), X), H), A)
        db = mv(F, b) + mv(C, mv(H, mv(X, r), transpose_a=True)) - mv(C, mv(H, mv(X, mv(H, b)), transpose_a=True)) + c
        dC = mm(F, C) - mm(mm(mm(mm(C, H, transpose_b=True), X), H), C) + mm(C, F, transpose_b=True) + LULT

        dx = pack_abc(dA,db,dC)

        return dx

    def body(carry, inputs):
        """ Body of the scan function for forward integration of A, b, C.

        Parameters:
            carry: (A, b, C)
            inputs: Time t

        Returns:
            carry: (A, b, C)
        """

        t = inputs
        A, b, C = carry

        x = pack_abc(A, b, C)
        x = diffeq_tf.rk4(f, dt, x, t)
        A, b, C = unpack_abc(x)
        C = 0.5 * (C + top(C))

        return A, b, C

    Ts = tf.range(0, steps, dtype=A0.dtype) * dt

    def singlePass(elem):
        """ Single pass of the sequential forward pass when using vectorized map over (t0, A0, b0, C0).

        Parameters:
            elem: (t0, A0, b0, C0)

        Returns:
            As, bs, Cs: Integrated values
        """
        _t0, _A, _b, _C = elem

        As, bs, Cs = tf.scan(body, _t0 + Ts,
                             initializer=(_A, _b, _C),
                             reverse=False)

        As = tf.concat([tf.expand_dims(_A, axis=0), As], axis=0)
        bs = tf.concat([tf.expand_dims(_b, axis=0), bs], axis=0)
        Cs = tf.concat([tf.expand_dims(_C, axis=0), Cs], axis=0)

        return As, bs, Cs

    if len(t0.shape) > 0:
        As_all, bs_all, Cs_all = tf.vectorized_map(singlePass, (t0, A0, b0, C0), fallback_to_while_loop=False)
    else:
        As_all, bs_all, Cs_all = singlePass((t0, A0, b0, C0))

    return As_all, bs_all, Cs_all

@tf.function
def clqt_combine_fwdbwd(Kxs, ds, Ss, vs, As, bs, Cs):
    """ Combine forward and backward pass to compute controls and states.

    Parameters:
        Kxs: Control gains
        ds: Control offsets
        Ss: Backward value function matrices
        vs: Backward value function vectors
        As, bs, Cs: Forward integrated value function parameters

    Returns:
        xs: States
        us: Controls
    """

    # As are retained as parameter for alternative way of computing these
    xs = tf.linalg.solve(tf.eye(Cs.shape[-1], dtype=Kxs.dtype) + mm(Cs, Ss), tf.expand_dims(bs + mv(Cs, vs), -1))[..., 0]
    us = -mv(Kxs, xs[..., :-1, :]) + ds
    return xs, us

###########################################################################
# Parallel computation of gains and value functions backwards
###########################################################################

def par_backwardpass_defaults(blocks, steps, XT, HT, rT, T):
    """ Default values for parallel backward pass.

    Parameters:
        blocks: Number of blocks
        steps: Number of steps per block
        XT: Terminal state cost
        HT: Terminal state output matrix
        rT: Terminal state output vector
        T: Terminal time

    Returns:
        dt, t0, ST, vT: Default values
    """
    dt = T / tf.cast(steps * blocks, dtype=T.dtype)
    t0 = tf.constant(0.0, dtype=XT.dtype)
    ST = mm(HT, mm(XT, HT), transpose_a=True)
    vT = mv(HT, mv(XT, rT), transpose_a=True)

    return dt, t0, ST, vT

@tf.function
def par_backwardpass_init_bw(blocks, steps, dt, t0, ST, vT, F_f, L_f, X_f, U_f, c_f, H_f, r_f):
    """ Initialize backward pass for parallel computation of gains and value functions using backward integration.

    Parameters:
        blocks: Number of blocks
        steps: Number of steps per block
        dt: Time step
        t0: Initial time
        ST: Terminal value function matrix
        vT: Terminal value function vector
        F_f, L_f, X_f, U_f, c_f, H_f, r_f : Functions of time for LQT parameters

    Returns:
        As, bs, Cs, etas, Js : Initial element values for A, b, C, eta, J
    """

    def f(x, t, p):
        """" ODE function for backward integration of A, b, C, eta, J.

        Parameters:
            x: Packed (A, b, C, eta, J)
            t: Time
            p: Unused

        Returns:
            dx: Time derivative of x
        """
        A, b, C, eta, J = unpack_abcej(x)

        U = U_f(t)
        L = L_f(t)
        F = F_f(t)
        X = X_f(t)
        H = H_f(t)
        r = r_f(t)
        c = c_f(t)

        LULT = mm(L, tf.linalg.solve(U, top(L)))

        dA = mm(mm(A, LULT), J) - mm(A, F)
        db = -mv(A, mv(LULT, eta)) - mv(A, c)
        dC = -mm(mm(A, LULT), A, transpose_b=True)
        deta = -mv(H, mv(X, r), transpose_a=True) + mv(J, mv(LULT, eta)) - mv(F, eta, transpose_a=True) + mv(J, c)
        dJ = -mm(mm(H, X, transpose_a=True), H) + mm(mm(J, LULT), J) - mm(J, F) - mm(F, J, transpose_a=True)

        dx = pack_abcej(dA,db,dC,deta,dJ)

        return dx

    def body(carry, inputs):
        """ Body function for scan over time for backward integration of A, b, C, eta, J.

        Parameters:
            carry: (A, b, C, eta, J)
            inputs: Time

        Returns:
            carry: (A, b, C, eta, J)
        """
        t = inputs
        A, b, C, eta, J = carry

        x = pack_abcej(A, b, C, eta, J)
        x = diffeq_tf.rk4(f, -dt, x, t + dt)
        A, b, C, eta, J = unpack_abcej(x)
        C = 0.5 * (C + top(C))
        J = 0.5 * (J + top(J))

        return A, b, C, eta, J


    A0 = tf.eye(ST.shape[0], dtype=ST.dtype)
    b0 = tf.zeros_like(A0[:, 0])
    C0 = tf.zeros_like(A0)
    eta0 = tf.zeros_like(A0[:, 0])
    J0 = tf.zeros_like(A0)
    Ts = tf.range(0, steps, dtype=A0.dtype) * dt

    def singlePass(elem):
        """ Single pass of backward integration of A, b, C, eta, J using vectorized map over t0.

        Parameters:
            elem: Initial time t0

        Returns:
            As, bs, Cs, etas, Js : Integrated values of A, b, C, eta, J
        """
        _t0 = elem

        As, bs, Cs, etas, Js = tf.scan(body, _t0 + Ts,
                                       initializer=(A0, b0, C0, eta0, J0),
                                       reverse=True)

        # XXX: This is a hack to force a concrete batch shape (for associative_scan):
        As = tf.concat([As, tf.expand_dims(A0, axis=0)], axis=0)[:-1]
        bs = tf.concat([bs, tf.expand_dims(b0, axis=0)], axis=0)[:-1]
        Cs = tf.concat([Cs, tf.expand_dims(C0, axis=0)], axis=0)[:-1]
        etas = tf.concat([etas, tf.expand_dims(eta0, axis=0)], axis=0)[:-1]
        Js = tf.concat([Js, tf.expand_dims(J0, axis=0)], axis=0)[:-1]

        # We store only the last (leading) value from the integration
        A1 = As[0, :, :]
        b1 = bs[0, :]
        C1 = Cs[0, :, :]
        eta1 = etas[0, :]
        J1 = Js[0, :, :]

        return A1, b1, C1, eta1, J1

    t0s = t0 + tf.range(0, blocks, dtype=A0.dtype) * steps * dt
    As, bs, Cs, etas, Js = tf.vectorized_map(singlePass, t0s, fallback_to_while_loop=False)

    AT = tf.zeros_like(ST)
    bT = b0
    CT = C0
    etaT = vT
    JT = ST

    As = tf.concat([As, tf.expand_dims(AT, axis=0)], axis=0)
    bs = tf.concat([bs, tf.expand_dims(bT, axis=0)], axis=0)
    Cs = tf.concat([Cs, tf.expand_dims(CT, axis=0)], axis=0)
    etas = tf.concat([etas, tf.expand_dims(etaT, axis=0)], axis=0)
    Js = tf.concat([Js, tf.expand_dims(JT, axis=0)], axis=0)

    return As, bs, Cs, etas, Js

@tf.function
def par_backwardpass_init_fw(blocks, steps, dt, t0, ST, vT, F_f, L_f, X_f, U_f, c_f, H_f, r_f):
    """ Initialize backward pass for parallel computation of gains and value functions using forward integration.

        blocks: Number of blocks
        steps: Number of steps per block
        dt: Time step
        t0: Initial time
        ST: Terminal value function matrix
        vT: Terminal value function vector
        F_f, L_f, X_f, U_f, c_f, H_f, r_f : Functions of time for LQT parameters

    Returns:
        As, bs, Cs, etas, Js : Initial element values for A, b, C, eta, J
    """

    def f(x, t, p):
        """" ODE function for forward integration of A, b, C, eta, J.

        Parameters:
            x: Packed (A, b, C, eta, J)
            t: Time
            p: Unused

        Returns:
            dx: Time derivative of x
        """
        A, b, C, eta, J = unpack_abcej(x)

        U = U_f(t)
        L = L_f(t)
        F = F_f(t)
        X = X_f(t)
        H = H_f(t)
        r = r_f(t)
        c = c_f(t)

        LULT = mm(L, tf.linalg.solve(U, top(L)))

        dA = mm(F, A) - mm(mm(mm(mm(C, H, transpose_b=True), X), H), A)
        db = mv(F, b) + mv(C, mv(H, mv(X, r), transpose_a=True)) - mv(C, mv(H, mv(X, mv(H, b)), transpose_a=True)) + c
        dC = mm(F, C) - mm(mm(mm(mm(C, H, transpose_b=True), X), H), C) + mm(C, F, transpose_b=True) + LULT

        deta = mv(A, mv(H, mv(X, r), transpose_a=True), transpose_a=True) - mv(A, mv(H, mv(X, mv(H, b)), transpose_a=True), transpose_a=True)
        dJ = mm(mm(mm(mm(A, H, transpose_a=True, transpose_b=True), X), H), A)

        dx = pack_abcej(dA,db,dC,deta,dJ)

        return dx

    def body(carry, inputs):
        """ Body function for scan over time for forward integration of A, b, C, eta, J.

        Parameters:
            carry: (A, b, C, eta, J)
            inputs: Time

        Returns:
            carry: (A, b, C, eta, J)
        """
        t = inputs
        A, b, C, eta, J = carry

        x = pack_abcej(A, b, C, eta, J)
        x = diffeq_tf.rk4(f, dt, x, t)
        A, b, C, eta, J = unpack_abcej(x)
        C = 0.5 * (C + top(C))
        J = 0.5 * (J + top(J))

        return A, b, C, eta, J


    A0 = tf.eye(ST.shape[0], dtype=ST.dtype)
    b0 = tf.zeros_like(A0[:, 0])
    C0 = tf.zeros_like(A0)
    eta0 = tf.zeros_like(A0[:, 0])
    J0 = tf.zeros_like(A0)
    Ts = tf.range(0, steps, dtype=A0.dtype) * dt

    def singlePass(elem):
        """ Single pass of forward integration of A, b, C, eta, J using vectorized map over t0.

        Parameters:
            elem: Initial time t0

        Returns:
            As, bs, Cs, etas, Js : Integrated values of A, b, C, eta, J
        """
        _t0 = elem

        As, bs, Cs, etas, Js = tf.scan(body, _t0 + Ts,
                                       initializer=(A0, b0, C0, eta0, J0),
                                       reverse=False)

        # XXX: This is a hack to force a concrete batch shape (for associative_scan):
        As = tf.concat([As, tf.expand_dims(A0, axis=0)], axis=0)[:-1]
        bs = tf.concat([bs, tf.expand_dims(b0, axis=0)], axis=0)[:-1]
        Cs = tf.concat([Cs, tf.expand_dims(C0, axis=0)], axis=0)[:-1]
        etas = tf.concat([etas, tf.expand_dims(eta0, axis=0)], axis=0)[:-1]
        Js = tf.concat([Js, tf.expand_dims(J0, axis=0)], axis=0)[:-1]

        # We store only the last value from the integration
        A1 = As[-1, :, :]
        b1 = bs[-1, :]
        C1 = Cs[-1, :, :]
        eta1 = etas[-1, :]
        J1 = Js[-1, :, :]

        return A1, b1, C1, eta1, J1

    t0s = t0 + tf.range(0, blocks, dtype=A0.dtype) * steps * dt
    As, bs, Cs, etas, Js = tf.vectorized_map(singlePass, t0s, fallback_to_while_loop=False)

    AT = tf.zeros_like(ST)
    bT = b0
    CT = C0
    etaT = vT
    JT = ST

    As = tf.concat([As, tf.expand_dims(AT, axis=0)], axis=0)
    bs = tf.concat([bs, tf.expand_dims(bT, axis=0)], axis=0)
    Cs = tf.concat([Cs, tf.expand_dims(CT, axis=0)], axis=0)
    etas = tf.concat([etas, tf.expand_dims(etaT, axis=0)], axis=0)
    Js = tf.concat([Js, tf.expand_dims(JT, axis=0)], axis=0)

    return As, bs, Cs, etas, Js

@tf.function
def clqt_par_backwardpass_extract(etas, Js, steps, dt, t0, F_f, L_f, X_f, U_f, c_f, H_f, r_f):
    """ Extract the value function and control law from the backward pass of the parallel CLQT algorithm.

    Parameters:
        etas: The eta parameters from the backward pass of the CLQT algorithm.
        Js: The J parameters from the backward pass of the CLQT algorithm.
        steps: The number of steps in the block backward pass.
        dt: The time step size.
        t0: The initial time.
        F_f, L_f, X_f, U_f, c_f, H_f, r_f : Functions of time for LQT parameters

    Returns:
        Ss, vs, Kxs, ds: Value function matrix and vector, control gain matrix and vector.
    """
    blocks = tf.shape(Js)[0] - 1
    t0s = t0 + tf.range(0, blocks, dtype=Js.dtype) * steps * dt

    Ss, vs, Kxs, ds = clqt_seq_backwardpass(steps, dt, t0s, Js[1:, ...], etas[1:, ...], F_f, L_f, X_f, U_f, c_f, H_f, r_f)

    Ss = Ss[:, 1:, :, :]
    vs = vs[:, 1:, :]

#    Ss = tf.reshape(Ss, shape=(Ss.shape[0] * Ss.shape[1], Ss.shape[2], Ss.shape[3]))
#    vs = tf.reshape(vs, shape=(vs.shape[0] * vs.shape[1], vs.shape[2]))
#    Kxs = tf.reshape(Kxs, shape=(Kxs.shape[0] * Kxs.shape[1], Kxs.shape[2], Kxs.shape[3]))
#    ds = tf.reshape(ds, shape=(ds.shape[0] * ds.shape[1], ds.shape[2]))

    Ss_shape = tf.shape(Ss)
    Ss = tf.reshape(Ss, shape=(Ss_shape[0] * Ss_shape[1], Ss_shape[2], Ss_shape[3]))
    vs_shape = tf.shape(vs)
    vs = tf.reshape(vs, shape=(vs_shape[0] * vs_shape[1], vs_shape[2]))
    Kxs_shape = tf.shape(Kxs)
    Kxs = tf.reshape(Kxs, shape=(Kxs_shape[0] * Kxs_shape[1], Kxs_shape[2], Kxs_shape[3]))
    ds_shape = tf.shape(ds)
    ds = tf.reshape(ds, shape=(ds_shape[0] * ds_shape[1], ds_shape[2]))

    S0 = Js[0, :, :]
    v0 = etas[0, :]

    Ss = tf.concat([tf.expand_dims(S0, axis=0), Ss], axis=0)
    vs = tf.concat([tf.expand_dims(v0, axis=0), vs], axis=0)

    return Ss, vs, Kxs, ds

@tf.function
def clqt_par_backwardpass(blocks, steps, dt, t0, ST, vT, F_f, L_f, X_f, U_f, c_f, H_f, r_f, forward=False, max_parallel=10000):
    """ Perform the backward pass of the CLQT algorithm in parallel.

    Parameters:
        blocks: The number of blocks in the backward pass.
        steps: The number of steps per block.
        dt: The time step size.
        t0: The initial time.
        ST: The final value function matrix.
        vT: The final value function vector.
        F_f, L_f, X_f, U_f, c_f, H_f, r_f : Functions of time for LQT parameters
        forward: Whether to integrated forwards in init (default: False).
        max_parallel: Parameter for tfp.math.scan_associative (default: 10000).

    Returns:
        As, bs, Cs, etas, Js: The value function elements from the backward pass of the CLQT algorithm.
    """

    if forward:
        As, bs, Cs, etas, Js = par_backwardpass_init_fw(blocks, steps, dt, t0, ST, vT, F_f, L_f, X_f, U_f, c_f, H_f, r_f)
    else:
        As, bs, Cs, etas, Js = par_backwardpass_init_bw(blocks, steps, dt, t0, ST, vT, F_f, L_f, X_f, U_f, c_f, H_f, r_f)

    # XXX: It would be better to have a function for this in discrete LQT:
    As_rev = tf.reverse(As, axis=[0])
    bs_rev = tf.reverse(bs, axis=[0])
    Cs_rev = tf.reverse(Cs, axis=[0])
    etas_rev = tf.reverse(etas, axis=[0])
    Js_rev = tf.reverse(Js, axis=[0])

    rev_elems = (As_rev, bs_rev, Cs_rev, etas_rev, Js_rev)

    rev_elems = tfp.math.scan_associative(lqt_tf.lqt_par_comb_V_rev,
                                          rev_elems,
                                          max_num_levels=max(math.ceil(math.log2(max_parallel)),1))

    etas = tf.reverse(rev_elems[3], axis=[0])
    Js = tf.reverse(rev_elems[4], axis=[0])

    # returns Ss, vs, Kxs, ds
    return clqt_par_backwardpass_extract(etas, Js, steps, dt, t0, F_f, L_f, X_f, U_f, c_f, H_f, r_f)

###########################################################################
# Parallel computation of states and controls forward
###########################################################################

@tf.function
def clqt_par_forwardpass_init_fw(blocks, steps, x0, Kxs, ds, dt, t0, F_f, L_f, c_f):
    """ Initialize the parallel forward pass using forward integration.

    Parameters:
        blocks: The number of blocks.
        steps: The number of steps per block.
        x0: The initial state.
        Kxs: The control gains.
        ds: The control offsets.
        dt: The time step size.
        t0: The initial time.
        F_f, L_f, c_f: Functions of time for LQT parameters.

    Returns:
        Psis, phis: The initial elements for the parallel forward pass.
    """

    def f(x,t,Kx_d):
        """ ODE function for the parallel forward pass Psi, phi using forward integration.

        Parameters:
            x: Packed (Psi, phi).
            t: The time.
            Kx_d: Parameter containing the control gains and offsets as (Kx, d).

        Returns:
            dx: The time derivative of x.
        """
        Kx, d = Kx_d

        Psi, phi = unpack_Psiphi(x)
        L = L_f(t)
        tF = F_f(t) - mm(L, Kx)
        tc = c_f(t) + mv(L, d)
        dPsi = mm(tF, Psi)
        dphi = mv(tF, phi) + tc
        dx = pack_Psiphi(dPsi, dphi)
        return dx

    def body(carry, inputs):
        """ Body function for the parallel forward pass for Psi, phi using forward integration.

        Parameters:
            carry: The previous Psi, phi.
            inputs: The current time, control gains, and offsets.

        Returns:
            Psi, phi: The updated Psi, phi.
        """
        t, Kx, d = inputs
        Psi, phi = carry

        x = pack_Psiphi(Psi, phi)
        x = diffeq_tf.rk4(f, dt, x, t, param=(Kx, d))
        Psi, phi = unpack_Psiphi(x)

        return Psi, phi

    Kxs = tf.reshape(Kxs, (blocks, steps, tf.shape(Kxs)[-2], tf.shape(Kxs)[-1]))
    ds = tf.reshape(ds, (blocks, steps, tf.shape(ds)[-1]))
    Psi0 = tf.eye(tf.shape(x0)[-1], dtype=x0.dtype)
    phi0 = tf.zeros_like(x0)
    Ts = tf.range(0, steps, dtype=x0.dtype) * dt

    def singlePass(elem):
        """ Single pass of the parallel forward (fw-int) pass for Psi, phi for vectorization over t0, Kx, ds.

        Parameters:
            elem: The current initial time, control gains, and offsets (t0, Kx, ds).

        Returns:
            Phis, psis: The computed Phis, psis for the parameters.
        """

        _t0, _Kxs, _ds = elem

        Psis, phis = tf.scan(body, (_t0 + Ts, _Kxs, _ds),
                             initializer=(Psi0, phi0),
                             reverse=False)

        # XXX: This is a hack to force a concrete batch shape (for associative_scan):
        Psis = tf.concat([Psis, tf.expand_dims(Psi0, axis=0)], axis=0)[:-1]
        phis = tf.concat([phis, tf.expand_dims(phi0, axis=0)], axis=0)[:-1]

        # We store only the last value from the integration
        Psi1 = Psis[-1, :, :]
        phi1 = phis[-1, :]

        return Psi1, phi1

    t0s = t0 + tf.range(0, blocks, dtype=x0.dtype) * steps * dt
    Psis, phis = tf.vectorized_map(singlePass, (t0s, Kxs, ds), fallback_to_while_loop=False)

    Psi1 = tf.eye(x0.shape[-1], dtype=x0.dtype)
    phi1 = x0

    Psis = tf.concat([tf.expand_dims(Psi1, axis=0), Psis], axis=0)
    phis = tf.concat([tf.expand_dims(phi1, axis=0), phis], axis=0)

    return Psis, phis

@tf.function
def clqt_par_forwardpass_init_bw(blocks, steps, x0, Kxs, ds, dt, t0, F_f, L_f, c_f):
    """ Initialize the parallel forward pass using backward integration.

    Parameters:
        blocks: The number of blocks.
        steps: The number of steps per block.
        x0: The initial state.
        Kxs: The control gains.
        ds: The control offsets.
        dt: The time step size.
        t0: The initial time.
        F_f, L_f, c_f: Functions of time for LQT parameters.

    Returns:
        Psis, phis: The initial elements for the parallel forward pass.
    """

    def f(x,t,Kx_d):
        """ ODE function for the parallel forward pass Psi, phi using backward integration.

        Parameters:
            x: Packed (Psi, phi).
            t: The time.
            Kx_d: Parameter containing the control gains and offsets as (Kx, d).

        Returns:
            dx: The time derivative of x.
        """
        Kx, d = Kx_d

        Psi, phi = unpack_Psiphi(x)
        L = L_f(t)
        tF = F_f(t) - mm(L, Kx)
        tc = c_f(t) + mv(L, d)
        dPsi = -mm(Psi, tF)
        dphi = -mv(Psi, tc)
        dx = pack_Psiphi(dPsi, dphi)
        return dx

    def body(carry, inputs):
        """ Body function for the parallel forward pass for Psi, phi using backward integration.

        Parameters:
            carry: The previous Psi, phi.
            inputs: The current time, control gains, and offsets.

        Returns:
            Psi, phi: The updated Psi, phi.
        """
        t, Kx, d = inputs
        Psi, phi = carry

        x = pack_Psiphi(Psi, phi)
        x = diffeq_tf.rk4(f, -dt, x, t + dt, param=(Kx, d))
        Psi, phi = unpack_Psiphi(x)

        return Psi, phi

    Kxs = tf.reshape(Kxs, (blocks, steps, tf.shape(Kxs)[-2], tf.shape(Kxs)[-1]))
    ds = tf.reshape(ds, (blocks, steps, tf.shape(ds)[-1]))

    Psi0 = tf.eye(tf.shape(x0)[-1], dtype=x0.dtype)
    phi0 = tf.zeros_like(x0)
    Ts = tf.range(0, steps, dtype=x0.dtype) * dt

    def singlePass(elem):
        """ Single pass of the parallel forward (bw-int) pass for Psi, phi for vectorization over t0, Kx, ds.

        Parameters:
            elem: The current initial time, control gains, and offsets (t0, Kx, ds).

        Returns:
            Phis, psis: The computed Phis, psis for the parameters.
        """
        _t0, _Kxs, _ds = elem

        Psis, phis = tf.scan(body, (_t0 + Ts, _Kxs, _ds),
                             initializer=(Psi0, phi0),
                             reverse=True)

        # XXX: This is a hack to force a concrete batch shape (for associative_scan):
        Psis = tf.concat([Psis, tf.expand_dims(Psi0, axis=0)], axis=0)[:-1]
        phis = tf.concat([phis, tf.expand_dims(phi0, axis=0)], axis=0)[:-1]

        # We store only the first value from the integration
        Psi1 = Psis[0, :, :]
        phi1 = phis[0, :]

        return Psi1, phi1

    t0s = t0 + tf.range(0, blocks, dtype=x0.dtype) * steps * dt
    Psis, phis = tf.vectorized_map(singlePass, (t0s, Kxs, ds), fallback_to_while_loop=False)

    Psi1 = tf.eye(x0.shape[-1], dtype=x0.dtype)
    phi1 = x0

    Psis = tf.concat([tf.expand_dims(Psi1, axis=0), Psis], axis=0)
    phis = tf.concat([tf.expand_dims(phi1, axis=0), phis], axis=0)

    return Psis, phis

@tf.function
def clqt_par_forwardpass_extract(Kxs, ds, Psis, phis, steps, dt, t0, F_f, L_f, c_f, u_zoh=False):
    """ Extract the control and state trajectories from the parallel forward pass.

    Parameters:
        Kxs: The control gains.
        ds: The control offsets.
        Psis, phis: The parallel forward pass elements.
        steps: The number of steps per block.
        dt: The time step size.
        t0: The initial time.
        F_f, L_f, c_f: Functions of time for LQT parameters.
        u_zoh: Whether to use zero-order hold for the control.

    Returns:
        xs, us: The state and control trajectories.
    """

    blocks = tf.shape(Psis)[0] - 1
    t0s = t0 + tf.range(0, blocks, dtype=Psis.dtype) * steps * dt

    Kxs = tf.reshape(Kxs, (blocks, steps, tf.shape(Kxs)[-2], tf.shape(Kxs)[-1]))
    ds = tf.reshape(ds, (blocks, steps, tf.shape(ds)[-1]))

    xs, us = clqt_seq_forwardpass(phis[:-1, ...], Kxs, ds, dt, t0s, F_f, L_f, c_f, u_zoh=u_zoh)

    xs = xs[:, :-1, :]

    xs_shape = tf.shape(xs)
    xs = tf.reshape(xs, shape=(xs_shape[0] * xs_shape[1], xs_shape[2]))
    us_shape = tf.shape(us)
    us = tf.reshape(us, shape=(us_shape[0] * us_shape[1], us_shape[2]))

    xT = phis[-1, :]

    xs = tf.concat([xs, tf.expand_dims(xT, axis=0)], axis=0)

    return xs, us


@tf.function
def clqt_par_forwardpass(blocks, steps, x0, Kxs, ds, dt, t0, F_f, L_f, c_f, forward=True, u_zoh=False, max_parallel=10000):
    """ Compute the parallel forward pass.

    Parameters:
        blocks: The number of blocks.
        steps: The number of steps per block.
        x0: The initial state.
        Kxs: The control gains.
        ds: The control offsets.
        dt: The time step size.
        t0: The initial time.
        F_f, L_f, c_f: Functions of time for LQT parameters.
        forward: Whether to initialize forward or backward.
        u_zoh: Whether to use zero-order hold for the control.
        max_parallel: The maximum number of parallel iterations for tfp.math.scan_associative.

    Returns:
        xs, us: The state and control trajectories.
    """

    if forward:
        Psis, phis = clqt_par_forwardpass_init_fw(blocks, steps, x0, Kxs, ds, dt, t0, F_f, L_f, c_f)
    else:
        Psis, phis = clqt_par_forwardpass_init_bw(blocks, steps, x0, Kxs, ds, dt, t0, F_f, L_f, c_f)

    elems = tfp.math.scan_associative(lqt_tf.lqt_par_comb_f,
                                      (Psis, phis),
                                      max_num_levels=max(math.ceil(math.log2(max_parallel)),1))

    Psis = elems[0]
    phis = elems[1]

    # Returns xs, us
    return clqt_par_forwardpass_extract(Kxs, ds, Psis, phis, steps, dt, t0, F_f, L_f, c_f, u_zoh=u_zoh)

###########################################################################
# Parallel computation of value functions forward
###########################################################################

def clqt_par_fwdbwdpass_defaults(blocks, steps, T):
    """ Compute the default values for the parallel forward/backward pass.

    Parameters:
        blocks: The number of blocks.
        steps: The number of steps per block.
        T: The final time.

    Returns:
        dt, t0: The default time step size and initial time.
    """
    dt = T / tf.cast(steps * blocks, dtype=T.dtype)
    t0 = tf.constant(0.0, dtype=T.dtype)
    return dt, t0

@tf.function
def par_fwdbwdpass_init(blocks, steps, x0, dt, t0, F_f, L_f, X_f, U_f, c_f, H_f, r_f, forward=True):
    """ Compute the initial elements for the parallel forward/backward pass.

    Parameters:
        blocks: The number of blocks.
        steps: The number of steps per block.
        x0: The initial state.
        dt: The time step size.
        t0: The initial time.
        F_f, L_f, X_f, U_f, c_f, H_f, r_f: Functions of time for LQT parameters.
        forward: Whether to initialize forward or backward.

    Returns:
        As, bs, Cs, etas, Js: The initial elements.
    """

    # The backward init has mostly the same elements, so let us use it
    ST = tf.eye(x0.shape[-1], dtype=x0.dtype)      # Dummy value
    vT = tf.zeros((x0.shape[-1],), dtype=x0.dtype) # Dummy value
    if forward:
        As, bs, Cs, etas, Js = par_backwardpass_init_fw(blocks, steps, dt, t0, ST, vT, F_f, L_f, X_f, U_f, c_f, H_f, r_f)
    else:
        As, bs, Cs, etas, Js = par_backwardpass_init_bw(blocks, steps, dt, t0, ST, vT, F_f, L_f, X_f, U_f, c_f, H_f, r_f)

    A0 = tf.zeros((x0.shape[0], x0.shape[0]), dtype=x0.dtype)
    b0 = x0
    C0 = tf.zeros_like(A0)
    eta0 = tf.zeros_like(A0[:, 0])
    J0 = tf.zeros_like(A0)

    As = tf.concat([tf.expand_dims(A0, axis=0), As[:-1]], axis=0)
    bs = tf.concat([tf.expand_dims(b0, axis=0), bs[:-1]], axis=0)
    Cs = tf.concat([tf.expand_dims(C0, axis=0), Cs[:-1]], axis=0)
    etas = tf.concat([tf.expand_dims(eta0, axis=0), etas[:-1]], axis=0)
    Js = tf.concat([tf.expand_dims(J0, axis=0), Js[:-1]], axis=0)

    return As, bs, Cs, etas, Js

@tf.function
def clqt_par_fwdbwdpass_extract(As, bs, Cs, Ss, vs, Kxs, ds, steps, dt, t0, F_f, L_f, X_f, U_f, c_f, H_f, r_f):
    """ Extract the state and control trajectories from the parallel forward/backward pass results.

    Parameters:
        As, bs, Cs: Forward integrated conditional value function parameters.
        Ss, vs: Backward integrated value function parameters.
        Kxs: The control gains.
        ds: The control offsets.
        steps: The number of steps per block.
        dt: The time step size.
        t0: The initial time.
        F_f, L_f, X_f, U_f, c_f, H_f, r_f: Functions of time for LQT parameters.

    Returns:
        xs, us: The state and control trajectories.
    """

    blocks = tf.shape(As)[0] - 1
    t0s = t0 + tf.range(0, blocks, dtype=As.dtype) * steps * dt

    As_all, bs_all, Cs_all = clqt_seq_fwdbwdpass(steps, dt, t0s, As[:-1], bs[:-1], Cs[:-1], F_f, L_f, X_f, U_f, c_f, H_f, r_f)

    AT = As_all[-1, -1, :, :]
    bT = bs_all[-1, -1, :]
    CT = Cs_all[-1, -1, :, :]

    As_all = As_all[:, :-1, :, :]
    bs_all = bs_all[:, :-1, :]
    Cs_all = Cs_all[:, :-1, :, :]

    As_all_shape = tf.shape(As_all)
    As_all = tf.reshape(As_all, shape=(As_all_shape[0] * As_all_shape[1], As_all_shape[2], As_all_shape[3]))
    bs_all_shape = tf.shape(bs_all)
    bs_all = tf.reshape(bs_all, shape=(bs_all_shape[0] * bs_all_shape[1], bs_all_shape[2]))
    Cs_all_shape = tf.shape(Cs_all)
    Cs_all = tf.reshape(Cs_all, shape=(Cs_all_shape[0] * Cs_all_shape[1], Cs_all_shape[2], Cs_all_shape[3]))

    As_all = tf.concat([As_all, tf.expand_dims(AT, axis=0)], axis=0)
    bs_all = tf.concat([bs_all, tf.expand_dims(bT, axis=0)], axis=0)
    Cs_all = tf.concat([Cs_all, tf.expand_dims(CT, axis=0)], axis=0)

    # Returns xs, us
    return clqt_combine_fwdbwd(Kxs, ds, Ss, vs, As_all, bs_all, Cs_all)

@tf.function
def clqt_par_fwdbwdpass(blocks, steps, x0, Ss, vs, Kxs, ds, dt, t0, F_f, L_f, X_f, U_f, c_f, H_f, r_f, forward=True, max_parallel=10000):
    """ Parallel forward/backward pass for CLQT.

    Parameters:
        blocks: The number of blocks.
        steps: The number of steps per block.
        x0: The initial state.
        Ss, vs: Backward integrated value function parameters.
        Kxs: The control gains.
        ds: The control offsets.
        dt: The time step size.
        t0: The initial time.
        F_f, L_f, X_f, U_f, c_f, H_f, r_f: Functions of time for LQT parameters.
        forward: Whether to initialize forward or backward (default: True).
        max_parallel: The maximum number of parallel operations for tfp.math.scan_associative (default: 10000).

    Returns:
        xs, us: The state and control trajectories.
    """

    As, bs, Cs, etas, Js = par_fwdbwdpass_init(blocks, steps, x0, dt, t0, F_f, L_f, X_f, U_f, c_f, H_f, r_f, forward)

    elems = tfp.math.scan_associative(lqt_tf.lqt_par_comb_V,
                                      (As, bs, Cs, etas, Js),
                                      max_num_levels=max(math.ceil(math.log2(max_parallel)),1))

    As = elems[0]
    bs = elems[1]
    Cs = elems[2]

    # Returns xs, us
    return clqt_par_fwdbwdpass_extract(As, bs, Cs, Ss, vs, Kxs, ds, steps, dt, t0, F_f, L_f, X_f, U_f, c_f, H_f, r_f)
