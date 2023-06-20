"""
Tensorflow versions sequential quadratic programming (with equality constraints).

@author: Simo Särkkä
"""

import tensorflow as tf
import math

# Abbreviations for tensorflow
mm = tf.linalg.matmul
mv = tf.linalg.matvec
top = tf.linalg.matrix_transpose

def lqe_solve(G, c, A, b):
    """
    Quadratic programming solver for

    min 1/2 x^T G x + c^T x \\
    A x = b

    Parameters:
        G: Quadratic cost matrix.
        c: Linear cost vector.
        A: Equality constraint matrix.
        b: Equality constraint vector.

    Returns:
        x: Optimal solution.
        lam: Lagrange multipliers.
    """

    Z = tf.zeros(b[..., 0].shape + (b.shape[-1], b.shape[-1]), dtype=G.dtype)
    Psi = tf.concat((tf.concat((G, -top(A)), axis=-1), tf.concat((A, Z), axis=-1)), axis=-2)
    xlam = tf.linalg.solve(Psi, tf.expand_dims(tf.concat((-c, b), axis=-1), -1))[..., 0]

    x = xlam[..., :c.shape[-1]]
    lam = xlam[..., c.shape[-1]:]

    return x, lam

@tf.function
def fun_value_grad(fun, x, par, batch=False):
    """ Evaluate function and its gradient.

    Parameters:
        fun: Function f(x,par).
        x: Input.
        par: Parameters.
        batch: If True, x has one batch dimension.

    Returns:
        value: Function value.
        grad: Gradient.
    """
    if not batch: # No batch dimensions
        with tf.GradientTape(watch_accessed_variables=False) as t:
            t.watch(x)
            value = fun(x, par)
        grad = t.jacobian(value, x)
    else:  # One batch dimension
        with tf.GradientTape(watch_accessed_variables=False) as t:
            t.watch(x)
            value = tf.expand_dims(fun(x, par), -1)
        grad = t.batch_jacobian(value, x)[..., 0, :]
        value = value[..., 0]

    return value, grad

@tf.function
def con_value_jac(con, x, par, batch=False):
    """ Evaluate constraint and its Jacobian.

    Parameters:
        con: Constraint function c(x,par).
        x: Input.
        par: Parameters.
        batch: If True, x has one batch dimension.

    Returns:
        value: Constraint value.
        jac: Jacobian.
    """
    if not batch: # No batch dimensions
        with tf.GradientTape(watch_accessed_variables=False) as t:
            t.watch(x)
            value = con(x, par)
        jac = t.jacobian(value, x)
    else:  # One batch dimension
        with tf.GradientTape(watch_accessed_variables=False) as t:
            t.watch(x)
            value = con(x, par)
        jac = t.batch_jacobian(value, x)

    return value, jac

@tf.function
def lag_value_grad_hess(fun, con, x, lam, par, batch=False):
    """ Evaluate Lagrangian, its gradient, and Hessian.

    Parameters:
        fun: Function f(x,par).
        con: Constraint function c(x,par).
        x: Input.
        lam: Lagrange multipliers.
        par: Parameters.
        batch: If True, x has one batch dimension.

    Returns:
        value: Lagrangian value.
        grad: Gradient.
        hess: Hessian.
    """
    if not batch: # No batch dimensions
        with tf.GradientTape(watch_accessed_variables=False) as t2:
            t2.watch(x)
            with tf.GradientTape() as t1:
                t1.watch(x)
                value = fun(x, par) - tf.reduce_sum(tf.math.multiply(lam, con(x, par)), axis=-1)
            grad = t1.jacobian(value, x)
        hess = t2.jacobian(grad, x)

    else:  # One batch dimension
        with tf.GradientTape(watch_accessed_variables=False) as t2:
            t2.watch(x)
            with tf.GradientTape() as t1:
                t1.watch(x)
                value = tf.expand_dims(fun(x, par) - tf.reduce_sum(tf.math.multiply(lam, con(x, par)), axis=-1),-1)
            grad = t1.batch_jacobian(value, x)[..., 0, :]
        hess = t2.batch_jacobian(grad, x)
        value = value[..., 0]

    return value, grad, hess


def local_eq_fast(fun, con, x0, par, num_iter=5, batch=False):
    """ Local sequential quadratic programming with equality constraints.

    Parameters:
        fun: Function f(x,par).
        con: Constraint function c(x,par).
        x0: Initial guess.
        par: Parameters.
        num_iter: Number of iterations.
        batch: If True, x has one batch dimension.

    Returns:
        xs: Optimal solutions.
        lams: Lagrange multipliers.
        crits: Convergence criteria.
    """
    def body(carry, inp):
        x, lam, _ = carry
        iter = inp

        f, Jf = fun_value_grad(fun, x, par, batch)
        c, Jc = con_value_jac(con, x, par, batch)
        l, Jl, HL = lag_value_grad_hess(fun, con, x, lam, par, batch)

        p, hlam = lqe_solve(HL, Jf, Jc, -c)

        x = x + p
        lam = hlam

        crit = tf.linalg.norm(Jl)

        new_carry = x, lam, crit

        return new_carry

    c = con(x0, par)
    lam0 = tf.zeros_like(c)

    xs, lams, crits = tf.scan(body, tf.range(0, num_iter, dtype=tf.int32),
                              initializer=(x0, lam0, tf.constant(1.0, dtype=x0.dtype)))

    return xs, lams, crits

