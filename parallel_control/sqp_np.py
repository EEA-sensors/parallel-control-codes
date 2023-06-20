"""
Numpy/Jax versions sequential quadratic programming (with equality constraints).

@author: Simo Särkkä
"""

from scipy import linalg
import numpy as np
import jax
import jax.numpy as jnp

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

    Z = jnp.zeros((A.shape[0], A.shape[0]))
    Psi = jax.numpy.block([[G, -A.T], [A, Z]])
    xlam = jnp.linalg.solve(Psi, jnp.concatenate((-c, b)))

    x = xlam[:c.shape[0]]
    lam = xlam[c.shape[0]:]

    return x, lam

def local_eq(fun, con, x0, max_iter=100, tol=1e-6, quiet=False):
    """
    Local sequential linear programming with equality constraints.

    Parameters:
        fun: Objective function.
        con: Equality constraint function.
        x0: Initial guess.
        max_iter: Maximum number of iterations.
        tol: Tolerance for stopping.
        quiet: If True, do not print anything.

    Returns:
        x: Optimal solution.
        iter: Number of iterations.
        crit: Final criterion value.
    """
    if not quiet:
        print('LOCAL_eq starting.')

    lag = lambda x, lam: fun(x) - jnp.dot(lam, con(x))

    dlag = jax.jacfwd(lag)
    dfun = jax.jacfwd(fun)
    dcon = jax.jacfwd(con)
    hlag = jax.jacfwd(jax.grad(lag))

    x = x0
    c = con(x)
    lam = jnp.zeros(c.shape)

    iter = 0
    crit = 1.0

    while iter < max_iter and (crit > tol):
#        f = fun(x)
        Jf = dfun(x)

        c = con(x)
        Jc = dcon(x)

        Jl = dlag(x, lam)
        HL = hlag(x, lam)

        p, hlam = lqe_solve(HL, Jf, Jc, -c)

        x = x + p
        lam = hlam

        iter = iter + 1
        crit = jnp.linalg.norm(Jl)


        if not quiet:
            print(f'{iter} / {max_iter} : x={x}, |Jl|={jnp.linalg.norm(Jl)}')

    if not quiet:
        print('LOCAL_eq done.')

    return x, iter, crit


def local_eq_fast(fun, con, x0, num_iter=5, quiet=True):
    """
    Local sequential linear programming, "fast" and differentiable version with fixed number of iterations.

    Parameters:
        fun: Objective function.
        con: Equality constraint function.
        x0: Initial guess.
        num_iter: Number of iterations.
        quiet: If True, do not print anything.

    Returns:
        x: Optimal solution.
        num_iter: Number of iterations.
        crit: Final criterion value.
    """
    if not quiet:
        print('LOCAL_eq_fast starting.')

    lag = lambda x, lam: fun(x) - jnp.dot(lam, con(x))

    dlag = jax.jacfwd(lag)
    dfun = jax.jacfwd(fun)
    dcon = jax.jacfwd(con)
    hlag = jax.jacfwd(jax.grad(lag))

    def body(carry, inp):
        x, lam, _ = carry
        iter = inp

#        f = fun(x)
        Jf = dfun(x)

        c = con(x)
        Jc = dcon(x)

        Jl = dlag(x, lam)
        HL = hlag(x, lam)

        p, hlam = lqe_solve(HL, Jf, Jc, -c)

        x = x + p
        lam = hlam

        iter = iter + 1
        crit = jnp.linalg.norm(Jl)

        if not quiet:
            print(f'{iter} / {num_iter} : x={x}, |Jl|={jnp.linalg.norm(Jl)}')

        new_carry = x, lam, crit

        return new_carry, new_carry

    c = con(x0)
    lam0 = jnp.zeros(c.shape)
    res, res_list = jax.lax.scan(body, (x0, lam0, 1.0), jnp.arange(0, num_iter, dtype=jnp.int32))

    x, lam, crit = res

    if not quiet:
        print('LOCAL_eq_fast done.')

    return x, num_iter, crit



#
# BFGS versions
#

def bfgs_basic_update(B, s, y, quiet=False):
    """ BFGS update of Hessian approximation.

    Parameters:
        B: Hessian approximation.
        s: Step.
        y: Gradient difference.
        quiet: If True, do not print anything.

    Returns:
        B: Updated Hessian approximation.
    """
    den = jnp.dot(s, y)
#    print(f'den = {den}')
    if den > 0.0:
        Bs = B @ s
        B = B + jnp.outer(y, y) / den - jnp.outer(Bs, Bs) / jnp.dot(s, Bs)
    else:
        if not quiet:
            print(f'Skipped update of B due to den = {den}')
    return B


def bfgs_damped_update(B, s, y, quiet=False):
    """ BFGS update of Hessian approximation with damping.

    Parameters:
        B: Hessian approximation.
        s: Step.
        y: Gradient difference.
        quiet: If True, do not print anything.

    Returns:
        B: Updated Hessian approximation.
    """
    Bs = B @ s
    if jnp.dot(s, y) >= 0.2 * jnp.dot(Bs, s):
        theta = 1.0
    else:
        theta = 0.8 * jnp.dot(Bs, s) / (jnp.dot(Bs, s) - jnp.dot(s, y))
    y = theta * y + (1.0 - theta) * Bs
    den = jnp.dot(s, y)
#    print(f'den = {den}')
    B = B + jnp.outer(y, y) / den - jnp.outer(Bs, Bs) / jnp.dot(s, Bs)
    return B


def bfgs_eq_bt(fun, con, x0, max_iter=10, tol=1e-6, max_backtrack=10, damped=True, quiet=False):
    """
    BFGS sequential linear programming.

    Parameters:
        fun: Objective function.
        con: Equality constraint function.
        x0: Initial guess.
        max_iter: Maximum number of iterations.
        tol: Tolerance for the norm of the Lagrangian gradient.
        max_backtrack: Maximum number of backtracking steps.
        damped: If True, use damped BFGS update.
        quiet: If True, do not print anything.

    Returns:
        x: Optimal solution.
        iter: Number of iterations.
        crit: Final criterion value.
    """

    if not quiet:
        print('BFGS_eq starting.')

    lag = lambda x, lam: fun(x) - jnp.dot(lam, con(x))

    dlag = jax.jacfwd(lag)
    dfun = jax.jacfwd(fun)
    dcon = jax.jacfwd(con)

    mu = 1.0
    merit = lambda x, mu: fun(x) + jnp.sum(jnp.abs(con(x))) / mu

    x = x0
    B = jnp.eye(x.shape[0])
    c = con(x)
    lam = jnp.zeros(c.shape)

    iter = 0
    crit = 1.0

    while iter < max_iter and (crit > tol):
#        f = fun(x)
        Jf = dfun(x)

        c = con(x)
        Jc = dcon(x)

        p, hlam = lqe_solve(B, Jf, Jc, -c)
        plam = hlam - lam

        alf = 1.0
        m0 = merit(x, mu)
        backtrack = 0
        while merit(x + alf * p, mu) > m0 and (backtrack < max_backtrack):
            alf = 0.5 * alf
            backtrack = backtrack + 1

        if backtrack == max_backtrack:
            if not quiet:
                print(f'-> Oops, function did not decrease, taking leap of faith.')
            alf = 1.0

        x_old = x
        s = alf * p
        x = x + s
        lam = lam + alf * plam

        Jl_old = dlag(x_old, lam)
        Jl_new = dlag(x, lam)
        y = Jl_new - Jl_old

        if damped:
            B = bfgs_damped_update(B, s, y, quiet)
        else:
            B = bfgs_basic_update(B, s, y, quiet)

        iter = iter + 1
        crit = jnp.linalg.norm(Jl_new)

        if not quiet:
            print(f'{iter} / {max_iter} : x={x}, |Jl|={jnp.linalg.norm(Jl_new)}, alpha={alf}')

    if not quiet:
        print('BFGS_eq done.')

    return x, iter, crit
