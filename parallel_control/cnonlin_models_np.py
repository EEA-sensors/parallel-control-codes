"""
Numpy/Jax versions of (1d) nonlinear models.

@author: Simo Särkkä
"""

from scipy import linalg
import numpy as np
import jax
import jax.numpy as jnp
from functools import partial

def linear_model(x_grid_steps = 100, u_grid_steps = 100, dtype=jnp.float64):
    """ Form simple linear model with quadratic cost.

    Parameters:
        x_grid_steps: Number of grid points in x (default 100).
        u_grid_steps: Number of grid points in u (default 100).
        dtype: Data type.

    Returns:
        f: Continuous-time dynamics function.
        L: Running cost function.
        LT: Terminal cost function.
        T: Time horizon.
        x_grid: Grid points in x.
        u_grid: Grid points in u.
    """
    X = 1.0
    U = 1.0
    XT = 2.0
    f = lambda x, u: x + u
    L = lambda x, u: 0.5 * X * x**2 + 0.5 * U * u**2
    LT = lambda x: 0.5 * XT * x**2
    x_grid = jnp.linspace(-2.0, 2.0, x_grid_steps, dtype=dtype)
    u_grid = jnp.linspace(-2.0, 2.0, u_grid_steps, dtype=dtype)
    T = 1.0

    return f, L, LT, T, x_grid, u_grid


def velocity_model(x_grid_steps = 100, u_grid_steps = 100, dtype=jnp.float64):
    """ Form nonlinear velocity model with quadratic cost.

    Parameters:
        x_grid_steps: Number of grid points in x (default 100).
        u_grid_steps: Number of grid points in u (default 100).
        dtype: Data type.

    Returns:
        f: Continuous-time dynamics function.
        L: Running cost function.
        LT: Terminal cost function.
        T: Time horizon.
        x_grid: Grid points in x.
        u_grid: Grid points in u.
    """
    beta = 0.1
    X = 1.0
    U = 1.0
    XT = 4.0
    f = lambda x, u: 1.0 - beta * x ** 2 + u
#    f = lambda x, u: 1.0 - beta * jnp.tanh(x) + u
    L = lambda x, u: 0.5 * X * x**2 + 0.5 * U * u**2
    LT = lambda x: 0.5 * XT * x**2
    x_grid = jnp.linspace(-3.0, 3.0, x_grid_steps, dtype=dtype)
    u_grid = jnp.linspace(-2.0, 2.0, u_grid_steps, dtype=dtype)
    T = 1.0

    return f, L, LT, T, x_grid, u_grid

def upwind_model(x_grid_steps = 100, u_grid_steps = 100, dtype=jnp.float64):
    """ Form nonlinear upwind model with terminal cost only.

    Parameters:
        x_grid_steps: Number of grid points in x (default 100).
        u_grid_steps: Number of grid points in u (default 100).
        dtype: Data type.

    Returns:
        f: Continuous-time dynamics function.
        L: Running cost function.
        LT: Terminal cost function.
        T: Time horizon.
        x_grid: Grid points in x.
        u_grid: Grid points in u.
    """
    f = lambda x, u: u * x
    L = lambda x, u: 0.0
    LT = lambda x: -x

    x_grid = jnp.linspace(-1.0, 1.0, x_grid_steps, dtype=dtype)
    u_grid = jnp.linspace(0.0, 1.0, u_grid_steps, dtype=dtype)
    T = 1.0

    return f, L, LT, T, x_grid, u_grid

def upwind_model_block_V(inf_value, block_dt, x_grid):
    """ Form conditional value function for a block for the upwind model.

    Parameters:
        inf_value: Value of the function outside the reachable area ("infinity").
        block_dt: Block length in time.
        x_grid: Grid points in x.

    Returns:
        V: Conditional value function evaluated on the grid.
    """

    xf_mesh, x0_mesh = jnp.meshgrid(x_grid, x_grid)
    inf_mesh = inf_value * jnp.ones(x0_mesh.shape, dtype=x0_mesh.dtype)

    a = jnp.exp(block_dt)
    cond1 = (x0_mesh > 0.0) * (xf_mesh >= x0_mesh) * (xf_mesh <= a * x0_mesh)
    cond2 = (x0_mesh <= 0.0) * (xf_mesh >= a * x0_mesh) * (xf_mesh <= x0_mesh)

    return (1 - jnp.maximum(cond1,cond2)) * inf_mesh




