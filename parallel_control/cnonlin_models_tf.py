"""
Tensorflow versions of (1d) nonlinear models.

@author: Simo Särkkä
"""

import tensorflow as tf

def linear_model(x_grid_steps = 100, u_grid_steps = 100, dtype=tf.float64):
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
    dp = lambda v: tf.constant(v, dtype=dtype)

    X = tf.constant(1.0, dtype=dtype)
    U = tf.constant(1.0, dtype=dtype)
    XT = tf.constant(2.0, dtype=dtype)

    f = lambda x, u: x + u
    L = lambda x, u: dp(0.5) * X * x**2 + dp(0.5) * U * u**2
    LT = lambda x: dp(0.5) * XT * x**2
    x_grid = tf.linspace(tf.constant(-2.0, dtype=dtype), tf.constant(2.0, dtype=dtype), x_grid_steps)
    u_grid = tf.linspace(tf.constant(-2.0, dtype=dtype), tf.constant(2.0, dtype=dtype), u_grid_steps)
    T = tf.constant(1.0, dtype=dtype)

    return f, L, LT, T, x_grid, u_grid


def velocity_model(x_grid_steps = 100, u_grid_steps = 100, dtype=tf.float64):
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
    dp = lambda v: tf.constant(v, dtype=dtype)

    beta = tf.constant(0.1, dtype=dtype)
    X = tf.constant(1.0, dtype=dtype)
    U = tf.constant(1.0, dtype=dtype)
    XT = tf.constant(4.0, dtype=dtype)

    f = lambda x, u: dp(1.0) - beta * x ** 2 + u
#    f = lambda x, u: dp(1.0) - beta * jnp.tanh(x) + u
    L = lambda x, u: dp(0.5) * X * x**2 + dp(0.5) * U * u**2
    LT = lambda x: dp(0.5) * XT * x**2
    x_grid = tf.linspace(tf.constant(-3.0, dtype=dtype), tf.constant(3.0, dtype=dtype), x_grid_steps)
    u_grid = tf.linspace(tf.constant(-2.0, dtype=dtype), tf.constant(2.0, dtype=dtype), u_grid_steps)
    T = tf.constant(1.0, dtype=dtype)

    return f, L, LT, T, x_grid, u_grid


def upwind_model(x_grid_steps = 100, u_grid_steps = 100, dtype=tf.float64):
    """ Form nonlinear upwind model with quadratic cost.

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
    L = lambda x, u: tf.constant(0.0, dtype=dtype)
    LT = lambda x: -x

    x_grid = tf.linspace(tf.constant(-1.0, dtype=dtype), tf.constant(1.0, dtype=dtype), x_grid_steps)
    u_grid = tf.linspace(tf.constant(0.0, dtype=dtype), tf.constant(1.0, dtype=dtype), u_grid_steps)
    T = tf.constant(1.0, dtype=dtype)

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

    xf_mesh, x0_mesh = tf.meshgrid(x_grid, x_grid)
    inf_mesh = inf_value * tf.ones(x0_mesh.shape, dtype=x0_mesh.dtype)
    zero_mesh = tf.zeros(x0_mesh.shape, dtype=x0_mesh.dtype)

    a = tf.exp(block_dt)
    cond1 = tf.logical_and(tf.logical_and(tf.greater_equal(x0_mesh, 0.0), tf.greater_equal(xf_mesh, x0_mesh)), tf.less_equal(xf_mesh, a * x0_mesh))
    cond2 = tf.logical_and(tf.logical_and(tf.greater_equal(x0_mesh, 0.0), tf.greater_equal(xf_mesh, a * x0_mesh)), tf.less_equal(xf_mesh, x0_mesh))

    return tf.where(tf.logical_or(cond1,cond2), zero_mesh, inf_mesh)
