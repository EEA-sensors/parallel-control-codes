"""
TensorFlow versions of the linear model. Not much as discrete LQT is directly convertable from
its NumPy version to TF. However, for CLQT we have a separate implementation.

@author: Simo Särkkä
"""

import tensorflow as tf
import numpy as np

# Abbreviations for convenience
mm = tf.linalg.matmul
mv = tf.linalg.matvec

def get_clqt(dtype=tf.float64):
    """ Get CLQT for the linear model.

    Parameters:
        dtype: Data type.

    Returns:
        x0: Initial state.
        T: Time horizon.
        XT, HT, rT, F_f, L_f, X_f, U_f, c_f, H_f, r_f: CLQT parameters
    """
    dp = lambda v: tf.constant(v, dtype=dtype)

    U_f = lambda t: dp(0.1) * tf.eye(2, dtype=dtype)
    H_f = lambda t: tf.constant([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=dtype)
    HT = tf.eye(4, dtype=dtype)
    X_f = lambda t: dp(1.0) * tf.eye(2, dtype=dtype)
    XT = dp(1.0) * tf.eye(4, dtype=dtype)

    F_f = lambda t: tf.constant([[0.0, 0.0, 1.0, 0.0],
                              [0.0, 0.0, 0.0, 1.0],
                              [0.0, 0.0, 0.0, 0.0],
                              [0.0, 0.0, 0.0, 0.0]], dtype=dtype)
    L_f = lambda t: tf.constant([[0.0, 0.0],
                               [0.0, 0.0],
                               [1.0, 0.0],
                               [0.0, 1.0]], dtype=dtype)
    c_f = lambda t: tf.zeros((4,), dtype=dtype)

    cx = tf.constant([5.7770, -2.6692, 1.1187, 0.1379, 0.5718, 1.1214,
                      0.2998, 0.3325, 0.7451, 0.2117, 0.6595, 0.0401, -0.2995], dtype=dtype)
    cy = tf.constant([4.3266, -1.4584, -1.2457, 1.1804, 0.2035, 0.5123,
                      1.0588, 0.2616, -0.6286, -0.3802, 0.2750, -0.0070, -0.0022], dtype=dtype)

    a = tf.constant(2.0 * np.pi / 50.0, dtype=dtype)

    r_f = lambda t: tf.stack([cx[0] +
                              cx[1] * tf.cos(dp(1.0) * a * t) + cx[2] * tf.sin(dp(1.0) * a * t) +
                              cx[3] * tf.cos(dp(2.0) * a * t) + cx[4] * tf.sin(dp(2.0) * a * t) +
                              cx[5] * tf.cos(dp(3.0) * a * t) + cx[6] * tf.sin(dp(3.0) * a * t) +
                              cx[7] * tf.cos(dp(4.0) * a * t) + cx[8] * tf.sin(dp(4.0) * a * t) +
                              cx[9] * tf.cos(dp(5.0) * a * t) + cx[10] * tf.sin(dp(5.0) * a * t) +
                              cx[11] * tf.cos(dp(6.0) * a * t) + cx[12] * tf.sin(dp(6.0) * a * t),
                              cy[0] +
                              cy[1] * tf.cos(dp(1.0) * a * t) + cy[2] * tf.sin(dp(1.0) * a * t) +
                              cy[3] * tf.cos(dp(2.0) * a * t) + cy[4] * tf.sin(dp(2.0) * a * t) +
                              cy[5] * tf.cos(dp(3.0) * a * t) + cy[6] * tf.sin(dp(3.0) * a * t) +
                              cy[7] * tf.cos(dp(4.0) * a * t) + cy[8] * tf.sin(dp(4.0) * a * t) +
                              cy[9] * tf.cos(dp(5.0) * a * t) + cy[10] * tf.sin(dp(5.0) * a * t) +
                              cy[11] * tf.cos(dp(6.0) * a * t) + cy[12] * tf.sin(dp(6.0) * a * t)])

    T = tf.constant(50.0, dtype=dtype)
    rT = tf.constant([4.9514, 4.4353, 0.0, 0.0], dtype=dtype)
    x0 = tf.constant([5.0, 5.0, 0.0, 0.0], dtype=dtype)

    return x0, T, XT, HT, rT, F_f, L_f, X_f, U_f, c_f, H_f, r_f
