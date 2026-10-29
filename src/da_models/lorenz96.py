"""One-layer and two-layer Lorenz-96 models.

Both right-hand sides accept a single state of shape (n,) or a stacked
ensemble of shape (n, Ne) (one member per column, as in enkf.py). Every
operation acts along axis 0 (the state index), so members never mix and a
single state gives exactly the same result as before.
"""

import numpy as np


def rhs(t, state, F=8.0, d=1.0):
    """Right-hand side of the one-layer Lorenz-96 system."""
    u = np.asarray(state)
    up1 = np.roll(u, -1, axis=0)
    um1 = np.roll(u, 1, axis=0)
    um2 = np.roll(u, 2, axis=0)
    return um1 * (up1 - um2) - d * u + F


def pack_two_layer(u, v):
    """Flatten u: (K,) and v: (K, J) into one state vector.

    For an ensemble, u: (K, Ne) and v: (K, J, Ne) give shape (K(1+J), Ne).
    """
    K, J = v.shape[:2]
    return np.concatenate([u, v.reshape((K * J,) + v.shape[2:])])


def unpack_two_layer(state, K, J):
    """Recover u (K,) and v (K, J) from a flattened state vector.

    For an ensemble of shape (K(1+J), Ne), u is (K, Ne) and v is (K, J, Ne).
    """
    u = state[:K]
    v = state[K:].reshape((K, J) + state.shape[1:])
    return u, v


def rhs_two_layer(t, state, K=40, J=5, F=8.0, d=1.0, gamma=0.05,
                   d_j=(0.2, 0.5, 1.0, 2.0, 5.0)):
    """Right-hand side of the two-layer Lorenz-96 system."""
    u, v = unpack_two_layer(np.asarray(state), K, J)
    d_j = np.asarray(d_j).reshape((1, J) + (1,) * (v.ndim - 2))

    up1 = np.roll(u, -1, axis=0)
    um1 = np.roll(u, 1, axis=0)
    um2 = np.roll(u, 2, axis=0)

    coupling = gamma * np.sum(v, axis=1)
    du = um1 * (up1 - um2) + coupling * u - d * u + F
    dv = -d_j * v - gamma * (u ** 2)[:, np.newaxis]

    return pack_two_layer(du, dv)
