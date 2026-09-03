"""One-layer and two-layer Lorenz-96 models."""

import numpy as np


def rhs(t, state, F=8.0, d=1.0):
    """Right-hand side of the one-layer Lorenz-96 system."""
    u = np.asarray(state)
    up1 = np.roll(u, -1)
    um1 = np.roll(u, 1)
    um2 = np.roll(u, 2)
    return um1 * (up1 - um2) - d * u + F


def pack_two_layer(u, v):
    """Flatten (u: shape (K,), v: shape (K,J)) into one state vector."""
    return np.concatenate([u, v.ravel()])


def unpack_two_layer(state, K, J):
    """Recover u (K,) and v (K,J) from a flattened state vector."""
    u = state[:K]
    v = state[K:].reshape(K, J)
    return u, v


def rhs_two_layer(t, state, K=40, J=5, F=8.0, d=1.0, gamma=0.05,
                   d_j=(0.2, 0.5, 1.0, 2.0, 5.0)):
    """Right-hand side of the two-layer Lorenz-96 system."""
    u, v = unpack_two_layer(state, K, J)
    d_j = np.asarray(d_j)

    up1 = np.roll(u, -1)
    um1 = np.roll(u, 1)
    um2 = np.roll(u, 2)

    coupling = gamma * np.sum(v, axis=1)
    du = um1 * (up1 - um2) + coupling * u - d * u + F
    dv = -d_j[np.newaxis, :] * v - gamma * (u ** 2)[:, np.newaxis]

    return pack_two_layer(du, dv)