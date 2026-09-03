"""Lorenz-63 model: right-hand side for use with scipy.integrate.solve_ivp."""

import numpy as np


def rhs(t, state, sigma=10.0, rho=28.0, beta=8.0 / 3.0):
    """Right-hand side of the Lorenz-63 system.

    Parameters
    ----------
    t : float
        Time (unused, but solve_ivp requires this signature).
    state : array_like, shape (3,)
        Current state [x, y, z].
    sigma, rho, beta : float
        Model parameters. Defaults are the classic benchmark values.

    Returns
    -------
    dstate : ndarray, shape (3,)
        Time derivative [dx/dt, dy/dt, dz/dt].
    """
    x, y, z = state
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return np.array([dx, dy, dz])