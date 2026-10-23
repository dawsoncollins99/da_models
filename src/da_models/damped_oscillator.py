"""Single-degree-of-freedom, viscously damped linear oscillator.

    z'' + 2 xi omega z' + omega^2 z = 0

As a first-order system in x = (z, z'):

    x' = A x,    A = [[0, 1], [-omega^2, -2 xi omega]].

Explicit Euler with step dt gives the linear map x_{k+1} = M x_k,
M = I + dt A, which is what the Kalman filter in HW 7 uses for both the
truth and the filter.
"""

import numpy as np


def rhs(t, state, omega=1.0, xi=1e-3):
    """Right-hand side, same signature as the other benchmark models."""
    z, zdot = state
    return np.array([zdot, -2.0 * xi * omega * zdot - omega ** 2 * z])


def generator(omega=1.0, xi=1e-3):
    """Continuous-time matrix A with x' = A x."""
    return np.array([[0.0, 1.0],
                     [-omega ** 2, -2.0 * xi * omega]])


def euler_matrix(omega=1.0, xi=1e-3, dt=1e-3):
    """One explicit-Euler step: M = I + dt A."""
    return np.eye(2) + dt * generator(omega, xi)


def euler_spectral_radius(omega=1.0, xi=1e-3, dt=1e-3):
    """Largest |eigenvalue| of M. Below 1 means the discrete solution decays.

    For an underdamped oscillator (xi < 1) the eigenvalues of A are
    -xi omega +/- i omega sqrt(1 - xi^2), so

        |1 + dt lambda|^2 = 1 - 2 dt xi omega + dt^2 omega^2,

    which is below 1 exactly when dt < 2 xi / omega. Larger omega (or
    smaller xi) therefore needs a smaller dt.
    """
    return np.max(np.abs(np.linalg.eigvals(euler_matrix(omega, xi, dt))))


def euler_max_stable_dt(omega=1.0, xi=1e-3):
    """Largest dt for which explicit Euler still decays (underdamped case)."""
    return 2.0 * xi / omega
