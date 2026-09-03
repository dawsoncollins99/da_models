"""Goodwin-Keen economic model."""

import numpy as np


def rhs(t, state, alpha=0.025, beta=0.02, delta=0.01, nu=3.0, r=0.03,
        phi0=0.0400641, phi1=0.0000641026,
        kappa0=-0.0065, kappa1=np.exp(-5), kappa2=20.0):
    """Right-hand side of the Goodwin-Keen economic model.

    state = [x1, x2, x3] = [wage share, employment rate, debt-to-output ratio]
    Time is in years.
    """
    x1, x2, x3 = state

    Phi = phi1 / (1.0 - x2) ** 2 - phi0
    K = kappa0 + kappa1 * np.exp(kappa2 * (1.0 - x1 - r * x3))

    dx1 = x1 * (Phi - alpha)
    dx2 = x2 * (K / nu - alpha - beta - delta)
    dx3 = x3 * (r - K / nu + delta) + K - (1.0 - x1)

    return np.array([dx1, dx2, dx3])