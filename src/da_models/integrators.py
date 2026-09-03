"""Generic integrators shared across all models."""

import numpy as np


def euler_maruyama(rhs, x0, t_span, dt, q=0.0, seed=None, rhs_kwargs=None):
    """Integrate dX = f(X) dt + q dW using the Euler-Maruyama scheme.

    Parameters
    ----------
    rhs : callable
        Right-hand side function with signature rhs(t, state, **rhs_kwargs).
    x0 : array_like
        Initial state.
    t_span : tuple (t0, tf)
        Start and end time.
    dt : float
        Fixed time step.
    q : float, default 0.0
        Noise strength. q=0 reduces exactly to forward Euler.
    seed : int, optional
        Random seed for reproducibility.
    rhs_kwargs : dict, optional
        Extra keyword arguments passed to rhs (e.g. model parameters).

    Returns
    -------
    t : ndarray, shape (n_steps+1,)
        Time points.
    X : ndarray, shape (n_steps+1, len(x0))
        State at each time point.
    """
    rhs_kwargs = rhs_kwargs or {}
    rng = np.random.default_rng(seed)

    t0, tf = t_span
    n_steps = int(round((tf - t0) / dt))
    t = t0 + dt * np.arange(n_steps + 1)

    x0 = np.asarray(x0, dtype=float)
    X = np.zeros((n_steps + 1, x0.shape[0]))
    X[0] = x0

    for n in range(n_steps):
        noise = q * np.sqrt(dt) * rng.normal(size=x0.shape)
        X[n + 1] = X[n] + dt * rhs(t[n], X[n], **rhs_kwargs) + noise

    return t, X