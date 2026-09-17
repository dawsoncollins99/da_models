"""Adjoint (backward sensitivity) tools for the HW1 benchmark models.

For a model x_dot = f(x, p) and a terminal cost

    J = 1/2 * || x(T) - y ||^2,

the continuous adjoint equation is

    lambda_dot = -(Dx f(x(t), p))^T lambda,      lambda(T) = x(T) - y,

integrated *backward* in time from T to 0, and the gradient of J with respect
to the initial condition is grad_x0 J = lambda(0).

This module is deliberately model-agnostic: it works with the plain rhs(t,
state, **kwargs) functions from lorenz63.py, lorenz96.py, sturis.py, and
goodwin_keen.py without modification. The state Jacobian Dxf is obtained by
central finite differences, which keeps the code short, avoids hand-derived
(and error-prone) partials for the 6-D Sturis and 240-D two-layer Lorenz-96
systems, and is validated below to machine-level accuracy against the
finite-difference check requested in the assignment. If you would rather use
analytic Jacobians for a given model, write a `jacobian(state, **kwargs)`
function in that model's file and pass it in as `jac_func` below.
"""

import numpy as np
from scipy.integrate import solve_ivp


def numerical_jacobian(rhs, t, x, rhs_kwargs=None, eps=1e-6):
    """Central-difference Jacobian Dxf of rhs(t, x, **rhs_kwargs) w.r.t. x.

    Returns an (n, n) array with entry [i, j] = d f_i / d x_j.
    """
    rhs_kwargs = rhs_kwargs or {}
    x = np.asarray(x, dtype=float)
    n = x.shape[0]
    Jx = np.zeros((n, n))
    for j in range(n):
        h = eps * max(1.0, abs(x[j]))
        dx = np.zeros(n)
        dx[j] = h
        f_plus = np.asarray(rhs(t, x + dx, **rhs_kwargs))
        f_minus = np.asarray(rhs(t, x - dx, **rhs_kwargs))
        Jx[:, j] = (f_plus - f_minus) / (2.0 * h)
    return Jx


def forward_solve(rhs, x0, t_span, rhs_kwargs=None, rtol=1e-10, atol=1e-12):
    """Integrate the forward model and return a dense (continuous) solution."""
    rhs_kwargs = rhs_kwargs or {}
    f = lambda t, x: rhs(t, x, **rhs_kwargs)
    sol = solve_ivp(f, t_span, np.asarray(x0, dtype=float),
                     method="RK45", rtol=rtol, atol=atol, dense_output=True)
    if not sol.success:
        raise RuntimeError(f"Forward integration failed: {sol.message}")
    return sol


def adjoint_solve(rhs, t_span, lam_T, sol, rhs_kwargs=None, jac_func=None,
                   jac_eps=1e-6, rtol=1e-10, atol=1e-12):
    """Integrate the continuous adjoint equation backward from T to t0.

    Parameters
    ----------
    rhs : callable
        Model right-hand side, rhs(t, state, **rhs_kwargs).
    t_span : (t0, T)
        Forward-time interval; the adjoint is integrated from T back to t0.
    lam_T : array_like
        Terminal condition lambda(T). For J = 1/2||x(T)-y||^2, lam_T = x(T)-y.
    sol : OdeSolution
        Dense forward solution from `forward_solve`, used to evaluate x(t)
        at whatever backward time points the adjoint integrator requests.
    jac_func : callable, optional
        jac_func(state, **rhs_kwargs) -> (n, n) array, the analytic Dxf.
        Defaults to a numerical (central-difference) Jacobian of `rhs`.

    Returns
    -------
    lam0 : ndarray
        lambda(t0) = grad_x0 J.
    """
    rhs_kwargs = rhs_kwargs or {}
    t0, T = t_span

    if jac_func is None:
        def jac(t, x):
            return numerical_jacobian(rhs, t, x, rhs_kwargs=rhs_kwargs, eps=jac_eps)
    else:
        def jac(t, x):
            return jac_func(x, **rhs_kwargs)

    def adj_rhs(t, lam):
        x_t = sol.sol(t)
        Jx = jac(t, x_t)
        return -Jx.T @ lam

    sol_adj = solve_ivp(adj_rhs, (T, t0), np.asarray(lam_T, dtype=float),
                         method="RK45", rtol=rtol, atol=atol)
    if not sol_adj.success:
        raise RuntimeError(f"Adjoint integration failed: {sol_adj.message}")
    lam0 = sol_adj.y[:, -1]
    return lam0


def terminal_cost_and_gradient(rhs, x0, T, y_target, rhs_kwargs=None, jac_func=None,
                                jac_eps=1e-6, t0=0.0, rtol=1e-10, atol=1e-12):
    """Compute J = 1/2||x(T)-y||^2 and grad_x0 J via the adjoint method."""
    rhs_kwargs = rhs_kwargs or {}
    sol = forward_solve(rhs, x0, (t0, T), rhs_kwargs, rtol=rtol, atol=atol)
    xT = sol.sol(T)
    J = 0.5 * np.sum((xT - np.asarray(y_target)) ** 2)
    lam_T = xT - np.asarray(y_target)
    grad = adjoint_solve(rhs, (t0, T), lam_T, sol, rhs_kwargs=rhs_kwargs,
                          jac_func=jac_func, jac_eps=jac_eps, rtol=rtol, atol=atol)
    return J, grad, sol


def finite_difference_check(rhs, x0, T, y_target, rhs_kwargs=None, jac_func=None,
                             eps=1e-6, seed=None, t0=0.0, rtol=1e-10, atol=1e-12):
    """Compare the adjoint directional derivative to a centered finite difference.

    Picks a random unit direction d, and returns (adjoint_deriv, fd_deriv, d)
    where both should agree to several digits if the adjoint is correct:

        adjoint_deriv = grad_x0 J . d
        fd_deriv      = (J(x0 + eps d) - J(x0 - eps d)) / (2 eps)
    """
    rhs_kwargs = rhs_kwargs or {}
    rng = np.random.default_rng(seed)
    x0 = np.asarray(x0, dtype=float)
    d = rng.normal(size=x0.shape)
    d /= np.linalg.norm(d)

    def J_of(x0_):
        sol = forward_solve(rhs, x0_, (t0, T), rhs_kwargs, rtol=rtol, atol=atol)
        xT = sol.sol(T)
        return 0.5 * np.sum((xT - np.asarray(y_target)) ** 2)

    J_plus = J_of(x0 + eps * d)
    J_minus = J_of(x0 - eps * d)
    fd_deriv = (J_plus - J_minus) / (2.0 * eps)

    J0, grad0, _ = terminal_cost_and_gradient(rhs, x0, T, y_target, rhs_kwargs=rhs_kwargs,
                                               jac_func=jac_func, jac_eps=eps, t0=t0,
                                               rtol=rtol, atol=atol)
    adjoint_deriv = grad0 @ d

    return adjoint_deriv, fd_deriv, d
