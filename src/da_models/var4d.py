"""One-window 4DVAR (HW 5, Problem 5).

Plugs into the forecast/analysis cycle in assimilation.run_cycle. At each
observation time t_k, with background mu_k (the forecast that arrived at t_k),
4DVAR chooses the state at t_k whose forecast best fits the *next*
observation y_{k+1}, so the forecast over [t_k, t_{k+1}] is already corrected
(and analyze(N, .) has nothing left to assimilate).

The gradient of the observation term comes from the continuous adjoint in
adjoint.py, using the analytic Jacobians in jacobians.py.
"""

import numpy as np
from scipy.optimize import minimize

from .adjoint import adjoint_solve, forward_solve
from .assimilation import error_norm_segments


def make_observation_term(rhs, params, jac_func, dt, H, R,
                          rtol=1e-9, atol=1e-9, adj_rtol=1e-9, adj_atol=1e-12):
    """Observation part of the 4DVAR cost over one window, with its gradient.

        J_o(x) = 1/2 (y - H M(x))^T R^{-1} (y - H M(x)),   M = flow over [0, dt]

    Gradient by the continuous adjoint: integrate the model forward from x,
    then integrate  lambda' = -(Df)^T lambda  backward from
    lambda(dt) = H^T R^{-1} (H M(x) - y); the gradient is lambda(0).
    Returns obs_term(x, y) -> (J_o, grad J_o).
    """
    R_inv = np.linalg.inv(R)

    def obs_term(x, y):
        sol = forward_solve(rhs, x, (0.0, dt), params, rtol=rtol, atol=atol)
        misfit = H @ sol.y[:, -1] - y
        weighted = R_inv @ misfit
        grad = adjoint_solve(rhs, (0.0, dt), H.T @ weighted, sol, rhs_kwargs=params,
                             jac_func=jac_func, rtol=adj_rtol, atol=adj_atol)
        return 0.5 * misfit @ weighted, grad

    return obs_term


def background_sqrt(B):
    """Square root L with B = L L^T (symmetric eigendecomposition).

    Works even when B is nearly singular: directions B says are (almost)
    impossible get (almost) zero columns instead of huge entries in B^{-1}.
    """
    evals, V = np.linalg.eigh(B)
    return V * np.sqrt(np.clip(evals, 0.0, None))


def make_4dvar(obs_term, B, Y, gtol=1e-6, ftol=1e-8, maxiter=500):
    """One-window 4DVAR (HW 5, eq. 4), solved with L-BFGS.

    At t_k with background mu_k, minimize
        L(x) = J_o(x; y_{k+1}) + 1/2 (x - mu_k)^T B^{-1} (x - mu_k).
    The search runs in the control variable w, x = mu_k + L_B w with
    B = L_B L_B^T, so the background term is 1/2 |w|^2 and
        grad_w = w + L_B^T grad_x J_o.
    This has the same minimizer as L(x) but never forms B^{-1} (the Sturis B
    has condition number ~1e11) and is much better conditioned for the
    optimizer. Per-cycle optimizer statistics are kept in analyze.stats.

    Stopping: L-BFGS-B stops when the relative decrease in the cost per
    iteration falls below `ftol` (or the projected gradient below `gtol`).
    The cost of 4DVAR depends directly on this choice; ftol = 1e-8 changes
    the analyses by far less than the observation noise in these experiments.
    """
    L_B = background_sqrt(B)
    n_obs = len(Y)
    stats = []

    def analyze(k, background):
        if k == 0:
            stats.clear()
        if k == n_obs:                       # no observation after t_N
            return background
        y = Y[k]                             # y_{k+1}

        def cost_and_grad(w):
            Jo, g_x = obs_term(background + L_B @ w, y)
            return 0.5 * w @ w + Jo, w + L_B.T @ g_x

        res = minimize(cost_and_grad, np.zeros_like(background), jac=True,
                       method="L-BFGS-B",
                       options=dict(gtol=gtol, ftol=ftol, maxiter=maxiter))
        stats.append(dict(k=k, nit=res.nit, nfev=res.nfev, success=res.success,
                          message=res.message, cost=res.fun))
        return background + L_B @ res.x

    analyze.stats = stats
    analyze.L_B = L_B
    return analyze


def time_integrated_rmse(truth_segments, est_segments, dt, skip_cycles=0):
    """Time-integrated RMSE, optionally leaving out the first windows.

    Same as assimilation.time_integrated_rmse when skip_cycles = 0. With
    skip_cycles = s the integral runs over [s*dt, T] (normalized by its own
    length), e.g. to leave out the initial spin-up of the filter.
    """
    truth_segments = truth_segments[skip_cycles:]
    est_segments = est_segments[skip_cycles:]
    n_cycles, n_pts, _ = truth_segments.shape
    h = dt / (n_pts - 1)
    w = np.full(n_pts, h)
    w[0] = w[-1] = 0.5 * h
    sq = error_norm_segments(truth_segments, est_segments) ** 2
    return np.sqrt(np.sum(sq @ w) / (n_cycles * dt))
