"""Cycled data assimilation: model propagation, synthetic observations,
3DVAR, and error metrics.

Conventions
-----------
* Observations arrive every `dt` time units at t_k = k*dt, k = 1, ..., N.
* M is "integrate the model forward by dt" (solve_ivp, HW 1 tolerances).
  The benchmark models are autonomous, so every window is integrated on
  [0, dt] and only the elapsed time matters.
* Each window is also sampled at `n_sub` sub-steps so the estimate can be
  plotted and the time-integrated RMSE computed between observation times.

The cycling loop is method-agnostic. At each observation time t_k the
method's `analyze(k, background)` turns the background (the forecast that
arrived at t_k) into the state that is carried forward from t_k:

    k = 0      background = initial guess
    for k = 0..N:
        analysis_k  = analyze(k, background_k)
        if k < N:   forecast analysis_k over [t_k, t_{k+1}] -> background_{k+1}

3DVAR uses y_k inside analyze(k, .) (and leaves k = 0 alone, since there is
no y_0). A no-assimilation "free run" is analyze = identity. The same loop
will drive 4DVAR in Problem 5.
"""

import time

import numpy as np
from scipy.integrate import solve_ivp


# ---------------------------------------------------------------------------
# Model propagation
# ---------------------------------------------------------------------------

def make_propagator(rhs, params, dt, n_sub=10, rtol=1e-9, atol=1e-9):
    """Return propagate(x) -> (n_sub+1, n) samples of the solution on [0, dt].

    Row 0 is x itself and row -1 is M(x).
    """
    params = params or {}
    t_eval = np.linspace(0.0, dt, n_sub + 1)

    def f(t, x):
        return rhs(t, x, **params)

    def propagate(x):
        sol = solve_ivp(f, (0.0, dt), np.asarray(x, dtype=float),
                        t_eval=t_eval, rtol=rtol, atol=atol)
        if not sol.success:
            raise RuntimeError(f"Propagation failed: {sol.message}")
        return sol.y.T

    propagate.dt = dt
    propagate.n_sub = n_sub
    return propagate


# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------

def selection_operator(n, observed=None):
    """Linear observation operator that picks out the listed state indices.

    observed=None gives the identity (full observation).
    """
    if observed is None:
        return np.eye(n)
    observed = np.asarray(observed, dtype=int)
    H = np.zeros((len(observed), n))
    H[np.arange(len(observed)), observed] = 1.0
    return H


def simulate_truth(propagate, x0, n_cycles):
    """True trajectory as a stack of windows: shape (n_cycles, n_sub+1, n)."""
    segments = []
    x = np.asarray(x0, dtype=float)
    for _ in range(n_cycles):
        seg = propagate(x)
        segments.append(seg)
        x = seg[-1]
    return np.array(segments)


def observe(truth_segments, H, sigma, standard_normal):
    """y_k = H x_k + sigma * eps_k at t_k = k*dt, k = 1..N.

    `standard_normal` is a pre-drawn (N, m) array of N(0, 1) samples, so the
    same noise realization can be reused across noise levels and methods.
    Returns Y with Y[k-1] = y_k.
    """
    x_obs_times = truth_segments[:, -1, :]          # x_1, ..., x_N
    return x_obs_times @ H.T + sigma * standard_normal


# ---------------------------------------------------------------------------
# Analysis schemes
# ---------------------------------------------------------------------------

def free_run():
    """No assimilation: the forecast is never corrected."""
    def analyze(k, background):
        return background
    return analyze


def make_3dvar(H, R, B, Y):
    """3DVAR with a fixed background covariance B.

    Minimizer of J = 1/2 |y - Hx|^2_{R^-1} + 1/2 |x - mu|^2_{B^-1}, written
    in gain form so B is never inverted:

        x_a = mu + K (y - H mu),     K = B H^T (H B H^T + R)^{-1}.

    B, H, and R are fixed, so K is computed once and reused every cycle.
    """
    S = H @ B @ H.T + R                               # (m, m), SPD
    K = np.linalg.solve(S, H @ B).T                   # = B H^T S^{-1}

    def analyze(k, background):
        if k == 0:                                    # no observation at t_0
            return background
        innovation = Y[k - 1] - H @ background
        return background + K @ innovation

    analyze.gain = K
    return analyze


# ---------------------------------------------------------------------------
# Cycling driver
# ---------------------------------------------------------------------------

def run_cycle(propagate, x_background0, n_cycles, analyze):
    """Run the forecast/analysis cycle on [0, n_cycles*dt].

    Returns a dict with
        segments     (N, n_sub+1, n)  forecast from each analysis over its window
        backgrounds  (N+1, n)         background at t_0..t_N (t_0: initial guess)
        analyses     (N+1, n)         state carried forward from t_0..t_N
        wall_time    seconds for the whole cycle (forecasts + analyses)
    """
    backgrounds, analyses, segments = [], [], []
    background = np.asarray(x_background0, dtype=float)

    tic = time.perf_counter()
    for k in range(n_cycles + 1):
        backgrounds.append(background)
        xa = analyze(k, background)
        analyses.append(xa)
        if k == n_cycles:
            break
        seg = propagate(xa)
        segments.append(seg)
        background = seg[-1]
    wall_time = time.perf_counter() - tic

    return dict(segments=np.array(segments), backgrounds=np.array(backgrounds),
                analyses=np.array(analyses), wall_time=wall_time)


# ---------------------------------------------------------------------------
# Error metrics
# ---------------------------------------------------------------------------

def error_norm_segments(truth_segments, est_segments, components=None):
    """||x_true(t) - x_est(t)||_2 on every sub-step: shape (N, n_sub+1)."""
    diff = truth_segments - est_segments
    if components is not None:
        diff = diff[..., components]
    return np.sqrt(np.sum(diff ** 2, axis=-1))


def time_integrated_rmse(truth_segments, est_segments, dt):
    """sqrt( (1/T) int_0^T ||x_true - x_est||_2^2 dt ), trapezoid per window.

    Each window is integrated separately, so the jumps at analysis times
    (where the estimate is discontinuous) are handled correctly.
    """
    n_cycles, n_pts, _ = truth_segments.shape
    h = dt / (n_pts - 1)
    w = np.full(n_pts, h)
    w[0] = w[-1] = 0.5 * h
    sq = error_norm_segments(truth_segments, est_segments) ** 2
    return np.sqrt(np.sum(sq @ w) / (n_cycles * dt))


def final_time_rmse(truth_segments, run):
    """||x_true(T) - x_est(T)||_2, using the state held at T (the final analysis)."""
    return np.linalg.norm(truth_segments[-1, -1] - run["analyses"][-1])


def flatten(segments, dt, final_state=None):
    """Concatenate windows for plotting.

    Window boundaries appear twice (end of one forecast, start of the next),
    so plotted lines show the analysis jumps as vertical strokes. If
    `final_state` is given it is appended at t = T.
    """
    n_cycles, n_pts, n = segments.shape
    tau = np.linspace(0.0, dt, n_pts)
    t = (np.arange(n_cycles)[:, None] * dt + tau[None, :]).ravel()
    x = segments.reshape(-1, n)
    if final_state is not None:
        t = np.append(t, n_cycles * dt)
        x = np.vstack([x, final_state])
    return t, x
