"""Stochastic (perturbed-observation) ensemble Kalman filter.

Conventions
-----------
* An ensemble is an (n, Ne) array: one column per member.
* Observations y_k arrive at t_k = k*dt, k = 1..N, stored as Y[k-1]; a row
  of NaNs means "no observation" (the filter just forecasts through it).
  k = 0 is the initial ensemble (no y_0), as in kalman.py.
* The forecast advances every member with the same deterministic solver
  used for the truth (solve_ivp, HW 1 tolerances), sampled at n_sub
  sub-steps per window so the ensemble can be plotted between analyses.
* Q = 0: no model noise, no inflation, no localization. The only random
  numbers inside the filter are the observation perturbations, drawn from
  the generator passed to run_enkf.

Update (HW 7, Section 7 of the notes)
-------------------------------------
    x_f^(i) = M(x_a^(i)),       xbar_f = mean_i x_f^(i),
    P_f     = (1/(Ne-1)) sum_i (x_f^(i) - xbar_f)(x_f^(i) - xbar_f)^T,
    K       = P_f H^T (H P_f H^T + R)^{-1}      (by solving S K^T = H P_f),
    x_a^(i) = x_f^(i) + K (y + v^(i) - H x_f^(i)),   v^(i) ~ N(0, R) i.i.d.,

and the analysis mean and covariance are recomputed from the updated members.
"""

import time

import numpy as np
from scipy.integrate import solve_ivp


# ---------------------------------------------------------------------------
# Ensemble propagation
# ---------------------------------------------------------------------------

def make_ensemble_propagator(rhs, params, dt, n_sub=10, rtol=1e-9, atol=1e-9,
                             vectorized=True):
    """Return propagate(E) -> (n_sub+1, n, Ne): every member over [0, dt].

    vectorized=True stacks all members into one solve_ivp call, which needs
    an rhs that broadcasts over a trailing member axis (state shape (n, Ne)).
    lorenz63.rhs does. Members do not interact, so this is the same ODE
    solved member by member, just with shared adaptive steps (the error
    control then covers every member at once). vectorized=False loops over
    members instead, for any rhs.
    """
    params = params or {}
    t_eval = np.linspace(0.0, dt, n_sub + 1)

    def _solve(f, x0):
        sol = solve_ivp(f, (0.0, dt), x0, t_eval=t_eval, rtol=rtol, atol=atol)
        if not sol.success:
            raise RuntimeError(f"Ensemble propagation failed: {sol.message}")
        return sol.y

    def propagate(E):
        E = np.asarray(E, dtype=float)
        n, Ne = E.shape
        if vectorized:
            def f(t, z):
                return np.asarray(rhs(t, z.reshape(n, Ne), **params)).ravel()
            Y = _solve(f, E.ravel())                       # (n*Ne, n_sub+1)
            return Y.T.reshape(n_sub + 1, n, Ne)
        out = np.empty((n_sub + 1, n, Ne))
        for i in range(Ne):
            out[:, :, i] = _solve(lambda t, x: rhs(t, x, **params), E[:, i]).T
        return out

    propagate.dt = dt
    propagate.n_sub = n_sub
    return propagate


# ---------------------------------------------------------------------------
# Ensemble statistics and initialization
# ---------------------------------------------------------------------------

def ensemble_mean_cov(E):
    """Sample mean (n,) and covariance (n, n) with the 1/(Ne-1) normalization."""
    mean = E.mean(axis=1)
    A = E - mean[:, None]
    return mean, A @ A.T / (E.shape[1] - 1)


def draw_initial_ensemble(m0, P0, Ne, rng):
    """Ne independent draws from N(m0, P0), as an (n, Ne) array."""
    m0 = np.asarray(m0, dtype=float)
    L = np.linalg.cholesky(np.asarray(P0, dtype=float))
    return m0[:, None] + L @ rng.standard_normal((m0.size, Ne))


# ---------------------------------------------------------------------------
# Forecast and analysis
# ---------------------------------------------------------------------------

def enkf_forecast(E_a, propagate):
    """Advance every member one window. Returns (segments, E_f).

    segments has shape (n_sub+1, n, Ne); E_f = segments[-1].
    """
    segments = propagate(E_a)
    return segments, segments[-1]


def enkf_analysis(E_f, y, H, R, rng):
    """Stochastic EnKF analysis with perturbed observations, linear H.

    Returns E_a and a dict of diagnostics:
        mean_f, P_f   forecast-ensemble mean and sample covariance
        K             (n, m) ensemble gain, from S K^T = H P_f
        innovations   (m, Ne) perturbed innovations y + v^(i) - H x_f^(i)
        innovation_mean   y - H xbar_f (the unperturbed mean innovation)
        increments    (n, Ne) K @ innovations, the change to each member
    """
    H = np.atleast_2d(H)
    R = np.atleast_2d(R)
    m, Ne = H.shape[0], E_f.shape[1]
    mean_f, P_f = ensemble_mean_cov(E_f)

    S = H @ P_f @ H.T + R                                   # (m, m), SPD
    K = np.linalg.solve(S, H @ P_f).T                       # = P_f H^T S^{-1}

    V = np.linalg.cholesky(R) @ rng.standard_normal((m, Ne))   # v^(i) ~ N(0, R)
    D = (y[:, None] + V) - H @ E_f                          # perturbed innovations
    increments = K @ D
    E_a = E_f + increments

    info = dict(mean_f=mean_f, P_f=P_f, K=K, innovations=D,
                innovation_mean=y - H @ mean_f, increments=increments)
    return E_a, info


# ---------------------------------------------------------------------------
# Cycling driver
# ---------------------------------------------------------------------------

def run_enkf(propagate, E0, Y, H, R, rng):
    """Cycle the stochastic EnKF over k = 1..N.

    Parameters
    ----------
    propagate : from make_ensemble_propagator (or any E -> (n_sub+1, n, Ne))
    E0        : (n, Ne) initial ensemble (the analysis at k = 0)
    Y         : (N, m) observations, Y[k-1] = y_k; a NaN row means no obs
    H, R      : linear observation operator and observation-error covariance
    rng       : numpy Generator for the observation perturbations only

    Returns a dict
        mean_a, cov_a     (N+1, n), (N+1, n, n)   analysis stats; row 0 from E0
        mean_f, cov_f     (N+1, n), (N+1, n, n)   forecast stats; row 0 from E0
        E_a, E_f          (N+1, n, Ne)            the ensembles themselves
        K                 (N, n, m)               gains (NaN if no obs)
        innovation_mean   (N, m)                  y_k - H xbar_f_k
        innovations       (N, m, Ne)              y_k + v_k^(i) - H x_f^(i)
        seg_mean, seg_std (N, n_sub+1, n)         ensemble mean / std inside
                                                  each forecast window
        wall_time         seconds for all forecasts + analyses
    """
    H = np.atleast_2d(np.asarray(H, dtype=float))
    R = np.atleast_2d(np.asarray(R, dtype=float))
    Y = np.asarray(Y, dtype=float).reshape(len(Y), -1)
    E0 = np.asarray(E0, dtype=float)
    N, m = Y.shape
    n, Ne = E0.shape

    E_a = np.empty((N + 1, n, Ne));  E_f = np.empty((N + 1, n, Ne))
    mean_a = np.empty((N + 1, n));   cov_a = np.empty((N + 1, n, n))
    mean_f = np.empty((N + 1, n));   cov_f = np.empty((N + 1, n, n))
    K = np.full((N, n, m), np.nan)
    innov = np.full((N, m), np.nan)
    innov_members = np.full((N, m, Ne), np.nan)
    seg_mean, seg_std = [], []

    E_a[0] = E_f[0] = E0
    mean_a[0], cov_a[0] = ensemble_mean_cov(E0)
    mean_f[0], cov_f[0] = mean_a[0], cov_a[0]

    tic = time.perf_counter()
    for k in range(1, N + 1):
        segments, E_f[k] = enkf_forecast(E_a[k - 1], propagate)
        seg_mean.append(segments.mean(axis=2))
        seg_std.append(segments.std(axis=2, ddof=1))
        y = Y[k - 1]
        if np.all(np.isnan(y)):
            E_a[k] = E_f[k]
            mean_f[k], cov_f[k] = ensemble_mean_cov(E_f[k])
        else:
            E_a[k], info = enkf_analysis(E_f[k], y, H, R, rng)
            mean_f[k], cov_f[k] = info["mean_f"], info["P_f"]
            K[k - 1], innov[k - 1] = info["K"], info["innovation_mean"]
            innov_members[k - 1] = info["innovations"]
        mean_a[k], cov_a[k] = ensemble_mean_cov(E_a[k])
    wall_time = time.perf_counter() - tic

    return dict(mean_a=mean_a, cov_a=cov_a, mean_f=mean_f, cov_f=cov_f,
                E_a=E_a, E_f=E_f, K=K, innovation_mean=innov,
                innovations=innov_members,
                seg_mean=np.array(seg_mean), seg_std=np.array(seg_std),
                wall_time=wall_time)
