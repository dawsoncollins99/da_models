"""Linear Kalman filter: forecast and analysis steps, a cycling driver,
twin-experiment helpers for linear models, and RMSE/spread diagnostics.

Indexing (matches HW 7)
-----------------------
* x_k is the state at t_k = k*dt, k = 0, ..., N.
* The prior X_0 ~ N(m_0, P_0) is the analysis at k = 0 (no y_0).
* Observations y_k arrive at k = 1, ..., N and are stored as Y[k-1].
  A row of NaNs in Y means "no observation at this step": the filter just
  forecasts through it, so sparser observation schedules need no new code.

The model may be a fixed matrix M or a callable M(k) returning M_{k-1}, the
matrix that maps x_{k-1} to x_k. The same goes for Q (default Q = 0).
"""

import numpy as np


# ---------------------------------------------------------------------------
# Linear twin-experiment helpers
# ---------------------------------------------------------------------------

def _as_step(op):
    """Wrap a fixed matrix (or None) as a function of k."""
    if op is None or callable(op):
        return op
    op = np.asarray(op, dtype=float)
    return lambda k: op


def simulate_linear(M, x0, n_steps):
    """True trajectory x_0, ..., x_N of x_k = M_{k-1} x_{k-1}: shape (N+1, n)."""
    M_at = _as_step(M)
    x0 = np.asarray(x0, dtype=float)
    X = np.empty((n_steps + 1, x0.size))
    X[0] = x0
    for k in range(1, n_steps + 1):
        X[k] = M_at(k) @ X[k - 1]
    return X


def synthetic_observations(truth, H, R, rng):
    """y_k = H x_k + v_k with v_k ~ N(0, R) i.i.d., for k = 1..N.

    `truth` has shape (N+1, n) (row 0 is x_0, which is not observed).
    Returns Y of shape (N, m) with Y[k-1] = y_k.
    """
    H = np.atleast_2d(H)
    R = np.atleast_2d(R)
    X = truth[1:]
    L = np.linalg.cholesky(R)
    V = rng.standard_normal((X.shape[0], H.shape[0])) @ L.T
    return X @ H.T + V


# ---------------------------------------------------------------------------
# Kalman filter steps
# ---------------------------------------------------------------------------

def kf_forecast(m, P, M, Q=None):
    """m_f = M m,  P_f = M P M^T + Q."""
    m_f = M @ m
    P_f = M @ P @ M.T
    if Q is not None:
        P_f = P_f + Q
    return m_f, P_f


def kf_analysis(m_f, P_f, y, H, R):
    """One Kalman analysis.

        S = H P_f H^T + R,   K = P_f H^T S^{-1},   d = y - H m_f,
        m_a = m_f + K d,
        P_a = (I - K H) P_f (I - K H)^T + K R K^T     (Joseph form).

    K is obtained by solving S K^T = H P_f (S and P_f symmetric), never by
    forming S^{-1}. The Joseph form equals (I - KH) P_f for the optimal K
    but stays symmetric positive semidefinite in floating point, which
    matters over the ~10^5 analyses of HW 7, where P_a becomes very small.

    Returns m_a, P_a, K, d.
    """
    n = m_f.size
    S = H @ P_f @ H.T + R
    K = np.linalg.solve(S, H @ P_f).T
    d = y - H @ m_f
    m_a = m_f + K @ d
    I_KH = np.eye(n) - K @ H
    P_a = I_KH @ P_f @ I_KH.T + K @ R @ K.T
    return m_a, P_a, K, d


def run_kf(M, H, R, Y, m0, P0, Q=None):
    """Cycle the Kalman filter over k = 1..N.

    Parameters
    ----------
    M : (n, n) array or callable k -> M_{k-1}
    H : (m, n) array
    R : (m, m) array or scalar (for m = 1)
    Y : (N, m) array, Y[k-1] = y_k; a row of NaNs means no observation
    m0, P0 : prior mean and covariance at k = 0
    Q : (n, n) array, callable, or None (perfect model)

    Returns a dict of arrays
        m_a, P_a      (N+1, n), (N+1, n, n)   analysis; row 0 is the prior
        m_f, P_f      (N+1, n), (N+1, n, n)   forecast; row 0 is the prior
        K             (N, n, m)               gain at k = 1..N (NaN if no obs)
        innovation    (N, m)                  y_k - H m_f_k   (NaN if no obs)
    """
    M_at, Q_at = _as_step(M), _as_step(Q)
    H = np.atleast_2d(np.asarray(H, dtype=float))
    R = np.atleast_2d(np.asarray(R, dtype=float))
    Y = np.asarray(Y, dtype=float).reshape(len(Y), -1)
    N, m = Y.shape
    n = H.shape[1]

    m_a = np.empty((N + 1, n));    P_a = np.empty((N + 1, n, n))
    m_f = np.empty((N + 1, n));    P_f = np.empty((N + 1, n, n))
    K = np.full((N, n, m), np.nan)
    innov = np.full((N, m), np.nan)

    m_a[0] = m_f[0] = np.asarray(m0, dtype=float)
    P_a[0] = P_f[0] = np.asarray(P0, dtype=float)

    for k in range(1, N + 1):
        Qk = None if Q_at is None else Q_at(k)
        m_f[k], P_f[k] = kf_forecast(m_a[k - 1], P_a[k - 1], M_at(k), Qk)
        y = Y[k - 1]
        if np.all(np.isnan(y)):
            m_a[k], P_a[k] = m_f[k], P_f[k]
            continue
        m_a[k], P_a[k], K[k - 1], innov[k - 1] = kf_analysis(m_f[k], P_f[k], y, H, R)

    return dict(m_a=m_a, P_a=P_a, m_f=m_f, P_f=P_f, K=K, innovation=innov)


# ---------------------------------------------------------------------------
# Independent check (perfect model only)
# ---------------------------------------------------------------------------

def perfect_model_posterior(M, H, R, Y, m0, P0, k):
    """Posterior of x_k given y_1..y_k, computed in one batch (Q = 0, fixed M).

    With a perfect model x_j = M^j x_0, so every observation is a linear
    measurement of x_0: y_j = (H M^j) x_0 + v_j. The posterior of x_0 is
    Gaussian with

        precision  L  = P0^{-1} + sum_j (H M^j)^T R^{-1} (H M^j),
        mean       mu = L^{-1} ( P0^{-1} m0 + sum_j (H M^j)^T R^{-1} y_j ),

    and x_k = M^k x_0 maps it to time k. This never runs the filter, so it
    is an independent check of run_kf.
    """
    H = np.atleast_2d(np.asarray(H, dtype=float))
    R = np.atleast_2d(np.asarray(R, dtype=float))
    Y = np.asarray(Y, dtype=float).reshape(len(Y), -1)
    n = H.shape[1]
    P0_inv = np.linalg.inv(P0)
    R_inv = np.linalg.inv(R)
    L = P0_inv.copy()
    b = P0_inv @ np.asarray(m0, dtype=float)
    Mj = np.eye(n)
    for j in range(1, k + 1):
        Mj = M @ Mj
        G = H @ Mj
        L += G.T @ R_inv @ G
        b += G.T @ R_inv @ Y[j - 1]
    cov0 = np.linalg.inv(L)
    mean0 = cov0 @ b
    return Mj @ mean0, Mj @ cov0 @ Mj.T


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def rmse_and_spread(means, covs, truth):
    """RMSE_k = sqrt(|m_k - x_k|^2 / n),  Spread_k = sqrt(tr(P_k) / n).

    `means` (N+1, n), `covs` (N+1, n, n), `truth` (N+1, n). Works for the
    Kalman filter (m_a, P_a) and for an ensemble (mean, sample covariance).
    """
    n = truth.shape[1]
    rmse = np.sqrt(np.sum((means - truth) ** 2, axis=1) / n)
    spread = np.sqrt(np.trace(covs, axis1=1, axis2=2) / n)
    return rmse, spread
