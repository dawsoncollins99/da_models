"""Ensemble score filter (EnSF), HW 9 Problem 3.

Follows Section 3 of the MATH 513R diffusion notes (Bao, Zhang, and Zhang,
2024) with the whole forecast ensemble used in every score evaluation.

Conventions (as in enkf.py)
---------------------------
* Ensembles are (n, Ne) arrays with one member per COLUMN. (The class
  notebook stores members in rows; that is only a transpose.)
* x_k at t_k = k*dt, k = 0..N; the initial ensemble is the analysis at k = 0;
  y_k (k = 1..N) is stored as Y[k-1], and a NaN row means "no observation".
* The analysis works in rescaled coordinates u = D^{-1}(x - c) with a fixed
  center c and fixed scales D (a 1-D array of the diagonal of D).
  Observations stay in their original units, so R is unchanged and

      grad_z log l(c + D z) = (H D)^T R^{-1} (y - H (c + D z)).

Artificial time tau runs from 1 (noisy end) to 0 (analysis); it is
unrelated to physical time t_k. Regularized schedule (notes eq. 19-20):

    a_tau = 1 - tau (1 - eps_a),       v_tau = eps_b + tau (1 - eps_b),
    b_tau = -(1 - eps_a) / a_tau,      g_tau^2 = (1 - eps_b) - 2 b_tau v_tau.

Reverse update (notes eq. 24), coefficients evaluated at tau_old:

    z_new = z + [-b z + g^2 s~(z, tau)] dtau + g sqrt(dtau) xi,
    s~(z, tau) = s_f(z, tau) + (1 - tau) grad_z log l(z)        (eq. 23)

where s_f is the exact score of the Gaussian mixture
(1/Ne) sum_j N(z; a_tau u_f^(j), v_tau I) built from the forecast members
(eq. 21-22).
"""

import time

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from .enkf import enkf_forecast, ensemble_mean_cov


# ---------------------------------------------------------------------------
# Schedule, scores, and the reverse sampler
# ---------------------------------------------------------------------------

def schedule(tau, eps_a=0.01, eps_b=0.01):
    """Regularized EnSF coefficients (a, v, b, g^2) at artificial time tau."""
    a = 1.0 - tau * (1.0 - eps_a)
    v = eps_b + tau * (1.0 - eps_b)
    b = -(1.0 - eps_a) / a
    g2 = (1.0 - eps_b) - 2.0 * b * v
    return a, v, b, g2


def mixture_score(Z, UF, a, v, UF_sq=None):
    """Score of p_f(z) = (1/Ne) sum_j N(z; a u_f^(j), v I) at each column of Z.

    Z     : (n, Ns) current sampler positions
    UF    : (n, Ne) forecast members in rescaled coordinates
    UF_sq : optional precomputed |u_f^(j)|^2, shape (Ne,)
    Returns (score, rho): score is (n, Ns); rho[i, j] is the local weight of
    forecast cloud j at sample i (rows sum to one).

    The largest log weight in each row is subtracted before exponentiating,
    so the nearest cloud always has weight exp(0) = 1 before normalizing and
    the weights cannot all underflow to zero.
    """
    if UF_sq is None:
        UF_sq = np.sum(UF * UF, axis=0)
    # squared distances |z_i - a u_j|^2 for all pairs, shape (Ns, Ne)
    d2 = (np.sum(Z * Z, axis=0)[:, None] + (a * a) * UF_sq[None, :]
          - (2.0 * a) * (Z.T @ UF))
    logw = -np.maximum(d2, 0.0) / (2.0 * v)
    logw -= logw.max(axis=1, keepdims=True)
    rho = np.exp(logw)
    rho /= rho.sum(axis=1, keepdims=True)
    score = (a * (UF @ rho.T) - Z) / v        # sum_j rho_ij (a u_j - z_i) / v
    return score, rho


def make_likelihood_score(y, H, R, c, D):
    """Return Z -> grad_z log l(c + D z) for the columns of Z.

    grad_z log l(c + D z) = (H D)^T R^{-1} (y - H c - (H D) z), with y, H, R
    in their original units (R is not rescaled). It is affine in z, so it is
    precomputed once per analysis as  beta - A z  with
        A    = (H D)^T R^{-1} (H D)       (n x n)
        beta = (H D)^T R^{-1} (y - H c)   (n,)
    where R^{-1} is applied by Cholesky solves, never an explicit inverse.
    """
    H = np.atleast_2d(np.asarray(H, dtype=float))
    HD = H * np.asarray(D, dtype=float)[None, :]          # H @ diag(D)
    r0 = np.asarray(y, dtype=float) - H @ c               # y - H c
    R_chol = cho_factor(np.atleast_2d(np.asarray(R, dtype=float)))
    A = HD.T @ cho_solve(R_chol, HD)
    beta = HD.T @ cho_solve(R_chol, r0)

    def likelihood_score(Z):
        return beta[:, None] - A @ Z

    likelihood_score.A = A
    likelihood_score.beta = beta
    return likelihood_score


def reverse_sample(Z1, score_fn, L=600, eps_a=0.01, eps_b=0.01, rng=None,
                   record_every=None):
    """Euler-Maruyama reverse sampler from tau = 1 to tau = 0 (notes eq. 24).

    Z1        : (n, Ns) starting positions (normally N(0, I) draws)
    score_fn  : (Z, tau, a, v) -> (n, Ns) approximate score at tau
    L         : number of reverse steps, dtau = 1/L
    rng       : Generator for the sampling noise (a fresh xi per sample and step)
    record_every : if an integer m, also return the positions at
                   tau = 1, 1 - m dtau, ..., 0 (for plotting sample paths)

    Returns Z0, or (Z0, taus, path) when record_every is set.
    """
    Z = np.array(Z1, dtype=float, copy=True)
    dtau = 1.0 / L
    taus, path = [1.0], [Z.copy()] if record_every else None
    for j in range(L):
        tau = (L - j) / L                                  # tau_old
        a, v, b, g2 = schedule(tau, eps_a, eps_b)
        s = score_fn(Z, tau, a, v)
        Z = Z + (-b * Z + g2 * s) * dtau + np.sqrt(g2 * dtau) * rng.standard_normal(Z.shape)
        if record_every and ((j + 1) % record_every == 0 or j == L - 1):
            taus.append((L - j - 1) / L)
            path.append(Z.copy())
    if record_every:
        return Z, np.array(taus), np.array(path)
    return Z


# ---------------------------------------------------------------------------
# One EnSF analysis and the cycling driver
# ---------------------------------------------------------------------------

def ensf_analysis(Ef, y, H, R, c, D, rng, L=600, eps_a=0.01, eps_b=0.01,
                  n_samples=None, record_every=None):
    """EnSF analysis at one observation time.

    Ef : (n, Ne) forecast ensemble in physical units (held fixed throughout)
    y  : observation in original units (held fixed; NOT perturbed)
    c, D : fixed center and scales, u = D^{-1}(x - c)
    rng  : Generator for the starting positions and the sampling noise
    n_samples : number of analysis samples (default Ne)

    Returns (Ea, info). Ea is (n, n_samples) in physical units. info has the
    forecast mean/covariance, the terminal local weights' effective sample
    size per analysis sample (1 / sum_j rho_ij^2 at the last reverse step),
    the number of distinct forecast members that dominate the terminal
    weights, and (if record_every) the reverse paths in physical units.
    """
    Ef = np.asarray(Ef, dtype=float)
    c = np.asarray(c, dtype=float)
    D = np.asarray(D, dtype=float)
    n, Ne = Ef.shape
    Ns = Ne if n_samples is None else n_samples

    UF = (Ef - c[:, None]) / D[:, None]                    # forecast in u coordinates
    UF_sq = np.sum(UF * UF, axis=0)
    lik = make_likelihood_score(y, H, R, c, D)
    last = {}

    def score_fn(Z, tau, a, v):
        s_f, rho = mixture_score(Z, UF, a, v, UF_sq)
        last["rho"] = rho
        return s_f + (1.0 - tau) * lik(Z)

    Z1 = rng.standard_normal((n, Ns))                      # new positions at tau = 1
    out = reverse_sample(Z1, score_fn, L, eps_a, eps_b, rng, record_every)
    Z0 = out[0] if record_every else out
    Ea = c[:, None] + D[:, None] * Z0                      # back to physical units

    rho = last["rho"]                                      # weights at the last step
    mean_f, P_f = ensemble_mean_cov(Ef)
    info = dict(mean_f=mean_f, P_f=P_f,
                ess=1.0 / np.sum(rho ** 2, axis=1),
                n_distinct=len(np.unique(np.argmax(rho, axis=1))))
    if record_every:
        _, taus, path = out
        info["taus"] = taus
        info["path"] = c[None, :, None] + D[None, :, None] * path
    return Ea, info


def finite_states(E):
    """Default physical-domain check: every entry finite (Lorenz-96 has no
    other restriction)."""
    return bool(np.all(np.isfinite(E)))


def run_ensf(propagate, E0, Y, H, R, c, D, rng, L=600, eps_a=0.01, eps_b=0.01,
             admissible=finite_states, record_k=None, record_every=10):
    """Cycle the EnSF over k = 1..N (Q = 0).

    Parameters
    ----------
    propagate  : from enkf.make_ensemble_propagator (E -> (n_sub+1, n, Ne)),
                 the same physical model as the truth and the EnKF
    E0         : (n, Ne) initial ensemble (the analysis at k = 0)
    Y          : (N, m) observations, Y[k-1] = y_k; a NaN row means no obs
    H, R       : linear observation operator and covariance (original units)
    c, D       : fixed center and scales of u = D^{-1}(x - c)
    rng        : Generator for the reverse-sampler noise only
    L, eps_a, eps_b : reverse steps and endpoint regularization
    admissible : E -> bool, the physical-domain check applied to the initial
                 ensemble and to every analysis ensemble before it is
                 propagated (and to every forecast)
    record_k   : optional analysis index whose reverse paths are kept

    Returns a dict like run_enkf's: mean_a, cov_a, mean_f, cov_f, E_a, E_f,
    seg_mean, seg_std, wall_time (forecasts + analyses + ensemble
    statistics), analysis_time (reverse sampling only), ess (N, Ne) and
    n_distinct (N,) terminal-weight diagnostics, paths (dict or None),
    failed_at (None or the step k), and message. After a failure every
    remaining entry is NaN: nothing is clipped, and the run is reported as
    failed rather than dropped.
    """
    H = np.atleast_2d(np.asarray(H, dtype=float))
    Y = np.asarray(Y, dtype=float).reshape(len(Y), -1)
    E0 = np.asarray(E0, dtype=float)
    N = Y.shape[0]
    n, Ne = E0.shape

    nan = np.nan
    E_a = np.full((N + 1, n, Ne), nan);  E_f = np.full((N + 1, n, Ne), nan)
    mean_a = np.full((N + 1, n), nan);   cov_a = np.full((N + 1, n, n), nan)
    mean_f = np.full((N + 1, n), nan);   cov_f = np.full((N + 1, n, n), nan)
    ess = np.full((N, Ne), nan);         n_distinct = np.full(N, nan)
    seg_mean = seg_std = None
    paths = None
    failed_at, message = None, ""
    analysis_time = 0.0

    E_a[0] = E_f[0] = E0
    mean_a[0], cov_a[0] = ensemble_mean_cov(E0)
    mean_f[0], cov_f[0] = mean_a[0], cov_a[0]
    if not admissible(E0):
        failed_at, message = 0, "inadmissible initial member"

    tic = time.perf_counter()
    for k in range(1, N + 1):
        if failed_at is not None:
            break
        try:
            segments, Ef = enkf_forecast(E_a[k - 1], propagate)
        except RuntimeError as err:
            failed_at, message = k, f"solver failure: {err}"
            break
        if seg_mean is None:
            seg_mean = np.full((N,) + segments.shape[:2], nan)
            seg_std = np.full((N,) + segments.shape[:2], nan)
        if not admissible(Ef):
            failed_at, message = k, "inadmissible forecast member"
            break
        E_f[k] = Ef
        seg_mean[k - 1] = segments.mean(axis=2)
        seg_std[k - 1] = segments.std(axis=2, ddof=1)
        y = Y[k - 1]
        if np.all(np.isnan(y)):
            E_a[k] = Ef
            mean_f[k], cov_f[k] = ensemble_mean_cov(Ef)
        else:
            t_an = time.perf_counter()
            rec = record_every if k == record_k else None
            Ea, info = ensf_analysis(Ef, y, H, R, c, D, rng, L, eps_a, eps_b,
                                     record_every=rec)
            analysis_time += time.perf_counter() - t_an
            mean_f[k], cov_f[k] = info["mean_f"], info["P_f"]
            ess[k - 1], n_distinct[k - 1] = info["ess"], info["n_distinct"]
            if rec:
                paths = dict(k=k, taus=info["taus"], path=info["path"])
            if not admissible(Ea):
                failed_at, message = k, "inadmissible terminal state"
                break
            E_a[k] = Ea
        mean_a[k], cov_a[k] = ensemble_mean_cov(E_a[k])
    wall_time = time.perf_counter() - tic

    if seg_mean is None:                       # failed before the first forecast
        n_pts = getattr(propagate, "n_sub", 0) + 1
        seg_mean = np.full((N, n_pts, n), nan)
        seg_std = np.full((N, n_pts, n), nan)

    return dict(mean_a=mean_a, cov_a=cov_a, mean_f=mean_f, cov_f=cov_f,
                E_a=E_a, E_f=E_f, seg_mean=seg_mean, seg_std=seg_std,
                ess=ess, n_distinct=n_distinct, paths=paths,
                wall_time=wall_time, analysis_time=analysis_time,
                failed_at=failed_at, message=message)
