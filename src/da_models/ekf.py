"""Extended Kalman filter (HW 8, Problem 2).

Conventions match kalman.py and enkf.py: x_k at t_k = k*dt, k = 0..N; the
prior N(m_0, P_0) is the analysis at k = 0; y_k (k = 1..N) is stored as
Y[k-1], and a row of NaNs means "no observation" (the filter forecasts
through it). Q = 0 throughout HW 8.

One cycle
---------
    m_f = M(m_a)                   the same numerical model map as the truth
                                   (assimilation.make_propagator)
    M   = DM(m_a)                  Jacobian of the flow map over the whole
                                   window, from the variational equations
    P_f = M P_a M^T                (+ Q, here 0), then symmetrized
    analysis                       kalman.kf_analysis: K from S K^T = H P_f
                                   (a linear solve, no inverse) and the
                                   Joseph update; P_a is then symmetrized

The flow-map Jacobian is NOT Df: it is Phi(dt), the solution of the
variational equations along the forecast trajectory,

    dx/dt = f(x),    dPhi/dt = Df(x(t)) Phi,    x(0) = m_a, Phi(0) = I,

integrated together as one (n + n^2)-dimensional ODE with solve_ivp.
"""

import time

import numpy as np
from scipy.integrate import solve_ivp

from .kalman import kf_analysis


# ---------------------------------------------------------------------------
# Flow-map Jacobian
# ---------------------------------------------------------------------------

def make_flow_jacobian(rhs, jac, params, dt, n_sub=10, rtol=1e-9, atol=1e-9,
                       method="RK45"):
    """Return flow_jacobian(x) -> Phi of shape (n_sub+1, n, n).

    Phi[s] is the Jacobian of the flow map from 0 to s*dt/n_sub, so Phi[0] = I
    and Phi[-1] = M = DM(x) over one observation window. The intermediate
    Phi[s] give the linearized covariance Phi[s] P_a Phi[s]^T inside the
    window (used only for plotting).
    """
    params = params or {}
    t_eval = np.linspace(0.0, dt, n_sub + 1)

    def flow_jacobian(x):
        x = np.asarray(x, dtype=float)
        n = x.size

        def aug(t, z):
            xx, Phi = z[:n], z[n:].reshape(n, n)
            return np.concatenate([rhs(t, xx, **params),
                                   (jac(xx, **params) @ Phi).ravel()])

        z0 = np.concatenate([x, np.eye(n).ravel()])
        sol = solve_ivp(aug, (0.0, dt), z0, t_eval=t_eval, method=method,
                        rtol=rtol, atol=atol)
        if not sol.success:
            raise RuntimeError(f"Variational solve failed: {sol.message}")
        return sol.y[n:].T.reshape(n_sub + 1, n, n)

    flow_jacobian.dt = dt
    flow_jacobian.n_sub = n_sub
    return flow_jacobian


def central_difference_check(propagate, flow_jacobian, x, q, eps_values):
    """Compare M q with (M(x + eps q) - M(x - eps q)) / (2 eps).

    Returns (Mq, rel_err) with rel_err[i] = ||fd_i - M q|| / ||M q|| for each
    eps in eps_values. `propagate` is the same model map the filter uses.
    """
    Mq = flow_jacobian(x)[-1] @ q
    rel_err = []
    for eps in eps_values:
        fd = (propagate(x + eps * q)[-1] - propagate(x - eps * q)[-1]) / (2 * eps)
        rel_err.append(np.linalg.norm(fd - Mq) / np.linalg.norm(Mq))
    return Mq, np.array(rel_err)


# ---------------------------------------------------------------------------
# Cycling driver
# ---------------------------------------------------------------------------

def _asymmetry(P):
    """Relative asymmetry ||P - P^T||_F / ||P||_F."""
    return np.linalg.norm(P - P.T) / np.linalg.norm(P)


def run_ekf(propagate, flow_jacobian, m0, P0, Y, H, R):
    """Cycle the EKF over k = 1..N (Q = 0).

    Parameters
    ----------
    propagate     : x -> (n_sub+1, n) model trajectory over one window
                    (assimilation.make_propagator; the truth's model map)
    flow_jacobian : x -> (n_sub+1, n, n) from make_flow_jacobian
    m0, P0        : prior mean and covariance at k = 0
    Y             : (N, m) observations, Y[k-1] = y_k; a NaN row means no obs
    H, R          : linear observation operator and observation covariance

    Returns a dict
        m_a, P_a        (N+1, n), (N+1, n, n)   analysis; row 0 is the prior
        m_f, P_f        (N+1, n), (N+1, n, n)   forecast; row 0 is the prior
        M               (N, n, n)    flow-map Jacobian used for step k (M_{k-1})
        K               (N, n, m)    gain (NaN if no obs)
        innovation      (N, m)       d_k = y_k - H m_f_k
        S               (N, m, m)    innovation covariance H P_f H^T + R
        asym_f, asym_a  (N,)         relative asymmetry of P_f and of the
                                     Joseph P_a before symmetrization
        seg_mean        (N, n_sub+1, n)  mean inside each forecast window
        seg_std         (N, n_sub+1, n)  sqrt(diag(Phi(t) P_a Phi(t)^T))
        wall_time       seconds for the whole cycle: forecasts of the mean,
                        Jacobian (variational) integrations, and analyses
        jacobian_time   seconds spent in the variational integrations alone
        failed_at       None, or the step k at which the run stopped
        message         reason for a failure ("" if none)
    After a failure every remaining entry is NaN; nothing is clipped or
    silently dropped.
    """
    H = np.atleast_2d(np.asarray(H, dtype=float))
    R = np.atleast_2d(np.asarray(R, dtype=float))
    Y = np.asarray(Y, dtype=float).reshape(len(Y), -1)
    m0 = np.asarray(m0, dtype=float)
    N, m = Y.shape
    n = m0.size
    n_sub = flow_jacobian.n_sub

    nan = np.nan
    m_a = np.full((N + 1, n), nan);      P_a = np.full((N + 1, n, n), nan)
    m_f = np.full((N + 1, n), nan);      P_f = np.full((N + 1, n, n), nan)
    M_all = np.full((N, n, n), nan);     K = np.full((N, n, m), nan)
    innov = np.full((N, m), nan);        S_all = np.full((N, m, m), nan)
    asym_f = np.full(N, nan);            asym_a = np.full(N, nan)
    seg_mean = np.full((N, n_sub + 1, n), nan)
    seg_std = np.full((N, n_sub + 1, n), nan)

    m_a[0] = m_f[0] = m0
    P_a[0] = P_f[0] = np.asarray(P0, dtype=float)
    failed_at, message = None, ""
    jac_time = 0.0

    tic = time.perf_counter()
    for k in range(1, N + 1):
        try:
            seg = propagate(m_a[k - 1])                       # m_f = M(m_a)
            t_jac = time.perf_counter()
            Phi = flow_jacobian(m_a[k - 1])                   # M = DM(m_a)
            jac_time += time.perf_counter() - t_jac
        except RuntimeError as err:
            failed_at, message = k, str(err)
            break
        Mk = Phi[-1]
        Pf = Mk @ P_a[k - 1] @ Mk.T                           # Q = 0
        asym_f[k - 1] = _asymmetry(Pf)
        Pf = 0.5 * (Pf + Pf.T)
        if not (np.all(np.isfinite(seg)) and np.all(np.isfinite(Pf))):
            failed_at, message = k, "non-finite forecast"
            break

        m_f[k], P_f[k], M_all[k - 1] = seg[-1], Pf, Mk
        seg_mean[k - 1] = seg
        PhiP = Phi @ P_a[k - 1]                               # (n_sub+1, n, n)
        seg_std[k - 1] = np.sqrt(np.einsum("sij,sij->si", PhiP, Phi))

        y = Y[k - 1]
        if np.all(np.isnan(y)):
            m_a[k], P_a[k] = m_f[k], P_f[k]
            continue
        S_all[k - 1] = H @ Pf @ H.T + R
        m_a[k], Pa, K[k - 1], innov[k - 1] = kf_analysis(m_f[k], Pf, y, H, R)
        asym_a[k - 1] = _asymmetry(Pa)
        P_a[k] = 0.5 * (Pa + Pa.T)                            # remove roundoff asymmetry
    wall_time = time.perf_counter() - tic

    return dict(m_a=m_a, P_a=P_a, m_f=m_f, P_f=P_f, M=M_all, K=K,
                innovation=innov, S=S_all, asym_f=asym_f, asym_a=asym_a,
                seg_mean=seg_mean, seg_std=seg_std, wall_time=wall_time,
                jacobian_time=jac_time, failed_at=failed_at, message=message)
