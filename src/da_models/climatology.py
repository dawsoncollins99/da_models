"""Climatological (long-time) statistics for the benchmark models.

Builds the approximate background covariance B used by 3DVAR/4DVAR:

    mu = (1/Dt) * int_{t1}^{tf} x(t) dt
    B  = (1/Dt) * int_{t1}^{tf} (x(t) - mu)(x(t) - mu)^T dt,   Dt = tf - t1.

The integrals are approximated with the trapezoid rule on the sampled
trajectory, so the statistics match the continuous definitions (note the
1/Dt normalization -- there is no N-1 "unbiased" correction here).
"""

import inspect
import json

import numpy as np
from scipy.integrate import solve_ivp


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

def model_params(rhs, **overrides):
    """Return the full parameter dict for `rhs`: its defaults plus overrides.

    Recording every parameter (not just the ones you changed) makes the saved
    climatology reproducible even if the defaults in the model file change.
    """
    sig = inspect.signature(rhs)
    params = {name: p.default for name, p in sig.parameters.items()
              if p.default is not inspect.Parameter.empty}
    unknown = set(overrides) - set(params)
    if unknown:
        raise KeyError(f"Unknown parameter(s) for {rhs.__name__}: {unknown}")
    params.update(overrides)
    return params


# ---------------------------------------------------------------------------
# Long run and time-averaged statistics
# ---------------------------------------------------------------------------

def long_run(rhs, x0, t_final, dt, params=None, rtol=1e-9, atol=1e-9,
             method="RK45"):
    """Integrate from t=0 to t_final, sampling every dt.

    Returns
    -------
    t : ndarray, shape (N,)
    X : ndarray, shape (N, n)   (one row per time sample)
    """
    params = params or {}
    n_steps = int(round(t_final / dt))
    t_eval = np.linspace(0.0, t_final, n_steps + 1)
    sol = solve_ivp(lambda t, x: rhs(t, x, **params), (0.0, t_final),
                    np.asarray(x0, dtype=float), t_eval=t_eval,
                    rtol=rtol, atol=atol, method=method)
    if not sol.success:
        raise RuntimeError(f"Long run failed: {sol.message}")
    return sol.t, sol.y.T


def _trapezoid_weights(t):
    """Weights w with sum_i w_i f(t_i) = trapezoid rule for int f dt."""
    w = np.empty_like(t)
    w[1:-1] = 0.5 * (t[2:] - t[:-2])
    w[0] = 0.5 * (t[1] - t[0])
    w[-1] = 0.5 * (t[-1] - t[-2])
    return w


def time_average_stats(t, X, t_start, t_end=None):
    """Time-averaged mean and covariance of X(t) on [t_start, t_end].

    Uses trapezoid weights rather than forming an (N, n, n) array of outer
    products, so it stays cheap for long runs of higher-dimensional models.
    """
    t_end = t[-1] if t_end is None else t_end
    mask = (t >= t_start) & (t <= t_end)
    t, X = t[mask], X[mask]
    w = _trapezoid_weights(t)
    duration = t[-1] - t[0]

    mu = (w @ X) / duration
    D = X - mu
    B = (D * w[:, None]).T @ D / duration
    B = 0.5 * (B + B.T)  # clean up round-off asymmetry
    return mu, B


def convergence_check(t, X, t_start):
    """Compare statistics from the two halves of the averaging window.

    Small relative differences mean the time averages have settled and the
    run was long enough. Returns (rel. diff in mu, rel. diff in B), measured
    in the 2-norm and Frobenius norm respectively.
    """
    t_mid = 0.5 * (t_start + t[-1])
    mu1, B1 = time_average_stats(t, X, t_start, t_mid)
    mu2, B2 = time_average_stats(t, X, t_mid)
    mu, B = time_average_stats(t, X, t_start)
    scale_mu = max(np.linalg.norm(mu), np.sqrt(np.trace(B)))
    return (np.linalg.norm(mu1 - mu2) / scale_mu,
            np.linalg.norm(B1 - B2) / np.linalg.norm(B))


def correlation_from_covariance(B):
    """Correlation matrix C_ij = B_ij / sqrt(B_ii B_jj)."""
    s = np.sqrt(np.diag(B))
    return B / np.outer(s, s)


# ---------------------------------------------------------------------------
# JSON persistence
# ---------------------------------------------------------------------------

def _to_builtin(obj):
    """json.dump fallback for NumPy scalars/arrays."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"Not JSON serializable: {type(obj)}")


def save_climatology(path, model, mu, B, *, params=None, state_labels=None,
                     t_start=None, t_final=None, dt=None, notes=None):
    """Save mu and B, plus the settings that produced them, to JSON."""
    record = {
        "model": model,
        "state_labels": state_labels,
        "params": params,
        "t_start": t_start,
        "t_final": t_final,
        "sample_dt": dt,
        "mu": np.asarray(mu).tolist(),
        "B": np.asarray(B).tolist(),
        "notes": notes,
    }
    with open(path, "w") as f:
        json.dump(record, f, indent=2, default=_to_builtin)


def load_climatology(path):
    """Load a saved climatology; mu and B come back as NumPy arrays."""
    with open(path) as f:
        record = json.load(f)
    record["mu"] = np.asarray(record["mu"])
    record["B"] = np.asarray(record["B"])
    return record


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_covariance_heatmaps(B, labels=None, title="", annotate=None):
    """Side-by-side heat maps of the covariance and correlation matrices.

    The covariance panel uses a symmetric color scale centered at zero; the
    correlation panel is fixed to [-1, 1] so models are directly comparable.
    Values are printed in each cell when the matrix is small (n <= 8).
    """
    import matplotlib.pyplot as plt

    n = B.shape[0]
    C = correlation_from_covariance(B)
    annotate = (n <= 8) if annotate is None else annotate
    vmax = np.abs(B).max()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
    panels = [(B, "Covariance $B$", -vmax, vmax, "{:.3g}"),
              (C, "Correlation", -1.0, 1.0, "{:.2f}")]
    for ax, (M, name, lo, hi, fmt) in zip(axes, panels):
        im = ax.imshow(M, cmap="RdBu_r", vmin=lo, vmax=hi)
        fig.colorbar(im, ax=ax, shrink=0.85)
        ax.set_title(name)
        if labels is not None and n <= 12:
            ax.set_xticks(range(n), labels, rotation=45 if n > 6 else 0)
            ax.set_yticks(range(n), labels)
        else:
            ax.set_xlabel("state index")
            ax.set_ylabel("state index")
        if annotate:
            for i in range(n):
                for j in range(n):
                    ax.text(j, i, fmt.format(M[i, j]), ha="center",
                            va="center", fontsize=8,
                            color="white" if abs(M[i, j]) > 0.6 * hi else "black")
    if title:
        fig.suptitle(title)
    return fig
