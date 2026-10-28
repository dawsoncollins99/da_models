"""HW 8 experiment definitions, shared by Problems 1-4 (EKF and EnKF).

Everything that must be identical across the two filters lives here: the two
Lorenz-96 models and their parameters, the deterministic solver, the
observation operators and noise, the state scales D, the prior, the random
streams, and the saved truth/observation data. Problem 1 generates and saves
one twin data set per model; Problems 2-4 only ever load it, so every filter
run sees the same truth and the same observations.

Indexing (as in kalman.py and enkf.py)
--------------------------------------
* x_k is the state at t_k = k*dt, k = 0, ..., L.
* The prior N(m_0, P_0) is the analysis at k = 0; there is no y_0.
* Observations y_k arrive at k = 1, ..., L and are stored as Y[k-1].

Models
------
* "l96":        one-layer Lorenz-96, K = 40, F = 8, d = 1 (n = 40).
* "l96_2layer": two-layer Lorenz-96 reduced to K = 10, J = 2, with
                gamma_j = 0.05 and (d_1, d_2) = (0.2, 0.5) (n = 30). State
                order (lorenz96.pack_two_layer): u_0..u_{K-1}, then
                v_{0,1}, v_{0,2}, v_{1,1}, v_{1,2}, ..., i.e. v_{k,j} sits at
                index K + k*J + (j - 1).
"""

import json
import time
from pathlib import Path

import numpy as np

from . import jacobians, lorenz96
from .assimilation import make_propagator, selection_operator, simulate_truth
from .climatology import load_climatology
from .ekf import make_flow_jacobian
from .kalman import synthetic_observations

# ---------------------------------------------------------------------------
# Common experiment settings
# ---------------------------------------------------------------------------

DT_OBS = 0.1                  # observation interval (not the solver step)
T_FINAL = 10.0                # assimilate on [0, T_FINAL]
L = int(round(T_FINAL / DT_OBS))
N_SUB = 10                    # plotting sub-steps per observation window
SOLVER = dict(method="RK45", rtol=1e-9, atol=1e-9)   # HW 1 solve_ivp settings
SEED = 513
PRIOR_SHIFT = 0.1             # m_0^a = x_0^dagger + 0.1 D 1
PRIOR_VAR = 0.01              # P_0^a = 0.01 D^2

MODELS = {
    "l96": dict(
        title="One-layer Lorenz-96",
        rhs=lorenz96.rhs,
        jac=jacobians.lorenz96_jacobian,
        params=dict(F=8.0, d=1.0),
        K=40, J=0,
        ic_file="lorenz96_ic.npy",
        clim_file="climatology_lorenz96.json",
    ),
    "l96_2layer": dict(
        title="Two-layer Lorenz-96 (K = 10, J = 2)",
        rhs=lorenz96.rhs_two_layer,
        jac=jacobians.lorenz96_two_layer_jacobian,
        params=dict(K=10, J=2, F=8.0, d=1.0, gamma=0.05, d_j=(0.2, 0.5)),
        K=10, J=2,
        ic_file="lorenz96_two_layer_K10_J2_ic.npy",
        clim_file="climatology_lorenz96_two_layer_K10_J2.json",
    ),
}


def _model_index(name):
    return list(MODELS).index(name)


# ---------------------------------------------------------------------------
# Random streams: one generator per purpose, never shared
# ---------------------------------------------------------------------------

def rng_data(name):
    """Observation noise v_k for the saved data (used once, in Problem 1)."""
    return np.random.default_rng([SEED, 0, _model_index(name)])


def rng_init(name, Ne, r):
    """Initial ensemble for ensemble size Ne, replicate r."""
    return np.random.default_rng([SEED, 1, _model_index(name), Ne, r])


def rng_pert(name, Ne, r):
    """EnKF observation perturbations for ensemble size Ne, replicate r."""
    return np.random.default_rng([SEED, 2, _model_index(name), Ne, r])


def rng_direction(name):
    """Direction q for the Problem 2(a) Jacobian check."""
    return np.random.default_rng([SEED, 3, _model_index(name)])


# ---------------------------------------------------------------------------
# State layout
# ---------------------------------------------------------------------------

def state_labels(name):
    """u0..u{K-1}, then v{k},{j} in pack_two_layer order (j is 1-based)."""
    K, J = MODELS[name]["K"], MODELS[name]["J"]
    labels = [f"u{k}" for k in range(K)]
    labels += [f"v{k},{j + 1}" for k in range(K) for j in range(J)]
    return labels


def variable_families(name):
    """Indices of each family of statistically identical variables.

    {"u": u_0..u_{K-1}} for the one-layer model, plus {"v1": v_{:,1},
    "v2": v_{:,2}} for the two-layer model.
    """
    K, J = MODELS[name]["K"], MODELS[name]["J"]
    fam = {"u": np.arange(K)}
    for j in range(J):
        fam[f"v{j + 1}"] = K + np.arange(K) * J + j
    return fam


def observed_indices(name):
    """Every second u_k: u_0, u_2, ..., u_{K-2}. All v_{k,j} are hidden."""
    return list(range(0, MODELS[name]["K"], 2))


# ---------------------------------------------------------------------------
# State scales and prior
# ---------------------------------------------------------------------------

def state_scales(name, B):
    """Diagonal of D: one scale per variable family.

    Lorenz-96 is invariant under cyclic shifts k -> k+1, so every u_k (and
    every v_{k,j} with the same j) has the same climatological variance. The
    scale of a family is the square root of its site-averaged climatological
    variance, sqrt(mean_k B_kk), which pools the sites instead of keeping
    the small site-to-site sampling differences.
    """
    var = np.diag(B)
    D = np.empty_like(var)
    for idx in variable_families(name).values():
        D[idx] = np.sqrt(np.mean(var[idx]))
    return D


def prior(x0, D):
    """m_0^a = x_0^dagger + 0.1 D 1,  P_0^a = 0.01 D^2."""
    m0 = np.asarray(x0, dtype=float) + PRIOR_SHIFT * D
    P0 = PRIOR_VAR * np.diag(D ** 2)
    return m0, P0


def scaled_rmse_spread(means, truth, D, covs=None):
    """Scaled diagnostics at each time (Problem 4 definitions):

        RMSE_k   = sqrt( |D^{-1}(m_k - x_k)|^2 / n )
        Spread_k = sqrt( tr(D^{-1} P_k D^{-T}) / n ) = sqrt( sum_i P_k,ii / D_i^2 / n )

    means and truth have shape (T, n), covs (T, n, n). Returns (rmse, spread);
    spread is None when covs is None (the free forecast has no spread).
    """
    n = truth.shape[1]
    rmse = np.sqrt(np.sum(((means - truth) / D) ** 2, axis=1) / n)
    if covs is None:
        return rmse, None
    var = np.diagonal(covs, axis1=1, axis2=2)
    spread = np.sqrt(np.sum(var / D ** 2, axis=1) / n)
    return rmse, spread


# ---------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------

def load_model(name, data_dir):
    """Everything about one experiment except the random data.

    Returns a dict with rhs, jac (analytic Df), params, x0 (spun-up true
    initial state), the climatology (mu, B), D, the prior (m0, P0), H, R,
    labels, a propagator for the model map over one observation window
    (sampled at N_SUB sub-steps), and flow_jacobian, its Jacobian from the
    variational equations (same solver settings).
    """
    spec = MODELS[name]
    data_dir = Path(data_dir)
    rhs, params = spec["rhs"], dict(spec["params"])
    x0 = np.load(data_dir / spec["ic_file"])
    clim = load_climatology(data_dir / spec["clim_file"])
    n = x0.size
    if clim["mu"].size != n:
        raise ValueError(f"{name}: climatology has size {clim['mu'].size}, state has {n}")

    D = state_scales(name, clim["B"])
    m0, P0 = prior(x0, D)
    observed = observed_indices(name)
    H = selection_operator(n, observed)
    R = np.eye(len(observed))
    propagate = make_propagator(rhs, params, DT_OBS, N_SUB,
                                rtol=SOLVER["rtol"], atol=SOLVER["atol"])
    flow_jacobian = make_flow_jacobian(rhs, spec["jac"], params, DT_OBS, N_SUB,
                                       **SOLVER)
    return dict(name=name, title=spec["title"], rhs=rhs, jac=spec["jac"],
                flow_jacobian=flow_jacobian, params=params,
                n=n, K=spec["K"], J=spec["J"], x0=x0, mu=clim["mu"],
                B=clim["B"], D=D, m0=m0, P0=P0, observed=observed, H=H, R=R,
                labels=state_labels(name), propagate=propagate)


# ---------------------------------------------------------------------------
# Twin data: truth, observations, free forecast
# ---------------------------------------------------------------------------

def build_twin(model):
    """Generate the truth, the observations, and the free forecast.

    truth:  x_0 = x0, x_k = M(x_{k-1}) for k = 1..L (deterministic, Q = 0)
    data:   y_k = H x_k + v_k, v_k ~ N(0, R) i.i.d., from rng_data(name)
    free:   the same model run from m_0^a with no assimilation.

    free_runtime (wall time of the free forecast) is returned for reference
    but not saved: runtimes in the Problem 4 table are measured on the same
    machine as the filters.
    """
    propagate = model["propagate"]
    truth_segments = simulate_truth(propagate, model["x0"], L)    # (L, N_SUB+1, n)
    x_true = np.vstack([model["x0"], truth_segments[:, -1]])      # (L+1, n)
    Y = synthetic_observations(x_true, model["H"], model["R"],
                               rng_data(model["name"]))           # (L, p)

    tic = time.perf_counter()
    free_segments = simulate_truth(propagate, model["m0"], L)
    free_runtime = time.perf_counter() - tic
    x_free = np.vstack([model["m0"], free_segments[:, -1]])

    return dict(x_true=x_true, truth_segments=truth_segments, Y=Y,
                x_free=x_free, free_segments=free_segments,
                free_runtime=free_runtime)


def twin_path(name, data_dir):
    return Path(data_dir) / f"hw8_twin_{name}.npz"


def save_twin(model, twin, data_dir):
    """Save the twin data plus every setting needed to reproduce it."""
    meta = dict(model=model["name"], params=model["params"],
                solver=SOLVER, dt_obs=DT_OBS, t_final=T_FINAL, L=L,
                n_sub=N_SUB, seed=SEED, data_stream=[SEED, 0, _model_index(model["name"])],
                observed=model["observed"], labels=model["labels"],
                prior_shift=PRIOR_SHIFT, prior_var=PRIOR_VAR)
    path = twin_path(model["name"], data_dir)
    arrays = {k: v for k, v in twin.items() if k != "free_runtime"}
    np.savez_compressed(
        path, meta=json.dumps(meta), x0=model["x0"], D=model["D"],
        m0=model["m0"], P0=model["P0"], H=model["H"], R=model["R"], **arrays)
    return path


def load_twin(name, data_dir):
    """Load the saved twin data; `meta` comes back as a dict."""
    with np.load(twin_path(name, data_dir)) as f:
        out = {key: f[key] for key in f.files}
    out["meta"] = json.loads(str(out["meta"]))
    return out
