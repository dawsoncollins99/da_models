"""HW 5 experiment definitions, shared by Problems 3 (3DVAR) and 5 (4DVAR).

Everything that must be identical across the two methods lives here: model
parameters and climatology, the true initial conditions, the observation
operators, the noise model, and the random seeds. Building a case twice with
the same arguments gives the same truth and the same observations.

Noise model: sigma = p * mean_i |mu_i| (one scalar per model, p = 1% or 10%),
treated as a standard deviation, so R = sigma^2 I.
"""

from pathlib import Path

import numpy as np

from ... import lorenz63, lorenz96, sturis
from .assimilation import make_propagator, observe, selection_operator, simulate_truth
from .climatology import load_climatology, long_run

DT_OBS = 0.1          # time between observations
T_FINAL = 10.0        # assimilate on [0, T_FINAL]
N_CYCLES = int(round(T_FINAL / DT_OBS))
N_SUB = 10            # plotting/RMSE sub-steps per observation window
NOISE_LEVELS = (0.01, 0.10)

MODELS = {
    "lorenz63": dict(
        rhs=lorenz63.rhs, time_unit="time",
        operators={"identity": None, "x only": [0]},
    ),
    "lorenz96": dict(
        rhs=lorenz96.rhs, time_unit="time",
        operators={"identity": None, "every other site": list(range(0, 40, 2))},
    ),
    "sturis": dict(
        rhs=sturis.rhs, time_unit="min",
        operators={"identity": None, "glucose only": [2]},
    ),
}


def load_model(name, data_dir):
    """Climatology (mu, B, params, labels) plus the true initial condition."""
    data_dir = Path(data_dir)
    clim = load_climatology(data_dir / f"climatology_{name}.json")
    rhs = MODELS[name]["rhs"]
    x0 = np.load(data_dir / f"{name}_ic.npy")
    if name == "sturis":
        # The saved IC is the I_G = 0 equilibrium; spin up onto the
        # I_G = 216 limit cycle with the same burn-in used for the climatology.
        _, X = long_run(rhs, x0, clim["t_start"], 1.0, clim["params"])
        x0 = X[-1]
    return dict(name=name, rhs=rhs, params=clim["params"], mu=clim["mu"],
                B=clim["B"], labels=clim["state_labels"], x_true0=x0,
                time_unit=MODELS[name]["time_unit"])


def build_case(model, operator_name, noise_level, seed=513):
    """Truth, observations, H, R, and the propagator for one experiment.

    The standard-normal noise draw depends only on (model, operator, seed),
    so the 1% and 10% cases share one realization (just rescaled), and
    3DVAR and 4DVAR see identical observations.
    """
    name = model["name"]
    n = len(model["mu"])
    observed = MODELS[name]["operators"][operator_name]
    H = selection_operator(n, observed)
    m = H.shape[0]

    sigma = noise_level * np.mean(np.abs(model["mu"]))
    R = sigma ** 2 * np.eye(m)

    propagate = make_propagator(model["rhs"], model["params"], DT_OBS, N_SUB)
    truth = simulate_truth(propagate, model["x_true0"], N_CYCLES)

    model_idx = list(MODELS).index(name)
    op_idx = list(MODELS[name]["operators"]).index(operator_name)
    rng = np.random.default_rng([seed, model_idx, op_idx])
    eps = rng.standard_normal((N_CYCLES, m))
    Y = observe(truth, H, sigma, eps)

    return dict(model=name, operator=operator_name, noise_level=noise_level,
                observed=observed, H=H, R=R, sigma=sigma, truth=truth, Y=Y,
                propagate=propagate, x_guess0=model["mu"].copy(),
                dt=DT_OBS, n_cycles=N_CYCLES)
