"""HW 9 Problem 3 settings: EnSF on the HW 8 Lorenz-96 experiments.

Everything about the physical experiment is reused unchanged from hw8.py and
the saved HW 8 twin data (truth, observations, parameters, solver, H, R,
observation interval, horizon, prior, free forecast, and the scales D). This
module only adds what EnSF needs:

* the fixed center c of the rescaled coordinates u = D^{-1}(x - c),
* the EnSF numerical settings (Ne, L, eps_a, eps_b), and
* a separate random stream for the reverse-sampler noise.
"""

import numpy as np

from . import hw8

NE = 50              # baseline ensemble size (same as the HW 8 EnKF baseline)
L_STEPS = 600        # reverse steps, dtau = 1/L
EPS_A = 0.01         # a_1 = eps_a: forecast structure left at the noisy end
EPS_B = 0.01         # v_0 = eps_b: blur variance (u units) at the analysis end
REPLICATES = hw8.REPLICATES   # (0, 1, 2)


def rng_ensf(name, Ne, r):
    """EnSF reverse-sampler noise (starting positions and every xi) for
    ensemble size Ne, replicate r. Stream 4 is new in HW 9; streams 0-3 are
    the HW 8 data, initial-ensemble, EnKF-perturbation, and Jacobian-check
    streams."""
    return np.random.default_rng([hw8.SEED, 4, hw8._model_index(name), Ne, r])


def rng_compare(name, Ne, r):
    """EnKF observation perturbations for the single-analysis comparison in
    Problem 3(e) (an EnKF analysis of an EnSF forecast ensemble)."""
    return np.random.default_rng([hw8.SEED, 5, hw8._model_index(name), Ne, r])


def state_center(name, mu):
    """Fixed center c: the climatological mean, pooled within each variable
    family exactly as D pools the climatological variance.

    The climatology comes from a long free model run made before any
    assimilation (HW 5 / HW 8 Problem 1), so c uses no observations and no
    knowledge of the truth during the experiment. Lorenz-96 is invariant
    under cyclic shifts, so every u_k (and every v_{k,j} with the same j)
    has the same climatological mean.
    """
    c = np.empty_like(mu)
    for idx in hw8.variable_families(name).values():
        c[idx] = np.mean(mu[idx])
    return c


def load_model(name, data_dir):
    """hw8.load_model plus the EnSF center c."""
    mdl = hw8.load_model(name, data_dir)
    mdl["c"] = state_center(name, mdl["mu"])
    return mdl


def to_u(X, mdl):
    """Physical states (n,) or (n, Ne) -> rescaled u = D^{-1}(x - c)."""
    c, D = mdl["c"], mdl["D"]
    X = np.asarray(X, dtype=float)
    return (X - c) / D if X.ndim == 1 else (X - c[:, None]) / D[:, None]
