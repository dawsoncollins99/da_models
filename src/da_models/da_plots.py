"""Plots for cycled data-assimilation experiments (HW 5, Problems 3 and 5).

Every figure uses the same layout: one column per noise level, with the
true state, the assimilated estimate, the no-assimilation free run, and the
observations, plus a bottom row with the full-state error on a log scale.
"""

import numpy as np
import matplotlib.pyplot as plt

from .assimilation import error_norm_segments, flatten

NOISE_COLORS = {0.01: "tab:blue", 0.10: "tab:orange"}


def _noise_label(p):
    return f"noise {p:.0%} of mean"


def _unobserved(case, n):
    if case["observed"] is None:
        return None
    return [i for i in range(n) if i not in set(case["observed"])]


def _error_panel(ax, model, case, run, free, color, method):
    """Full-state error norm (and its unobserved part) vs. time, log scale."""
    dt = case["dt"]
    n = len(model["mu"])
    t, _ = flatten(run["segments"], dt)
    e = error_norm_segments(case["truth"], run["segments"]).ravel()
    ax.semilogy(t, e, color=color, lw=1.2, label=f"{method}: full state")

    unobs = _unobserved(case, n)
    if unobs:
        e_u = error_norm_segments(case["truth"], run["segments"], unobs).ravel()
        ax.semilogy(t, e_u, color=color, lw=1.0, ls=":", label=f"{method}: unobserved part")

    tf, _ = flatten(free["segments"], dt)
    ef = error_norm_segments(case["truth"], free["segments"]).ravel()
    ax.semilogy(tf, ef, color="0.5", ls="--", lw=1.0, label="free run (no DA)")
    ax.axhline(np.sqrt(np.trace(model["B"])), color="k", lw=0.8, ls="-.",
               label=r"climatological spread $\sqrt{\mathrm{tr}\,B}$")
    ax.set_ylabel(r"$\|\hat x - \tilde x\|_2$")
    ax.set_xlabel(f"t ({model['time_unit']})")


def plot_case_timeseries(model, cases, runs, free, method="3DVAR"):
    """Per-variable time series (rows) for each noise level (columns).

    Observed components show the observations as dots. Vertical strokes in
    the estimate are the analysis updates.
    """
    labels = model["labels"]
    n = len(labels)
    ncol = len(cases)
    fig, axes = plt.subplots(n + 1, ncol, figsize=(6.2 * ncol, 1.9 * (n + 1)),
                             sharex=True, squeeze=False, constrained_layout=True)
    dt = cases[0]["dt"]
    t_truth, x_truth = flatten(cases[0]["truth"], dt)
    t_free, x_free = flatten(free["segments"], dt)

    for j, (case, run) in enumerate(zip(cases, runs)):
        color = NOISE_COLORS.get(case["noise_level"], "tab:green")
        t_est, x_est = flatten(run["segments"], dt, run["analyses"][-1])
        t_obs = dt * np.arange(1, case["n_cycles"] + 1)
        observed = list(range(n)) if case["observed"] is None else list(case["observed"])

        for i in range(n):
            ax = axes[i, j]
            ax.plot(t_truth, x_truth[:, i], color="k", lw=1.6, label="truth")
            ax.plot(t_free, x_free[:, i], color="0.5", ls="--", lw=1.0, label="free run")
            ax.plot(t_est, x_est[:, i], color=color, lw=1.0, label=method)
            if i in observed:
                row = observed.index(i)
                ax.plot(t_obs, case["Y"][:, row], ".", ms=3, color="tab:red",
                        alpha=0.7, label="observations")
            ax.set_ylabel(labels[i] + ("" if i in observed else "  (hidden)"))
            if i == 0:
                ax.set_title(_noise_label(case["noise_level"]) +
                             f"  (σ = {case['sigma']:.3g})")
        _error_panel(axes[n, j], model, case, run, free, color, method)

    seen = {}
    for ax in axes[:n, 0]:
        for h, l in zip(*ax.get_legend_handles_labels()):
            seen.setdefault(l, h)
    axes[0, -1].legend(seen.values(), seen.keys(), loc="upper right", fontsize=8)
    axes[n, -1].legend(loc="lower left", fontsize=7)
    fig.suptitle(f"{model['name']}: {method}, observing {cases[0]['operator']}")
    return fig


def _hovmoller_grid(segments, dt, final_state):
    """Drop duplicated window endpoints so time is strictly increasing."""
    n_cycles, n_pts, n = segments.shape
    tau = np.linspace(0.0, dt, n_pts)[:-1]
    t = (np.arange(n_cycles)[:, None] * dt + tau[None, :]).ravel()
    x = segments[:, :-1, :].reshape(-1, n)
    return np.append(t, n_cycles * dt), np.vstack([x, final_state])


def plot_case_hovmoller(model, cases, runs, free, method="3DVAR"):
    """Space-time (Hovmöller) plots for spatially extended models (Lorenz-96).

    Rows: truth, estimate, |error|, and the full-state error norm.
    """
    ncol = len(cases)
    fig, axes = plt.subplots(4, ncol, figsize=(6.2 * ncol, 10.5), sharex=True,
                             squeeze=False, constrained_layout=True,
                             gridspec_kw=dict(height_ratios=[1, 1, 1, 0.9]))
    dt = cases[0]["dt"]
    n = len(model["mu"])
    sites = np.arange(1, n + 1)
    t, x_true = _hovmoller_grid(cases[0]["truth"], dt, cases[0]["truth"][-1, -1])
    vmax = np.abs(x_true).max()
    err_max = 0.0
    est = []
    for run in runs:
        _, x_est = _hovmoller_grid(run["segments"], dt, run["analyses"][-1])
        est.append(x_est)
        err_max = max(err_max, np.abs(x_true - x_est).max())

    for j, (case, run, x_est) in enumerate(zip(cases, runs, est)):
        color = NOISE_COLORS.get(case["noise_level"], "tab:green")
        im0 = axes[0, j].pcolormesh(t, sites, x_true.T, cmap="RdBu_r", vmin=-vmax,
                                    vmax=vmax, shading="nearest")
        axes[1, j].pcolormesh(t, sites, x_est.T, cmap="RdBu_r", vmin=-vmax,
                              vmax=vmax, shading="nearest")
        im2 = axes[2, j].pcolormesh(t, sites, np.abs(x_true - x_est).T, cmap="magma",
                                    vmin=0, vmax=err_max, shading="nearest")
        axes[0, j].set_title(_noise_label(case["noise_level"]) +
                             f"  (σ = {case['sigma']:.3g})")
        for r, name in enumerate(["truth", method, "|error|"]):
            axes[r, j].set_ylabel(f"{name}\nsite")
        if case["observed"] is not None:
            for r in range(3):
                axes[r, j].plot(np.zeros(len(case["observed"])),
                                np.asarray(case["observed"]) + 1, ">", color="lime",
                                ms=3, clip_on=False)
        _error_panel(axes[3, j], model, case, run, free, color, method)

    fig.colorbar(im0, ax=axes[:2, -1], shrink=0.8, label="state value")
    fig.colorbar(im2, ax=axes[2, -1], shrink=0.8, label="|error|")
    axes[3, -1].legend(loc="lower left", fontsize=7)
    note = "" if cases[0]["observed"] is None else "  (green ticks = observed sites)"
    fig.suptitle(f"{model['name']}: {method}, observing {cases[0]['operator']}{note}")
    return fig


def plot_error_summary(models, results, free_runs, method="3DVAR"):
    """Full-state error vs. time for every case, one panel per model."""
    names = list(models)
    fig, axes = plt.subplots(1, len(names), figsize=(5.4 * len(names), 4),
                             constrained_layout=True, squeeze=False)
    styles = ["-", "--"]
    for ax, name in zip(axes[0], names):
        model = models[name]
        ops = list(dict.fromkeys(case["operator"] for case, _ in results[name]))
        for case, run in results[name]:
            t, _ = flatten(run["segments"], case["dt"])
            e = error_norm_segments(case["truth"], run["segments"]).ravel()
            ax.semilogy(t, e, color=NOISE_COLORS.get(case["noise_level"]),
                        ls=styles[ops.index(case["operator"]) % 2], lw=1.1,
                        label=f"{case['operator']}, {case['noise_level']:.0%}")
        free = free_runs[name]
        t, _ = flatten(free["segments"], results[name][0][0]["dt"])
        e = error_norm_segments(results[name][0][0]["truth"], free["segments"]).ravel()
        ax.semilogy(t, e, color="0.5", ls=":", label="free run")
        ax.set_title(name)
        ax.set_xlabel(f"t ({model['time_unit']})")
        ax.set_ylabel(r"$\|\hat x - \tilde x\|_2$")
        ax.legend(fontsize=7, loc="lower left")
    fig.suptitle(f"{method}: full-state error for every case")
    return fig
