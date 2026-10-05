"""3DVAR vs. 4DVAR comparison plot (HW 5, Problem 5).

Uses the same conventions as da_plots.py (colors mark the noise level).
"""

import matplotlib.pyplot as plt

from .assimilation import error_norm_segments, flatten
from .da_plots import NOISE_COLORS


def plot_method_comparison(models, results_by_method, free_runs):
    """3DVAR vs. 4DVAR error curves: rows = models, columns = operators.

    `results_by_method` maps a method name to {model: [(case, run), ...]}.
    Colors mark the noise level; line style marks the method.
    """
    names = list(models)
    methods = list(results_by_method)
    styles = ["-", "--", "-.", ":"]
    fig, axes = plt.subplots(len(names), 2, figsize=(12, 3.3 * len(names)),
                             constrained_layout=True, squeeze=False)
    for i, name in enumerate(names):
        first = results_by_method[methods[0]][name]
        ops = list(dict.fromkeys(case["operator"] for case, _ in first))
        for j, op in enumerate(ops):
            ax = axes[i, j]
            for s, method in enumerate(methods):
                for case, run in results_by_method[method][name]:
                    if case["operator"] != op:
                        continue
                    t, _ = flatten(run["segments"], case["dt"])
                    e = error_norm_segments(case["truth"], run["segments"]).ravel()
                    ax.semilogy(t, e, color=NOISE_COLORS.get(case["noise_level"]),
                                ls=styles[s], lw=1.0,
                                label=f"{method}, {case['noise_level']:.0%}")
            case0 = first[0][0]
            t, _ = flatten(free_runs[name]["segments"], case0["dt"])
            e = error_norm_segments(case0["truth"], free_runs[name]["segments"]).ravel()
            ax.semilogy(t, e, color="0.6", ls=":", label="free run")
            ax.set_title(f"{name}: observing {op}")
            ax.set_xlabel(f"t ({models[name]['time_unit']})")
            ax.set_ylabel(r"$\|\hat x - \tilde x\|_2$")
            ax.legend(fontsize=7, loc="lower left", ncol=2)
    fig.suptitle("Full-state error: " + " vs. ".join(methods))
    return fig
