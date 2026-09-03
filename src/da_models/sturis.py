"""Sturis ultradian glucose-insulin model."""

import numpy as np


def rhs(t, state,
        Vp=3.0, Vi=11.0, Vg=10.0, E=0.2, tp=6.0, ti=100.0, td=12.0,
        Rm=209.0, a1=6.6, C1=300.0, C2=144.0, C3=100.0, C4=80.0, C5=26.0,
        Ub=72.0, U0=4.0, Um=94.0, Rg=180.0, alpha=7.5, beta=1.772,
        IG=0.0):
    """Right-hand side of the Sturis ultradian glucose-insulin model.

    state = [Ip, Ii, G, h1, h2, h3]
    IG is the glucose infusion rate (default 0, per the assignment).
    Time is in minutes.
    """
    Ip, Ii, G, h1, h2, h3 = state

    kappa = (1.0 / C4) * (1.0 / Vi + 1.0 / (E * ti))

    f1 = Rm / (1.0 + np.exp(-G / (Vg * C1) + a1))
    f2 = Ub * (1.0 - np.exp(-G / (C2 * Vg)))
    f3 = (1.0 / (C3 * Vg)) * (U0 + (Um - U0) / (1.0 + (kappa * Ii) ** (-beta)))
    f4 = Rg / (1.0 + np.exp(alpha * (h3 / (C5 * Vp) - 1.0)))

    dIp = f1 - E * (Ip / Vp - Ii / Vi) - Ip / tp
    dIi = E * (Ip / Vp - Ii / Vi) - Ii / ti
    dG = f4 + IG - f2 - f3 * G
    dh1 = (Ip - h1) / td
    dh2 = (h1 - h2) / td
    dh3 = (h2 - h3) / td

    return np.array([dIp, dIi, dG, dh1, dh2, dh3])