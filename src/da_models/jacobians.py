"""Analytic state Jacobians Df(x) for the HW 5 and HW 8 models.

Each function has the signature jac(state, **params) with the same keyword
parameters (and defaults) as the matching rhs, so it plugs directly into the
`jac_func` hook of adjoint.adjoint_solve and into ekf.make_flow_jacobian:

    lorenz63_jacobian           <->  lorenz63.rhs
    lorenz96_jacobian           <->  lorenz96.rhs            (one-layer model)
    sturis_jacobian             <->  sturis.rhs
    lorenz96_two_layer_jacobian <->  lorenz96.rhs_two_layer  (HW 8)

Entry [i, j] of each Jacobian is d f_i / d x_j. All agree with central
finite differences to ~1e-10 (see the checks in hw5_problem5.ipynb and
hw8_problem2.ipynb).
"""

import numpy as np

JACOBIANS = {}  # filled in at the bottom: model name -> Jacobian function


def lorenz63_jacobian(state, sigma=10.0, rho=28.0, beta=8.0 / 3.0):
    """Analytic state Jacobian Df(x), entry [i, j] = d f_i / d x_j.

    Signature matches the `jac_func(state, **rhs_kwargs)` hook in adjoint.py.
    """
    x, y, z = state
    return np.array([
        [-sigma, sigma, 0.0],
        [rho - z, -1.0, -x],
        [y, x, -beta],
    ])


def lorenz96_jacobian(state, F=8.0, d=1.0):
    """Analytic Jacobian of the one-layer model, entry [k, j] = d f_k / d u_j.

    f_k = u_{k-1} (u_{k+1} - u_{k-2}) - d u_k + F  (indices periodic), so row k
    has four nonzeros:
        d f_k / d u_{k-2} = -u_{k-1}
        d f_k / d u_{k-1} =  u_{k+1} - u_{k-2}
        d f_k / d u_k     = -d
        d f_k / d u_{k+1} =  u_{k-1}
    F does not appear (it is accepted so the signature matches rhs).
    """
    u = np.asarray(state)
    K = u.shape[0]
    k = np.arange(K)
    up1, um1, um2 = np.roll(u, -1), np.roll(u, 1), np.roll(u, 2)
    Jx = np.zeros((K, K))
    Jx[k, (k - 2) % K] = -um1
    Jx[k, (k - 1) % K] = up1 - um2
    Jx[k, k] = -d
    Jx[k, (k + 1) % K] = um1
    return Jx


def sturis_jacobian(state,
             Vp=3.0, Vi=11.0, Vg=10.0, E=0.2, tp=6.0, ti=100.0, td=12.0,
             Rm=209.0, a1=6.6, C1=300.0, C2=144.0, C3=100.0, C4=80.0, C5=26.0,
             Ub=72.0, U0=4.0, Um=94.0, Rg=180.0, alpha=7.5, beta=1.772,
             IG=0.0):
    """Analytic state Jacobian, entry [i, j] = d f_i / d x_j.

    state = [Ip, Ii, G, h1, h2, h3]. Derivatives of the four nonlinear
    functions (each depends on a single variable):
        f1(G)  = Rm / (1 + e1),            e1 = exp(-G/(Vg C1) + a1)
                 f1' = Rm e1 / (Vg C1 (1 + e1)^2)
        f2(G)  = Ub (1 - exp(-G/(C2 Vg)))
                 f2' = Ub exp(-G/(C2 Vg)) / (C2 Vg)
        f3(Ii) = (U0 + (Um - U0)/(1 + q)) / (C3 Vg),   q = (kappa Ii)^(-beta)
                 f3' = (Um - U0) beta q / (C3 Vg Ii (1 + q)^2)
        f4(h3) = Rg / (1 + e4),            e4 = exp(alpha (h3/(C5 Vp) - 1))
                 f4' = -Rg alpha e4 / (C5 Vp (1 + e4)^2)
    IG is a constant input, so it does not appear in the Jacobian.
    """
    Ip, Ii, G, h1, h2, h3 = state
    kappa = (1.0 / C4) * (1.0 / Vi + 1.0 / (E * ti))

    e1 = np.exp(-G / (Vg * C1) + a1)
    df1 = Rm * e1 / (Vg * C1 * (1.0 + e1) ** 2)
    df2 = Ub * np.exp(-G / (C2 * Vg)) / (C2 * Vg)
    q = (kappa * Ii) ** (-beta)
    f3 = (1.0 / (C3 * Vg)) * (U0 + (Um - U0) / (1.0 + q))
    df3 = (Um - U0) * beta * q / (C3 * Vg * Ii * (1.0 + q) ** 2)
    e4 = np.exp(alpha * (h3 / (C5 * Vp) - 1.0))
    df4 = -Rg * alpha * e4 / (C5 * Vp * (1.0 + e4) ** 2)

    Jx = np.zeros((6, 6))
    # dIp/dt = f1(G) - E (Ip/Vp - Ii/Vi) - Ip/tp
    Jx[0, 0] = -E / Vp - 1.0 / tp
    Jx[0, 1] = E / Vi
    Jx[0, 2] = df1
    # dIi/dt = E (Ip/Vp - Ii/Vi) - Ii/ti
    Jx[1, 0] = E / Vp
    Jx[1, 1] = -E / Vi - 1.0 / ti
    # dG/dt = f4(h3) + IG - f2(G) - f3(Ii) G
    Jx[2, 1] = -df3 * G
    Jx[2, 2] = -df2 - f3
    Jx[2, 5] = df4
    # delay chain dh_i/dt = (input - h_i)/td
    Jx[3, 0], Jx[3, 3] = 1.0 / td, -1.0 / td
    Jx[4, 3], Jx[4, 4] = 1.0 / td, -1.0 / td
    Jx[5, 4], Jx[5, 5] = 1.0 / td, -1.0 / td
    return Jx


def lorenz96_two_layer_jacobian(state, K=40, J=5, F=8.0, d=1.0, gamma=0.05,
                                d_j=(0.2, 0.5, 1.0, 2.0, 5.0)):
    """Analytic Jacobian of the two-layer model (HW 8), matches rhs_two_layer.

    State order (lorenz96.pack_two_layer): u_0..u_{K-1}, then v_{k,j} at
    index K + k*J + j (j = 0..J-1 here). With the same gamma for every j,

        du_k/dt     = u_{k-1}(u_{k+1} - u_{k-2}) + gamma sum_j v_{k,j} u_k - d u_k + F
        dv_{k,j}/dt = -d_j v_{k,j} - gamma u_k^2

    so the nonzero blocks are
        d(du_k)/d(u)       = one-layer Jacobian row k, plus gamma sum_j v_{k,j}
                             on the diagonal
        d(du_k)/d(v_{k,j}) = gamma u_k
        d(dv_{k,j})/d(u_k) = -2 gamma u_k
        d(dv_{k,j})/d(v_{k,j}) = -d_j
    Each v_{k,j} couples only to its own site u_k.
    """
    state = np.asarray(state)
    u = state[:K]
    v = state[K:].reshape(K, J)
    d_j = np.asarray(d_j)
    n = K * (1 + J)

    Jx = np.zeros((n, n))
    Jx[:K, :K] = lorenz96_jacobian(u, F=F, d=d) + np.diag(gamma * v.sum(axis=1))
    k = np.repeat(np.arange(K), J)          # site of each v, in state order
    j = np.tile(np.arange(J), K)            # family of each v
    iv = K + k * J + j                      # state index of v_{k,j}
    Jx[k, iv] = gamma * u[k]
    Jx[iv, k] = -2.0 * gamma * u[k]
    Jx[iv, iv] = -d_j[j]
    return Jx


JACOBIANS.update(lorenz63=lorenz63_jacobian, lorenz96=lorenz96_jacobian,
                 sturis=sturis_jacobian,
                 lorenz96_two_layer=lorenz96_two_layer_jacobian)
