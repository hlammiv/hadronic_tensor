"""Corrected quantitative panel: 101-qubit W^{00} ridge vs the parameter-free
two-band model, identical one-sided narrow-window transforms on both sides
(sigma_x = 0.83 kills |x| > 2, where the model and data correlators are both
volume-converged -- no volume extension, no normalization).

Also: integral Ward/continuity check on the raw production grids,
  C00(x,t) - C00(x,0) = - int_0^t dt' [C10(x+1/4,t') - C10(x-1/4,t')]
via Simpson integration (error ~ (w*dt)^4/180 ~ 2%), which avoids the ~30%
sinc bias of naive time differentiation.

  PYTHONPATH=. .venv/bin/python scripts/final_ridge_overlay.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import cumulative_simpson

from htensor import Z2Lattice, exact, spectroscopy, analysis
from htensor import currents as cur
from scripts_helpers_ridge import two_band_space, model_correlator, onesided_ft

OI = ["#0072B2", "#D55E00", "#009E73", "#E69F00"]
plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "stix",
    "font.size": 11, "axes.labelsize": 12,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
})

M0, G2, ETA = 0.7, 1.1, 1.3
SIG_T, SIG_X = 8.0 / 3.0, 0.83
Q1S = np.array([2 * np.pi / 5, -2 * np.pi / 5, 4 * np.pi / 5, -4 * np.pi / 5])
Q0 = np.arange(-0.6, 1.601, 0.04)

# ---------------- ns=10 two-band model correlators
lat10 = Z2Lattice(10, pbc=True)
sts, Ek, vac10, e0 = two_band_space(lat10, M0, G2, ETA)
band1 = spectroscopy.meson_band(lat10, M0, G2, ETA, n_states=14,
                                matrix_free=True)
TIMES = np.arange(0.0, 8.01, 0.5)
x10 = analysis.ring_fold((np.arange(lat10.ns) - 4) / 2, lat10.nx)
model_W = {}
for k0 in (0.0, 2 * np.pi / 5):
    mix = spectroscopy.optimize_interpolator(lat10, band1, k0=k0,
                                             sigma_x=0.75, x0=2)
    wp, _ = spectroscopy.meson_wavepacket(lat10, band1, k0=k0, sigma_x=0.75,
                                          x0=2, mix=mix["mix"])
    f = np.array([np.vdot(s, wp) for s in sts])
    Gm, _ = model_correlator(lat10, sts, Ek, f, 4, TIMES, M0, G2, ETA)
    model_W[k0] = onesided_ft(TIMES, x10, Gm, Q0, Q1S, SIG_T, SIG_X, 0.5, 0.5)
    print(f"model k0={k0:.2f} ready", flush=True)

# ---------------- production data, same one-sided transform
data_W = {}
ward = {}
for k0, path in ((0.0, "data/w_meson_ns50_k0.00_v3.npz"),
                 (2 * np.pi / 5, "data/w_meson_ns50_k1.26_v3.npz")):
    d = np.load(path)
    ns, vc = int(d["ns"]), int(d["center"])
    lat = Z2Lattice(ns, pbc=True)
    times = d["times"]

    def conn(corr, one, ins):
        return analysis.subtract(corr, None, one, complex(ins))

    G = conn(d["corr_wp"], d["one_pt_wp"], d["insert_1pt_wp"]) \
        - conn(d["corr_vac"], d["one_pt_vac"], d["insert_1pt_vac"])
    x = analysis.ring_fold((np.arange(ns) - vc) / 2, lat.nx)
    data_W[k0] = onesided_ft(times, x, G[:, :ns], Q0, Q1S, SIG_T, SIG_X,
                             0.5, 0.5)

    # integral Ward check on raw wp grids
    c00 = d["corr_wp"][:, :ns]
    c10 = d["corr_wp"][:, ns:]
    div = c10 - np.roll(c10, 1, axis=1)
    lhs = c00 - c00[0][None, :]
    rhs = -cumulative_simpson(div, x=times, axis=0, initial=0.0)
    ward[k0] = float(np.abs(lhs - rhs).max() / np.abs(lhs).max())
    print(f"data k0={k0:.2f}: integral Ward residual = {ward[k0]:.3f}", flush=True)

# ---------------- figure
fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8), constrained_layout=True,
                         sharey=True)
for ax, k0, ttl in ((axes[0], 0.0, r"rest, $\bar k = 0$"),
                    (axes[1], 2 * np.pi / 5, r"boosted, $\bar k = 2\pi/5$")):
    for i, q in enumerate([2 * np.pi / 5, -2 * np.pi / 5]):
        j = np.argmin(np.abs(Q1S - q))
        ax.plot(Q0, data_W[k0][:, j], color=OI[i], lw=1.9,
                label=rf"101-qubit data, $q^1 = {q:+.2f}$")
        ax.plot(Q0, model_W[k0][:, j], color=OI[i], lw=1.5, ls="--",
                label="two-band model (param-free)")
    ax.axhline(0, color="0.4", lw=0.7)
    ax.set_xlabel(r"$q^0$")
    ax.set_title(ttl, fontsize=10.5)
    ax.legend(fontsize=8, framealpha=0.9)
axes[0].set_ylabel(r"$W^{00}$ (one-sided, $\sigma_x = 0.83$)")
fig.savefig("data/w_ridge_final.pdf", dpi=200)
print("wrote data/w_ridge_final.pdf")

for k0 in (0.0, 2 * np.pi / 5):
    for q in Q1S:
        j = np.argmin(np.abs(Q1S - q))
        r = data_W[k0][:, j] - model_W[k0][:, j]
        print(f"k0={k0:.2f} q1={q:+.2f}: rms(data-model)/max = "
              f"{np.sqrt(np.mean(r**2)) / np.abs(data_W[k0][:, j]).max():.3f}")
