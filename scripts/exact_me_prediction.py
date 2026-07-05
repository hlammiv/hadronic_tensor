"""Exact-matrix-element prediction for the elastic region of W^{00}.

From the volume-converged ns=10 band eigenstates compute
    M(k, q) = <band, k+q | J0~(q) | band, k>,   J0~(q) = sum_v e^{-i q x_v} J0(v),
build the predicted elastic profile
    W_pred(q0, q1) = A * sum_k w_k |M(k,q1)|^2 G_{1/sigma_t}(q0 - [E(k+q1)-E(k)])
with w_k the packet momentum weights, and compare against the measured
101-qubit W^{00}(q0) cuts at the shared momenta q1 = +-2pi/5, +-4pi/5
(one common normalization A fitted across ALL curves and both configs).

  PYTHONPATH=. .venv/bin/python scripts/exact_me_prediction.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline

from htensor import Z2Lattice, exact, spectroscopy, analysis
from htensor import currents as cur

OI = ["#0072B2", "#D55E00", "#009E73", "#E69F00"]
plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "stix",
    "font.size": 11, "axes.labelsize": 12,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
})

M0, G2, ETA = 0.7, 1.1, 1.3
SIGMA_T = 8.0 / 3.0
SIG_K = 2.0 / 3.0

# ---------------- exact MEs on the ns=10 band
lat = Z2Lattice(10, pbc=True)
band = spectroscopy.meson_band(lat, M0, G2, ETA, n_states=14, matrix_free=True)
kgrid = band["k"]
states = {round(k, 6): s for k, s in zip(band["k"], band["states"])}
egrid = {round(k, 6): e for k, e in zip(band["k"], band["energy"])}
vc10 = 4


def fold(k):
    return round(float((k + np.pi) % (2 * np.pi) - np.pi), 6)


def j0_q(q, psi):
    out = np.zeros_like(psi)
    for v in range(lat.ns):
        xv = (v - vc10) / 2
        op = cur.charge_density(lat, v)
        out = out + np.exp(-1j * q * xv) * exact.apply_pauli_sum(op, psi)
    return out


ks = sorted(states.keys())
M2 = {}
for k in ks:
    jpsi_cache = {}
    for q in ks:  # q on the same 5-point grid
        kp = fold(k + q)
        if kp not in states:
            continue
        me = np.vdot(states[kp], j0_q(q, states[k]))
        M2[(k, q)] = abs(me) ** 2
print("elastic |M(k,q)|^2 table (ns=10 band):")
for k in ks:
    row = "  ".join(f"{M2.get((k, q), np.nan):7.4f}" for q in ks)
    print(f"  k={k:+.3f}:  {row}")

# ---------------- dispersion spline (volume-converged points)
k_ed = np.array([0.0, 1.2566, 1.5708, 2.0944, 2.5133, np.pi])
e_ed = np.array([2.7451, 2.8188, 2.8560, 2.9275, 2.9886, 3.0778])
E = CubicSpline(np.concatenate([-k_ed[:0:-1], k_ed]),
                np.concatenate([e_ed[:0:-1], e_ed]))

# 2D interpolation of |M|^2 over (k, q), periodic; 5x5 grid -> smooth model
from scipy.interpolate import griddata
pts, vals = [], []
for (k, q), v in M2.items():
    for dk in (-2 * np.pi, 0, 2 * np.pi):
        for dq in (-2 * np.pi, 0, 2 * np.pi):
            pts.append((k + dk, q + dq))
            vals.append(v)
pts, vals = np.array(pts), np.array(vals)


def m2_interp(k, q):
    return np.maximum(griddata(pts, vals, (k, q), method="cubic"), 0.0)


def predict(q0, q1, kbar):
    kk = np.linspace(kbar - 3 * SIG_K, kbar + 3 * SIG_K, 61)
    wk = np.exp(-((kk - kbar) ** 2) / (2 * SIG_K**2))
    prof = np.zeros_like(q0)
    for k, w in zip(kk, wk):
        om = E((np.asarray(k + q1) + np.pi) % (2 * np.pi) - np.pi) \
            - E((k + np.pi) % (2 * np.pi) - np.pi)
        me2 = float(m2_interp(k, q1))
        prof = prof + w * me2 * np.exp(-((q0 - om) ** 2) * SIGMA_T**2 / 2)
    return prof


# ---------------- compare with production data at shared momenta
d0 = np.load("data/w_meson_ns50_k0.00_v3_W.npz")
db = np.load("data/w_meson_ns50_k1.26_v3_W.npz")
q0 = d0["q0"]
q1g = d0["q1"]
elwin = (q0 > -0.8) & (q0 < 1.6)
shared = [2 * np.pi / 5, -2 * np.pi / 5, 4 * np.pi / 5, -4 * np.pi / 5]

# single global normalization from all boosted+rest shared cuts
num = den = 0.0
preds, datas = {}, {}
for cfg, d, kbar in (("rest", d0, 0.0), ("boost", db, 2 * np.pi / 5)):
    for q in shared:
        j = np.argmin(np.abs(q1g - q))
        w_data = d["W"].real[:, j]
        w_pred = predict(q0, q, kbar)
        preds[(cfg, q)] = w_pred
        datas[(cfg, q)] = (w_data, d["spread"][:, j])
        num += np.sum(w_data[elwin] * w_pred[elwin])
        den += np.sum(w_pred[elwin] ** 2)
A = num / den
print(f"\nglobal normalization A = {A:.4f}")
for key in preds:
    w_data, _ = datas[key]
    r = w_data[elwin] - A * preds[key][elwin]
    print(f"  {key[0]:5s} q1={key[1]:+.3f}: rms resid / peak = "
          f"{np.sqrt(np.mean(r**2)) / np.abs(w_data[elwin]).max():.3f}")

# ---------------- figure: curve-on-curve at +-2pi/5, boosted and rest
fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8), constrained_layout=True,
                         sharey=True)
for ax, cfg, kbar, ttl in ((axes[0], "rest", 0.0, r"rest, $\bar k=0$"),
                           (axes[1], "boost", 2 * np.pi / 5,
                            r"boosted, $\bar k = 2\pi/5$")):
    for i, q in enumerate([2 * np.pi / 5, -2 * np.pi / 5]):
        w_data, spr = datas[(cfg, q)]
        ax.fill_between(q0[elwin], (w_data - spr / 2)[elwin],
                        (w_data + spr / 2)[elwin], color=OI[i], alpha=0.2, lw=0)
        ax.plot(q0[elwin], w_data[elwin], color=OI[i], lw=1.8,
                label=rf"data $q^1={q:+.2f}$")
        ax.plot(q0[elwin], A * preds[(cfg, q)][elwin], color=OI[i], lw=1.6,
                ls="--", label=rf"ED-ME pred.")
    ax.axhline(0, color="0.4", lw=0.7)
    ax.set_xlabel(r"$q^0$")
    ax.set_title(ttl + "  (one global norm.)", fontsize=10.5)
    ax.legend(fontsize=8, framealpha=0.9)
axes[0].set_ylabel(r"$W^{00}(q^0, q^1)$, elastic window")
fig.savefig("data/w_elastic_vs_prediction.pdf", dpi=200)
print("wrote data/w_elastic_vs_prediction.pdf")
