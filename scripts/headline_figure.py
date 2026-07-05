"""Headline figure: W^{00}(q0, q1) on rest and boosted meson wavepackets at
101 qubits, with the elastic-ridge dispersion compared to the ED prediction
omega_el(q1) = E(kbar + q1) - E(kbar).

  PYTHONPATH=. .venv/bin/python scripts/headline_figure.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline

OI = ["#0072B2", "#D55E00", "#009E73", "#E69F00"]
plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "stix",
    "font.size": 11, "axes.labelsize": 12,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
})

# measured single-meson dispersion (ED, volume-converged ns=8/10)
k_ed = np.array([0.0, 1.2566, 1.5708, 2.0944, 2.5133, np.pi])
e_ed = np.array([2.7451, 2.8188, 2.8560, 2.9275, 2.9886, 3.0778])
kk = np.concatenate([-k_ed[:0:-1], k_ed])          # even extension
ee = np.concatenate([e_ed[:0:-1], e_ed])
E = CubicSpline(kk, ee)


def fold_pi(k):
    return (np.asarray(k) + np.pi) % (2 * np.pi) - np.pi


ELWIN = (-0.8, 1.6)
SIG_K = 2.0 / 3.0  # packet momentum spread, sigma_x = 0.75


def centroid(q0, col):
    """First moment of the positive part over the elastic window: robust on
    window-flattened peaks where argmax is noise-dominated."""
    m = (q0 > ELWIN[0]) & (q0 < ELWIN[1])
    w = np.clip(col[m], 0, None)
    return float(np.sum(q0[m] * w) / np.sum(w))


def ridge(Wfile):
    d = np.load(Wfile)
    q0, q1, W, spread = d["q0"], d["q1"], d["W"].real, d["spread"]
    pk = np.array([centroid(q0, W[:, j]) for j in range(len(q1))])
    # window-scan systematic propagated through the same estimator
    err = np.array([abs(centroid(q0, W[:, j] + spread[:, j] / 2)
                        - centroid(q0, W[:, j] - spread[:, j] / 2)) / 2
                    + 0.02 for j in range(len(q1))])
    return d, pk, err


def smeared_prediction(q1_arr, kbar, sigma_t):
    """Same centroid applied to the ED-predicted elastic profile: lines at
    E(fold(k + q1)) - E(k) weighted by the packet momentum content, convolved
    with the exact Gaussian window kernel (sigma_omega = 1/sigma_t)."""
    q0 = np.arange(ELWIN[0], ELWIN[1], 0.01)
    kgrid = np.linspace(kbar - 3 * SIG_K, kbar + 3 * SIG_K, 61)
    wk = np.exp(-((kgrid - kbar) ** 2) / (2 * SIG_K**2))
    out = []
    for q in q1_arr:
        prof = np.zeros_like(q0)
        for k, w in zip(kgrid, wk):
            om = E(fold_pi(k + q)) - E(fold_pi(k))
            prof += w * np.exp(-((q0 - om) ** 2) * sigma_t**2 / 2)
        out.append(centroid(q0, prof))
    return np.array(out)


d0, pk0, er0 = ridge("data/w_meson_ns50_k0.00_v3_W.npz")
db, pkb, erb = ridge("data/w_meson_ns50_k1.26_v3_W.npz")
q1 = d0["q1"]
kbar = float(db["k0"])
SIGMA_T = 8.0 / 3.0

fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.9), constrained_layout=True,
                         gridspec_kw={"width_ratios": [1, 1, 1.15]})
vmax = max(np.percentile(d0["W"].real, 99.5), np.percentile(db["W"].real, 99.5))
for ax, d, ttl in ((axes[0], d0, r"rest, $\bar k = 0$"),
                   (axes[1], db, rf"boosted, $\bar k = {kbar:.2f}$")):
    pm = ax.pcolormesh(d["q1"], d["q0"], d["W"].real, cmap="Blues",
                       shading="nearest", vmin=0, vmax=vmax, rasterized=True)
    ax.set_xlabel(r"$q^1$")
    ax.set_title(ttl, fontsize=11)
    ax.grid(False)
    ax.set_ylim(-1, 4)
axes[0].set_ylabel(r"$q^0$")
fig.colorbar(pm, ax=axes[1], pad=0.02, label=r"$W^{00}$")

# Boost-asymmetry observable A(q1) = <q0>(+q1) - <q0>(-q1): odd in q1, so
# the circuit-chirality preparation systematic largely cancels and the REST
# packet provides the null test.
ax = axes[2]
pos = q1 > 0.3
qp = q1[pos]


def asym(pk):
    out = []
    for q in qp:
        i = np.argmin(np.abs(q1 - q))
        j = np.argmin(np.abs(q1 + q))
        out.append(pk[i] - pk[j])
    return np.array(out)


qf = np.linspace(0.35, np.pi, 60)
predA_b = smeared_prediction(qf, kbar, SIGMA_T) - smeared_prediction(-qf, kbar, SIGMA_T)
ax.plot(qf, predA_b, color=OI[1], lw=1.7, label="smeared ED prediction, boosted")
ax.axhline(0, color=OI[0], lw=1.2, ls="--", label="prediction, rest (null)")
eA0, eAb = asym(er0) * 0 + np.sqrt(2) * er0[np.argmax(pos):][:len(qp)], \
    np.sqrt(2) * erb[np.argmax(pos):][:len(qp)]
ax.errorbar(qp, asym(pk0), yerr=eA0, fmt="o", ms=4.5, color=OI[0],
            capsize=2, lw=1, label="rest (null test)")
ax.errorbar(qp, asym(pkb), yerr=eAb, fmt="s", ms=4.5, color=OI[1],
            capsize=2, lw=1, label="boosted")
ax.set_xlabel(r"$q^1$")
ax.set_ylabel(r"$A(q^1) = \langle q^0\rangle_{+q^1} - \langle q^0\rangle_{-q^1}$")
ax.set_title("boost asymmetry vs prediction", fontsize=11)
ax.legend(fontsize=8, framealpha=0.9)

fig.savefig("data/w_headline_ns50.pdf", dpi=200)
print("wrote data/w_headline_ns50.pdf")
predA_data = (smeared_prediction(qp, kbar, SIGMA_T)
              - smeared_prediction(-qp, kbar, SIGMA_T))
Ab, A0 = asym(pkb), asym(pk0)
print(f"boosted A(q1) RMS vs prediction: {np.sqrt(np.mean((Ab-predA_data)**2)):.3f}")
print(f"null (rest) A(q1) RMS: {np.sqrt(np.mean(A0**2)):.3f}  "
      f"boosted signal RMS: {np.sqrt(np.mean(Ab**2)):.3f}")
