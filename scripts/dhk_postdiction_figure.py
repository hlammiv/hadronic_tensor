"""Paper figure (fig:postdiction): zero-free-parameter postdiction of the
DHK (arXiv:2505.20408) 27-qubit collision return probability from
small-volume nonperturbative input.

(a) the input: single-meson dispersion E(K) measured at volumes ns<=20
    (merged), with DHK's own exact/DMRG finite-volume energies overlaid at
    the derived physical momentum K = 2k -- the convention test.
(b) the output: R(t) = |A_1(t) A_2(t)|^2 with A_i = sum_k |Psi_i(k)|^2
    e^{-i E(2k) t}, packets from their Table I, vs their MPS-TDVP ideal
    curve (vector-extracted from their figure).  The naive K = k curve
    shows the convention is discriminated by the data.

  PYTHONPATH=. .venv/bin/python scripts/dhk_postdiction_figure.py
"""
import numpy as np
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "stix",
                     "font.size": 9, "axes.grid": True, "grid.alpha": 0.25,
                     "grid.linewidth": 0.6})
BLUE, ORANGE = "#0072B2", "#D55E00"          # Okabe-Ito

NP, SIGMA, KBAR = 13, 3 * np.pi / 13, 2 * np.pi / 13   # their Table I


def disp_from(volumes):
    kpts = {}
    for ns in volumes:
        d = np.load(f"data/deep_levels_dhk_ns{ns}.npz")
        b = defaultdict(list)
        for g, p in zip(d["gaps"], d["phases"]):
            if g > 0.1:
                b[abs(round(float(p), 4))].append(float(g))
        for k in b:
            kpts[k] = min(kpts.get(k, 99), min(b[k]))
    ks = np.array(sorted(kpts))
    return ks, np.array([kpts[k] for k in ks])


def factorized_R(times, ks, Es, doubling):
    k = np.pi / NP * np.arange(-NP, NP)
    k = k[(k >= -np.pi / 2 - 1e-9) & (k < np.pi / 2 - 1e-9)]
    w1 = np.exp(-((k - KBAR) ** 2) / (2 * SIGMA ** 2))
    w2 = np.exp(-((k + KBAR) ** 2) / (2 * SIGMA ** 2))
    w1, w2 = w1 / w1.sum(), w2 / w2.sum()
    Kf = (doubling * k + np.pi) % (2 * np.pi) - np.pi
    E = np.interp(np.abs(Kf), ks, Es)
    A1 = np.array([(w1 * np.exp(-1j * E * t)).sum() for t in times])
    A2 = np.array([(w2 * np.exp(-1j * E * t)).sum() for t in times])
    return np.abs(A1 * A2) ** 2


ks, Es = disp_from((12, 16, 20))
true = np.load("data/dhk_Rt_true.npz")
t, R_dhk = true["t"].astype(float), true["R_ideal"]
tt = np.linspace(0, 20, 400)
R2 = factorized_R(tt, ks, Es, 2)
R1 = factorized_R(tt, ks, Es, 1)
rms2 = float(np.sqrt(np.mean((factorized_R(t, ks, Es, 2) - R_dhk) ** 2)))
rms1 = float(np.sqrt(np.mean((factorized_R(t, ks, Es, 1) - R_dhk) ** 2)))
k12, E12 = disp_from((8, 10, 12))
rms12 = float(np.sqrt(np.mean((factorized_R(t, k12, E12, 2) - R_dhk) ** 2)))
print(f"RMS derived {rms2:.4f} | naive {rms1:.4f} | ns<=12 input {rms12:.4f}")

# DHK exact/DMRG finite-volume energies (their appendix tables), gaps above
# vacuum, plotted at the derived physical momentum K = 2k
dhk13_k = 2 * np.pi / 13 * np.arange(7)                    # K = 2*(j*pi/13)
dhk13_E = 2.7488 + np.array([0, 0.0068, 0.0263, 0.0567, 0.0949, 0.1373, 0.1803])
dhk5_k = np.array([0, 2 * np.pi / 5, 4 * np.pi / 5])
dhk5_E = 2.7487 + np.array([0, 0.0435, 0.1460])

fig, (aT, aB) = plt.subplots(2, 1, figsize=(3.5, 4.5), constrained_layout=True)

# ---- (a) dispersion: our measured input + their spectra at K=2k ----
kk = np.linspace(0, np.pi, 300)
aT.plot(kk, np.interp(kk, ks, Es), "-", color=BLUE, lw=1.4,
        label=r"this work, volumes $N_s\leq 20$")
aT.plot(ks, Es, "o", color=BLUE, ms=3.5)
aT.plot(dhk13_k, dhk13_E, "s", mfc="none", mec="0.2", ms=5, mew=1.1,
        label=r"Ref. DMRG, $N_P{=}13$")
aT.plot(dhk5_k, dhk5_E, "D", mfc="none", mec=ORANGE, ms=5, mew=1.1,
        label=r"Ref. exact, $N_P{=}5$")
# packet |Psi|^2 sampling weight along the baseline
wK = np.exp(-((kk - 2 * KBAR) ** 2) / (2 * (2 * SIGMA) ** 2))
aT.fill_between(kk, 2.735, 2.735 + 0.035 * wK, color=BLUE, alpha=0.15, lw=0)
aT.set_xlabel(r"physical momentum $K$")
aT.set_ylabel(r"$E(K) - E_\Omega$")
aT.set_xlim(0, np.pi); aT.set_ylim(2.735, 2.96)
aT.legend(fontsize=6.6, loc="upper left", framealpha=0.9)
aT.text(0.955, 0.06, "(a)", transform=aT.transAxes, fontsize=9)

# ---- (b) R(t): their ideal curve vs our zero-parameter postdiction ----
aB.plot(t, R_dhk, "s", color="0.25", ms=4.5, label=r"Ref. MPS ideal (27 qb)")
aB.plot(tt, R2, "-", color=BLUE, lw=1.5,
        label=rf"postdiction (RMS ${rms2:.3f}$)")
aB.set_xlabel(r"time $t$")
aB.set_ylabel(r"$\mathcal{R}(t)$")
aB.set_xlim(0, 20); aB.set_ylim(0, 1.05)
aB.legend(fontsize=6.6, loc="lower left", framealpha=0.9)
aB.text(0.955, 0.9, "(b)", transform=aB.transAxes, fontsize=9)

fig.savefig("data/dhk_postdiction_fig.pdf", dpi=200)
print("wrote data/dhk_postdiction_fig.pdf")
