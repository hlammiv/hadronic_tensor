"""Merged postdictions figure (fig:postdictions, figure*): zero-parameter
postdictions of both published collision studies from small-volume input.

(a) CGK (arXiv:2505.21240) L=30 vector dispersion at eps=1.0;
(b) CGK vector+scalar at eps=0.2;
(c) DHK (arXiv:2505.20408) 27-qubit return probability R(t).

Curves = our ns<=20 predictions; points = their published values
(vector-extracted from the arXiv-source figures).  Supersedes
dhk_postdiction_figure.py + cgk_postdiction_figure.py.

  PYTHONPATH=. .venv/bin/python scripts/postdictions_figure.py
"""
import numpy as np
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paper_style import *  # usetex + OI palette + dev()
BLUE, ORANGE = "#0072B2", "#D55E00"


# ---------------- CGK bands from small-volume deep levels ----------------
def bands(tag):
    b1, b2 = {}, {}
    for ns in (8, 10, 12, 14, 16, 18, 20):
        d = np.load(f"data/deep_levels_cgk{tag}_ns{ns}.npz")
        cl = defaultdict(list)
        for g, p in zip(d["gaps"], d["phases"]):
            if g > 0.05:
                cl[abs(round(float(p), 4))].append(float(g))
        for k, gs in cl.items():
            gs = sorted(gs)
            mult = 1 if (k < 1e-3 or abs(k - np.pi) < 1e-3) else 2
            b1[k] = gs[0]
            if len(gs) > mult:
                b2[k] = gs[mult]
    k1 = np.array(sorted(b1)); e1 = np.array([b1[k] for k in k1])
    k2 = np.array(sorted(b2)); e2 = np.array([b2[k] for k in k2])
    return (k1, e1), (k2, e2)


def glue_curve(tag, jj):
    (k1, e1), (k2, e2) = bands(tag)
    k = 2 * np.pi * np.abs(jj) / 30
    return np.where(k <= np.pi / 2,
                    np.interp(2 * k, k1, e1),
                    np.interp(2 * np.pi - 2 * k, k2, e2))


# ---------------- DHK factorized R(t) from small-volume dispersion -------
def dhk_curve(times):
    kpts = {}
    for ns in (12, 16, 20):
        d = np.load(f"data/deep_levels_dhk_ns{ns}.npz")
        b = defaultdict(list)
        for g, p in zip(d["gaps"], d["phases"]):
            if g > 0.1:
                b[abs(round(float(p), 4))].append(float(g))
        for k in b:
            kpts[k] = min(kpts.get(k, 99), min(b[k]))
    ks = np.array(sorted(kpts)); Es = np.array([kpts[k] for k in ks])
    NP, SIGMA, KBAR = 13, 3 * np.pi / 13, 2 * np.pi / 13
    k = np.pi / NP * np.arange(-NP, NP)
    k = k[(k >= -np.pi / 2 - 1e-9) & (k < np.pi / 2 - 1e-9)]
    w1 = np.exp(-((k - KBAR) ** 2) / (2 * SIGMA ** 2))
    w2 = np.exp(-((k + KBAR) ** 2) / (2 * SIGMA ** 2))
    w1, w2 = w1 / w1.sum(), w2 / w2.sum()
    Kf = (2 * k + np.pi) % (2 * np.pi) - np.pi
    E = np.interp(np.abs(Kf), ks, Es)
    A1 = np.array([(w1 * np.exp(-1j * E * t)).sum() for t in times])
    A2 = np.array([(w2 * np.exp(-1j * E * t)).sum() for t in times])
    return np.abs(A1 * A2) ** 2


cmp = np.load("data/cgk_spectrum_compare.npz")
true = np.load("data/dhk_Rt_true.npz")
jj = np.linspace(-13.5, 14.5, 400)

fig, (aA, aB, aC) = plt.subplots(1, 3, figsize=(7.1, 2.45),
                                 constrained_layout=True)
# (a) CGK eps=1.0 vector
aA.plot(jj, glue_curve("el", jj), "-", color=BLUE, lw=1.4,
        label=r"ED, this work ($N_s\leq20$)")
pa = cmp["panel_a"]
aA.plot(pa[:, 0], pa[:, 1], "o", mfc="none", mec="0.2", ms=4, mew=1.0,
        label=r"published TN ($N_s{=}30$)")
aA.set_xlabel(r"$k \times (L/2\pi)$")
aA.set_ylabel(r"$\Delta E$")
aA.legend(fontsize=6.3, loc="upper center", framealpha=0.9)
aA.text(0.04, 0.05, r"$\varepsilon{=}1.0$", transform=aA.transAxes, fontsize=8)
aA.text(0.96, 0.05, "(a)", transform=aA.transAxes, fontsize=9, ha="right")

# (b) CGK eps=0.2 vector + scalar
aB.plot(jj, glue_curve("inA", jj), "-", color=BLUE, lw=1.4)
pc = cmp["panel_c"]
aB.plot(pc[:, 0], pc[:, 1], "o", mfc="none", mec="0.2", ms=4, mew=1.0)
(k1, e1), (k2, e2) = bands("inA")
js = np.linspace(-7.4, 7.4, 200)
aB.plot(js, np.interp(2 * (2 * np.pi * np.abs(js) / 30), k2, e2), "-",
        color=ORANGE, lw=1.4)
sc = cmp["scalar_c"]
aB.plot(sc[:, 0], sc[:, 1], "D", mfc="none", mec="0.2", ms=4, mew=1.0)
aB.text(-2.1, 1.335, "scalar", color=ORANGE, fontsize=7.5)
aB.text(3.6, 1.02, "vector", color=BLUE, fontsize=7.5)
aB.set_xlabel(r"$k \times (L/2\pi)$")
aB.text(0.04, 0.05, r"$\varepsilon{=}0.2$", transform=aB.transAxes, fontsize=8)
aB.text(0.96, 0.05, "(b)", transform=aB.transAxes, fontsize=9, ha="right")

# (c) DHK R(t)
t, R_dhk = true["t"].astype(float), true["R_ideal"]
tt = np.linspace(0, 20, 300)
aC.plot(t, R_dhk, "s", color="0.25", ms=3.8, label=r"published collision ($N_s{=}26$)")
aC.plot(tt, dhk_curve(tt), "-", color=BLUE, lw=1.5, label="dephasing model, this work")
aC.set_xlabel(r"time $t$")
aC.set_ylabel(r"$\mathcal{R}(t)$")
aC.set_xlim(0, 20); aC.set_ylim(0, 1.05)
aC.legend(fontsize=6.3, loc="lower left", framealpha=0.9)
aC.text(0.96, 0.9, "(c)", transform=aC.transAxes, fontsize=9, ha="right")

fig.savefig("data/postdictions_fig.pdf", dpi=200)
print("wrote data/postdictions_fig.pdf")
