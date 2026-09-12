"""Merged synthesis-tolerance figure: the 17-qubit pilot (lines) overlaid
with the 101-qubit ladder (markers).  Left: ridge RMS error / peak -- an
INTENSIVE quantity, so the two scales fall together (tolerance is scale
invariant).  Right: |Delta<H>| -- an EXTENSIVE quantity, so the 101-qubit
points sit above the pilot (energy error grows with system size).  Colours =
rounding mode; lines = 17q pilot, filled markers = 101q.

  PYTHONPATH=. .venv/bin/python scripts/synthesis_merged.py
"""
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullFormatter, NullLocator
from matplotlib.lines import Line2D
import numpy as np

from htensor import Z2Lattice, analysis
from scripts_helpers_ridge import onesided_ft

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paper_style import *  # usetex + OI palette + dev()
DET, STOCH = "#0072B2", "#D55E00"          # Okabe-Ito blue / orange

# ================= 101-qubit ladder (markers) =================
lat = Z2Lattice(50, pbc=True)
xg = analysis.ring_fold((np.arange(50) - 24) / 2, lat.nx)
Q0 = np.arange(-0.6, 1.601, 0.04)


def ridge(f, W0, peak):
    d = np.load(f)
    G = d["corr"] - d["one_pt"] * complex(d["insert_1pt"])
    W = onesided_ft(d["times"], xg, G, Q0, np.array([np.pi / 2]),
                    8 / 3, lat.nx / 6, 0.5, 0.5)[:, 0]
    return np.sqrt(np.mean(np.abs(W - W0) ** 2)) / peak


def herr(f, H0):
    return abs(complex(np.load(f)["H"]).real - H0)


d0 = np.load("data/synthladder_exact.npz")
G0 = d0["corr"] - d0["one_pt"] * complex(d0["insert_1pt"])
W0 = onesided_ft(d0["times"], xg, G0, Q0, np.array([np.pi / 2]),
                 8 / 3, lat.nx / 6, 0.5, 0.5)[:, 0]
peak = np.abs(W0).max()
H0 = complex(d0["H"]).real
denomsL = [128, 64, 32, 16]
deltaL = np.pi / np.array(denomsL)


def collectL(metric):
    sm, ss, rn = [], [], []
    for d in denomsL:
        e = [metric(f) for f in glob.glob(f"data/synthladder_stoc_pi{d}*.npz")]
        sm.append(np.mean(e)); ss.append(np.std(e) if len(e) > 1 else 0)
        rf = f"data/synthladder_roun_pi{d}_s0.npz"
        if not os.path.exists(rf):
            rf = f"data/synthladder_roun_pi{d}_s0_chi256.npz"
        rn.append(metric(rf) if os.path.exists(rf) else np.nan)
    return np.array(sm), np.array(ss), np.array(rn)


L_r_sm, L_r_ss, L_r_rn = collectL(lambda f: ridge(f, W0, peak))
L_h_sm, L_h_ss, L_h_rn = collectL(lambda f: herr(f, H0))

# coarse-delta points from bond-capped runs (chi <= 256): value = chi256,
# systematic = |chi256 - chi128| (converged to <1% at both points)
CAPPED = [8, 4]
deltaC = np.pi / np.array(CAPPED)
C_r, C_r_e, C_h = [], [], []
for d in CAPPED:
    r256 = ridge(f"data/synthladder_stoc_pi{d}_s1_chi256.npz", W0, peak)
    r128 = ridge(f"data/synthladder_stoc_pi{d}_s1_chi128.npz", W0, peak)
    C_r.append(r256); C_r_e.append(abs(r256 - r128))
    C_h.append(herr(f"data/synthladder_stoc_pi{d}_s1_chi256.npz", H0))
C_r, C_r_e, C_h = map(np.array, (C_r, C_r_e, C_h))
# deterministic companions at the same coarse deltas: the snapped angles
# collapse toward zero, entanglement dies, and the chi cap is idle
# (chi256-chi128 difference is the quoted systematic)
D_r, D_r_e, D_h = [], [], []
for d in CAPPED:
    r256 = ridge(f"data/synthladder_roun_pi{d}_s0_chi256.npz", W0, peak)
    r128 = ridge(f"data/synthladder_roun_pi{d}_s0_chi128.npz", W0, peak)
    D_r.append(r256); D_r_e.append(abs(r256 - r128))
    D_h.append(herr(f"data/synthladder_roun_pi{d}_s0_chi256.npz", H0))
D_r, D_r_e, D_h = map(np.array, (D_r, D_r_e, D_h))
cliff_r = ridge("data/synthladder_clifford.npz", W0, peak)
cliff_h = herr("data/synthladder_clifford.npz", H0)

# ================= 17-qubit pilot (lines) =================
dp = np.load("data/synthesis_pilot_ns8.npz")
rows = dp["rows"]                          # [delta, mode(0 det,1 stoch), seed, gap, err]
gap0 = float(dp["gap0"])
deltaP = np.unique(rows[:, 0])


def pilot(mode, kind):
    m, s = [], []
    for dl in deltaP:
        sel = rows[(rows[:, 0] == dl) & (rows[:, 1] == mode)]
        v = np.abs(sel[:, 3] - gap0) if kind == "H" else sel[:, 4]
        m.append(np.mean(v)); s.append(np.std(v) if len(v) > 1 else 0.0)
    return np.array(m), np.array(s)


# ================= plot =================
def d2t(z):
    return 3 * np.log2(2 / np.clip(z, 1e-12, None))
def t2d(t):
    return 2 / 2 ** (np.asarray(t) / 3)


def style(ax, ylab):
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"synthesis grid spacing $\delta$ (rad)")
    ax.set_ylabel(ylab)
    ax.set_xlim(0.02, 2.0)
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    sx = ax.secondary_xaxis("top", functions=(d2t, t2d))
    sx.set_xlabel(r"$\sim T$ gates per rotation (Ross--Selinger)")
    sx.set_xticks([4, 8, 12, 16])
    sx.xaxis.set_major_formatter(FuncFormatter(lambda t, _: f"{t:.0f}"))
    sx.xaxis.set_minor_locator(NullLocator())


fig, (aL, aR) = plt.subplots(1, 2, figsize=(6.6, 2.55),
                             constrained_layout=True)
for ax, kind, ls, lss, Lsm, Lss, Lrn, cl, ylab in (
    (aL, "ridge", None, None, L_r_sm, L_r_ss, L_r_rn, cliff_r, r"ridge RMS error / peak"),
    (aR, "H", None, None, L_h_sm, L_h_ss, L_h_rn, cliff_h, r"$|\Delta\langle H\rangle|$ error"),
):
    # 17-qubit pilot: unconnected open markers (seed scatter as bars)
    for mode, c, mk in ((0, DET, "o"), (1, STOCH, "s")):
        m, s = pilot(mode, kind)
        ax.errorbar(deltaP, m, yerr=s, fmt=mk, color=c, ms=4.5, mfc="none",
                    mew=1.0, lw=0, elinewidth=0.7, capsize=1.5, alpha=0.9)
    # BOTH candidate scalings drawn in BOTH panels so each panel visibly
    # selects one: the unscaled pilot curve is the solid line above; the
    # sqrt(N_rot) ~ sqrt(50/8) rescaled pilot is the dotted guide.  The
    # 101q points follow the solid line on the left (intensive ridge
    # error) and the dotted guide on the right (extensive energy error).
    m, _ = pilot(1, kind)
    ax.plot(deltaP, m * np.sqrt(50 / 8), ":", color=STOCH, lw=1.3,
            alpha=0.85)
    # 101-qubit ladder: large filled markers (+ seed error bars)
    ax.errorbar(deltaL, Lrn, fmt="o", color=DET, ms=6, lw=0)
    ax.errorbar(deltaL, Lsm, yerr=Lss, fmt="s", color=STOCH, ms=6, lw=0,
                elinewidth=0.9, capsize=2.5)
    Dy = D_r if kind == "ridge" else D_h
    De = D_r_e if kind == "ridge" else None
    ax.errorbar(deltaC, Dy, yerr=De, fmt="o", color=DET, ms=6, lw=0,
                elinewidth=0.9, capsize=2.5)
    Cy = C_r if kind == "ridge" else C_h
    Ce = C_r_e if kind == "ridge" else None
    ax.errorbar(deltaC, Cy, yerr=Ce, fmt="s", color=STOCH, ms=6, lw=0,
                elinewidth=0.9, capsize=2.5)
    ax.axhline(cl, color="0.55", ls=":", lw=0.9)
    # offset in points -> identical visual spacing on both panels
    ax.annotate("Clifford", xy=(0.012, cl),
                xycoords=("axes fraction", "data"),
                xytext=(0, 2.5), textcoords="offset points",
                fontsize=7, color="0.4", ha="left", va="bottom")
    style(ax, ylab)
aL.set_ylim(0.009, 1.0)
aR.set_ylim(0.004, 30)          # headroom for the Clifford line at ~9.4

# split legends into the empty lower-right corners so neither covers data:
# colour = rounding (left panel), marker style = qubit count (right panel)
h_color = [Line2D([], [], color=DET, lw=2, label="deterministic"),
           Line2D([], [], color=STOCH, lw=2, label="stochastic")]
h_size = [Line2D([], [], color="0.4", lw=0, marker="o", ms=4.5, mfc="none",
                 mew=1.0, label="17 qubits"),
          Line2D([], [], color="0.4", lw=0, marker="o", ms=6,
                 label="101 qubits")]
aL.legend(handles=h_color, fontsize=7.5, loc="lower right", framealpha=0.9)
aR.legend(handles=h_size, fontsize=7.5, loc="lower right", framealpha=0.9)

fig.savefig("data/synthesis_merged.pdf", dpi=200)
print(f"pilot ridge det/stoch at pi/128: {pilot(0,'ridge')[0][0]:.3f}/"
      f"{pilot(1,'ridge')[0][0]:.3f}; 101q: {L_r_rn[0]:.3f}/{L_r_sm[0]:.3f}")
print("wrote data/synthesis_merged.pdf")
