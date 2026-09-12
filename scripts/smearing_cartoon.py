"""Schematic: the interplay of finite volume, Gaussian smearing, and
resolution for smeared spectral densities, on synthetic data (fig:ordered).

A cartoon density S(E) is sampled by discrete finite-volume levels at a
small volume (spacing D_A) and a large volume (D_B < D_A).  Smearing at
sigma > D_A is volume-universal but kernel-flattened; sigma < D_A resolves
the small volume's own levels (volume-specific wiggles); sigma ~ D_B is
sharp AND universal.  PYTHONPATH=. python scripts/smearing_cartoon.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_style import OI

E = np.linspace(0.0, 1.0, 800)


def S_true(e):
    """Cartoon density: threshold rise into a skewed resonance peak."""
    thr = np.clip(e - 0.06, 0, None) ** 0.5
    peak = np.exp(-0.5 * ((e - 0.55) / 0.16) ** 2)
    return thr * peak / 0.66           # unit peak height


def levels(delta):
    en = np.arange(0.08, 1.0, delta)
    return en, S_true(en) * delta      # weights: density x spacing


def smear(en, wn, sig):
    return (wn[:, None] * np.exp(-(E[None] - en[:, None]) ** 2 /
            (2 * sig ** 2))).sum(0) / (np.sqrt(2 * np.pi) * sig)


DA, DB = 0.085, 0.028                  # small / large volume spacings
SIG1, SIG2 = 0.11, 0.035               # coarse / fine kernels
eA, wA = levels(DA)
eB, wB = levels(DB)

fig, ax = plt.subplots(figsize=(3.4, 2.9), constrained_layout=True)
ax.plot(E, S_true(E), "-", color="k", lw=1.6, zorder=5,
        label="infinite volume, unsmeared")
ax.plot(E, smear(eA, wA, SIG1), "-", color="0.45", lw=1.3, zorder=3,
        label=r"$\sigma_E > \Delta_A$: universal, flattened")
ax.plot(E, smear(eA, wA, SIG2), "-", color=OI[1], lw=1.1, zorder=2,
        label=r"$\sigma_E < \Delta_A$: resolves levels")
ax.plot(E, smear(eB, wB, SIG2), "--", color=OI[0], lw=1.4, zorder=4,
        label=r"$\sigma_E \gtrsim \Delta_B$: sharp + universal")
# level combs (ticks at the bottom, two rows)
for en, y0, col, lab in ((eA, -0.115, "0.45", r"$\Delta_A$ (small $N_x$)"),
                         (eB, -0.215, OI[0], r"$\Delta_B$ (large $N_x$)")):
    ax.vlines(en, y0, y0 + 0.075, color=col, lw=1.0)
    ax.text(1.005, y0 + 0.02, lab, fontsize=6.2, color=col, va="bottom")
ax.axhline(0, color="0.8", lw=0.5)
ax.set_xlim(0, 1.32)
ax.set_ylim(-0.25, 1.32)
ax.set_xticks([]); ax.set_yticks([])
ax.set_xlabel("$E$")
ax.set_ylabel(r"$S_{\sigma_E}(E)$")
ax.legend(fontsize=6.3, loc="upper right", framealpha=0.95,
          handlelength=1.6)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.savefig("data/smearing_cartoon.pdf", dpi=200)
print("wrote data/smearing_cartoon.pdf")
