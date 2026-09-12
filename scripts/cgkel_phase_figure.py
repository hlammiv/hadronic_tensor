"""Elastic phase shift at the Chai epsilon=1.0 point (cgkel), against
relative momentum p (threshold-smooth; an energy axis has a sqrt kink at
2M).  Top axis: two-meson energy E via the measured dispersion.  Band:
the predicted -pi/2 crossing E = 5.01(2), the falsifiable target for the
collision program; shading beyond the inelastic threshold E_MM'.

  PYTHONPATH=. python scripts/cgkel_phase_figure.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_style import OI, MARKERS

d = np.load("data/predict_tables.npz", allow_pickle=True)
pts = d["cgkel_points"]                # (ns, E2, p, delta)
thr_inel = float(d["cgkel_thr"])
vols = sorted(set(int(r[0]) for r in pts))
style = {v: (OI[i], MARKERS[i]) for i, v in enumerate(vols)}

# monotone E<->p map from the pooled level data
srt = pts[np.argsort(pts[:, 2])]
P, E = srt[:, 2], srt[:, 1]
p2E = lambda x: np.interp(x, P, E)
E2p = lambda x: np.interp(x, E[np.argsort(E)], P[np.argsort(E)])

# -pi/2 crossings per volume -> prediction band
crossE = []
for v in vols:
    sel = pts[pts[:, 0] == v]
    sel = sel[np.argsort(sel[:, 1])]
    dl = np.where((sel[:, 1] > 4.95) & (sel[:, 3] > 0),
                  sel[:, 3] - np.pi, sel[:, 3])
    for i in range(len(dl) - 1):
        if (dl[i] + np.pi / 2) * (dl[i + 1] + np.pi / 2) < 0:
            t = (-np.pi / 2 - dl[i]) / (dl[i + 1] - dl[i])
            crossE.append(sel[i, 1] + t * (sel[i + 1, 1] - sel[i, 1]))
Ec, Ece = float(np.mean(crossE)), float(np.std(crossE))
# draw the crossing in the variable it is quoted in (E), converted
# through the same map as the top axis, so line and label agree
pc = float(E2p(Ec))
p_lo, p_hi = float(E2p(Ec - Ece)), float(E2p(Ec + Ece))

# near-threshold ERE on this branch: phi = delta + pi/2 vanishes at
# threshold (the 1D theorem's repulsive-channel image of fig:delta's
# pi/2); fit p cot(phi) = 1/a + r p^2 / 2 below the crossing region
mask = pts[:, 2] <= 1.0
pp, ph = pts[mask, 2], np.where(
    (pts[mask, 1] > 4.95) & (pts[mask, 3] > 0),
    pts[mask, 3] - np.pi, pts[mask, 3]) + np.pi / 2
Xf = np.vstack([np.ones(mask.sum()), pp ** 2 / 2]).T
cf, *_ = np.linalg.lstsq(Xf, pp / np.tan(ph), rcond=None)
a_thr = 1.0 / cf[0]
pf = np.linspace(0.004, 1.05, 400)
phi_f = np.arctan(pf / (cf[0] + cf[1] * pf ** 2 / 2))
print(f"threshold ERE: 1/a = {cf[0]:.4f} -> a = {a_thr:.1f}, "
      f"r = {cf[1]:.3f}")

# Chai et al. elastic collision kinematics: kbar = 6 x 2pi/30, sigma_k =
# 2pi/30 -> relative momentum band
KBAR, SK = 6 * 2 * np.pi / 30, 2 * np.pi / 30 / np.sqrt(2)

fig, ax = plt.subplots(figsize=(3.3, 2.7), constrained_layout=True)
ax.axvspan(p_lo, p_hi, color="#0072B2", alpha=0.20, lw=0)
ax.axvline(pc, color="#0072B2", lw=0.9)
p_inel = float(E2p(thr_inel))
ax.axvspan(p_inel, 2.85, color="0.85", alpha=0.5, lw=0)
for v in vols:
    sel = pts[pts[:, 0] == v]
    dl = np.where((sel[:, 1] > 4.95) & (sel[:, 3] > 0),
                  sel[:, 3] - np.pi, sel[:, 3])
    c, mk = style[v]
    ax.plot(sel[:, 2], dl, mk, color=c, ms=5, lw=0,
            label=f"$N_x={v//2}$")
ax.plot(pf, phi_f - np.pi / 2, "-", color="0.35", lw=1.1, zorder=1)
ax.axhline(-np.pi / 2, color="0.7", lw=0.6, ls=":")
ax.text(0.03, -np.pi / 2 + 0.03, r"$-\pi/2$", fontsize=7, color="0.4")
ax.text(p_lo - 0.05, -0.62, r"$E_{\delta=-\pi/2} = 5.01(2)$", fontsize=7,
        color="#0072B2", ha="right")
ax.text(1.30, -0.90, r"$\tau = -9.5(2.8)$", fontsize=6.6,
        color="#D55E00", ha="left")
# slope segment: their measured Wigner advance tau = -9.5(2.8) ->
# ddelta/dE = tau/2, converted to the p axis via the local dE/dp;
# vertical position borrowed from our curve at their energy
dEdp = (p2E(KBAR + 0.05) - p2E(KBAR - 0.05)) / 0.1
sl_p = -9.5 / 2 * dEdp
d_at = np.interp(KBAR, np.sort(pts[:, 2]),
                 np.where((pts[:, 1] > 4.95) & (pts[:, 3] > 0),
                          pts[:, 3] - np.pi,
                          pts[:, 3])[np.argsort(pts[:, 2])])
seg = np.linspace(KBAR - 0.22, KBAR + 0.22, 60)
dsl = 2.8 / 2 * dEdp
ax.fill_between(seg, d_at + (sl_p - dsl) * (seg - KBAR),
                d_at + (sl_p + dsl) * (seg - KBAR),
                color="#D55E00", alpha=0.22, lw=0, zorder=5)
ax.plot(seg, d_at + sl_p * (seg - KBAR), "-", color="#D55E00",
        lw=1.6, zorder=6)
ax.text(p_inel + 0.05, -1.05, "inelastic", fontsize=6.5, color="0.45",
        rotation=90, va="center")
ax.set_xlim(0.0, 2.85)
ax.set_xlabel("relative momentum $p$")
ax.set_ylabel(r"elastic phase shift $\delta(p)$")
sx = ax.secondary_xaxis("top", functions=(p2E, E2p))
sx.set_xlabel(r"$E$ (two-meson energy)", fontsize=8)
sx.set_xticks([4.7, 4.8, 4.9, 5.0, 5.1])
ax.legend(fontsize=6.0, ncol=4, loc="lower left",
          bbox_to_anchor=(0.01, 0.015), framealpha=0.9,
          columnspacing=0.55, handletextpad=0.25,
          labelspacing=0.45, borderpad=0.3, handlelength=1.1)
fig.savefig("data/cgkel_phase.pdf", dpi=200)
print(f"wrote data/cgkel_phase.pdf; crossing E = {Ec:.3f}+-{Ece:.3f} -> p = {pc:.3f}, "
      f"E = {p2E(pc):.3f}; inelastic at p = {p_inel:.3f}")
print(f"Chai collision: E = {p2E(KBAR):.3f} "
      f"[{p2E(KBAR-SK):.3f}, {p2E(KBAR+SK):.3f}]")
