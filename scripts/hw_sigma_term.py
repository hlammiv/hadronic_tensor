"""Hardware meson sigma-term from the certificate charge profiles.

sigma = dM/dm0 = sum_v (-1)^v [ <J0(v)>_packet - <J0(v)>_vacuum ], the
staggered sum of the connected (packet - vacuum) charge profile -- a
diagonal, ancilla-free observable.  The staggered sum is immune to a uniform
readout bias, and a same-device vacuum cancels the common bias; the residual
is a uncanceled staggered bias that, by locality, must appear as a flat floor
in the WINGS where the meson has no support.  Subtracting that (blind) wing
floor recovers sigma, with the window choice giving an honest systematic band.

  PYTHONPATH=. .venv/bin/python scripts/hw_sigma_term.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OI = ["#0072B2", "#D55E00", "#009E73", "#E69F00"]
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paper_style import *  # usetex + OI palette + dev()

NS, C = 50, 24
v = np.arange(NS)
sgn = (-1.0) ** v

# hardware: fez packet certificate + fez same-device (depth-matched) vacuum
d = np.load("data/hw/job_tier1_d96f12l2su3c739gsg0g.npz", allow_pickle=True)
Jp = np.array([float(d[f"J0_{i}"]) for i in range(NS)])
Je = np.array([float(d[f"J0_{i}_err"]) for i in range(NS)])
Jv = np.load("data/hw/vaccert_fez_matched.npz")["J0"]
s_hw = sgn * (Jp - Jv)                       # staggered connected integrand (sums to sigma)
s_err = np.sqrt(2) * Je                       # shot error on each s_v (packet + vacuum)

# kingston: forward-campaign harvest, passes 1+2 pooled (t0.0 - vac pubs)
kg = np.load("data/hw/losch_harvest_pooled.npz")
s_kg, s_ke = kg["s_kg"], kg["s_ke"]

# MPS truth for the same object, one curve per packet actually measured:
# fez ran the rest packet, kingston the boosted one (ibm_hardware.K0TAG).
def mps_truth(tag):
    m = np.load(f"data/w_meson_ns50_{tag}_v3.npz")
    return sgn * (np.asarray(m["one_pt_wp"])[0, :NS].real
                  - np.asarray(m["one_pt_vac"])[0, :NS].real)
s_mps = mps_truth("k0.00")     # rest    -> fez
s_mps_k = mps_truth("k1.26")   # boosted -> kingston

# --- blind wing-subtraction panel: connected profile -> 0 in the wings ---
Ws = np.array([8, 10, 12, 14, 16])
def sigma_W(sv, W):
    return (sv - sv[np.abs(v - C) > W].mean()).sum()
sig_by_W = np.array([sigma_W(s_hw, W) for W in Ws])
sig_raw = s_hw.sum()

# error decomposition by Monte Carlo over the shot noise
rng = np.random.default_rng(1)
NT = 4000
def panel(sv):
    return np.mean([sigma_W(sv, W) for W in Ws])
mc = np.array([panel(sgn * ((Jp + rng.normal(0, Je)) - (Jv + rng.normal(0, Je))))
               for _ in range(NT)])
sig = float(np.mean(sig_by_W))
stat = float(mc.std())
syst = float(sig_by_W.std())
print(f"raw (no wing subtraction) sigma = {sig_raw:.2f}")
print(f"blind wing-subtracted sigma = {sig:.2f} +- {stat:.2f}(stat) +- {syst:.2f}(sys) "
      f"[exact dM/dm0 = 1.88]")
print(f"kingston (harvest): {float(kg['sig']):.2f} +- {float(kg['stat']):.2f} "
      f"+- {float(kg['syst']):.2f}")

# ---------------- figure: single column, wing-scan inset ----------------
fig, axL = plt.subplots(figsize=(3.4, 3.0), constrained_layout=True)

# main: staggered connected integrand, hardware vs MPS, with the wing floor
x = v - C
floor = s_hw[np.abs(v - C) > 12].mean()
axL.axhspan(0, floor, color="0.6", alpha=0.25, lw=0, label="wing floor (bias)")
kfloor = float(s_kg[np.abs(v - C) > 12].mean())
axL.axhline(kfloor, color=OI[2], lw=0.9, ls="--", zorder=1)
axL.plot(x, s_mps, "-", color="k", lw=1.3, zorder=3, label="MPS (rest)")
axL.plot(x, s_mps_k, "--", color="k", lw=1.1, zorder=3, label="MPS (boosted)")
neg = s_hw < 0
axL.errorbar(x[~neg], s_hw[~neg], yerr=s_err[~neg], fmt="o", color=OI[1],
             ms=3, lw=0, elinewidth=0.7, capsize=1.2, zorder=4,
             label=dev("fez"))
# the two negative wing points are off the y>=0 scale: draw them as arrows
for xv in x[neg]:
    axL.annotate("", xy=(xv, 0.004), xytext=(xv, 0.075),
                 arrowprops=dict(arrowstyle="-|>", color=OI[1], lw=1.1),
                 zorder=4)
negk = s_kg < 0
axL.errorbar(x[~negk] + 0.25, s_kg[~negk], yerr=s_ke[~negk], fmt="s",
             color=OI[2], ms=2.6, lw=0, elinewidth=0.7, capsize=1.2,
             zorder=4, label=dev("kingston"))
for xv in x[negk]:
    axL.annotate("", xy=(xv + 0.25, 0.004), xytext=(xv + 0.25, 0.075),
                 arrowprops=dict(arrowstyle="-|>", color=OI[2], lw=1.0),
                 zorder=4)
axL.axhline(0, color="0.5", lw=0.6)
axL.set_xlim(-25, 25)
axL.set_ylim(0, 0.58)
axL.set_xlabel(r"$v - v_{\rm packet}$")
axL.set_ylabel(r"$(-1)^v[\langle J^0(v)\rangle_\Phi-\langle J^0(v)\rangle_\Omega]$")
axL.legend(fontsize=6.5, loc="upper right", framealpha=0.9)

# inset: dM/dm0 vs wing window, zoomed (no-subtraction line omitted)
axR = axL.inset_axes([0.14, 0.58, 0.28, 0.38])
axR.axhline(1.88, color="k", ls="--", lw=1.0)
axR.fill_between([Ws.min() - 1, Ws.max() + 1], sig - syst, sig + syst,
                 color=OI[1], alpha=0.18, lw=0)
axR.errorbar(Ws, sig_by_W, yerr=stat, fmt="o", color=OI[1], ms=3.2, lw=0,
             elinewidth=0.8, capsize=1.8, zorder=4)
axR.fill_between([Ws.min() - 1, Ws.max() + 1],
                 float(kg["sig"]) - float(kg["syst"]),
                 float(kg["sig"]) + float(kg["syst"]),
                 color=OI[2], alpha=0.15, lw=0)
axR.axhline(1.802, color=OI[2], ls="--", lw=0.9)
axR.errorbar(kg["Ws"] + 0.35, kg["sig_by_W"], yerr=float(kg["stat"]),
             fmt="s", color=OI[2], ms=2.8, lw=0, elinewidth=0.7,
             capsize=1.4, zorder=4)
axR.set_xlim(Ws.min() - 1, Ws.max() + 1)
axR.set_ylim(1.3, 3.0)
axR.set_xlabel(r"wing window $W$", fontsize=6.5, labelpad=1)
axR.set_ylabel(r"$\mathrm{d}M/\mathrm{d}m_0$", fontsize=6.5, labelpad=1)
axR.tick_params(labelsize=6)

fig.savefig("data/hw_sigma_term.pdf", dpi=200)
fig.savefig("data/hw_sigma_term.png", dpi=200)
print("wrote data/hw_sigma_term.pdf")
