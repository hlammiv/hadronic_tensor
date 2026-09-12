"""Candidate Fig. 10 with tensor-derived curves: (a) CGK eps=1.0 dispersion, (c) DHK R(t).
Inputs: data/rt_wvac_ridge_fullx.json (scripts/rt_wvac_ridge.py with RT_XWIN=full).
Panel (b) (eps=0.2) is added when data/rt_wvac_cgkinA_ns30_T20.npz exists in the json."""
import json, sys, os, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from postdictions_figure import glue_curve, dhk_curve
BLUE, ORANGE, GREEN = "#0072B2", "#D55E00", "#009E73"
R = json.load(open("data/rt_wvac_ridge_fullx.json"))
cmp = np.load("data/cgk_spectrum_compare.npz"); true = np.load("data/dhk_Rt_true.npz")
KEY_B = "cgkinA_ns30_T12_chi256" if "cgkinA_ns30_T12_chi256" in R else "cgkinA_ns30"
has_b = KEY_B in R
fig, axs = plt.subplots(1, 3 if has_b else 2, figsize=(7.1 if has_b else 4.9, 2.45), constrained_layout=True)
def panel(ax, key, pubkey, label):
    jj = np.linspace(-13.5, 14.5, 400)
    ax.plot(jj, glue_curve(key[3:6] if key.startswith("cgkin") else "el", jj), "-", color="0.6", lw=1.2, label=r"ED, $N_s\leq20$")
    pa = cmp[pubkey]; ax.plot(pa[:, 0], pa[:, 1], "o", mfc="none", mec="0.2", ms=4, mew=1.0, label=r"published TN ($N_s{=}30$)")
    rows = [r for r in R[key]["rows"] if r["E1"] is not None]
    j = np.array([r["j"] for r in rows]); e = np.array([r["E1"] for r in rows])
    ax.plot(np.r_[-j[::-1], j], np.r_[e[::-1], e], "s", color=BLUE, ms=4.5, label=r"$W^{00}_{\rm vac}$ ridge ($N_s{=}30$, $t\leq6$)")
    ax.set_xlabel(r"$k \times (L/2\pi)$"); ax.text(0.04, 0.05, label, transform=ax.transAxes, fontsize=8)
panel(axs[0], "cgkel_ns30", "panel_a", r"$\varepsilon{=}1.0$"); axs[0].set_ylabel(r"$\Delta E$"); axs[0].legend(fontsize=6, loc="upper center")
axs[0].text(0.96, 0.05, "(a)", transform=axs[0].transAxes, fontsize=9, ha="right")
if has_b:
    panel(axs[1], KEY_B, "panel_c", r"$\varepsilon{=}0.2$"); axs[1].text(0.96, 0.05, "(b)", transform=axs[1].transAxes, fontsize=9, ha="right")
aC = axs[-1]
t = true["t"].astype(float); tt = np.linspace(0, 20, 300)
aC.plot(t, true["R_ideal"], "s", color="0.25", ms=3.8, label=r"published collision ($N_s{=}26$)")
aC.plot(tt, dhk_curve(tt), "-", color="0.6", lw=1.2, label="dephasing, ED dispersion")
# dephasing with the tensor dispersion (same model as rt_wvac_ridge.py)
rows = [r for r in R["dhk_ns26"]["rows"] if r["E1"] is not None]
ks_t = np.array([r["q1"] for r in rows]); Es_t = np.array([r["E1"] for r in rows])
A = np.stack([np.ones_like(ks_t), np.cos(ks_t), np.cos(2*ks_t)], 1); c, *_ = np.linalg.lstsq(A, Es_t, rcond=None)
NP, SIG, KB = 13, 3*np.pi/13, 2*np.pi/13
k = np.pi/NP*np.arange(-NP, NP); k = k[(k >= -np.pi/2-1e-9) & (k < np.pi/2-1e-9)]
w1 = np.exp(-((k-KB)**2)/(2*SIG**2)); w2 = np.exp(-((k+KB)**2)/(2*SIG**2)); w1, w2 = w1/w1.sum(), w2/w2.sum()
Kf = np.abs((2*k+np.pi) % (2*np.pi) - np.pi); E = c[0] + c[1]*np.cos(Kf) + c[2]*np.cos(2*Kf)
Rt = np.abs(np.array([(w1*np.exp(-1j*E*x)).sum() for x in tt]) * np.array([(w2*np.exp(-1j*E*x)).sum() for x in tt]))**2
aC.plot(tt, Rt, "--", color=BLUE, lw=1.5, label=r"dephasing, $W^{00}_{\rm vac}$ dispersion")
aC.set_xlabel(r"time $t$"); aC.set_ylabel(r"$\mathcal{R}(t)$"); aC.set_xlim(0, 20); aC.set_ylim(0, 1.05)
aC.legend(fontsize=6, loc="lower left"); aC.text(0.96, 0.9, "(c)" if has_b else "(b)", transform=aC.transAxes, fontsize=9, ha="right")
fig.savefig("data/rt_wvac_fig10_candidate.pdf", dpi=200); print("wrote data/rt_wvac_fig10_candidate.pdf")
