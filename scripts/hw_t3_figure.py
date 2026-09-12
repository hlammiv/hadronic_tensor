"""Tier-3 hardware figure: C(t,x) on ibm_kingston vs raw / ideal / truth.

Part a (t=0.5) for now; grows to a multi-slice figure as parts b/c land.

  PYTHONPATH=. .venv/bin/python scripts/hw_t3_figure.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paper_style import *  # usetex + OI palette + dev()
import numpy as np

K0TAG = "k1.26"
NS, CENTER = 50, 24
JOBS = [("data/hw/job_tier3k_T05MERGED.npz", 0.5, "277k"),
        ("data/hw/job_tier3i_T10RZZ.npz", 1.0, "20k"),
        ("data/hw/job_tier3h_T15RZZ.npz", 1.5, "70k"),
        ("data/hw/job_tier3f_T20RZZ.npz", 2.0, "140k"),
        ("data/hw/job_tier3g_T30RZZ.npz", 3.0, "70k")]
idl = np.load(f"data/hw_t3_ideal50_{K0TAG}.npz")
tru = np.load(f"data/w_meson_ns50_{K0TAG}_v3.npz")
x = np.arange(NS) - CENTER

fig, axes = plt.subplots(1, len(JOBS), figsize=(3.2 * len(JOBS), 3.2),
                         constrained_layout=True, sharey=True)
for a, (path, T, sh) in zip(np.atleast_1d(axes), JOBS):
    d = np.load(path)
    C_cal, C_err = d["C_cal"][0].real, d["C_err"][0]
    C_raw = d["C"][0].real
    kap = d["kappa_v"][0]
    id_a, c_a, id_b = float(d["id_a"]), float(d["c_a"]), d["id_b"]
    ti = int(np.argmin(np.abs(tru["times"] - T)))
    C_tru = tru["corr_wp"][ti, :NS].real
    row = int(np.argmin(np.abs(idl["times"] - T)))
    sxi = np.array([idl[f"XB_{v}"][row] for v in range(NS)]) \
        - id_b * idl["X"][row]
    syi = np.array([idl[f"YB_{v}"][row] for v in range(NS)]) \
        - id_b * idl["Y"][row]
    bi = np.array([idl[f"B_{v}"][row] for v in range(NS)])
    A0i = idl[f"B_{CENTER}"][0]
    C_idl = (c_a * (sxi + 1j * syi) + id_a * (bi - id_b)
             + id_b * (A0i - id_a) + id_a * id_b).real

    a.plot(x, C_tru, "k-", lw=1.3, zorder=3, label=r"exact (MPS, $dt\to0$)")
    a.plot(x, C_idl, "--", color="#0072B2", lw=1.1, zorder=4,
           label=r"ideal circuit ($dt=0.5$)")
    a.plot(x, C_raw, ":", color="0.55", lw=0.8, zorder=1)
    a.plot(x, C_raw, "x", color="0.45", ms=3.5, zorder=2,
           label="raw device (uncalibrated)")
    # mirror damping too small to invert -> site omitted entirely: the
    # calibrated value there is raw/kappa with kappa ~ 0.01, i.e. noise
    # amplified ~100x (error bars 4-10x the axis range at t=2).  Plotting
    # those points as scattered circles only misleads the eye.
    ok = kap > 0.05
    a.errorbar(x[ok], C_cal[ok], yerr=C_err[ok], fmt="o", color="#009E73",
               ms=3.4, lw=0, elinewidth=0.9, capsize=1.6, zorder=5,
               label=dev("kingston") + " (mirror-cal.)")
    a.text(0.98, 0.96,
           rf"{int((~ok).sum())}/{NS} unrecoverable ($\kappa_v<0.05$)",
           color="C3", fontsize=6.5, ha="right", va="top",
           transform=a.transAxes)
    a.axhline(0.25, color="0.7", lw=0.7, ls="-.", zorder=0)
    a.set_xlabel(r"$x - x_{\rm packet}$  (site)")
    a.set_xlim(-25, 25)
    a.set_title(rf"$t={T}$  ({sh} shots, "
                rf"$\kappa_{{\rm c}}={kap[CENTER]:.2f}$)", fontsize=9)
axes[0].text(-23.5, 0.262, "noise fixed point", color="0.5", fontsize=6.5)
axes[0].set_ylabel(r"Re $C(t,x)$")
axes[0].set_ylim(-0.17, 0.62)
axes[0].legend(fontsize=6.4, loc="upper left", handlelength=1.6)
fig.suptitle(r"$W^{00}$ integrand on 101 qubits (" + dev("kingston") + ")",
             fontsize=10)
fig.savefig("data/hw_t3_slices.pdf", dpi=200)
fig.savefig("data/hw_t3_slices.png", dpi=200)
print("wrote data/hw_t3_slices.{pdf,png}")
