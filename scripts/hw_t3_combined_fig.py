"""Single-figure candidate for the tier-3 hardware section: C(t,x) slices
(top row) + assembled W00 (bottom row).  Two-column PRD figure*.

  PYTHONPATH=. .venv/bin/python scripts/hw_t3_combined_fig.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

K0TAG = "k1.26"
NS, CENTER = 50, 24
JOBS = [("data/hw/job_tier3a_d9d3emkinv1c73aoguc0.npz", 0.5, "50k"),
        ("data/hw/job_tier3b_d9d45acinv1c73aohsj0.npz", 1.0, "150k"),
        ("data/hw/job_tier3d_d9d518sjeosc73fh2prg.npz", 2.0, "120k")]
idl = np.load(f"data/hw_t3_ideal50_{K0TAG}.npz")
tru = np.load(f"data/w_meson_ns50_{K0TAG}_v3.npz")
w = np.load("data/hw_w00_coarse_t2.npz")
x = np.arange(NS) - CENTER

fig = plt.figure(figsize=(7.0, 5.6), constrained_layout=True)
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.15])

for i, (path, T, sh) in enumerate(JOBS):
    a = fig.add_subplot(gs[0, i])
    d = np.load(path)
    C_cal, C_err = d["C_cal"][0].real, d["C_err"][0]
    C_raw = d["C"][0].real
    kap = d["kappa_v"][0]
    id_a, c_a, id_b = float(d["id_a"]), float(d["c_a"]), d["id_b"]
    ti = int(np.argmin(np.abs(tru["times"] - T)))
    row = int(np.argmin(np.abs(idl["times"] - T)))
    sxi = np.array([idl[f"XB_{v}"][row] for v in range(NS)]) \
        - id_b * idl["X"][row]
    syi = np.array([idl[f"YB_{v}"][row] for v in range(NS)]) \
        - id_b * idl["Y"][row]
    bi = np.array([idl[f"B_{v}"][row] for v in range(NS)])
    C_idl = (c_a * (sxi + 1j * syi) + id_a * (bi - id_b)
             + id_b * (idl["B_24"][0] - id_a) + id_a * id_b).real
    a.plot(x, tru["corr_wp"][ti, :NS].real, "k-", lw=1.1, zorder=3)
    a.plot(x, C_idl, "--", color="C0", lw=1.0, zorder=4)
    a.plot(x, C_raw, "x", color="0.55", ms=2.8, zorder=1)
    ok = kap > 0.05
    a.errorbar(x[ok], C_cal[ok], yerr=C_err[ok], fmt="o", color="C3",
               ms=2.9, lw=0, elinewidth=0.8, capsize=1.3, zorder=5)
    a.plot(x[~ok], C_cal[~ok], "o", mfc="none", mec="C3", ms=2.9,
           mew=0.7, zorder=5)
    a.text(0.02, 0.02, f"{int((~ok).sum())} masked", color="C3",
           fontsize=6, transform=a.transAxes)
    a.set_xlim(-25, 25)
    a.set_ylim(-0.17, 0.60)
    a.set_title(rf"$t={T}$ ({sh}, $\kappa_c={kap[CENTER]:.2f}$)",
                fontsize=8.5)
    a.set_xlabel(r"$x - x_{\rm packet}$", fontsize=8)
    if i == 0:
        a.set_ylabel(r"Re $C(t,x)$")
        a.plot([], [], "k-", lw=1.1, label="exact")
        a.plot([], [], "--", color="C0", lw=1.0, label=r"ideal ($dt{=}0.5$)")
        a.plot([], [], "x", color="0.55", ms=2.8, label="raw")
        a.plot([], [], "o", color="C3", ms=2.9, label="mirror-cal.")
        a.legend(fontsize=6, loc="upper left", handlelength=1.3)
    else:
        a.set_yticklabels([])

q0, q1 = w["q0"], w["q1"]
vmax = max(np.abs(w["W_ref"]).max(), np.abs(w["W_hw"]).max())
for i, (Wp, ttl) in enumerate([(w["W_hw"], "ibm_kingston"),
                               (w["W_ref"], "MPS, same window")]):
    ax = fig.add_subplot(gs[1, i])
    pm = ax.pcolormesh(q1, q0, Wp, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                       shading="nearest")
    ax.set_xlabel(r"$q^1$")
    ax.set_title(ttl, fontsize=8.5)
    if i == 0:
        ax.set_ylabel(r"$q^0$")
    else:
        ax.set_yticklabels([])
fig.colorbar(pm, ax=fig.axes[3:5], shrink=0.8, pad=0.015,
             label=r"$W^{00}$")
axc = fig.add_subplot(gs[1, 2])
for j, col in zip((21, 17, 15), ("C1", "C2", "C0")):
    tot = w["W_err"] + w["band"][:, j] / 2
    axc.fill_between(q0, w["W_hw"][:, j] - tot, w["W_hw"][:, j] + tot,
                     color=col, alpha=0.18, lw=0)
    axc.plot(q0, w["W_hw"][:, j], color=col, lw=1.2,
             label=rf"$q^1={q1[j]:.2f}$")
    axc.plot(q0, w["W_ref"][:, j], color=col, lw=0.9, ls="--")
axc.axhline(0, color="0.6", lw=0.6)
axc.set_xlabel(r"$q^0$")
axc.set_title("cuts (dashed: MPS)", fontsize=8.5)
axc.legend(fontsize=6.2, loc="upper right")
fig.suptitle(r"$W^{00}$ from ibm_kingston: integrand slices (top), "
             rf"assembled tensor, $t\leq2$, $\delta q^0\sim1.6$ (bottom)",
             fontsize=10)
fig.savefig("data/hw_t3_combined.pdf", dpi=200)
fig.savefig("data/hw_t3_combined.png", dpi=200)
print("wrote data/hw_t3_combined.{pdf,png}")
