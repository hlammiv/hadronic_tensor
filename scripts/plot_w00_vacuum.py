"""Figure: vacuum-polarization W^{00}(q0, q1) at ns=50 (101 qubits).

Left: heatmap of W over (q1, q0) with the small-volume ED meson dispersion
overlaid.  Right: W(q0) cuts at selected q1 with window-scan systematic bands.

  PYTHONPATH=. .venv/bin/python scripts/plot_w00_vacuum.py [data/w00_vac_ns50.npz]
"""

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from htensor import Z2Lattice, analysis, spectroscopy

# Okabe-Ito, fixed assignment order (legacy paper style)
OI = ["#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7", "#000000"]
plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "stix",
    "font.size": 11, "axes.labelsize": 12,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
})

path = sys.argv[1] if len(sys.argv) > 1 else "data/w00_vac_ns50.npz"
d = np.load(path)
ns, vc = int(d["ns"]), int(d["vc"])
lat = Z2Lattice(ns, pbc=True)
times = d["times"]

# ---- analysis chain: connected correlator -> time completion -> windowed FT
c_conn = analysis.subtract(d["corr"], None, d["probe_1pt"], complex(d["insert_1pt"]))
x = analysis.ring_fold((np.arange(ns) - vc) / 2, lat.nx)
grid = analysis.CorrelatorGrid(times, x, c_conn)
t_full, c_full = analysis.complete_time(grid)

q0 = np.arange(-0.5, 5.501, 0.04)
ks = np.arange(0, lat.nx // 2 + 1)
q1 = 2 * np.pi * ks / lat.nx
sigma_t, sigma_x = times[-1] / 3.0, lat.nx / 6.0
W, spread = analysis.window_scan(t_full, x, c_full, q0, q1, sigma_t, sigma_x)
Wr = W.real

# ---- small-volume ED dispersion for overlay (volume-converged by ns=8)
band = spectroscopy.meson_band(Z2Lattice(8, pbc=True), float(d["m0"]),
                               float(d["g2"]), float(d["eta"]))
k_ed = np.abs(band["k"])
e_ed = band["energy"]

fig, (axL, axR) = plt.subplots(
    1, 2, figsize=(9.2, 3.8), constrained_layout=True,
    gridspec_kw={"width_ratios": [1.05, 1]})

# ---- left: heatmap (sequential single hue), dispersion overlay
pm = axL.pcolormesh(q1, q0, Wr, cmap="Blues", shading="nearest",
                    vmin=0.0, vmax=np.percentile(Wr, 99.5), rasterized=True)
axL.plot(k_ed, e_ed, "o", ms=7, mfc="none", mew=1.8, color=OI[1],
         label=r"ED meson $E(k)$ ($N_s=8$)")
axL.set_xlabel(r"$q^1$")
axL.set_ylabel(r"$q^0$")
axL.set_title(rf"$W^{{00}}_{{\rm vac}}(q^0,q^1)$,  $N_s={ns}$ (101 qubits, MPS)",
              fontsize=11)
axL.legend(loc="upper left", framealpha=0.9, fontsize=9)
axL.grid(False)
fig.colorbar(pm, ax=axL, pad=0.02, label=r"$W^{00}$")

# ---- right: cuts with window-systematic bands
cut_ks = [k for k in (0, len(ks) // 3, 2 * len(ks) // 3, len(ks) - 1)]
for i, j in enumerate(cut_ks):
    axR.fill_between(q0, Wr[:, j] - spread[:, j] / 2, Wr[:, j] + spread[:, j] / 2,
                     color=OI[i], alpha=0.22, lw=0)
    axR.plot(q0, Wr[:, j], color=OI[i], lw=1.8)
    ipk = np.argmax(Wr[:, j])
    axR.annotate(rf"$q^1={q1[j]:.2f}$", (q0[ipk], Wr[ipk, j]),
                 textcoords="offset points", xytext=(6, 4),
                 color=OI[i], fontsize=9)
axR.axhline(0, color="0.4", lw=0.7)
axR.set_xlabel(r"$q^0$")
axR.set_ylabel(r"$W^{00}_{\rm vac}(q^0, q^1)$")
axR.set_title("cuts, band = window-scan systematic", fontsize=11)

out = path.replace(".npz", ".pdf")
fig.savefig(out, dpi=200)
np.savez(path.replace(".npz", "_W.npz"), q0=q0, q1=q1, W=W, spread=spread,
         sigma_t=sigma_t, sigma_x=sigma_x)
print("wrote", out)
