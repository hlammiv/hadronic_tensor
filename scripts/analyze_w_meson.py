"""Assemble W^{00}(q0,q1) from a meson-wavepacket production run, check the
lattice continuity equation on the raw grids, and produce the figure.

  PYTHONPATH=. .venv/bin/python scripts/analyze_w_meson.py data/w_meson_ns50_k0.00_v3.npz
"""

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from htensor import Z2Lattice, analysis

OI = ["#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7"]
plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "stix",
    "font.size": 11, "axes.labelsize": 12,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
})

path = sys.argv[1]
d = np.load(path)
ns, vc, k0 = int(d["ns"]), int(d["center"]), float(d["k0"])
lat = Z2Lattice(ns, pbc=True)
times = d["times"]
nJ0 = ns  # probes: J0 at all sites, then J1 at all bonds

# connected-minus-connected: [C_wp - <J><J>_wp] - [C_vac - <J><J>_vac]
def connected(corr, one_pt, ins):
    return analysis.subtract(corr, None, one_pt, complex(ins))

g_wp = connected(d["corr_wp"], d["one_pt_wp"], d["insert_1pt_wp"])
g_vac = connected(d["corr_vac"], d["one_pt_vac"], d["insert_1pt_vac"])
G = g_wp - g_vac

# ---------------- continuity check on raw (unsubtracted) wp grids
c00 = d["corr_wp"][:, :nJ0]
c10 = d["corr_wp"][:, nJ0:]
dt = times[1] - times[0]
lhs = (c00[2:] - c00[:-2]) / (2 * dt)          # d/dt C^{00}(v, t)
div = c10 - np.roll(c10, 1, axis=1)            # J1_{v+1/2} - J1_{v-1/2}
rhs = -div[1:-1]
num = np.abs(lhs - rhs).max()
den = np.abs(lhs).max()
print(f"continuity check (raw data): max|dC00/dt + div C10| = {num:.4f}, "
      f"max|dC00/dt| = {den:.4f}, ratio = {num/den:.3f} "
      f"(expect ~O(dt^2 + Trotter) ~ few %)")

# ---------------- W^{00}
x0 = analysis.ring_fold((np.arange(ns) - vc) / 2, lat.nx)
grid = analysis.CorrelatorGrid(times, x0, G[:, :nJ0])
t_full, c_full = analysis.complete_time(grid)
q0 = np.arange(-1.0, 6.001, 0.04)
ks = np.arange(-(lat.nx // 2), lat.nx // 2 + 1)  # +-q1: boost asymmetry
q1 = 2 * np.pi * ks / lat.nx
W, spread = analysis.window_scan(t_full, x0, c_full, q0, q1,
                                 sigma_t=times[-1] / 3, sigma_x=lat.nx / 6)
print(f"W^00: max|Im|/max|Re| = {np.abs(W.imag).max()/np.abs(W.real).max():.4f}")
np.savez(path.replace(".npz", "_W.npz"), q0=q0, q1=q1, W=W, spread=spread, k0=k0)

# ---------------- figure
fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.4, 3.8),
                               constrained_layout=True,
                               gridspec_kw={"width_ratios": [1.05, 1]})
pm = axL.pcolormesh(q1, q0, W.real, cmap="Blues", shading="nearest",
                    vmin=0, vmax=np.percentile(W.real, 99.5), rasterized=True)
axL.set_xlabel(r"$q^1$")
axL.set_ylabel(r"$q^0$")
axL.set_title(rf"$W^{{00}}(q^0,q^1)$ meson packet, $k_0={k0:.2f}$, "
              rf"$N_s={ns}$ (101 qubits)", fontsize=10.5)
axL.grid(False)
fig.colorbar(pm, ax=axL, pad=0.02)

for i, j in enumerate([lat.nx // 2 + 2, lat.nx // 2 + 5,
                       lat.nx // 2 + 8, lat.nx // 2 + 12]):
    axR.fill_between(q0, W.real[:, j] - spread[:, j] / 2,
                     W.real[:, j] + spread[:, j] / 2,
                     color=OI[i], alpha=0.22, lw=0)
    axR.plot(q0, W.real[:, j], color=OI[i], lw=1.8,
             label=rf"$q^1={q1[j]:.2f}$")
axR.axhline(0, color="0.4", lw=0.7)
axR.set_xlabel(r"$q^0$")
axR.set_ylabel(r"$W^{00}(q^0,q^1)$")
axR.legend(fontsize=8.5, framealpha=0.9)
axR.set_title("cuts; band = window systematic", fontsize=10.5)
out = path.replace(".npz", "_W.pdf")
fig.savefig(out, dpi=200)
print("wrote", out)
