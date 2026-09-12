"""All four components W^{mu nu}(q0, q1) from the production MPS grids, with
the lattice Ward / symmetry residual table and a paper-style figure.

  PYTHONPATH=. .venv/bin/python scripts/analyze_w_tensor.py \
      data/w_meson_ns50_k1.26_v3.npz data/w_meson_j1_ns50_k1.26.npz

Writes data/w_tensor_ns{ns}_k{k0:.2f}_W.npz (W_00 .. W_11, spread_*, q0, q1)
and data/w_tensor_ns{ns}_k{k0:.2f}.pdf.  Conventions: htensor/tensor.py.
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_style import OI  # noqa: E402
from htensor import tensor  # noqa: E402

path_j0, path_j1 = sys.argv[1], sys.argv[2]
out_tag = sys.argv[3] if len(sys.argv) > 3 else None

g = tensor.load_mps_components(path_j0, path_j1)
lat, k0 = g.lat, g.k0
tag = out_tag or f"data/w_tensor_ns{lat.ns}_k{k0:.2f}"

q0 = np.arange(-1.0, 6.001, 0.04)
ks = np.arange(-(lat.nx // 2), lat.nx // 2 + 1)
q1 = 2 * np.pi * ks / lat.nx
W = tensor.assemble_W(g, q0, q1)

print(f"# {path_j0} + {path_j1}: ns={lat.ns} k0={k0:.4f} t_max={g.times[-1]}")
print("continuity, integral form (raw wp grids) :", tensor.continuity_residual_xt(g))
print("continuity, integral form (connected)    :", tensor.continuity_residual_xt(g, use_raw=False))
print("Ward, q-space q0 W^{0nu} vs qhat W^{1nu} :", tensor.ward_residual_q(W, q0, q1))
print("insertion-side Ward (eigenstate only)    :", tensor.insertion_ward_residual(W, q0, q1))
if abs(k0) < 1e-9:
    print("parity about insertion (rest; 00/10 approx):", tensor.parity_residuals(g))
for comp in tensor.COMPONENTS:
    w, s = W[comp]
    print(f"W^{comp}: max|W| = {np.abs(w).max():.4f}  window spread/max = "
          f"{s.max()/np.abs(w).max():.3f}  C(0,0) = {g.g[comp][0][np.argmin(np.abs(g.x[comp]))]:.4f}")

np.savez(tag + "_W.npz", q0=q0, q1=q1, k0=k0, ns=lat.ns,
         **{f"W_{c}": W[c][0] for c in W}, **{f"spread_{c}": W[c][1] for c in W})

# ---------------------------------------------------------------- figure
fig = plt.figure(figsize=(11.2, 6.6), constrained_layout=True)
gs = fig.add_gridspec(2, 4, height_ratios=[1.15, 1])
labels = {"00": r"$W^{00}$", "10": r"$W^{10}$", "01": r"$W^{01}$", "11": r"$W^{11}$"}
for i, comp in enumerate(tensor.COMPONENTS):
    ax = fig.add_subplot(gs[0, i])
    w = W[comp][0]
    if comp == "00":
        pm = ax.pcolormesh(q1, q0, w, cmap="Blues", shading="nearest", vmin=0,
                           vmax=np.percentile(w, 99.5), rasterized=True)
    else:
        v = np.percentile(np.abs(w), 99.5)
        pm = ax.pcolormesh(q1, q0, w, cmap="RdBu_r", shading="nearest", vmin=-v, vmax=v,
                           rasterized=True)
    ax.set_xlabel(r"$q^1$")
    if i == 0:
        ax.set_ylabel(r"$q^0$")
    ax.set_title(labels[comp] + rf"$(q^0,q^1)$, $k_0={k0:.2f}$", fontsize=10)
    ax.grid(False)
    fig.colorbar(pm, ax=ax, pad=0.02)

# bottom left: direct W^{11} vs Ward-converted (q0/qhat)^2 W^{00}
ax = fig.add_subplot(gs[1, 0:2])
conv = tensor.ward_converted_W11(W["00"][0], q0, q1)
for i, j in enumerate([lat.nx // 2 + 2, lat.nx // 2 + 5, lat.nx // 2 + 8]):
    ax.plot(q0, W["11"][0][:, j], color=OI[i], lw=1.8, label=rf"$q^1={q1[j]:.2f}$ direct")
    ax.plot(q0, conv[:, j], color=OI[i], lw=1.2, ls="--", label="Ward from $W^{00}$")
ax.axhline(0, color="0.4", lw=0.7)
ax.set_xlabel(r"$q^0$")
ax.set_ylabel(r"$W^{11}$")
ax.set_title(r"$W^{11}$: direct vs $(q^0/\hat q)^2 W^{00}$", fontsize=10)
ax.legend(fontsize=7.5, ncol=2, framealpha=0.9)

# bottom right: the two mixed components, cuts
ax = fig.add_subplot(gs[1, 2:4])
for i, j in enumerate([lat.nx // 2 + 2, lat.nx // 2 + 5, lat.nx // 2 + 8]):
    ax.plot(q0, W["10"][0][:, j], color=OI[i], lw=1.8, label=rf"$W^{{10}}$, $q^1={q1[j]:.2f}$")
    ax.plot(q0, W["01"][0][:, j], color=OI[i], lw=1.2, ls=":", label=r"$W^{01}$")
ax.axhline(0, color="0.4", lw=0.7)
ax.set_xlabel(r"$q^0$")
ax.set_ylabel(r"$W^{10},\,W^{01}$")
ax.set_title("mixed components (cuts)", fontsize=10)
ax.legend(fontsize=7.5, ncol=2, framealpha=0.9)
fig.savefig(tag + ".pdf", dpi=200)
print("wrote", tag + "_W.npz", tag + ".pdf")
