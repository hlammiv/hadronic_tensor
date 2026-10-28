"""The spatial-current map three ways: measured W^{1nu}, the W^{1nu} that
continuity derives from the measured charge map (no J1 probe used), and the
MPS reference, odd parts under q1 -> -q1 with the matched-filter amplitude
of each against the reference on the q0 >= 0 wedge.

    PYTHONPATH=. python scripts/ward_derived_map_figure.py <map.npz> [--nu 0] [--out PREFIX]

<map.npz> is a reduction written by scripts/hw_w_tensor.py; nu = 0 pairs
(00, 10), nu = 1 pairs (01, 11).  Masked J0 points are filled from the
reference before the derivative, as in scripts/ward_scale.py.
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_style import OI_DIV, dev  # noqa: E402
from htensor import asymmetry, tensor  # noqa: E402

DT, DX = 0.5, 0.5
p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("npz")
p.add_argument("--nu", type=int, default=0, choices=(0, 1))
p.add_argument("--out", default=None)
a = p.parse_args()

d = np.load(a.npz, allow_pickle=True)
c0, c1 = ("00", "10") if a.nu == 0 else ("01", "11")
q0, q1 = d["q0"], d["q1"]
times, x0, x1 = d[f"times_{c0}"], d[f"x_{c0}"], d[f"x_{c1}"]
sig_t, sig_x = float(d[f"sig_t_{c0}"]), float(d["sig_x"])
G0_ref = np.ascontiguousarray(d[f"G_ref_{c0}"].real)
G0_hw = np.where(d[f"mask_{c0}"], d[f"G_hw_{c0}"], G0_ref)
W_der = tensor.ward_derived_W1(times, x0, x1, G0_hw, q0, q1, sig_t, sig_x, DT, DX)
W_ctl = tensor.ward_derived_W1(times, x0, x1, G0_ref, q0, q1, sig_t, sig_x, DT, DX)
W_ref, W_hw = d[f"W_ref_{c1}"], d[f"W_hw_{c1}"]
odd = asymmetry.odd
# continuity divides by qhat(q1): the q1 = 0 column is the zero mode it does
# not fix, and the rows next to it amplify the J0 map's noise by q0 / qhat.
# The derived panels blank |q1| < Q1_SHOW and every amplitude is quoted on
# |q1| >= Q1_FIT, the cut scripts/ward_scale.py quotes.
Q1_SHOW, Q1_FIT = 0.5, 1.1
panels = [("MPS reference", W_ref, False), ("measured, J1 probe", W_hw, False),
          ("derived from the J0 map by continuity", W_der, True),
          ("derived from the reference J0 map (control)", W_ctl, True)]

fig, axes = plt.subplots(1, len(panels), figsize=(2.7 * len(panels) + 0.8, 3.3), layout="compressed",
                         sharex=True, sharey=True)
reg = asymmetry.region_mask(q0)
fit = np.abs(q1) >= Q1_FIT
vmax = max(float(np.abs(odd(W)).max()) for _, W, der in panels if not der)
for ax, (label, W, der) in zip(axes, panels):
    Wf = np.where(np.isfinite(W), W, 0.0)
    T = odd(W_ref)[reg][:, fit]
    A = float((T * odd(Wf)[reg][:, fit]).sum() / (T * T).sum())
    shown = np.where((np.abs(q1) >= Q1_SHOW)[None, :], odd(Wf), np.nan) if der else odd(Wf)
    pm = ax.pcolormesh(q0, q1, shown.T, cmap=OI_DIV, vmin=-vmax, vmax=vmax, shading="nearest", rasterized=True)
    ax.grid(False)
    ax.set_aspect("equal")
    ax.set_xlim(*d["q0_plot_limits"])
    ax.set_ylim(-np.pi, np.pi)
    ax.set_yticks([-np.pi, 0, np.pi], [r"$-\pi$", "$0$", r"$\pi$"])
    ax.set_xlabel(r"$q^0$")
    ax.set_title(rf"$W^{{{c1}}}_{{\mathrm{{odd}}}}$  {label}" + f"\n$A = {A:.2f}$ on $|q^1| \\geq {Q1_FIT}$", fontsize=8)
axes[0].set_ylabel(r"$q^1$")
fig.colorbar(pm, ax=axes, shrink=0.9, pad=0.015, aspect=25)
label = str(d["label"]).replace("ibm_phoenix", dev("phoenix"))
fig.suptitle(rf"{label}: odd part of the spatial-current map, $t\leq{float(np.max(times)):g}$; "
             rf"derived panels blank $|q^1| < {Q1_SHOW}$, where continuity amplifies the charge map's noise by $q^0/\hat q$", fontsize=9)
out = a.out or os.path.splitext(a.npz)[0] + f"_ward_derived_nu{a.nu}"
for ext in ("png", "pdf"):
    fig.savefig(out + "." + ext, dpi=220)
print(out + ".png")
