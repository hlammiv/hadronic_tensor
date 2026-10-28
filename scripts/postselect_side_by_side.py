"""Side by side: the W^{mu nu} maps without post-selection and at Gauss windows
1 and 2, from the maps scripts/hw_w_tensor.py wrote for each window.

    PYTHONPATH=. python scripts/postselect_side_by_side.py \
        --base data/hw/runs/phoenix/2026-10-06/analysis/2026-10-28_postselect \
        --card prod --tag prod_t6 [--sub kapmin0.10] [--part odd]

Columns: MPS reference (matched window and mask of the unfiltered reduction),
then the hardware map at windows 0, 1, 2; rows: the components the map holds.
One colour scale per row.  Each hardware panel carries the matched-filter
amplitude A of its odd part on the q0 >= 0 wedge (htensor.asymmetry).
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
from htensor import asymmetry  # noqa: E402

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--base", default="data/hw/runs/phoenix/2026-10-06/analysis/2026-10-28_postselect")
p.add_argument("--tag", default="prod_t6", help="map file stem, e.g. prod_t6 or relA_t6")
p.add_argument("--sub", default="", help="subdirectory under w{W}, e.g. kapmin0.10")
p.add_argument("--part", default=None, choices=(None, "even", "odd"))
p.add_argument("--windows", type=int, nargs="+", default=(0, 1, 2))
p.add_argument("--out", default=None)
a = p.parse_args()

maps = {w: np.load(os.path.join(a.base, f"w{w}", a.sub, a.tag + ".npz"), allow_pickle=True) for w in a.windows}
ref = maps[a.windows[0]]
q0, q1 = ref["q0"], ref["q1"]
comps = [c for c in ("00", "10", "01", "11") if f"W_hw_{c}" in ref.files]
key = (lambda kind, c: f"W_{kind}_{a.part}_{c}") if a.part else (lambda kind, c: f"W_{kind}_{c}")
cols = [("MPS reference", lambda c: ref[key("ref", c)], None)] + \
       [(f"window {w}" if w else "no post-selection", (lambda c, w=w: maps[w][key("hw", c)]), w) for w in a.windows]

fig, axes = plt.subplots(len(comps), len(cols), figsize=(2.6 * len(cols) + 0.8, 2.5 * len(comps) + 0.9),
                         layout="compressed", squeeze=False, sharex=True, sharey=True)
for i, c in enumerate(comps):
    vmax = max(max(float(np.abs(f(c)).max()) for _, f, _ in cols), 1e-15)
    for j, (label, f, w) in enumerate(cols):
        ax = axes[i, j]
        pm = ax.pcolormesh(q0, q1, f(c).T, cmap=OI_DIV, vmin=-vmax, vmax=vmax, shading="nearest", rasterized=True)
        ax.grid(False)
        ax.set_aspect("equal")
        ax.set_xlim(*ref["q0_plot_limits"])
        ax.set_ylim(-np.pi, np.pi)
        ax.set_yticks([-np.pi, 0, np.pi], [r"$-\pi$", "$0$", r"$\pi$"])
        sym = rf"$W^{{{c}}}$" + (rf"$_{{\mathrm{{{a.part}}}}}$" if a.part else "")
        title = sym + "  " + label
        if w is not None:
            A = asymmetry.amplitude(maps[w][f"W_hw_{c}"], maps[w][f"W_ref_{c}"], q0)
            title += f"\n$A = {A:.2f}$"
        ax.set_title(title, fontsize=8)
        if i == len(comps) - 1:
            ax.set_xlabel(r"$q^0$")
        if j == 0:
            ax.set_ylabel(r"$q^1$")
    fig.colorbar(pm, ax=axes[i, :], shrink=0.9, pad=0.015, aspect=25)
label = str(ref["label"]).replace("ibm_phoenix", dev("phoenix"))
tmax = float(np.max(ref["times"]))
fig.suptitle(rf"{label}: Re-correlator tensor, $t\leq{tmax:g}$; Gauss post-selection window = checks each side of the insertion"
             + (f"; kappa floor {a.sub.replace('kapmin', '')}" if a.sub.startswith("kapmin") else ""), fontsize=9)
out = a.out or os.path.join(a.base, f"side_by_side_{a.tag}" + (f"_{a.sub}" if a.sub else "") + (f"_{a.part}" if a.part else ""))
for ext in ("png", "pdf"):
    fig.savefig(out + "." + ext, dpi=220)
print(out + ".png")
