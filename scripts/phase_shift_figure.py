"""delta(p) collapse figure across six volumes (replaces the phase-shift
prose table).  PYTHONPATH=. python scripts/phase_shift_figure.py"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

d = np.load("data/phase_shifts_6vol.npz")
rows = d["rows"]                     # ns, E2, p, n, delta
M = float(d["M"])
cmap = {8: "C0", 10: "C1", 12: "C2", 14: "C3", 16: "C4", 18: "C5", 20: "C6"}
fig, ax = plt.subplots(figsize=(3.3, 3.0), constrained_layout=True)
for ns in sorted(set(int(r[0]) for r in rows)):
    m = rows[:, 0] == ns
    ax.plot(rows[m, 2], rows[m, 4], "o", color=cmap[ns], ms=6,
            label=f"$N_x={ns//2}$")
ax.axhline(0, color="0.7", lw=0.6)
ax.axhline(np.pi / 2, color="0.7", lw=0.6, ls=":")
ax.text(0.05, np.pi / 2 + 0.03, r"$\pi/2$", fontsize=7, color="0.4")
ax.set_xlabel("relative momentum $p$")
ax.set_ylabel(r"elastic phase shift $\delta(p)$")
ax.legend(fontsize=6.5, ncol=2, loc="upper right")
fig.savefig("data/phase_shift_collapse.pdf", dpi=200)
print("wrote data/phase_shift_collapse.pdf  (%d points, %d volumes)"
      % (len(rows), len(set(rows[:, 0]))))
