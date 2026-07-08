"""Synthesis-error ladder figure at 101 qubits: ridge deviation vs rotation
tolerance, stochastic (seed-averaged) vs deterministic rounding, spanning
the exact baseline to the Clifford endpoint.

  PYTHONPATH=. python scripts/ladder_figure.py
"""

import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from htensor import Z2Lattice, analysis
from scripts_helpers_ridge import onesided_ft

lat = Z2Lattice(50, pbc=True)
x = analysis.ring_fold((np.arange(50) - 24) / 2, lat.nx)
Q0 = np.arange(-0.6, 1.601, 0.04)


def ridge_err(f, W0, peak):
    d = np.load(f)
    G = d["corr"] - d["one_pt"] * complex(d["insert_1pt"])
    W = onesided_ft(d["times"], x, G, Q0, np.array([np.pi / 2]),
                    8 / 3, lat.nx / 6, 0.5, 0.5)[:, 0]
    return np.sqrt(np.mean(np.abs(W - W0) ** 2)) / peak


d0 = np.load("data/synthladder_exact.npz")
G0 = d0["corr"] - d0["one_pt"] * complex(d0["insert_1pt"])
W0 = onesided_ft(d0["times"], x, G0, Q0, np.array([np.pi / 2]),
                 8 / 3, lat.nx / 6, 0.5, 0.5)[:, 0]
peak = np.abs(W0).max()

denoms = [128, 64, 32, 16]
tpr = [3 * np.log2(2 * d / np.pi * np.pi) for d in denoms]  # ~ T/rotation
tpr = [3 * np.log2(1 / (np.pi / d / 2)) for d in denoms]
stoch_m, stoch_s, rnd = [], [], []
for d in denoms:
    sf = glob.glob(f"data/synthladder_stoc_pi{d}*.npz")
    e = [ridge_err(f, W0, peak) for f in sf]
    stoch_m.append(np.mean(e)); stoch_s.append(np.std(e) if len(e) > 1 else 0)
    rf = f"data/synthladder_roun_pi{d}_s0.npz"
    rnd.append(ridge_err(rf, W0, peak) if os.path.exists(rf) else np.nan)

fig, ax = plt.subplots(figsize=(3.4, 3.0), constrained_layout=True)
ax.errorbar(tpr, stoch_m, yerr=stoch_s, fmt="o-", color="C0", capsize=3,
            label="stochastic (seeds)")
ax.plot(tpr, rnd, "s--", color="C1", label="deterministic")
cliff = ridge_err("data/synthladder_clifford.npz", W0, peak)
ax.axhline(cliff, color="0.5", ls=":", lw=1)
ax.text(tpr[-1], cliff + 0.02, "Clifford", fontsize=7, color="0.4")
ax.set_xlabel("$T$ gates / rotation")
ax.set_ylabel(r"ridge deviation (rms/peak)")
ax.set_title("Synthesis tolerance, 101 qubits", fontsize=9.5)
ax.legend(fontsize=8)
ax.invert_xaxis()
fig.savefig("data/ladder_figure.pdf", dpi=200)
print("T/rot:", [round(t, 1) for t in tpr])
print("stochastic:", [round(s, 4) for s in stoch_m])
print("deterministic:", [round(r, 4) for r in rnd])
print(f"Clifford endpoint: {cliff:.3f}")
print("wrote data/ladder_figure.pdf")
