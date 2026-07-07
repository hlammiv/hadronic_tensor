"""Task 13: synthesis-ladder analysis at 101 qubits (runs on whatever
ladder points exist; rerun as coarse-delta points land).

Per point: on-circuit <H> shift vs exact MPS baseline, and rms deviation
of the one-sided W^{00}(q^0) ridge at q^1 = pi/2 (same transform for all
configs), normalized to the exact ridge peak.  Stochastic points with
seed ensembles report mean +/- spread.

  PYTHONPATH=. .venv/bin/python scripts/ladder_analysis.py
"""

import glob
import re

import numpy as np

from htensor import Z2Lattice, analysis
from scripts_helpers_ridge import onesided_ft

NS, CENTER = 50, 24
lat = Z2Lattice(NS, pbc=True)
x = analysis.ring_fold((np.arange(NS) - CENTER) / 2, lat.nx)
Q0 = np.arange(-0.6, 1.601, 0.04)
Q1 = np.array([np.pi / 2])


def ridge(path):
    d = np.load(path)
    G = d["corr"] - d["one_pt"] * complex(d["insert_1pt"])
    W = onesided_ft(d["times"], x, G, Q0, Q1, 8 / 3, lat.nx / 6, 0.5, 0.5)
    return W[:, 0], float(d["H"]), d


W0, H0, _ = ridge("data/synthladder_exact.npz")
peak = np.abs(W0).max()
print(f"exact baseline: <H> = {H0:.4f}, ridge peak = {peak:.4f}\n")
print(f"{'point':<16s} {'T/rot':>6s} {'<H> shift':>10s} {'ridge rms/peak':>16s}")

groups = {}
for f in sorted(glob.glob("data/synthladder_*.npz")):
    m = re.match(r".*/synthladder_(stoc|roun)_pi(\d+)(?:_s(\d+))?\.npz", f)
    if not m:
        continue
    key = (m.group(1), int(m.group(2)))
    groups.setdefault(key, []).append(f)

for (mode, denom), files in sorted(groups.items(), key=lambda kv: -kv[0][1]):
    eps = np.pi / denom / 2
    tcount = 3 * np.log2(1 / eps)
    dh, dr = [], []
    for f in files:
        W, H, d = ridge(f)
        dh.append(H - H0)
        dr.append(float(np.sqrt(np.mean(np.abs(W - W0) ** 2)) / peak))
    lab = f"{mode} pi/{denom} (x{len(files)})"
    if len(files) > 1:
        print(f"{lab:<16s} {tcount:6.1f} {np.mean(dh):+8.4f}+-{np.std(dh):.4f}"
              f" {np.mean(dr):12.4f}+-{np.std(dr):.4f}")
    else:
        print(f"{lab:<16s} {tcount:6.1f} {dh[0]:+10.4f} {dr[0]:16.4f}")

np.savez("data/ladder_analysis.npz", q0=Q0, W_exact=W0, H_exact=H0)
print("\nsaved data/ladder_analysis.npz (exact ridge reference)")
