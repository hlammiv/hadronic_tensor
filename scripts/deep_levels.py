"""Physical-sector spectrum + T2 momentum phases -> data/deep_levels_ns{N}.npz

Replaces the earlier ad-hoc heredoc runs (whose ns=14 relaunch died without
a log line).  Usage:

  PYTHONPATH=. .venv/bin/python scripts/deep_levels.py <ns> <k> [ncv]

RAM: matrix-free Lanczos holds ~ncv complex vectors of 2^(2 ns) amplitudes
(ns=14: 4 GiB each -> ncv=16 is ~64 GiB).  Check free RAM before launching.
"""

import sys
import time

import numpy as np

from htensor import Z2Lattice, exact
from htensor.spectroscopy import _t2_phases

M0, G2, ETA = 0.7, 1.1, 1.3
t0 = time.time()


def log(m):
    print(f"[{time.time()-t0:7.0f}s] {m}", flush=True)


ns = int(sys.argv[1])
k = int(sys.argv[2])
ncv = int(sys.argv[3]) if len(sys.argv) > 3 else None
lat = Z2Lattice(ns, pbc=True)
log(f"ns={ns} ({lat.n_qubits} qubits, dim 2^{lat.n_qubits}), k={k}, ncv={ncv}")

energies, vecs = exact.lowest_physical_states(lat, M0, G2, ETA, k=k,
                                              matrix_free=True, ncv=ncv)
log(f"eigsh done: E = {np.round(energies, 4)}")

states = [np.ascontiguousarray(vecs[:, i]) for i in range(vecs.shape[1])]
del vecs
_, phases = _t2_phases(states, energies, lat)
gaps = energies - energies[0]
np.savez(f"data/deep_levels_ns{ns}.npz", gaps=gaps, phases=np.asarray(phases),
         energies=energies, m0=M0, g2=G2, eta=ETA)
log(f"saved data/deep_levels_ns{ns}.npz")
for g, p in zip(gaps, phases):
    log(f"  gap {g:8.4f}   T2 phase {p:+8.4f}")
log("done")
