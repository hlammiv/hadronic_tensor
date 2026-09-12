"""Compute the raw (dE, residue) spectrum for ONE volume of the CGK-A
factorization analysis and save it -- the expensive eigsh done remotely,
candidate selection/rho/|F|^2 done locally by factorization.py, which reads
data/fact_raw_ns{ns}.npz when present.

  OMP_NUM_THREADS=4 PYTHONPATH=. python scripts/fact_raw_volume.py 24 [k]
"""
import sys

import numpy as np
import scipy.sparse.linalg as spla

from htensor import Z2Lattice
from htensor import hamiltonian as ham
from htensor.hamiltonian import hop_term
from htensor.gaugefixed import PhysicalBasis

M0, G2, ETA = 0.1, 0.4, 1.0            # CGK-A
ns = int(sys.argv[1])
k = int(sys.argv[2]) if len(sys.argv) > 2 else 160

lat = Z2Lattice(ns, pbc=True)
basis = PhysicalBasis(lat)
sel = np.flatnonzero(basis.q == 0)
H = basis.matrix(ham.build_hamiltonian(lat, M0, G2, ETA), sub=sel).real
O = basis.matrix(sum(hop_term(lat, b, ETA) for b in range(lat.n_links)),
                 sub=sel).real
print(f"ns={ns}: Q=0 dim {H.shape[0]}, eigsh k={k}", flush=True)
k = min(k, H.shape[0] - 2)
w, v = spla.eigsh(H, k=k, which="SA")
o = np.argsort(w); w, v = w[o], v[:, o]
Ov = O @ v[:, 0]
res = np.abs(v.conj().T @ Ov) ** 2
np.savez(f"data/fact_raw_ns{ns}.npz", dE=w - w[0], res=res, ns=ns,
         m0=M0, g2=G2, eta=ETA, k=k)
print(f"saved data/fact_raw_ns{ns}.npz  (dE range "
      f"[{(w-w[0])[1]:.3f}, {(w-w[0])[-1]:.3f}])", flush=True)
