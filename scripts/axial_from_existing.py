"""Task 8: axial-channel correlators from EXISTING production data.

For charge densities J^0(v) = (-1)^v/2 - Z_v/2 and the axial density
J_ax(v) = 1/2 - (-1)^v Z_v/2, the CONNECTED correlators involve only the
Z-Z piece, so exactly

    G_ax(v, c; t) = (-1)^{v+c} G_vec(v, c; t)

pointwise: the full axial-axial channel is a staggered sign flip of the
already-measured connected vector correlator (equivalently the q1 -> q1 +
2pi staggered shift).  No new circuits, no new inversion.

  PYTHONPATH=. .venv/bin/python scripts/axial_from_existing.py data/w_meson_ns50_k0.00_v3.npz
"""

import sys

import numpy as np

from htensor import Z2Lattice, analysis
from scripts_helpers_ridge import onesided_ft

path = sys.argv[1] if len(sys.argv) > 1 else "data/w_meson_ns50_k0.00_v3.npz"
d = np.load(path)
ns, vc = int(d["ns"]), int(d["center"])
lat = Z2Lattice(ns, pbc=True)
times = d["times"]


def conn(corr, one, ins):
    return analysis.subtract(corr, None, one, complex(ins))


G_vec = conn(d["corr_wp"], d["one_pt_wp"], d["insert_1pt_wp"])[:, :ns] \
    - conn(d["corr_vac"], d["one_pt_vac"], d["insert_1pt_vac"])[:, :ns]
sgn = (-1.0) ** np.arange(ns)
G_ax = sgn[None, :] * sgn[vc] * G_vec

x = analysis.ring_fold((np.arange(ns) - vc) / 2, lat.nx)
q0 = np.arange(-0.6, 5.501, 0.04)
q1 = 2 * np.pi * np.arange(-(lat.nx // 2), lat.nx // 2 + 1) / lat.nx
W_ax = onesided_ft(times, x, G_ax, q0, q1, 8 / 3, lat.nx / 6, 0.5, 0.5)
W_vec = onesided_ft(times, x, G_vec, q0, q1, 8 / 3, lat.nx / 6, 0.5, 0.5)
out = path.replace(".npz", "_axial.npz")
np.savez(out, q0=q0, q1=q1, W_ax=W_ax, W_vec=W_vec)

for tag, W in (("axial", W_ax), ("vector", W_vec)):
    i, j = np.unravel_index(np.argmax(np.abs(W)), W.shape)
    print(f"{tag}: max |W| = {np.abs(W).max():.4f} at (q0, q1) = "
          f"({q0[i]:.2f}, {q1[j]:.2f})")
print("wrote", out)
