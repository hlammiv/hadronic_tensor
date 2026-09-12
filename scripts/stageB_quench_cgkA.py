"""Stage B(i): vacuum-quench two-point function of the translation-summed
hop interpolator at the CGK-A couplings, at their production size ns=30
(L=15) -- the real-time route to the smeared spectral density S(E) whose
line shape fig:factorization extracts from finite-volume levels.

C(t) = <0| O(t) O(0) |0>_conn,  O = sum_b hop_b
     = Ns * sum_b <hop_b(t) hop_c(0)>_conn      (translation invariance)

so one center insertion + all-bond probes (production W00 pattern).

  PYTHONPATH=. python scripts/stageB_quench_cgkA.py [ns] [tmax]
"""
import sys
import time

import numpy as np

from htensor import Z2Lattice, stateprep, backends
from htensor.currents import pauli_sum

M0, G2, ETA = 0.1, 0.4, 1.0            # CGK-A
NS = int(sys.argv[1]) if len(sys.argv) > 1 else 30
TMAX = float(sys.argv[2]) if len(sys.argv) > 2 else 20.0
TIMES = np.arange(0.0, TMAX + 1e-9, 0.5)
DT_TARGET = 0.25
TRUNC = 1e-8


def hop_op(lat, bond, eta=1.0):
    """Gauge-invariant hop bilinear on `bond` -- the (XX+YY) quadrature of
    bond_current; matches htensor.hamiltonian.hop_term conventions."""
    qa, qb = lat.site_qubit(bond), lat.site_qubit(bond + 1)
    ql = lat.link_qubit(bond)
    string = {q: "Z" for q in lat.seam_string_qubits()} \
        if lat.is_seam(bond) else {}
    return pauli_sum(lat.n_qubits, [
        ({qa: "X", qb: "X", ql: "Z", **string}, eta / 4),
        ({qa: "Y", qb: "Y", ql: "Z", **string}, eta / 4),
    ])


t0 = time.time()
lat = Z2Lattice(NS, pbc=True)
z = np.load("data/cgkA_vacuum_thetas.npz")
prep = stateprep.vacuum_ansatz(lat, z["thetas"])
print(f"[{time.time()-t0:6.0f}s] vacuum ansatz ({len(z['thetas'])} angles), "
      f"ns={NS}", flush=True)

bc = NS // 2                            # center bond
insert = hop_op(lat, bc, ETA)
probes = [hop_op(lat, b, ETA) for b in range(lat.n_links)]

rows = []
for t in TIMES:
    d = backends.hadamard_correlator_aer(
        lat, prep, insert, probes, M0, G2, ETA, [t], dt_target=DT_TARGET,
        method="matrix_product_state", mps_trunc=TRUNC,
        stationary_1pt=False if t == 0.0 else True)
    rows.append(d)
    print(f"[{time.time()-t0:6.0f}s] t={t:4.1f} done  "
          f"C(t, b=c) = {d.correlator[0, bc]:.6f}", flush=True)

corr = np.concatenate([d.correlator for d in rows], axis=0)
probe_1pt = np.concatenate([d.probe_expect for d in rows], axis=0)
np.savez(f"data/stageB_quench_cgkA_ns{NS}.npz", times=TIMES, corr=corr,
         probe_1pt=probe_1pt, insert_1pt=rows[0].insert_expect, ns=NS,
         bc=bc, m0=M0, g2=G2, eta=ETA, dt_target=DT_TARGET, trunc=TRUNC)
print(f"[{time.time()-t0:6.0f}s] saved data/stageB_quench_cgkA_ns{NS}.npz",
      flush=True)
