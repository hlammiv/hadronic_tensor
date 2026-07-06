"""Quasi-PDF bilinears on the 101-qubit boosted packet (task 9).

Measures C_R(z), C_I(z) = <O_R/I(center, z)> for z = -16..16 on
  (a) the boosted (k0 = 2pi/5) wavepacket state, and
  (b) the bare vacuum (connected subtraction),
via equal-time expectation values on the stored cap-512 MPS.  Also saves
<H> for both states as the usual on-circuit certificate.

  PYTHONPATH=. .venv/bin/python scripts/quasipdf_ns50.py [k0tag]
"""

import sys
import time

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

from htensor import Z2Lattice, stateprep, wavepacket, backends
from htensor import hamiltonian as ham
from htensor.quasipdf import wilson_bilinear

M0, G2, ETA = 0.7, 1.1, 1.3
NS, CENTER = 50, 24
K0TAG = sys.argv[1] if len(sys.argv) > 1 else "k1.26"
ZS = [z for z in range(-16, 17) if z != 0]
t0 = time.time()


def log(m):
    print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)


lat = Z2Lattice(NS, pbc=True)
TH = stateprep.optimize_vacuum(Z2Lattice(6, pbc=True), M0, G2, ETA,
                               n_layers=2, restarts=2)["thetas"]
z = np.load(f"data/wp10reg_params_{K0TAG}_L3.npz", allow_pickle=True)
params = wavepacket.params_from_vector(z["vec"], list(z["offsets"]),
                                       int(z["L"]))
H50 = ham.build_hamiltonian(lat, M0, G2, ETA)

ops = {}
for zz in ZS:
    ops[f"R{zz}"], ops[f"I{zz}"] = wilson_bilinear(lat, CENTER, zz)
ops["H"] = H50

out = {}
for state in ("wp", "vac"):
    prep = stateprep.vacuum_ansatz(lat, TH)
    if state == "wp":
        prep.compose(wavepacket.block_circuit(lat, CENTER, params),
                     inplace=True)
    mps, perm = backends.prepare_state_mps(lat, prep, CENTER * 2,
                                           cap=512, trunc=1e-10)
    log(f"{state}: stored MPS ready")
    n_tot = lat.n_qubits + 1
    qc = QuantumCircuit(n_tot)
    qc.set_matrix_product_state(mps)
    for name, op in ops.items():
        qc.save_expectation_value(backends.permute_pauli(op, perm, n_tot),
                                  list(range(n_tot)), label=name)
    sim = AerSimulator(method="matrix_product_state",
                       matrix_product_state_truncation_threshold=1e-10,
                       max_parallel_threads=4)
    data = sim.run(qc).result().data()
    for name in ops:
        out[f"{state}_{name}"] = float(np.real(data[name]))
    log(f"{state}: {len(ops)} expectation values done, "
        f"<H> = {out[f'{state}_H']:.4f}")

np.savez(f"data/quasipdf_ns50_{K0TAG}.npz",
         zs=np.array(ZS),
         wp_R=np.array([out[f"wp_R{zz}"] for zz in ZS]),
         wp_I=np.array([out[f"wp_I{zz}"] for zz in ZS]),
         vac_R=np.array([out[f"vac_R{zz}"] for zz in ZS]),
         vac_I=np.array([out[f"vac_I{zz}"] for zz in ZS]),
         wp_H=out["wp_H"], vac_H=out["vac_H"], center=CENTER, k0tag=K0TAG)
log(f"saved data/quasipdf_ns50_{K0TAG}.npz")
