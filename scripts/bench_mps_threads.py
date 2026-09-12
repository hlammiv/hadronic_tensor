"""Benchmark Aer-MPS threading options on the real two-packet state.

  prep mode:  PYTHONPATH=. python scripts/bench_mps_threads.py prep <cap>
      builds the j=3 two-packet state at <cap>, pickles (mps, perm) to
      data/bench_mps_chi<cap>.pkl
  unit mode:  PYTHONPATH=. python scripts/bench_mps_threads.py unit <cap> <lapack:0|1> <threads>
      loads the pickle and times ONE unit of Trotter evolution (4 steps,
      dt=0.25) + one density save under the given options.

OMP_NUM_THREADS must be set in unit mode BEFORE qiskit_aer is imported,
so it is exported here from argv prior to any qiskit import.
"""
import os
import sys
import pickle
import time

MODE = sys.argv[1]
CAP = int(sys.argv[2])
if MODE == "unit":
    LAPACK = bool(int(sys.argv[3]))
    THREADS = int(sys.argv[4])
    os.environ["OMP_NUM_THREADS"] = str(THREADS)

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp

from htensor import Z2Lattice, stateprep, wavepacket, backends, trotter
from htensor.backends import permute_circuit, permute_pauli, _AER_BASIS

M0, G2, ETA = 0.1, 0.4, 1.0
NS = 30
PKL = f"data/bench_mps_chi{CAP}.pkl"

lat = Z2Lattice(NS, pbc=True)

if MODE == "prep":
    zv = np.load("data/cgkA_vacuum_thetas.npz")
    prep = stateprep.vacuum_ansatz(lat, zv["thetas"])
    for ktag, center in (("1.26", 8), ("-1.26", 22)):
        z = np.load(f"data/wpCGKA_params_k{ktag}_L3.npz")
        params = wavepacket.params_from_vector(z["vec"], list(z["offsets"]),
                                               int(z["L"]))
        prep.compose(wavepacket.block_circuit(lat, center, params),
                     inplace=True)
    t0 = time.time()
    mps, perm = backends.prepare_state_mps(lat, prep, NS, cap=CAP,
                                           trunc=1e-10)
    with open(PKL, "wb") as f:
        pickle.dump({"mps": mps, "perm": perm}, f)
    print(f"prep cap={CAP}: {time.time()-t0:.0f}s -> {PKL}", flush=True)
    sys.exit(0)

with open(PKL, "rb") as f:
    d = pickle.load(f)
mps, perm = d["mps"], d["perm"]
n_tot = lat.n_qubits + 1

unit = trotter.trotter_circuit(lat, M0, G2, ETA, 1.0, 4)
unit_t = transpile(permute_circuit(unit, perm, n_tot),
                   basis_gates=_AER_BASIS, optimization_level=1)
zop = permute_pauli(
    SparsePauliOp.from_sparse_list([("Z", [lat.site_qubit(NS // 2)], 1.0)],
                                   num_qubits=lat.n_qubits), perm, n_tot)

from qiskit_aer import AerSimulator
opts = dict(method="matrix_product_state",
            matrix_product_state_max_bond_dimension=CAP,
            matrix_product_state_truncation_threshold=1e-8,
            max_parallel_threads=THREADS)
sim = AerSimulator(**opts)
if LAPACK:
    try:
        sim.set_options(mps_lapack=True)
    except Exception as e:
        print(f"mps_lapack unavailable: {e}", flush=True)
        sys.exit(1)

qc = QuantumCircuit(n_tot)
qc.set_matrix_product_state(mps)
qc.compose(unit_t, inplace=True)
qc.save_expectation_value(zop, list(range(n_tot)), label="z")
t0 = time.time()
res = sim.run(qc).result().data()
print(f"unit cap={CAP} lapack={int(LAPACK)} threads={THREADS}: "
      f"{time.time()-t0:.1f}s  <Z>={np.real(res['z']):+.6f}", flush=True)
