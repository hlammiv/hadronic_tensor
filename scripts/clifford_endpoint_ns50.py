"""Clifford endpoint of the synthesis ladder at 101 qubits (delta = pi/2).

At delta = pi/2 every rotation snaps to a Clifford angle, so the whole
pipeline is a stabilizer circuit -- volume-law entangled (hence MPS-
intractable, the entanglement wall that stalls the intermediate coarse-
delta points) but exactly and cheaply simulable in the stabilizer
formalism.  This is the physics-destroyed endpoint Henry asked for
("all the way up to a straight Clifford simulation").

Produces data/synthladder_clifford.npz in the ladder format so
scripts/ladder_analysis.py picks it up alongside the exact + snapped
points.

  PYTHONPATH=. .venv/bin/python scripts/clifford_endpoint_ns50.py
"""

import time

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

from htensor import Z2Lattice, stateprep, wavepacket, backends, synthesis, clifford
from htensor import currents as cur
from htensor import hamiltonian as ham

M0, G2, ETA = 0.7, 1.1, 1.3
NS, CENTER = 50, 24
TIMES = np.arange(0.0, 6.01, 0.5)
BASIS = ["cx", "cz", "rz", "sx", "x", "h", "ry"]
t0 = time.time()


def log(m):
    print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)


TH = stateprep.optimize_vacuum(Z2Lattice(6, pbc=True), M0, G2, ETA,
                               n_layers=2, restarts=2)["thetas"]
z = np.load("data/wp10reg_params_k0.00_L3.npz", allow_pickle=True)
params = wavepacket.params_from_vector(z["vec"], list(z["offsets"]), int(z["L"]))
lat = Z2Lattice(NS, pbc=True)
prep = stateprep.vacuum_ansatz(lat, TH)
prep.compose(wavepacket.block_circuit(lat, CENTER, params), inplace=True)
probes = [cur.charge_density(lat, v) for v in range(NS)]
insert = cur.charge_density(lat, CENTER)
H50 = ham.build_hamiltonian(lat, M0, G2, ETA)


def cliffordize(c):
    return clifford.to_clifford_circuit(synthesis.snap_angles(c, np.pi / 2,
                                                              "round", 0))


# <H> on the Clifford-snapped prepared state
pt = transpile(prep, basis_gates=BASIS, optimization_level=1)
qc = cliffordize(pt)
qc.save_expectation_value(H50, list(range(lat.n_qubits)), label="H")
sim = AerSimulator(method="stabilizer")
e = float(np.real(sim.run(qc).result().data()["H"]))
log(f"Clifford <H> = {e:.4f} (exact vacuum+packet ~ -49.4)")

rows = []
for t in TIMES:
    d = backends.hadamard_correlator_aer(
        lat, prep, insert, probes, M0, G2, ETA, [t], dt_target=0.5,
        method="stabilizer", circuit_transform=cliffordize, fold=False)
    rows.append(d)
    log(f"t={t:.1f} done")
corr = np.concatenate([r.correlator for r in rows], axis=0)
one = np.concatenate([r.probe_expect for r in rows], axis=0)
np.savez("data/synthladder_clifford.npz", times=TIMES, corr=corr, one_pt=one,
         insert_1pt=rows[0].insert_expect, H=e, delta=np.pi / 2,
         mode="clifford", seed=0)
log("saved data/synthladder_clifford.npz")
