"""h(0) addendum for the qPDF campaign: connected <J0(center)> on the
K=1.2566 packet, matching the production m=0 convention.

  PYTHONPATH=. python scripts/qpdf_h0_addendum.py <m0> <g2> <eta> <out>
Run inside a htq_<tag> clone (uses its data/ packet + vacuum files).
"""
import sys
import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from htensor import Z2Lattice, stateprep, wavepacket, backends
from htensor import currents as cur

M0, G2, ETA = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
OUT = sys.argv[4]
NS, CENTER = 30, 14
lat = Z2Lattice(NS, pbc=True)
TH = np.load("data/cgkA_vacuum_thetas.npz")["thetas"]
z = np.load("data/wpCGKA_params_k1.26_L3.npz", allow_pickle=True)
params = wavepacket.params_from_vector(z["vec"], list(z["offsets"]),
                                       int(z["L"]))
op = cur.charge_density(lat, CENTER)
vals = {}
for state in ("wp", "vac"):
    prep = stateprep.vacuum_ansatz(lat, TH)
    if state == "wp":
        prep.compose(wavepacket.block_circuit(lat, CENTER, params),
                     inplace=True)
    mps, perm = backends.prepare_state_mps(lat, prep, CENTER * 2, cap=512,
                                           trunc=1e-10, max_threads=1,
                                           sim_opts={"mps_lapack": True})
    n_tot = lat.n_qubits + 1
    qc = QuantumCircuit(n_tot)
    qc.set_matrix_product_state(mps)
    qc.save_expectation_value(backends.permute_pauli(op, perm, n_tot),
                              list(range(n_tot)), label="J0c")
    sim = AerSimulator(method="matrix_product_state",
                       matrix_product_state_max_bond_dimension=512,
                       matrix_product_state_truncation_threshold=1e-8,
                       mps_lapack=True)
    vals[state] = complex(sim.run(qc).result().data()["J0c"])
    print(f"{state}: <J0(c)> = {vals[state]:.6f}", flush=True)
np.savez(OUT, h0=vals["wp"] - vals["vac"], wp=vals["wp"], vac=vals["vac"])
print(f"saved {OUT}: h0 = {vals['wp'] - vals['vac']:.6f}")
