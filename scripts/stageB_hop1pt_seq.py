"""One-point time series <hop_b(t)> on the ansatz vacuum -- the per-t
disconnected piece for the sequential quench.  The ansatz vacuum is not
an exact eigenstate (F=0.9994), so <hop_b(t)> oscillates at the ~2e-3
level; the translation-summed connected C(t) needs the subtraction
<O_b(t)><O_c(0)> at EACH t (the frozen-t=0 convention leaves a real
background ~50% of the elastic-window signal -- found 2026-07-24).

Single sequential capped-chi pass, no ancilla, checkpointed per chunk.

  PYTHONPATH=. python scripts/stageB_hop1pt_seq.py <cap> [ns] [tmax]
"""
import os
import pickle
import sys
import time

import numpy as np
from qiskit import QuantumCircuit, transpile

from htensor import Z2Lattice, stateprep, backends, trotter
from htensor.backends import permute_circuit, permute_pauli, _AER_BASIS
from htensor.currents import pauli_sum

M0, G2, ETA = 0.1, 0.4, 1.0            # CGK-A
CAP = int(sys.argv[1]) if len(sys.argv) > 1 else 256
NS = int(sys.argv[2]) if len(sys.argv) > 2 else 40
TMAX = float(sys.argv[3]) if len(sys.argv) > 3 else 40.0
DT_SNAP = 0.5
STEPS = 2
TRUNC = 1e-8
OUT = f"data/hop1pt_ns{NS}_chi{CAP}.npz"
CKPT = OUT.replace(".npz", "_ckpt.pkl")


def hop_op(lat, bond, eta=1.0):
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
n_tot = lat.n_qubits + 1
probes = [hop_op(lat, b, ETA) for b in range(lat.n_links)]
nt = int(round(TMAX / DT_SNAP)) + 1
times = np.arange(nt) * DT_SNAP

zv = np.load("data/cgkA_vacuum_thetas.npz")
prep = stateprep.vacuum_ansatz(lat, zv["thetas"])
mps, perm = backends.prepare_state_mps(lat, prep, 0, cap=CAP, trunc=1e-10,
                                       max_threads=1,
                                       sim_opts={"mps_lapack": True})
print(f"[{time.time()-t0:6.0f}s] vacuum MPS prepared (ns={NS})", flush=True)

unit = trotter.trotter_circuit(lat, M0, G2, ETA, DT_SNAP, STEPS)
unit_t = transpile(permute_circuit(unit, perm, n_tot),
                   basis_gates=_AER_BASIS, optimization_level=1)
pops = [permute_pauli(p, perm, n_tot) for p in probes]

from qiskit_aer import AerSimulator
_KW = dict(method="matrix_product_state",
           matrix_product_state_max_bond_dimension=CAP,
           matrix_product_state_truncation_threshold=TRUNC,
           max_parallel_threads=1)
sim_fast = AerSimulator(**_KW, mps_lapack=True)
sim_safe = AerSimulator(**_KW)

start = 0
p1t = np.full((nt, len(probes)), np.nan, dtype=complex)
if os.path.exists(CKPT):
    with open(CKPT, "rb") as f:
        ck = pickle.load(f)
    mps, start = ck["mps"], ck["it"] + 1
    p1t[:start] = ck["p1t"][:start]
    print(f"[{time.time()-t0:6.0f}s] resumed at t={times[ck['it']]:.1f}",
          flush=True)

for it in range(start, nt):
    qc = QuantumCircuit(n_tot)
    qc.set_matrix_product_state(mps)
    if it:
        qc.compose(unit_t, inplace=True)
    for j, op in enumerate(pops):
        qc.save_expectation_value(op, list(range(n_tot)), label=f"p{j}")
    qc.save_matrix_product_state(label="mps")
    try:
        data = sim_fast.run(qc).result().data()
    except Exception as e:
        print(f"    lapack SVD failed ({type(e).__name__}); default SVD",
              flush=True)
        data = sim_safe.run(qc).result().data()
    mps = data["mps"]
    p1t[it] = [complex(data[f"p{j}"]) for j in range(len(probes))]
    np.savez(OUT, times=times, p1t=p1t, ns=NS, cap=CAP, m0=M0, g2=G2,
             eta=ETA, dt_snap=DT_SNAP, done_units=it)
    with open(CKPT + ".tmp", "wb") as f:
        pickle.dump({"mps": mps, "it": it, "p1t": p1t}, f)
    os.replace(CKPT + ".tmp", CKPT)
    print(f"[{time.time()-t0:6.0f}s] t={times[it]:5.1f} done  "
          f"mean Re<hop> = {p1t[it].real.mean():+.6f}", flush=True)
print(f"[{time.time()-t0:6.0f}s] saved {OUT}", flush=True)
