"""Sequential single-pass real-time vacuum quench (fast rewrite of
stageB_quench_cgkA.py): C(t) = <O(t) O(0)> for the translation-summed hop
interpolator at CGK-A, via ONE evolution per insertion Pauli term instead
of one per time slice.

The Hadamard-test insertion acts at t=0, before all evolution, so the
whole time series comes from a single capped-chi pass: prepare vacuum +
ancilla, apply H(anc) and the controlled insertion term, then evolve in
DT_SNAP chunks with <X_anc x hop_b> / <Y_anc x hop_b> saved on every bond
at each snapshot.  mps_lapack + default-SVD fallback + per-chunk disk
checkpoint/resume as in stageB_collision_density.py.  The two insertion
quadratures (terms 0,1) are independent -- run them as parallel jobs and
assemble  corr(t,b) = sum_a c_a (re_a + i im_a)  at analysis time.

  PYTHONPATH=. python scripts/stageB_quench_seq.py <term|all> <cap> [ns] [tmax]
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
from htensor.measure import (split_current, controlled_pauli,
                             _with_ancilla, _pauli_only)

M0, G2, ETA = 0.1, 0.4, 1.0            # CGK-A
TERM = sys.argv[1]                      # '0', '1', or 'all'
CAP = int(sys.argv[2]) if len(sys.argv) > 2 else 256
NS = int(sys.argv[3]) if len(sys.argv) > 3 else 40
TMAX = float(sys.argv[4]) if len(sys.argv) > 4 else 40.0
DT_SNAP = 0.5
STEPS = 2                               # Trotter steps per snapshot (0.25)
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
n_sys, n_tot, anc = lat.n_qubits, lat.n_qubits + 1, lat.n_qubits
bc = NS // 2                            # center bond insertion
insert = hop_op(lat, bc, ETA)
probes = [hop_op(lat, b, ETA) for b in range(lat.n_links)]
id_a, terms_a = split_current(insert)
assert abs(id_a) < 1e-12                # pure-Pauli insertion, no identity
terms = [int(TERM)] if TERM in ("0", "1") else [0, 1]
anc_site = min(terms_a[0][0])

zv = np.load("data/cgkA_vacuum_thetas.npz")
prep = stateprep.vacuum_ansatz(lat, zv["thetas"])
mps0, perm = backends.prepare_state_mps(lat, prep, anc_site, cap=CAP,
                                        trunc=1e-10, max_threads=1,
                                        sim_opts={"mps_lapack": True})
print(f"[{time.time()-t0:6.0f}s] vacuum MPS prepared (ns={NS}, cap={CAP})",
      flush=True)

nt = int(round(TMAX / DT_SNAP)) + 1
times = np.arange(nt) * DT_SNAP

unit = trotter.trotter_circuit(lat, M0, G2, ETA, DT_SNAP, STEPS)
unit_t = transpile(permute_circuit(unit, perm, n_tot),
                   basis_gates=_AER_BASIS, optimization_level=1)

obs = {}
for j, p in enumerate(probes):
    bp = _pauli_only(p)
    obs[f"re{j}"] = permute_pauli(_with_ancilla(bp, "X"), perm, n_tot)
    obs[f"im{j}"] = permute_pauli(_with_ancilla(bp, "Y"), perm, n_tot)

from qiskit_aer import AerSimulator
_SIM_KW = dict(method="matrix_product_state",
               matrix_product_state_max_bond_dimension=CAP,
               matrix_product_state_truncation_threshold=TRUNC,
               max_parallel_threads=1)
sim_fast = AerSimulator(**_SIM_KW, mps_lapack=True)
sim_safe = AerSimulator(**_SIM_KW)


def run_chunk(qc):
    try:
        return sim_fast.run(qc).result().data()
    except Exception as e:
        print(f"    lapack SVD failed ({type(e).__name__}); "
              f"redoing chunk with default SVD", flush=True)
        return sim_safe.run(qc).result().data()


# ---- one-point pass (stationary vacuum: t-independent), from mps0
qc = QuantumCircuit(n_tot)
qc.set_matrix_product_state(mps0)
for j, p in enumerate(probes):
    qc.save_expectation_value(permute_pauli(p, perm, n_tot),
                              list(range(n_tot)), label=f"p{j}")
qc.save_expectation_value(permute_pauli(insert, perm, n_tot),
                          list(range(n_tot)), label="ins")
d1 = run_chunk(qc)
probe_1pt = np.array([complex(d1[f"p{j}"]) for j in range(len(probes))])
insert_1pt = complex(d1["ins"])
print(f"[{time.time()-t0:6.0f}s] one-points done  <O_c> = {insert_1pt:.6f}",
      flush=True)

for a in terms:
    ops_a, c_a = terms_a[a]
    OUT = f"data/quench_seq_ns{NS}_chi{CAP}_term{a}.npz"
    # the checkpoint is working state, not a result: data/work/ (gitignored)
    CKPT = OUT.replace("data/", "data/work/").replace(".npz", "_ckpt.pkl")
    os.makedirs("data/work", exist_ok=True)
    if os.path.exists(CKPT):
        with open(CKPT, "rb") as f:
            ck = pickle.load(f)
        mps, start = ck["mps"], ck["it"] + 1
        reX = np.full((nt, len(probes)), np.nan)
        imY = np.full((nt, len(probes)), np.nan)
        reX[:start], imY[:start] = ck["reX"][:start], ck["imY"][:start]
        print(f"[{time.time()-t0:6.0f}s] term {a}: resumed at "
              f"t={times[ck['it']]:.1f}", flush=True)
    else:
        head = QuantumCircuit(n_tot)
        head.h(anc)
        controlled_pauli(head, anc, ops_a)
        head_t = transpile(permute_circuit(head, perm, n_tot),
                           basis_gates=_AER_BASIS, optimization_level=1)
        mps, start = (mps0, 0)
        reX = np.full((nt, len(probes)), np.nan)
        imY = np.full((nt, len(probes)), np.nan)
    for it in range(start, nt):
        qc = QuantumCircuit(n_tot)
        qc.set_matrix_product_state(mps)
        if it == 0:
            qc.compose(head_t, inplace=True)
        else:
            qc.compose(unit_t, inplace=True)
        for lbl, op in obs.items():
            qc.save_expectation_value(op, list(range(n_tot)), label=lbl)
        qc.save_matrix_product_state(label="mps")
        data = run_chunk(qc)
        mps = data["mps"]
        reX[it] = [np.real(data[f"re{j}"]) for j in range(len(probes))]
        imY[it] = [np.real(data[f"im{j}"]) for j in range(len(probes))]
        np.savez(OUT, times=times, reX=reX, imY=imY, c_a=c_a, term=a,
                 probe_1pt=probe_1pt, insert_1pt=insert_1pt, ns=NS,
                 cap=CAP, bc=bc, m0=M0, g2=G2, eta=ETA, trunc=TRUNC,
                 dt_snap=DT_SNAP, done_units=it)
        with open(CKPT + ".tmp", "wb") as f:
            pickle.dump({"mps": mps, "it": it, "reX": reX, "imY": imY}, f)
        os.replace(CKPT + ".tmp", CKPT)
        print(f"[{time.time()-t0:6.0f}s] term {a} t={times[it]:5.1f} done  "
              f"re(c) = {reX[it, bc]:+.5f}", flush=True)
    print(f"[{time.time()-t0:6.0f}s] saved {OUT}", flush=True)
