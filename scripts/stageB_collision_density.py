"""Stage B(ii'): single-pass density map for the CGK-A two-packet collision.

Replaces the per-slice restart protocol of stageB_collision_cgkA.py, whose
cost grew ~x3.5 per unit time (unusable past t~3): here ONE capped-chi Aer
MPS state is evolved 0 -> tmax, carried between unit-time chunks via
save/set_matrix_product_state, with <Z_n> snapshots on every site after
each unit.  The full rho(n,t) grid therefore costs a single evolution,
logged and written to the npz after every unit.  A chi=256 twin quantifies
the truncation systematic (synthesis-ladder pattern).

mps_lapack=True routes the per-gate SVDs through LAPACK: benchmarked 3.4x
faster than Aer's default at identical (6-digit) expectation values;
extra threads gain nothing, so runs are pinned to one core.

The evolved MPS is pickled to disk each unit; on relaunch the script
resumes from the checkpoint instead of re-prepping and re-evolving.

  PYTHONPATH=. python scripts/stageB_collision_density.py <j> <cap> [ns] [tmax]
j=3: K=+-1.2566 (on-resonance; predicted Wigner delay ~ +4.2)
j=2: K=+-0.8378 (off-resonance null; predicted ~ +0.2)
"""
import os
import pickle
import sys
import time

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp

from htensor import Z2Lattice, stateprep, wavepacket, backends, trotter
from htensor.backends import permute_circuit, permute_pauli, _AER_BASIS

M0, G2, ETA = 0.1, 0.4, 1.0            # CGK-A
JTAG = sys.argv[1]                      # "3" both packets; "3R"/"3L" single
J = int(JTAG[0])
MODE = JTAG[1:] or "RL"
CAP = int(sys.argv[2])
NS = int(sys.argv[3]) if len(sys.argv) > 3 else 30
TMAX = float(sys.argv[4]) if len(sys.argv) > 4 else 44.0
STEPS_PER_UNIT = 4                     # dt = 0.25
TRUNC = 1e-8
KTAG = {3: ("1.26", "-1.26"), 2: ("0.84", "-0.84")}[J]
# packet centers: even staggered sites, antipodal on the cell ring so the
# wrap-around re-encounter sits as late as possible
C1, C2 = (8, 22) if NS == 30 else (NS // 4, 3 * NS // 4)
OUT = f"data/stageB_colldens_j{JTAG}_chi{CAP}_ns{NS}.npz"
CKPT = f"data/work/stageB_colldens_j{JTAG}_chi{CAP}_ns{NS}_ckpt.pkl"

t0 = time.time()
lat = Z2Lattice(NS, pbc=True)
n_sys = lat.n_qubits
n_tot = n_sys + 1                       # prepare_state_mps register: +ancilla
n_units = int(round(TMAX))
times = np.arange(0, n_units + 1, dtype=float)
sgn = np.array([(-1) ** n for n in range(NS)])

if os.path.exists(CKPT):
    with open(CKPT, "rb") as f:
        ck = pickle.load(f)
    mps, perm, start = ck["mps"], ck["perm"], ck["it"] + 1
    zex = np.full((len(times), NS), np.nan)
    zex[:start] = ck["zex"][:start]
    print(f"[{time.time()-t0:6.0f}s] resumed from checkpoint at "
          f"t={ck['it']} (cap={CAP})", flush=True)
else:
    zv = np.load("data/cgkA_vacuum_thetas.npz")
    prep = stateprep.vacuum_ansatz(lat, zv["thetas"])
    packets = [p for tag, p in (("R", (KTAG[0], C1)), ("L", (KTAG[1], C2)))
               if tag in MODE]
    for ktag, center in packets:
        z = np.load(f"data/wpCGKA_params_k{ktag}_L3.npz")
        params = wavepacket.params_from_vector(z["vec"], list(z["offsets"]),
                                               int(z["L"]))
        prep.compose(wavepacket.block_circuit(lat, center, params),
                     inplace=True)
        print(f"[{time.time()-t0:6.0f}s] packet K={ktag} at site {center}",
              flush=True)
    mps, perm = backends.prepare_state_mps(
        lat, prep, NS, cap=CAP, trunc=1e-10, max_threads=1,
        sim_opts={"mps_lapack":
                  os.environ.get("PREP_LAPACK", "1") == "1"})
    start = 0
    zex = np.full((len(times), NS), np.nan)
    print(f"[{time.time()-t0:6.0f}s] two-packet MPS prepared (cap={CAP})",
          flush=True)

# one unit of Trotterized time, permuted + transpiled once, reused each chunk
unit = trotter.trotter_circuit(lat, M0, G2, ETA, 1.0, STEPS_PER_UNIT)
unit_t = transpile(permute_circuit(unit, perm, n_tot),
                   basis_gates=_AER_BASIS, optimization_level=1)

# <Z_site(n)> in the folded layout; J0(n) = ((-1)^n - <Z_n>)/2
zops = [permute_pauli(
            SparsePauliOp.from_sparse_list([("Z", [lat.site_qubit(n)], 1.0)],
                                           num_qubits=n_sys), perm, n_tot)
        for n in range(NS)]

from qiskit_aer import AerSimulator
_SIM_KW = dict(method="matrix_product_state",
               matrix_product_state_max_bond_dimension=CAP,
               matrix_product_state_truncation_threshold=TRUNC,
               max_parallel_threads=1)
sim_fast = AerSimulator(**_SIM_KW, mps_lapack=True)
sim_safe = AerSimulator(**_SIM_KW)     # default (Jacobi) SVD


def run_unit(qc):
    """LAPACK SVD (3.4x faster), falling back to the default SVD for any
    unit where gesvd hits a degenerate matrix and fails to converge."""
    try:
        return sim_fast.run(qc).result().data()
    except Exception as e:
        print(f"    lapack SVD failed ({type(e).__name__}); "
              f"redoing unit with default SVD", flush=True)
        return sim_safe.run(qc).result().data()

for it in range(start, len(times)):
    qc = QuantumCircuit(n_tot)
    qc.set_matrix_product_state(mps)
    if it:
        qc.compose(unit_t, inplace=True)
    for n, op in enumerate(zops):
        qc.save_expectation_value(op, list(range(n_tot)), label=f"z{n}")
    qc.save_matrix_product_state(label="mps")
    data = run_unit(qc)
    mps = data["mps"]
    zex[it] = [np.real(data[f"z{n}"]) for n in range(NS)]
    density = (sgn[None, :] - zex) / 2
    np.savez(OUT, times=times, density=density, zexp=zex, ns=NS, c1=C1,
             c2=C2, j=J, cap=CAP, m0=M0, g2=G2, eta=ETA,
             steps_per_unit=STEPS_PER_UNIT, trunc=TRUNC, done_units=it)
    with open(CKPT + ".tmp", "wb") as f:
        pickle.dump({"mps": mps, "perm": perm, "it": it, "zex": zex}, f)
    os.replace(CKPT + ".tmp", CKPT)
    dens = density[it]
    print(f"[{time.time()-t0:6.0f}s] t={times[it]:4.0f} done  "
          f"max|dJ0| = {np.abs(dens - dens.mean()).max():.4f}", flush=True)

print(f"[{time.time()-t0:6.0f}s] saved {OUT}", flush=True)
