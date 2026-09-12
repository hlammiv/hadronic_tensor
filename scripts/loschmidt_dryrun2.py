"""Bit-level damping-model Loschmidt gate (task #21 phase 2).

Samples real shots from noisy circuits (Pauli trajectories at the
validated g*=1 scale) and stores raw bits; the two-sector READOUT split
(sites nominal, links 3.7x, asymmetric ro01/ro10) is applied classically
in analysis, exactly as in hw_sq_2param.py.  Measurement bases: sites Z,
links X (free Gauss syndromes), ancilla X (Re-part estimator).

Modes:
  sample <s0> <s1>   seeds s0..s1-1 -> data/work/dryrun/tmp_losch2_<seed>.npz
                     (bit arrays per circuit: t0_phys, {t}_{kind})
  analyze            (written after sampling lands) readout split +
                     estimators + kappa + wing metric + syndrome cuts

  PYTHONPATH=. .venv/bin/python scripts/loschmidt_dryrun2.py sample 0 4
"""
import os
import sys
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

sys.path.insert(0, "scripts")
import hw_t3_dryrun as base
from hw_t3_dryrun import build_prep, insertion, log, noise_transform
from loschmidt_dryrun import variant_circuit, CAP
from htensor import backends

P2, P1 = base.P2, base.P1
TS = [0.5, 1.0]
KINDS = ("phys", "mir0", "losch")
PER = 512                              # shots per circuit per seed


def setup():
    lat, prep = build_prep()
    n_tot = lat.n_qubits + 1
    mps, perm = backends.prepare_state_mps(lat, prep, base.CENTER * 2,
                                           cap=CAP, trunc=1e-10,
                                           max_threads=1,
                                           sim_opts={"mps_lapack": True})
    log(f"prep MPS ready (cap={CAP})")
    _, ins_ops, _ = insertion(lat)
    tqcs = {}
    for t in TS + [0.0]:
        for kind in (KINDS if t > 0 else ("phys",)):
            qcp = backends.permute_circuit(
                variant_circuit(lat, t, ins_ops, kind), perm, n_tot)
            tqcs[(t, kind)] = transpile(qcp, basis_gates=backends._AER_BASIS,
                                        optimization_level=1)
    # basis-change layer (permuted frame): H on ancilla + all links
    rot = QuantumCircuit(n_tot)
    rot.h(perm[lat.n_qubits])
    for q in [lat.link_qubit(b) for b in range(lat.n_links)]:
        rot.h(perm[q])
    return lat, mps, perm, tqcs, rot


def sample(s0, s1):
    lat, mps, perm, tqcs, rot = setup()
    n_tot = lat.n_qubits + 1
    sim = AerSimulator(method="matrix_product_state",
                       matrix_product_state_max_bond_dimension=CAP,
                       matrix_product_state_truncation_threshold=1e-8,
                       max_parallel_threads=1, mps_lapack=True)
    for seed in range(s0, s1):
        rng = np.random.default_rng(10_000 + seed)
        rec = {}
        for (t, kind), tqc in sorted(tqcs.items()):
            noisy = noise_transform(tqc, rng, P2, P1)
            full = QuantumCircuit(n_tot, n_tot)
            full.set_matrix_product_state(mps)
            full.compose(noisy, inplace=True)
            full.compose(rot, qubits=range(n_tot), inplace=True)
            full.measure(range(n_tot), range(n_tot))
            try:
                res = sim.run(full, shots=PER, memory=True).result()
            except Exception:
                safe = AerSimulator(
                    method="matrix_product_state",
                    matrix_product_state_max_bond_dimension=CAP,
                    matrix_product_state_truncation_threshold=1e-8,
                    max_parallel_threads=1)
                res = safe.run(full, shots=PER, memory=True).result()
            mem = res.get_memory()
            bits = np.array([[int(c) for c in s[::-1]] for s in mem],
                            dtype=np.uint8)
            rec[f"{t}_{kind}"] = bits
        os.makedirs("data/work/dryrun", exist_ok=True)
        np.savez_compressed(f"data/work/dryrun/tmp_losch2_{seed}.npz", **rec,
                            perm_logical=np.array(
                                [perm[q] for q in range(n_tot)]))
        log(f"seed {seed} saved")


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "sample":
        sample(int(sys.argv[2]), int(sys.argv[3]))
    else:
        raise SystemExit("analyze mode is built after sampling lands")
