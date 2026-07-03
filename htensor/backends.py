"""Aer execution layer for the Hadamard-test correlator protocol.

Runs the same circuit families as measure.hadamard_correlator_sv but through
qiskit-aer, so the identical driver covers:
  - method='statevector'          exact, <= ~28 qubits
  - method='matrix_product_state' 50-200 qubits (the production engine);
    bond dimension / truncation exposed via mps_* options
  - shots=None                    exact expectation values (save_expectation_value)
(Sampled/shot-based execution with basis rotations is the hardware layer, M5.)

State preparation is a CIRCUIT here (e.g. stateprep.vacuum_ansatz output),
not a vector -- at 100 qubits there is no vector.
"""

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp

from .lattice import Z2Lattice
from . import trotter
from .measure import (CorrelatorData, split_current, controlled_pauli,
                      _with_ancilla, _pauli_only)

_AER_BASIS = ["cx", "cz", "rz", "rx", "ry", "sx", "x", "h"]


def _simulator(method: str, mps_max_bond: int | None, mps_trunc: float,
               max_threads: int):
    from qiskit_aer import AerSimulator

    opts = {"method": method, "max_parallel_threads": max_threads}
    if method == "matrix_product_state":
        if mps_max_bond is not None:
            opts["matrix_product_state_max_bond_dimension"] = mps_max_bond
        opts["matrix_product_state_truncation_threshold"] = mps_trunc
    return AerSimulator(**opts)


def hadamard_correlator_aer(lat: Z2Lattice, prep: QuantumCircuit,
                            insert_op: SparsePauliOp,
                            probe_ops: list[SparsePauliOp],
                            m0, g2, eta, times, dt_target: float = 0.05,
                            method: str = "matrix_product_state",
                            mps_max_bond: int | None = None,
                            mps_trunc: float = 1e-12,
                            max_threads: int = 4) -> CorrelatorData:
    """Hadamard-test correlator grid via Aer with exact expectation values.

    One circuit per (insertion Pauli term, time); every probe observable is
    read from the same circuit via multiple save_expectation_value slots --
    the x-multiplexing that makes this protocol cheap.
    """
    times = np.asarray(times, dtype=float)
    n_sys = lat.n_qubits
    sim = _simulator(method, mps_max_bond, mps_trunc, max_threads)
    id_a, terms_a = split_current(insert_op)
    probes_p = [_pauli_only(op) for op in probe_ops]
    id_b = np.array([split_current(op)[0] for op in probe_ops])

    def run(circ, observables, qubits):
        # transpile first: save instructions cannot pass the basis translator;
        # basis-only transpilation does no routing, so indices are stable
        tqc = transpile(circ, basis_gates=_AER_BASIS, optimization_level=1)
        for lbl, op in observables.items():
            tqc.save_expectation_value(op, qubits, label=lbl)
        return sim.run(tqc).result().data()

    # ---- plain single-current expectations (disconnected/identity pieces)
    probe_expect = np.empty((len(times), len(probe_ops)), dtype=complex)
    insert_expect = None
    for i, t in enumerate(times):
        qc = QuantumCircuit(n_sys)
        qc.compose(prep, inplace=True)
        n = max(1, int(np.ceil(abs(t) / dt_target))) if t != 0 else 0
        qc.compose(trotter.trotter_circuit(lat, m0, g2, eta, t, n), inplace=True)
        obs = {f"p{j}": op for j, op in enumerate(probe_ops)}
        if i == 0:
            obs["ins"] = insert_op
        data = run(qc, obs, list(range(n_sys)))
        probe_expect[i] = [data[f"p{j}"] for j in range(len(probe_ops))]
        if i == 0:
            insert_expect = complex(data["ins"])
    probe_p_expect = probe_expect - id_b[None, :]

    # ---- ancilla circuits
    corr = np.zeros((len(times), len(probe_ops)), dtype=complex)
    anc = n_sys
    for ops_a, c_a in terms_a:
        for i, t in enumerate(times):
            qc = QuantumCircuit(n_sys + 1)
            qc.compose(prep, qubits=range(n_sys), inplace=True)
            qc.h(anc)
            controlled_pauli(qc, anc, ops_a)
            n = max(1, int(np.ceil(abs(t) / dt_target))) if t != 0 else 0
            qc.compose(trotter.trotter_circuit(lat, m0, g2, eta, t, n),
                       qubits=range(n_sys), inplace=True)
            obs = {}
            for j, bp in enumerate(probes_p):
                obs[f"re{j}"] = _with_ancilla(bp, "X")
                obs[f"im{j}"] = _with_ancilla(bp, "Y")
            data = run(qc, obs, list(range(n_sys + 1)))
            for j in range(len(probes_p)):
                corr[i, j] += c_a * (np.real(data[f"re{j}"])
                                     + 1j * np.real(data[f"im{j}"]))

    for j in range(len(probe_ops)):
        corr[:, j] += (id_a * probe_p_expect[:, j]
                       + id_b[j] * (insert_expect - id_a) + id_a * id_b[j])
    return CorrelatorData(times, corr, probe_expect, insert_expect)
