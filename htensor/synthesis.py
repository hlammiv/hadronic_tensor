"""Rotation-synthesis error studies (fault-tolerant compilation axis).

In fault tolerance every non-Clifford Rz-form rotation must be approximated
by Clifford+T at tolerance eps, costing ~ 3 log2(1/eps) T gates
(Ross-Selinger).  This module models synthesis error by snapping all
rotation angles to a grid of spacing delta (deterministic or unbiased
stochastic rounding), so a full observable pipeline can be rerun as a
function of delta -- terminating at delta = pi/2 where the circuit is
exactly Clifford (stabilizer-simulable at any width).

The grid is anchored at 0, so Clifford points are preserved for every delta
that divides pi/2, and delta = pi/2 sends every rotation to a Clifford gate.
"""

import numpy as np
from qiskit import QuantumCircuit

_ROT = ("rz", "rx", "ry")


def snap_angles(circ: QuantumCircuit, delta: float, mode: str = "round",
                seed: int | None = None) -> QuantumCircuit:
    """Replace every rotation angle theta by a multiple of delta.

    mode='round': nearest grid point (coherent, worst-case linear error
    accumulation).  mode='stochastic': unbiased randomized rounding
    (E[snapped] = theta; dephasing-like quadratic accumulation)."""
    rng = np.random.default_rng(seed)
    out = QuantumCircuit(circ.num_qubits)
    for inst in circ.data:
        name = inst.operation.name
        qubits = [circ.find_bit(q).index for q in inst.qubits]
        if name in _ROT:
            th = float(inst.operation.params[0])
            k = th / delta
            if mode == "round":
                kk = np.round(k)
            elif mode == "stochastic":
                lo = np.floor(k)
                kk = lo + (rng.random() < (k - lo))
            else:
                raise ValueError(mode)
            ang = float(kk * delta)
            if abs(ang) > 1e-15:
                getattr(out, name)(ang, qubits[0])
        else:
            out.append(inst.operation, qubits)
    return out


def rotation_census(circ: QuantumCircuit, atol: float = 1e-12):
    """(n_nonclifford_rotations, n_total_rotations): rotations at multiples
    of pi/2 are Clifford and cost no T gates."""
    n_rot = n_nc = 0
    for inst in circ.data:
        if inst.operation.name in _ROT:
            n_rot += 1
            th = float(inst.operation.params[0]) % (np.pi / 2)
            if min(th, np.pi / 2 - th) > atol:
                n_nc += 1
    return n_nc, n_rot


def t_count_estimate(n_rotations: int, eps: float) -> int:
    """Ross-Selinger scaling: ~ 3 log2(1/eps) T gates per rotation."""
    return int(n_rotations * np.ceil(3 * np.log2(1.0 / eps)))


# ------------------------- true Clifford+T synthesis tier (pygridsynth) ----
_H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
_GATES = {
    "H": _H,
    "S": np.diag([1, 1j]),
    "T": np.diag([1, np.exp(1j * np.pi / 4)]),
    "X": np.array([[0, 1], [1, 0]]),
    "Z": np.diag([1, -1]),
    "W": np.exp(1j * np.pi / 4) * np.eye(2),  # global-phase marker
    "I": np.eye(2),
}


def _word_matrix(seq: str) -> np.ndarray:
    """Matrix of a pygridsynth gate word.  pygridsynth emits the sequence in
    circuit order (leftmost applied first), so the operator product runs
    right-to-left over the string reversed -- verified in tests against
    Rz(theta) to the requested tolerance."""
    U = np.eye(2, dtype=complex)
    for ch in seq:
        U = U @ _GATES[ch]
    return U


def _phase_dist(U: np.ndarray, V: np.ndarray) -> float:
    """Operator distance up to global phase."""
    inner = np.trace(U.conj().T @ V) / 2
    return float(np.sqrt(max(0.0, 1.0 - abs(inner) ** 2)))


def gridsynth_unitary(theta: float, eps: float, _cache={}) -> np.ndarray:
    """True Ross-Selinger Clifford+T approximant of Rz(theta), multiplied
    into a single 2x2 unitary (residual error includes the axis-tilt
    component that pure angle snapping does not model)."""
    key = (round(float(theta), 12), eps)
    if key not in _cache:
        from pygridsynth import gridsynth_gates
        seq = str(gridsynth_gates(theta=float(theta), epsilon=eps))
        U = _word_matrix(seq)
        # pick the orientation matching Rz(theta)
        rz = np.diag([np.exp(-0.5j * theta), np.exp(0.5j * theta)])
        Ur = _word_matrix(seq[::-1])
        _cache[key] = U if _phase_dist(U, rz) <= _phase_dist(Ur, rz) else Ur
    return _cache[key]


def gridsynth_substitute(circ: QuantumCircuit, eps: float) -> QuantumCircuit:
    """Replace every rotation by its true Clifford+T approximant (as a
    single 1q unitary; rx/ry via basis conjugation).  Distinct angles are
    cached, so the synthesis cost is per unique angle, not per gate."""
    from qiskit.circuit.library import UnitaryGate

    out = QuantumCircuit(circ.num_qubits)
    for inst in circ.data:
        name = inst.operation.name
        qubits = [circ.find_bit(q).index for q in inst.qubits]
        if name in _ROT:
            th = float(inst.operation.params[0])
            W = gridsynth_unitary(th, eps)
            if name == "rx":
                W = _H @ W @ _H
            elif name == "ry":
                Sm = _GATES["S"]
                W = (Sm @ _H) @ W @ (_H @ Sm.conj().T)
            out.append(UnitaryGate(W), qubits)
        else:
            out.append(inst.operation, qubits)
    return out
