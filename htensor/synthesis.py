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
