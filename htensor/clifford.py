"""Clifford-point utilities: full-width stabilizer validation of the pipeline.

At parameter points where every rotation angle is a multiple of pi/2 the
Trotter circuits are Clifford, so the identical circuit-generation code can be
validated (and mitigation rehearsed) at 100+ qubits with Aer's stabilizer
method / stim.  For the palindromic trotter_step (which contains dt/2
half-layers) the Clifford-point conditions are
    m0*dt in {k pi},  g2*dt in {k pi},  eta*dt in {2 k pi}
(e.g. m0 = g2 = 1, eta = 2, dt = pi).

Gates are expanded with fixed textbook decompositions (not the transpiler,
whose generic two-qubit synthesis emits non-Clifford rz(pi/4) pairs that only
cancel globally).  Global phase is not tracked -- irrelevant for stabilizer
expectation values.
"""

import numpy as np
from qiskit import QuantumCircuit

_PASSTHROUGH = {"h", "x", "y", "z", "s", "sdg", "sx", "sxdg", "cx", "cz", "cy", "swap"}


def _rz_clifford(out: QuantumCircuit, angle: float, q: int, atol: float):
    angle = float(angle) % (2 * np.pi)
    k = round(angle / (np.pi / 2))
    if abs(angle - k * np.pi / 2) > atol and abs(angle - 2 * np.pi) > atol:
        raise ValueError(f"rz({angle}) is not at a Clifford point")
    k %= 4
    if k == 1:
        out.s(q)
    elif k == 2:
        out.z(q)
    elif k == 3:
        out.sdg(q)


def _rzz(out: QuantumCircuit, angle: float, qa: int, qb: int, atol: float):
    out.cx(qa, qb)
    _rz_clifford(out, angle, qb, atol)
    out.cx(qa, qb)


def to_clifford_circuit(circ: QuantumCircuit, atol: float = 1e-9) -> QuantumCircuit:
    """Rewrite a circuit whose rotations all sit at Clifford angles into
    explicit Clifford gates (accepted by Aer 'stabilizer', stim, and
    qiskit.quantum_info.Clifford).  Raises ValueError off Clifford points."""
    out = QuantumCircuit(circ.num_qubits)
    for inst in circ.data:
        name = inst.operation.name
        q = [circ.find_bit(b).index for b in inst.qubits]
        if name in _PASSTHROUGH:
            out.append(inst.operation, q)
        elif name in ("barrier", "id"):
            continue
        elif name == "rz":
            _rz_clifford(out, inst.operation.params[0], q[0], atol)
        elif name == "rx":
            out.h(q[0])
            _rz_clifford(out, inst.operation.params[0], q[0], atol)
            out.h(q[0])
        elif name == "rzz":
            _rzz(out, inst.operation.params[0], q[0], q[1], atol)
        elif name == "rxx":
            for i in q:
                out.h(i)
            _rzz(out, inst.operation.params[0], q[0], q[1], atol)
            for i in q:
                out.h(i)
        elif name == "ryy":
            for i in q:  # V^dag with V = S.H  (V Z V^dag = Y)
                out.sdg(i)
                out.h(i)
            _rzz(out, inst.operation.params[0], q[0], q[1], atol)
            for i in q:
                out.h(i)
                out.s(i)
        else:
            raise ValueError(f"unsupported gate for Clifford rewrite: {name}")
    return out
