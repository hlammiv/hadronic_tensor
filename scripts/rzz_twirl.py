"""Transpile-time Pauli twirling for ISA circuits containing cz + rzz.

The runtime's built-in gate twirling rejects fractional gates (error
1519), but pre-twirled circuits are just circuits.  Twirl rules:
  cz      : full 16-pair Clifford twirl, compensation P' = CZ (P1 x P2) CZ
  rzz(th) : restricted 8-pair commutant twirl ({I,Z}x{I,Z} u {X,Y}x{X,Y});
            these commute with ZZ, so the compensating pair equals the
            inserted pair (self-inverse) and the angle is untouched.
Inserted Paulis are merged into the neighbouring 1q layers by a
routing-free 1q re-synthesis, so the 2q skeleton is unchanged.

self-test:  PYTHONPATH=. .venv/bin/python scripts/rzz_twirl.py
"""
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import IGate, XGate, YGate, ZGate

PAULI = {"I": IGate(), "X": XGate(), "Y": YGate(), "Z": ZGate()}
RZZ_PAIRS = [("I", "I"), ("I", "Z"), ("Z", "I"), ("Z", "Z"),
             ("X", "X"), ("X", "Y"), ("Y", "X"), ("Y", "Y")]
# CZ (P1 x P2) CZ = (P1', P2') up to phase
CZ_MAP = {"I": {"I": ("I", "I"), "X": ("Z", "X"), "Y": ("Z", "Y"),
                "Z": ("I", "Z")},
          "X": {"I": ("X", "Z"), "X": ("Y", "Y"), "Y": ("Y", "X"),
                "Z": ("X", "I")},
          "Y": {"I": ("Y", "Z"), "X": ("X", "Y"), "Y": ("X", "X"),
                "Z": ("Y", "I")},
          "Z": {"I": ("Z", "I"), "X": ("I", "X"), "Y": ("I", "Y"),
                "Z": ("Z", "Z")}}
ALL16 = [(a, b) for a in "IXYZ" for b in "IXYZ"]


def twirl_circuit(qc, rng, basis=("rz", "sx", "x", "cz", "rzz")):
    """One twirled instance of an ISA circuit; logical unitary unchanged
    (up to global phase), 2q skeleton identical."""
    out = QuantumCircuit(*qc.qregs, *qc.cregs)
    out.global_phase = qc.global_phase
    for inst in qc.data:
        name = inst.operation.name
        if name in ("cz", "rzz"):
            q0, q1 = (qc.find_bit(q).index for q in inst.qubits)
            if name == "rzz":
                p1, p2 = RZZ_PAIRS[rng.integers(len(RZZ_PAIRS))]
                c1, c2 = p1, p2
            else:
                p1, p2 = ALL16[rng.integers(16)]
                c1, c2 = CZ_MAP[p1][p2]
            for p, q in ((p1, q0), (p2, q1)):
                if p != "I":
                    out.append(PAULI[p], [q])
            out.append(inst.operation, [q0, q1])
            for c, q in ((c1, q0), (c2, q1)):
                if c != "I":
                    out.append(PAULI[c], [q])
        else:
            out.append(inst.operation,
                       [qc.find_bit(q).index for q in inst.qubits],
                       [qc.find_bit(c).index for c in inst.clbits])
    # merge inserted Paulis into 1q layers; no routing, no layout change
    return transpile(out, basis_gates=list(basis), optimization_level=1,
                     routing_method="none")


if __name__ == "__main__":
    from qiskit.quantum_info import Operator
    rng = np.random.default_rng(7)
    qc = QuantumCircuit(4)
    rng2 = np.random.default_rng(1)
    for layer in range(6):
        for q in range(4):
            qc.rz(rng2.uniform(0, 2 * np.pi), q)
            qc.sx(q)
        qc.cz(0, 1)
        qc.rzz(rng2.uniform(0, np.pi / 2), 1, 2)
        qc.cz(2, 3)
        qc.rzz(rng2.uniform(0, np.pi / 2), 0, 1)
    U = Operator(qc)
    ok = 0
    for k in range(20):
        tq = twirl_circuit(qc, rng)
        n2_a = sum(1 for i in qc.data if i.operation.num_qubits == 2)
        n2_b = sum(1 for i in tq.data if i.operation.num_qubits == 2)
        assert n2_a == n2_b, "2q skeleton changed"
        V = Operator(tq)
        # equal up to global phase
        M = U.adjoint().compose(V).data
        ph = M[0, 0] / abs(M[0, 0])
        if np.allclose(M / ph, np.eye(16), atol=1e-10):
            ok += 1
    print(f"twirl self-test: {ok}/20 instances unitary-equivalent, "
          f"2q skeleton preserved")
