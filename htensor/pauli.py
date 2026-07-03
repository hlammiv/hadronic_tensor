"""Unambiguous SparsePauliOp construction from {qubit: pauli} maps."""

from qiskit.quantum_info import SparsePauliOp


def pauli_term(n_qubits: int, ops: dict[int, str], coeff: complex) -> SparsePauliOp:
    """One Pauli string given as {qubit_index: 'X'|'Y'|'Z'}.

    Builds the dense little-endian label explicitly (leftmost character is
    qubit n_qubits-1) so no sparse-list index convention can bite.
    """
    label = ["I"] * n_qubits
    for q, p in ops.items():
        if not 0 <= q < n_qubits:
            raise ValueError(f"qubit {q} out of range")
        if label[n_qubits - 1 - q] != "I":
            raise ValueError(f"duplicate qubit {q}")
        label[n_qubits - 1 - q] = p
    return SparsePauliOp("".join(label), coeffs=[coeff])


def pauli_sum(n_qubits: int, terms: list[tuple[dict[int, str], complex]]) -> SparsePauliOp:
    op = sum(pauli_term(n_qubits, ops, c) for ops, c in terms)
    return op.simplify()
