"""Jordan-Wigner-mapped Z2 gauge Hamiltonian, Gauss law, and symmetry operators.

H = (g2/2) sum_links sigma^x_link
    - (m0/2) sum_n (-1)^n Z_n
    + (eta/4) sum_bonds (X_n X_{n+1} + Y_n Y_{n+1}) sigma^z_{n,n+1}

The PBC seam hop (bond ns-1 -> 0) carries the JW string over the interior
matter qubits 1..ns-2; on Gauss-law states this equals the bulk hop gate with
coupling eta -> (-1)^(ns/2 + 1) eta (fermion-parity replacement).
"""

from qiskit.quantum_info import SparsePauliOp

from .lattice import Z2Lattice
from .pauli import pauli_term, pauli_sum


def gauge_term(lat: Z2Lattice, g2: float = 1.0) -> SparsePauliOp:
    return pauli_sum(
        lat.n_qubits,
        [({q: "X"}, g2 / 2) for q in lat.link_qubits],
    )


def mass_term(lat: Z2Lattice, m0: float) -> SparsePauliOp:
    return pauli_sum(
        lat.n_qubits,
        [({lat.site_qubit(n): "Z"}, -(m0 / 2) * (-1) ** n) for n in range(lat.ns)],
    )


def hop_term(lat: Z2Lattice, bond: int, eta: float = 1.0) -> SparsePauliOp:
    """(eta/4)(X_n X_{n+1} + Y_n Y_{n+1}) sigma^z on one bond, with the JW
    string over interior matter qubits when the bond is the PBC seam."""
    qa, qb = lat.site_qubit(bond), lat.site_qubit(bond + 1)
    ql = lat.link_qubit(bond)
    string = {q: "Z" for q in lat.seam_string_qubits()} if lat.is_seam(bond) else {}
    return pauli_sum(
        lat.n_qubits,
        [
            ({qa: "X", qb: "X", ql: "Z", **string}, eta / 4),
            ({qa: "Y", qb: "Y", ql: "Z", **string}, eta / 4),
        ],
    )


def hopping_term(lat: Z2Lattice, eta: float = 1.0, parity: int | None = None) -> SparsePauliOp:
    """Sum of hop terms; parity=0/1 selects even/odd bonds (the internally
    commuting Trotter groups H2/H3), None takes all bonds."""
    bonds = [b for b in lat.bonds if parity is None or b % 2 == parity]
    return sum(hop_term(lat, b, eta) for b in bonds).simplify()


def build_hamiltonian(lat: Z2Lattice, m0: float, g2: float = 1.0, eta: float = 1.0) -> SparsePauliOp:
    return (gauge_term(lat, g2) + mass_term(lat, m0) + hopping_term(lat, eta)).simplify()


def gauss_operator(lat: Z2Lattice, n: int) -> SparsePauliOp:
    """G_n = (-1)^n Z_n sigma^x_{n-1,n} sigma^x_{n,n+1}; +1 on physical states.

    With OBC the boundary sites have only one adjacent link (the missing link
    operator is dropped, i.e. treated as the +1 eigenstate of a frozen field).
    """
    n = n % lat.ns
    ops = {lat.site_qubit(n): "Z"}
    if lat.pbc or n > 0:
        ops[lat.link_qubit(n - 1)] = "X"
    if lat.pbc or n < lat.ns - 1:
        ops[lat.link_qubit(n)] = "X"
    return pauli_term(lat.n_qubits, ops, (-1) ** n)


def fermion_parity(lat: Z2Lattice) -> SparsePauliOp:
    return pauli_term(lat.n_qubits, {q: "Z" for q in lat.matter_qubits}, 1.0)


def total_charge(lat: Z2Lattice) -> SparsePauliOp:
    """Q = sum_v J^0(v) with the normal-ordered staggered charge."""
    from .currents import charge_density

    return sum(charge_density(lat, v) for v in range(lat.ns)).simplify()
