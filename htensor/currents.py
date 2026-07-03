"""Corrected vector-current operators (PLAN.md sec. 2, items 3-4).

J^0(v)       = ((-1)^v - Z_v)/2                     (normal-ordered charge)
J^1_{v+1/2}  = (eta/4)(Y_v X_{v+1} - X_v Y_{v+1}) sigma^z_{v,v+1}   (bond current)

The bond current is exactly conserved: i[H, J^0(v)] = -(J^1_{v+1/2} - J^1_{v-1/2}),
so Z_V = 1 and q_mu W^{mu nu} = 0 holds as an exact lattice Ward identity.
The seam bond carries the same JW string as the seam hop.
"""

from qiskit.quantum_info import SparsePauliOp

from .lattice import Z2Lattice
from .pauli import pauli_sum


def charge_density(lat: Z2Lattice, v: int) -> SparsePauliOp:
    v = v % lat.ns
    return pauli_sum(
        lat.n_qubits,
        [({}, (-1) ** v / 2), ({lat.site_qubit(v): "Z"}, -0.5)],
    )


def bond_current(lat: Z2Lattice, bond: int, eta: float = 1.0) -> SparsePauliOp:
    """J^1 on bond (v, v+1), position x = v + 1/2 in staggered units."""
    qa, qb = lat.site_qubit(bond), lat.site_qubit(bond + 1)
    ql = lat.link_qubit(bond)
    string = {q: "Z" for q in lat.seam_string_qubits()} if lat.is_seam(bond) else {}
    return pauli_sum(
        lat.n_qubits,
        [
            ({qa: "Y", qb: "X", ql: "Z", **string}, eta / 4),
            ({qa: "X", qb: "Y", ql: "Z", **string}, -eta / 4),
        ],
    )


def site_current(lat: Z2Lattice, v: int, eta: float = 1.0) -> SparsePauliOp:
    """Site-symmetrized J^1(v) = (J^1_{v-1/2} + J^1_{v+1/2})/2. Conserved only
    to O(a^2); prefer bond_current for Ward-identity-exact analyses."""
    return (0.5 * (bond_current(lat, v - 1, eta) + bond_current(lat, v, eta))).simplify()


def axial_charge_density(lat: Z2Lattice, v: int) -> SparsePauliOp:
    """(-1)^v-staggered partner of J^0 (axial density), for symmetry checks."""
    v = v % lat.ns
    return pauli_sum(
        lat.n_qubits,
        [({}, 0.5), ({lat.site_qubit(v): "Z"}, -0.5 * (-1) ** v)],
    )
