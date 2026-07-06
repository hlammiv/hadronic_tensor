"""Equal-time Wilson-line bilinears for quasi-PDFs (task 9).

The gauge-invariant staggered bilinear chi^dag(v0+z) W[v0+z, v0] chi(v0)
becomes, after Jordan-Wigner in the interleaved site/link layout, a SINGLE
contiguous Pauli string: the fermionic string (Z on intermediate sites) and
the Wilson line (sigma^z on intermediate links) interleave into Z on every
qubit strictly between the endpoint site qubits.  Hermitian split:

  O_R = (1/2)(X Z..Z X + Y Z..Z Y)   ~  chi^dag W chi + h.c.
  O_I = (1/2)(X Z..Z Y - Y Z..Z X)   ~  i(chi^dag W chi - h.c.)

Equal-time expectation values on a stored MPS -- no ancilla, no evolution.
Restriction: the string must not cross the PBC seam (keep the endpoints on
the arc not containing qubit 0), so |z| < ns/2 around a bulk-centered
packet.
"""

from qiskit.quantum_info import SparsePauliOp

from .lattice import Z2Lattice
from .pauli import pauli_term


def wilson_bilinear(lat: Z2Lattice, v0: int, z: int):
    """(O_R, O_I) for separation z != 0 staggered sites, seam-free arc."""
    if z == 0:
        raise ValueError("z = 0 is the charge density; use currents.J0")
    va, vb = (v0, v0 + z) if z > 0 else (v0 + z, v0)
    if va < 0 or vb >= lat.ns:
        raise ValueError("string would cross the PBC seam")
    qa, qb = lat.site_qubit(va), lat.site_qubit(vb)
    string = {q: "Z" for q in range(qa + 1, qb)}
    n = lat.n_qubits

    def op(pa, pb, c):
        return pauli_term(n, {qa: pa, **string, qb: pb}, c)

    o_r = (op("X", "X", 0.5) + op("Y", "Y", 0.5)).simplify()
    o_i = (op("X", "Y", 0.5) - op("Y", "X", 0.5)).simplify()
    return o_r, o_i
