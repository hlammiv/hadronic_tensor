"""Exact-diagonalization reference machinery (small volumes, sparse matrices).

Ground truth for unit tests and for validating Trotter circuits, state
preparation, and the correlator/W pipeline at ns <= ~8 (16 qubits).
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from qiskit.quantum_info import SparsePauliOp

from .lattice import Z2Lattice
from . import hamiltonian as ham


def to_sparse(op: SparsePauliOp) -> sp.csr_matrix:
    return op.to_matrix(sparse=True).tocsr()


def strong_coupling_vacuum(lat: Z2Lattice) -> np.ndarray:
    """Product state: even matter sites |0>, odd |1>, links |+>.

    Satisfies G_n = +1 for all n and Q = 0; the m0 -> infinity vacuum and the
    reference state for adiabatic/variational preparation.
    """
    zero = np.array([1.0, 0.0])
    one = np.array([0.0, 1.0])
    plus = np.array([1.0, 1.0]) / np.sqrt(2)
    single = {}
    for n in range(lat.ns):
        single[lat.site_qubit(n)] = zero if n % 2 == 0 else one
    for q in lat.link_qubits:
        single[q] = plus
    psi = np.array([1.0])
    for q in reversed(range(lat.n_qubits)):  # accumulate from highest qubit down
        psi = np.kron(psi, single[q])
    return psi


def penalized_hamiltonian(
    lat: Z2Lattice, m0: float, g2: float = 1.0, eta: float = 1.0,
    lam_gauss: float = 10.0, lam_charge: float = 10.0,
) -> SparsePauliOp:
    """H - lam_gauss * sum_n G_n + lam_charge * Q^2: pushes the physical
    (all G_n = +1), charge-zero sector to the bottom of the spectrum."""
    H = ham.build_hamiltonian(lat, m0, g2, eta)
    G = sum(ham.gauss_operator(lat, n) for n in range(lat.ns)).simplify()
    Q = ham.total_charge(lat)
    return (H - lam_gauss * G + lam_charge * (Q @ Q)).simplify()


def lowest_physical_states(
    lat: Z2Lattice, m0: float, g2: float = 1.0, eta: float = 1.0, k: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """(energies, states) of the k lowest eigenstates of H within the
    physical, Q=0 sector. Energies are of H itself (penalties subtracted by
    evaluating <H> on the penalized eigenvectors)."""
    Hp = to_sparse(penalized_hamiltonian(lat, m0, g2, eta))
    H = to_sparse(ham.build_hamiltonian(lat, m0, g2, eta))
    v0 = strong_coupling_vacuum(lat)
    _, vecs = spla.eigsh(Hp, k=k, which="SA", v0=v0)
    energies = np.array([np.real(np.vdot(v, H @ v)) for v in vecs.T])
    order = np.argsort(energies)
    return energies[order], vecs[:, order]


def evolve(H: sp.spmatrix, psi: np.ndarray, t: float) -> np.ndarray:
    return spla.expm_multiply(-1j * t * H.tocsc(), psi)


def two_current_correlator(
    H: sp.spmatrix, op_later: sp.spmatrix, op_earlier: sp.spmatrix,
    psi: np.ndarray, times: np.ndarray,
) -> np.ndarray:
    """C(t) = <psi| e^{iHt} B e^{-iHt} A |psi> for B = op_later, A = op_earlier.

    Evolves |psi> and A|psi> stepwise over the (uniformly spaced or not)
    time grid; exact up to expm_multiply tolerance.
    """
    Hc = H.tocsc()
    ket = op_earlier @ psi
    bra = psi.copy()
    out = np.empty(len(times), dtype=complex)
    t_prev = 0.0
    for i, t in enumerate(times):
        dt = t - t_prev
        if dt != 0.0:
            ket = spla.expm_multiply(-1j * dt * Hc, ket)
            bra = spla.expm_multiply(-1j * dt * Hc, bra)
        out[i] = np.vdot(bra, op_later @ ket)
        t_prev = t
    return out


def expectation(op: sp.spmatrix | SparsePauliOp, psi: np.ndarray) -> complex:
    if isinstance(op, SparsePauliOp):
        op = to_sparse(op)
    return complex(np.vdot(psi, op @ psi))
