"""Variational vacuum preparation (SC-ADAPT-inspired, fixed Hamiltonian pool).

Layered variational-Hamiltonian ansatz on top of the strong-coupling vacuum:
every generator is a translation-invariant sum of Gauss-law-commuting terms
(the Hamiltonian's own hop / mass / gauge structures), so the ansatz cannot
leak out of the physical sector at ANY parameter value, and the PBC seam is
handled by the same parity trick as time evolution.

Because the theory is gapped and confining, optimal per-layer angles become
volume-independent once ns exceeds the correlation length: optimize
classically at small ns (statevector), then reuse the SAME angles at large ns
(the scalable-circuits trick of arXiv:2308.04481, adapted to explicit links).

Layer l:  exp(-i th_e^l H_even-hop) exp(-i th_o^l H_odd-hop)
          exp(-i th_m^l/2 sum (-1)^n Z_n) exp(-i th_g^l/2 sum X_link)
"""

import numpy as np
import scipy.optimize
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from .lattice import Z2Lattice
from . import hamiltonian as ham
from . import exact
from .trotter import strong_coupling_vacuum_circuit, _hop_layer, _single_qubit_layer

N_PARAMS_PER_LAYER = 4


def vacuum_ansatz(lat: Z2Lattice, thetas: np.ndarray) -> QuantumCircuit:
    """thetas: flat array of length 4*L -> (th_e, th_o, th_m, th_g) per layer."""
    thetas = np.asarray(thetas, dtype=float).reshape(-1, N_PARAMS_PER_LAYER)
    qc = strong_coupling_vacuum_circuit(lat)
    for th_e, th_o, th_m, th_g in thetas:
        _hop_layer(qc, lat, eta=1.0, dt=th_e, parity=0)
        _hop_layer(qc, lat, eta=1.0, dt=th_o, parity=1)
        _single_qubit_layer(qc, lat, m0=1.0, g2=0.0, dt=th_m)
        _single_qubit_layer(qc, lat, m0=0.0, g2=1.0, dt=th_g)
    return qc


def ansatz_state(lat: Z2Lattice, thetas) -> np.ndarray:
    return np.asarray(Statevector.from_instruction(vacuum_ansatz(lat, thetas)))


def vacuum_energy(lat: Z2Lattice, m0, g2, eta, thetas, H_sparse=None) -> float:
    if H_sparse is None:
        H_sparse = exact.to_sparse(ham.build_hamiltonian(lat, m0, g2, eta))
    psi = ansatz_state(lat, thetas)
    return float(np.real(np.vdot(psi, H_sparse @ psi)))


def optimize_vacuum(lat: Z2Lattice, m0, g2, eta, n_layers: int = 2,
                    x0: np.ndarray | None = None, restarts: int = 3,
                    seed: int = 7) -> dict:
    """Minimize <H> over the layered ansatz. Deterministic given `seed`.

    Returns {thetas, energy, exact_energy, fidelity} (exact via sparse ED)."""
    H = exact.to_sparse(ham.build_hamiltonian(lat, m0, g2, eta))
    rng = np.random.default_rng(seed)

    def cost(th):
        psi = ansatz_state(lat, th)
        return float(np.real(np.vdot(psi, H @ psi)))

    best = None
    starts = []
    if x0 is not None:
        starts.append(np.asarray(x0, dtype=float))
    while len(starts) < restarts:
        starts.append(0.15 * rng.standard_normal(N_PARAMS_PER_LAYER * n_layers))
    for s in starts:
        res = scipy.optimize.minimize(cost, s, method="BFGS",
                                      options={"gtol": 1e-8, "maxiter": 500})
        if best is None or res.fun < best.fun:
            best = res

    energies, vecs = exact.lowest_physical_states(lat, m0, g2, eta, k=1)
    psi = ansatz_state(lat, best.x)
    fidelity = float(abs(np.vdot(vecs[:, 0], psi)) ** 2)
    return {"thetas": best.x, "energy": best.fun,
            "exact_energy": float(energies[0]), "fidelity": fidelity}
