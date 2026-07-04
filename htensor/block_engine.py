"""Fast numpy engine for the wavepacket block: forward states + adjoint
(reverse-mode) gradients of the overlap fidelity.

Cost of one full gradient = ~2 forward passes, independent of the number of
parameters -- this is what makes deep (L >= 2) blocks trainable, where the
displacement-type single-layer block saturates at F ~ 2/3 from multi-meson
leakage.

Gate order and conventions MATCH htensor.wavepacket.block_circuit exactly
(verified by test), so trained vectors drop straight into circuits.
"""

import numpy as np
import scipy.linalg

from .lattice import Z2Lattice
from .wavepacket import KINDS, _bond_generator, window_offsets


def _apply_local(psi: np.ndarray, U: np.ndarray, qubits: list[int], n: int):
    """Apply a k-qubit dense gate to a statevector (bit q of index = qubit q).

    U follows the Qiskit convention: qubits[0] is the LEAST significant bit
    of the gate matrix, so it must be the last (fastest) tensor axis of the
    combined gate index."""
    k = len(qubits)
    t = psi.reshape((2,) * n)
    axes = [n - 1 - q for q in reversed(qubits)]  # qubits[-1] first => MSB
    t = np.moveaxis(t, axes, range(k))
    shp = t.shape
    t = U @ t.reshape(2**k, -1)
    t = np.moveaxis(t.reshape(shp), range(k), axes)
    return t.reshape(-1)


class BlockEngine:
    """Precomputed gate list for a (layers x kinds x offsets) block."""

    def __init__(self, lat: Z2Lattice, center: int, n_layers: int,
                 offsets: list[int] | None = None,
                 max_offset: int | None = None):
        self.lat = lat
        self.n = lat.n_qubits
        self.offsets = window_offsets(lat.ns) if offsets is None else offsets
        if max_offset is not None:
            self.offsets = [o for o in self.offsets if abs(o) <= max_offset]
        self.n_layers = n_layers
        self.gates = []  # (key, generator matrix (2^k, 2^k), qubits, eigh cache)
        for l in range(n_layers):
            for kind in KINDS:
                for off in sorted(self.offsets):
                    key = (l, kind, off)
                    if kind in ("cur", "hop"):
                        op3, qubits = _bond_generator(
                            lat, (center + off) % lat.ns, kind)
                        # local qubit order in _bond_generator is [qa, qb, ql]
                        # = local indices 0,1,2; op labels are little-endian
                        G = op3.to_matrix()
                    elif kind == "site":
                        qubits = [lat.site_qubit(center + off)]
                        G = np.diag([0.5, -0.5]).astype(complex)   # rz: Z/2
                    else:
                        qubits = [lat.link_qubit(center + off)]
                        G = 0.5 * np.array([[0, 1], [1, 0]], complex)  # rx: X/2
                    w, V = np.linalg.eigh(G)
                    self.gates.append((key, G, qubits, (w, V)))
        self.keys = [g[0] for g in self.gates]

    def _unitary(self, gate, theta):
        _, _, _, (w, V) = gate
        return (V * np.exp(-1j * theta * w)) @ V.conj().T

    def state(self, vac: np.ndarray, vec: np.ndarray) -> np.ndarray:
        psi = vac
        for gate, th in zip(self.gates, vec):
            psi = _apply_local(psi, self._unitary(gate, th), gate[2], self.n)
        return psi

    def fidelity_and_grad(self, vac: np.ndarray, target: np.ndarray,
                          vec: np.ndarray):
        """F = |<target|U(vec)|vac>|^2 and dF/dvec via adjoint sweep.

        Constant memory (3 live states): the pre-gate state is recomputed
        backwards by unitary reversal instead of caching the forward pass,
        so training volumes are limited by statevector size only
        (ns = 12 -> 268 MB/state)."""
        post = vac
        for gate, th in zip(self.gates, vec):
            post = _apply_local(post, self._unitary(gate, th), gate[2], self.n)
        o = np.vdot(target, post)
        lam = target
        grad = np.empty(len(vec))
        for g in range(len(self.gates) - 1, -1, -1):
            gate, th = self.gates[g], vec[g]
            # d/dth <lam|U_g|s_pre> = <lam|(-i G) U_g|s_pre> = <lam|(-iG)|s_post>
            Gs = _apply_local(post, gate[1], gate[2], self.n)
            grad[g] = 2 * np.real(np.conj(o) * np.vdot(lam, -1j * Gs))
            Udag = self._unitary(gate, th).conj().T
            post = _apply_local(post, Udag, gate[2], self.n)  # s_{g-1}
            lam = _apply_local(lam, Udag, gate[2], self.n)
        return float(abs(o) ** 2), grad


def train_adjoint(lat: Z2Lattice, vac_state: np.ndarray, target: np.ndarray,
                  center: int, n_layers: int = 2, seed: int = 11,
                  maxiter: int = 2000, inits=None,
                  max_offset: int | None = None, l2: float = 0.0) -> dict:
    """L-BFGS-B with analytic adjoint gradients.  inits: list of start
    vectors; defaults to the lsq direction at several amplitudes plus one
    random start.  max_offset restricts the block window (locality: train
    with a window the packet fits inside, so angles transfer).

    l2 > 0 adds an angle penalty l2*sum(theta^2): among the degenerate
    parameter sets preparing the same state it selects small-angle
    solutions, whose gate-by-gate creation path stays low-entanglement --
    required for efficient MPS execution of the block at large volume."""
    import scipy.optimize
    from .wavepacket import lsq_init, params_from_vector

    eng = BlockEngine(lat, center, n_layers, max_offset=max_offset)
    offsets = eng.offsets
    rng = np.random.default_rng(seed)

    def cost_grad(vec):
        f, g = eng.fidelity_and_grad(vac_state, target, vec)
        return 1.0 - f + l2 * np.dot(vec, vec), -g + 2 * l2 * vec

    if inits is None:
        direction = lsq_init(lat, vac_state, target, center, offsets, n_layers)
        inits = [a * direction + 0.02 * rng.standard_normal(direction.size)
                 for a in (0.5, 1.0, 1.6)]
        inits.append(0.2 * rng.standard_normal(direction.size))

    best = None
    for x0 in inits:
        res = scipy.optimize.minimize(cost_grad, x0, jac=True,
                                      method="L-BFGS-B",
                                      options={"maxiter": maxiter})
        if best is None or res.fun < best.fun:
            best = res
    return {"params": params_from_vector(best.x, offsets, n_layers),
            "fidelity": 1.0 - best.fun, "vector": best.x,
            "offsets": offsets, "n_layers": n_layers, "keys": eng.keys}
