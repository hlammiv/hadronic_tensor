"""Scalable meson-wavepacket preparation circuit (M4c).

A local unitary block applied on top of the (volume-transferred) vacuum
ansatz.  Generators are gauge-invariant bond operators in a window around
the packet center:

  hop quadrature      (X X + Y Y) sigma^z / 4     (real amplitude)
  current quadrature  (X Y - Y X) sigma^z / 4     (imaginary amplitude --
                                                   to leading order this IS
                                                   the interpolating current,
                                                   so U ~ 1 - i sum phi_b J_b
                                                   creates the meson channel)
plus per-site Z and per-link X rotations for phase/string sculpting.  Every
generator commutes with all Gauss operators, so the block cannot leave the
physical sector at any parameter value.

Angles are keyed by (layer, kind, OFFSET from the packet center) so a block
trained at small volume (against the exact band-projected wavepacket)
translates verbatim to any larger lattice: locality is enforced by
construction because only offsets present at the training volume exist.
Boosted packets (k0 != 0) are trained per k0 -- train at a k0 on the shared
momentum grid of the training and production volumes (e.g. pi/2).
"""

import numpy as np
import scipy.optimize
from qiskit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import Statevector

from .lattice import Z2Lattice
from .pauli import pauli_sum
from . import stateprep

KINDS = ("cur", "hop", "site", "link")


def _bond_generator(lat: Z2Lattice, bond: int, kind: str):
    """3-qubit generator as (compact SparsePauliOp, [qubit list]) so circuit
    synthesis never touches the full-width Hilbert space.
    Local qubit order: 0 = site a, 1 = site b, 2 = link."""
    qa, qb = lat.site_qubit(bond), lat.site_qubit(bond + 1)
    ql = lat.link_qubit(bond)
    if lat.is_seam(bond):  # parity trick: valid on physical states
        from .trotter import seam_sign
        s = seam_sign(lat)
    else:
        s = 1
    if kind == "hop":
        terms = [({0: "X", 1: "X", 2: "Z"}, s / 4), ({0: "Y", 1: "Y", 2: "Z"}, s / 4)]
    else:  # cur
        terms = [({0: "X", 1: "Y", 2: "Z"}, s / 4), ({0: "Y", 1: "X", 2: "Z"}, -s / 4)]
    return pauli_sum(3, terms), [qa, qb, ql]


def _bond_generator_full(lat: Z2Lattice, bond: int, kind: str):
    """Same generator as a full-width SparsePauliOp (for classical linear
    algebra in lsq_init)."""
    op3, qubits = _bond_generator(lat, bond, kind)
    terms = []
    for lab, c in zip(op3.paulis.to_labels(), op3.coeffs):
        ops = {qubits[q]: lab[2 - q] for q in range(3) if lab[2 - q] != "I"}
        terms.append((ops, complex(c).real))
    return pauli_sum(lat.n_qubits, terms)


def window_offsets(train_ns: int) -> list[int]:
    """All distinct ring offsets at the training volume (the whole small
    ring); on larger lattices these become a local window."""
    return list(range(-(train_ns // 2 - 1), train_ns // 2 + 1))


def block_circuit(lat: Z2Lattice, center: int, params: dict) -> QuantumCircuit:
    """params: {(layer, kind, offset): angle}. center = staggered site index
    of the packet center (bond offsets measured from bond `center`)."""
    qc = QuantumCircuit(lat.n_qubits)
    layers = sorted({k[0] for k in params})
    for l in layers:
        for kind in KINDS:
            keys = sorted(k for k in params if k[0] == l and k[1] == kind)
            for (_, _, off) in keys:
                ang = params[(l, kind, off)]
                if isinstance(ang, (int, float)) and abs(ang) < 1e-14:
                    continue
                if kind in ("cur", "hop"):
                    op, qubits = _bond_generator(lat, (center + off) % lat.ns, kind)
                    qc.append(PauliEvolutionGate(op, time=ang), qubits)
                elif kind == "site":
                    qc.rz(ang, lat.site_qubit(center + off))
                else:  # link
                    qc.rx(ang, lat.link_qubit(center + off))
    return qc


def params_from_vector(vec: np.ndarray, offsets: list[int], n_layers: int) -> dict:
    p, i = {}, 0
    for l in range(n_layers):
        for kind in KINDS:
            for off in offsets:
                p[(l, kind, off)] = vec[i]
                i += 1
    return p


def n_params(offsets: list[int], n_layers: int) -> int:
    return n_layers * len(KINDS) * len(offsets)


def full_circuit(lat: Z2Lattice, vac_thetas: np.ndarray, center: int,
                 params: dict) -> QuantumCircuit:
    qc = stateprep.vacuum_ansatz(lat, vac_thetas)
    qc.compose(block_circuit(lat, center, params), inplace=True)
    return qc


def _template(lat: Z2Lattice, center: int, offsets, n_layers):
    """Parameterized block circuit built and SYNTHESIZED once: transpiling to
    standard gates keeps angles as linear ParameterExpressions, so per-eval
    assign_parameters is pure substitution (no PauliEvolutionGate
    re-synthesis, which otherwise dominates the optimizer runtime)."""
    from qiskit import transpile
    from qiskit.circuit import Parameter

    names, syms = [], {}
    for l in range(n_layers):
        for kind in KINDS:
            for off in offsets:
                key = (l, kind, off)
                names.append(key)
                syms[key] = Parameter(f"p_{l}_{kind}_{off}")
    qc = block_circuit(lat, center, syms)
    qc = transpile(qc, basis_gates=["cx", "rz", "rx", "ry", "h", "sx", "x"],
                   optimization_level=0)
    return qc, names


def block_state(vac_sv: Statevector, tpl, names, vec) -> np.ndarray:
    return np.asarray(vac_sv.evolve(
        tpl.assign_parameters(_order_vec(tpl, names, vec))))


def _order_vec(tpl, names, vec):
    """Map our (layer, kind, offset) vector onto the template's alphabetical
    Parameter ordering."""
    by_name = {f"p_{l}_{k}_{o}": v for (l, k, o), v in zip(names, vec)}
    return [by_name[p.name] for p in tpl.parameters]


def lsq_init(lat: Z2Lattice, vac_state: np.ndarray, target: np.ndarray,
             center: int, offsets, n_layers: int) -> np.ndarray:
    """Linearized initialization: to first order the block creates
    -i sum theta_i O_i |vac>; choose real theta maximizing the generalized
    Rayleigh quotient |<target| sum theta_i M_i>|^2 / ||sum theta_i M_i||^2
    with M_i = -i O_i |vac> (vacuum component projected out).
    Only layer-0 cur/hop angles are seeded; everything else starts at 0."""
    import scipy.linalg

    cols, keys = [], []
    for kind in ("cur", "hop"):
        for off in offsets:
            full = _bond_generator_full(lat, (center + off) % lat.ns, kind)
            m = -1j * (full.to_matrix(sparse=True) @ vac_state)
            m = m - vac_state * np.vdot(vac_state, m)
            cols.append(m)
            keys.append((0, kind, off))
    a = np.array([np.vdot(target, m) for m in cols])
    G = np.array([[np.vdot(mi, mj) for mj in cols] for mi in cols])
    A = np.real(np.outer(np.conj(a), a))
    vals, vecs = scipy.linalg.eigh((A + A.T) / 2, np.real(G) + 1e-10 * np.eye(len(a)))
    theta = np.real(vecs[:, -1])
    theta = theta / (np.abs(theta).max() + 1e-12)
    full_vec = np.zeros(n_params(offsets, n_layers))
    lut = {k: i for i, k in enumerate(
        [(l, kind, off) for l in range(n_layers) for kind in KINDS
         for off in offsets])}
    for k, th in zip(keys, theta):
        full_vec[lut[k]] = th
    return full_vec


def train(lat: Z2Lattice, vac_thetas: np.ndarray, target: np.ndarray,
          center: int, n_layers: int = 2, seed: int = 11,
          maxiter: int = 300, amplitudes=(0.5, 1.0, 1.6)) -> dict:
    """Maximize |<target|U(params)|vac_ansatz>|^2 at the training volume.

    Starts from the linearized (least-squares) direction at several overall
    amplitudes -- the small-angle limit of the block IS the interpolating
    operator, so this lands in the right basin immediately."""
    offsets = window_offsets(lat.ns)
    rng = np.random.default_rng(seed)
    vac_sv = Statevector.from_instruction(stateprep.vacuum_ansatz(lat, vac_thetas))
    vac_state = np.asarray(vac_sv)
    tpl, names = _template(lat, center, offsets, n_layers)

    def cost(vec):
        return 1.0 - abs(np.vdot(target, block_state(vac_sv, tpl, names, vec))) ** 2

    direction = lsq_init(lat, vac_state, target, center, offsets, n_layers)
    best = None
    for amp in amplitudes:
        x0 = amp * direction + 0.02 * rng.standard_normal(direction.size)
        res = scipy.optimize.minimize(cost, x0, method="L-BFGS-B",
                                      options={"maxiter": maxiter})
        if best is None or res.fun < best.fun:
            best = res
    return {"params": params_from_vector(best.x, offsets, n_layers),
            "fidelity": 1.0 - best.fun, "vector": best.x,
            "offsets": offsets, "n_layers": n_layers}
