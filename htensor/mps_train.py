"""SKETCH (WS2-B phase 2, 2026-09): MPS-based block trainer for very wide
packets (sigma_x = 2, 3 spatial sites) at ns = 40-44, where no exact basis
exists (gf Q=0 dimension 2 C(ns, ns/2): ns=28 -> 80M, ns=40 -> 2.8e11).

Status: state builder + observables implemented and checked against
gf_engine at small ns (see _selftest); the trainer is a stub with the cost
function chosen but NOT exercised on long jobs.  Nothing here is used by
the production chain (scripts/train_packets.py).

Why a different cost.  Below ns ~ 26 the block is trained by fidelity to an
EXACT band-projected target (gf_engine.gf_packet_target).  At ns = 40 the
band states |k> are not available (momentum sectors are non-local in an
MPS), so the target-free cost

    C(theta) = Var_H(psi) / M^2  +  lam_E (E(psi) - E_vac - E(K))^2
               + lam_x sum_v (rho_v(psi) - rho_v^target)^2

is used instead: energy variance selects single-band content (a
band-projected packet with sigma_k has Var_H ~ (v_g sigma_k)^2, i.e. the
minimum achievable, everything else -- vacuum leakage, two-meson content --
raises it), the energy term pins the mean momentum through the dispersion
E(K) (known from data/deep_levels_ns20.npz), and the charge-profile term
rho_v = <J0(v)>_psi - <J0(v)>_vac pins the spatial width to the Gaussian
envelope.  Momentum sign is fixed by the warm start: the gf-trained block
at the largest exact volume (ns = 24-26, same K), zero-padded to the wider
window (scripts/train_packets.py:remap_vector), which already carries the
right chirality; training only has to widen it.

Gradients: quimb supports autodiff through tensor networks (autoray +
jax/torch backends), but the cleanest route for ~250 parameters with a
bond-dimension-capped MPS is the parameter-shift rule per gate
(exact for the rz/rx generators with eigenvalues +-1/2; the bond generators
have eigenvalues {0, +-1/2} so the 4-term shift rule applies), i.e. ~2-4
MPS simulations per parameter per step.  At ns = 44 (88 qubits), chi = 128,
one simulation of vacuum + block is ~5-10 s (backends.prepare_state_mps
timings), so one gradient is ~1 h: a background job of a few hours per
packet, or an SPSA variant when only a few hundred steps are needed.

Circuit geometry on the MPS: qubits in the natural order (site n -> 2n,
link -> 2n+1) make every BULK bond gate a contiguous 3-site gate
(2b, 2b+1, 2b+2) -- no swaps; the seam bond (2ns-2, 2ns-1, 0) is non-local
and is applied with quimb's gate_nonlocal (an MPO with identities in
between).  The wide block never touches the seam (window centred mid-ring),
so only the vacuum layers pay for it.
"""

import numpy as np

from .lattice import Z2Lattice
from .wavepacket import KINDS, _bond_generator
from .stateprep import N_PARAMS_PER_LAYER
from . import hamiltonian as ham


# ------------------------------------------------------------ dense gates
def _bond_unitary(lat: Z2Lattice, bond: int, kind: str, theta: float) -> np.ndarray:
    """exp(-i theta G) as an 8x8 matrix on (site a, site b, link) in
    QISKIT little-endian order (qubits[0] = site a is the least significant
    bit), the convention of block_engine.BlockEngine."""
    op3, _ = _bond_generator(lat, bond, kind)
    G = op3.to_matrix()
    w, V = np.linalg.eigh(G)
    return (V * np.exp(-1j * theta * w)) @ V.conj().T


def _to_quimb_order(U8: np.ndarray) -> np.ndarray:
    """Qiskit little-endian (qubits[0] = LSB) -> quimb big-endian
    (where[0] = most significant) for a 3-qubit gate."""
    T = U8.reshape((2,) * 6)                     # (out2 out1 out0, in2 in1 in0)
    T = T.transpose(2, 1, 0, 5, 4, 3)            # reverse qubit order
    return T.reshape(8, 8)


def _rz(theta):
    return np.diag(np.exp([-0.5j * theta, 0.5j * theta]))


def _rx(theta):
    c, s = np.cos(theta / 2), -1j * np.sin(theta / 2)
    return np.array([[c, s], [s, c]])


# ------------------------------------------------------------ circuits
def build_circuit(lat: Z2Lattice, vac_thetas, center: int, params: dict,
                  link_ref: str = "+", max_bond: int = 128, cutoff: float = 1e-10):
    """quimb CircuitMPS of stateprep.vacuum_ansatz + wavepacket.block_circuit
    (same gate order as block_engine / gf_engine).  params: {(layer, kind,
    offset): angle}."""
    import quimb.tensor as qtn

    ns, nq = lat.ns, lat.n_qubits
    circ = qtn.CircuitMPS(N=nq, max_bond=max_bond, cutoff=cutoff)
    # strong-coupling reference: odd sites |1>, links |+> (or |->)
    for n in range(1, ns, 2):
        circ.apply_gate("X", lat.site_qubit(n))
    for q in lat.link_qubits:
        circ.apply_gate("H", q)
        if link_ref == "-":
            circ.apply_gate("Z", q)

    def hop(bond, theta):
        qa, qb, ql = lat.site_qubit(bond), lat.site_qubit(bond + 1), lat.link_qubit(bond)
        U = _to_quimb_order(_bond_unitary(lat, bond % ns, "hop", theta))
        _apply3(circ, U, (qa, qb, ql))

    th = np.asarray(vac_thetas, dtype=float).reshape(-1, N_PARAMS_PER_LAYER)
    for th_e, th_o, th_m, th_g in th:
        for b in range(0, ns, 2):
            hop(b, th_e)                         # exp(-i th_e (s/4)(XX+YY)Z)
        for b in range(1, ns, 2):
            hop(b, th_o)
        for n in range(ns):
            circ.apply_gate_raw(_rz(-(-1) ** n * th_m), (lat.site_qubit(n),))
        for q in lat.link_qubits:
            circ.apply_gate_raw(_rx(th_g), (q,))
    layers = sorted({k[0] for k in params})
    for l in layers:
        for kind in KINDS:
            for (_, _, off) in sorted(k for k in params if k[0] == l and k[1] == kind):
                ang = params[(l, kind, off)]
                if abs(ang) < 1e-14:
                    continue
                if kind in ("cur", "hop"):
                    b = (center + off) % ns
                    qa, qb, ql = lat.site_qubit(b), lat.site_qubit(b + 1), lat.link_qubit(b)
                    _apply3(circ, _to_quimb_order(_bond_unitary(lat, b, kind, ang)), (qa, qb, ql))
                elif kind == "site":
                    circ.apply_gate_raw(_rz(ang), (lat.site_qubit(center + off),))
                else:
                    circ.apply_gate_raw(_rx(ang), (lat.link_qubit(center + off),))
    return circ


def _apply3(circ, U, where):
    """3-site gate: contiguous in the chain for bulk bonds; the seam bond
    (qubits nq-2, nq-1, 0) goes through the non-local MPO path."""
    where = tuple(int(w) for w in where)
    lo, hi = min(where), max(where)
    if hi - lo == 2:
        circ.apply_gate_raw(U, where)
    else:
        # circ.psi returns a COPY; the non-local seam gate must act on the
        # circuit's own MPS (quimb 1.14: CircuitMPS._psi)
        circ._psi.gate_nonlocal(U.reshape((2,) * 6), where, inplace=True,
                                max_bond=circ.gate_opts.get("max_bond"),
                                cutoff=circ.gate_opts.get("cutoff", 1e-10))


# ------------------------------------------------------------ observables
def _pauli_mpo(lat: Z2Lattice, op):
    """SparsePauliOp -> quimb MPO (sum of Pauli strings via MPO addition)."""
    import quimb as qu
    import quimb.tensor as qtn

    nq = lat.n_qubits
    total = None
    for lab, c in zip(op.paulis.to_labels(), op.coeffs):
        mats = [qu.pauli(lab[nq - 1 - q]) if lab[nq - 1 - q] != "I" else np.eye(2)
                for q in range(nq)]
        mpo = qtn.MPO_product_operator(mats)
        mpo *= complex(c)
        total = mpo if total is None else (total + mpo)
        if total.max_bond() > 64:
            total.compress(cutoff=1e-14)
    return total


def observables(lat: Z2Lattice, circ, m0, g2, eta, want_variance: bool = True) -> dict:
    """<H>, Var_H (via H|psi> as an MPO application), J0 profile."""
    import quimb.tensor as qtn

    psi = circ.psi
    H = _pauli_mpo(lat, ham.build_hamiltonian(lat, m0, g2, eta))
    Hpsi = H.apply(psi, compress=True, cutoff=1e-12)
    e = complex(psi.H @ Hpsi).real
    out = {"E": e}
    if want_variance:
        out["var_H"] = complex(Hpsi.H @ Hpsi).real - e ** 2
    zexp = np.array([complex(psi.H @ psi.gate(np.diag([1.0, -1.0]), lat.site_qubit(n))).real
                     for n in range(lat.ns)])
    out["j0"] = 0.5 * ((-1.0) ** np.arange(lat.ns) - zexp)
    return out


# ------------------------------------------------------------ trainer (stub)
def variance_cost(lat, vac_thetas, center, offsets, n_layers, vec, m0, g2, eta,
                  E_target, rho_target=None, lam_E=1.0, lam_x=0.0, e_vac=None,
                  j0_vac=None, max_bond=128):
    """C(theta) of the module docstring (one MPS simulation)."""
    from .wavepacket import params_from_vector

    circ = build_circuit(lat, vac_thetas, center, params_from_vector(vec, offsets, n_layers),
                         max_bond=max_bond)
    o = observables(lat, circ, m0, g2, eta)
    M2 = E_target ** 2
    c = o["var_H"] / M2 + lam_E * ((o["E"] - e_vac) - E_target) ** 2
    if rho_target is not None and lam_x > 0:
        c += lam_x * float(np.sum((o["j0"] - j0_vac - rho_target) ** 2))
    return c, o


def train_mps(*args, **kwargs):                      # pragma: no cover
    """NOT IMPLEMENTED: parameter-shift / SPSA loop over variance_cost,
    warm-started from the gf-trained block (see module docstring).  Left as
    a stub deliberately -- a run at ns = 40 is a multi-hour background job
    that must be scheduled explicitly."""
    raise NotImplementedError("mps_train.train_mps: sketch only (WS2-B phase 2)")


# ------------------------------------------------------------ self-test
def _selftest(ns: int = 8, max_bond: int = 256):
    """MPS circuit vs gf_engine at ns=8 (exact at this bond dimension)."""
    from . import gf_engine as gfe
    from .wavepacket import params_from_vector

    lat = Z2Lattice(ns, pbc=True)
    th = np.array([3.089753, -3.742056, -0.785397, 1.570794,
                   -0.461068, 0.194181, 0.785389, 3.141611])
    space = gfe.GFSpace(lat)
    vac = gfe.gf_vacuum_state(space, th)
    eng = gfe.GFEngine(space, 4, 2, max_offset=2)
    vec = 0.4 * np.random.default_rng(3).standard_normal(len(eng.keys))
    psi_gf = eng.state(vac, vec)
    circ = build_circuit(lat, th, 4, params_from_vector(vec, eng.offsets, 2), max_bond=max_bond)
    full = circ.psi.to_dense().ravel()
    # quimb's to_dense is big-endian in site index; qiskit indexing is bit q = qubit q
    full = full.reshape((2,) * lat.n_qubits).transpose(list(range(lat.n_qubits))[::-1]).ravel()
    ov = abs(np.vdot(gfe.embed_vector(space, psi_gf), full))
    o = observables(lat, circ, 0.7, 1.1, 1.3)
    H = gfe.GFHamiltonian(space, 0.7, 1.1, 1.3)
    e, s = H.energy_stats(psi_gf)
    return dict(overlap=ov, E_mps=o["E"], E_gf=e, var_mps=o["var_H"], var_gf=s ** 2,
                j0_err=np.abs(o["j0"] - space.j0_profile(psi_gf)).max())
