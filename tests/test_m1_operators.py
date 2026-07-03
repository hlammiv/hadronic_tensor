"""M1 validation: conventions, Gauss law, seam trick, continuity, ED sector.

Couplings are deliberately generic (m0=0.7, g2=1.1, eta=1.3) so coefficient
errors cannot hide behind symmetric values.
"""

import numpy as np
import pytest
import scipy.sparse.linalg as spla

from htensor import Z2Lattice, exact
from htensor import hamiltonian as ham
from htensor import currents as cur
from htensor.pauli import pauli_term, pauli_sum

M0, G2, ETA = 0.7, 1.1, 1.3


def spnorm(op):
    """Frobenius norm of a SparsePauliOp via its Pauli coefficients."""
    s = op.simplify()
    return float(np.linalg.norm(s.coeffs))


def comm(a, b):
    return (a @ b - b @ a).simplify()


# ---------------------------------------------------------------- conventions
def test_pauli_term_convention():
    z0 = pauli_term(2, {0: "Z"}, 1.0).to_matrix()
    assert np.allclose(np.diag(z0), [1, -1, 1, -1])  # qubit 0 = fast index
    z1 = pauli_term(2, {1: "Z"}, 1.0).to_matrix()
    assert np.allclose(np.diag(z1), [1, 1, -1, -1])


def test_vacuum_kron_ordering():
    lat = Z2Lattice(2, pbc=False)  # qubits: site0, link, site1 -> |0>,|+>,|1>
    psi = exact.strong_coupling_vacuum(lat)
    n0 = exact.expectation(pauli_term(3, {0: "Z"}, 1.0), psi)
    nl = exact.expectation(pauli_term(3, {1: "X"}, 1.0), psi)
    n1 = exact.expectation(pauli_term(3, {2: "Z"}, 1.0), psi)
    assert np.allclose([n0, nl, n1], [1.0, 1.0, -1.0])


# ---------------------------------------------------------------- Gauss law
@pytest.mark.parametrize("ns,pbc", [(4, True), (6, True), (4, False), (5, False)])
def test_gauss_algebra(ns, pbc):
    lat = Z2Lattice(ns, pbc=pbc)
    H = ham.build_hamiltonian(lat, M0, G2, ETA)
    gs = [ham.gauss_operator(lat, n) for n in range(ns)]
    for n, g in enumerate(gs):
        assert spnorm((g @ g) - pauli_term(lat.n_qubits, {}, 1.0)) < 1e-12
        assert spnorm(comm(g, H)) < 1e-12, f"[G_{n}, H] != 0"
        for g2 in gs:
            assert spnorm(comm(g, g2)) < 1e-12


@pytest.mark.parametrize("ns", [4, 6])
def test_gauss_product_is_parity(ns):
    lat = Z2Lattice(ns, pbc=True)
    prod = ham.gauss_operator(lat, 0)
    for n in range(1, ns):
        prod = (prod @ ham.gauss_operator(lat, n)).simplify()
    expected = (-1) ** (ns // 2) * ham.fermion_parity(lat)
    assert spnorm(prod - expected) < 1e-12


# ---------------------------------------------------------------- seam trick
@pytest.mark.parametrize("ns", [4, 6])
def test_seam_hop_parity_identity(ns):
    """String-dressed seam hop == -P_f (XX+YY) sigma^z /4 * eta on seam pair."""
    lat = Z2Lattice(ns, pbc=True)
    seam = ns - 1
    qa, qb, ql = lat.site_qubit(seam), lat.site_qubit(0), lat.link_qubit(seam)
    bare = pauli_sum(
        lat.n_qubits,
        [({qa: "X", qb: "X", ql: "Z"}, ETA / 4), ({qa: "Y", qb: "Y", ql: "Z"}, ETA / 4)],
    )
    expected = (-1.0) * (ham.fermion_parity(lat) @ bare)
    assert spnorm(ham.hop_term(lat, seam, ETA) - expected) < 1e-12


# ---------------------------------------------------------------- currents
@pytest.mark.parametrize("ns,pbc", [(4, True), (6, True), (5, False)])
def test_continuity_equation(ns, pbc):
    """i[H, J^0(v)] = -(J^1_{v+1/2} - J^1_{v-1/2}) on interior/all sites."""
    lat = Z2Lattice(ns, pbc=pbc)
    H = ham.build_hamiltonian(lat, M0, G2, ETA)
    sites = range(ns) if pbc else range(1, ns - 1)
    for v in sites:
        lhs = (1j * comm(H, cur.charge_density(lat, v))).simplify()
        rhs = (-(cur.bond_current(lat, v, ETA) - cur.bond_current(lat, v - 1, ETA))).simplify()
        assert spnorm(lhs - rhs) < 1e-12, f"continuity fails at v={v}"


@pytest.mark.parametrize("ns", [4, 6])
def test_currents_gauge_invariant(ns):
    lat = Z2Lattice(ns, pbc=True)
    gs = [ham.gauss_operator(lat, n) for n in range(ns)]
    for v in range(ns):
        for g in gs:
            assert spnorm(comm(g, cur.charge_density(lat, v))) < 1e-12
            assert spnorm(comm(g, cur.bond_current(lat, v, ETA))) < 1e-12


def test_total_charge_conserved():
    lat = Z2Lattice(6, pbc=True)
    H = ham.build_hamiltonian(lat, M0, G2, ETA)
    assert spnorm(comm(H, ham.total_charge(lat))) < 1e-12


# ---------------------------------------------------------------- vacuum & ED
@pytest.mark.parametrize("ns", [4, 6])
def test_strong_coupling_vacuum_sector(ns):
    lat = Z2Lattice(ns, pbc=True)
    psi = exact.strong_coupling_vacuum(lat)
    assert np.allclose(np.linalg.norm(psi), 1.0)
    for n in range(ns):
        assert np.isclose(exact.expectation(ham.gauss_operator(lat, n), psi).real, 1.0)
    assert abs(exact.expectation(ham.total_charge(lat), psi)) < 1e-12
    pf = exact.expectation(ham.fermion_parity(lat), psi).real
    assert np.isclose(pf, (-1) ** (ns // 2))


@pytest.mark.parametrize("ns", [4, 6])
def test_ed_ground_state_physical(ns):
    lat = Z2Lattice(ns, pbc=True)
    energies, vecs = exact.lowest_physical_states(lat, M0, G2, ETA, k=3)
    v0 = vecs[:, 0]
    for n in range(ns):
        assert np.isclose(exact.expectation(ham.gauss_operator(lat, n), v0).real, 1.0, atol=1e-8)
    assert abs(exact.expectation(ham.total_charge(lat), v0)) < 1e-8
    # Interacting vacuum carries a staggered charge profile <J0(v)> = (-1)^v rho
    # (pair fluctuations); C symmetry + translation force the alternation and
    # the vanishing of the cell average -- but NOT pointwise zero, so the
    # disconnected piece <J0(x)><J0(y)> must be subtracted pointwise (PLAN 2.9).
    rho = [exact.expectation(cur.charge_density(lat, v), v0).real for v in range(ns)]
    for v in range(ns):
        assert np.isclose(rho[v], (-1) ** v * rho[0], atol=1e-8)
    assert abs(sum(rho)) < 1e-8
    assert energies[0] < energies[1] - 1e-10  # gapped


# ---------------------------------------------------------------- correlator
def test_correlator_hermiticity_and_t0():
    lat = Z2Lattice(4, pbc=True)
    Hs = exact.to_sparse(ham.build_hamiltonian(lat, M0, G2, ETA))
    _, vecs = exact.lowest_physical_states(lat, M0, G2, ETA, k=1)
    psi = vecs[:, 0]
    A = exact.to_sparse(cur.charge_density(lat, 0))
    B = exact.to_sparse(cur.charge_density(lat, 1))
    ts = np.linspace(0.0, 2.0, 5)
    c_ba = exact.two_current_correlator(Hs, B, A, psi, ts)
    # t=0 value against direct matrix element
    direct = np.vdot(psi, (B @ (A @ psi)))
    assert np.isclose(c_ba[0], direct, atol=1e-10)
    # <psi|A B(t)|psi> = conj(<psi|B(t) A|psi>) for Hermitian A, B
    c_ab_rev = exact.two_current_correlator(Hs, A, B, psi, ts)  # <B(t') ... > not needed;
    # instead check reversed-order via conjugation identity:
    #   <psi| A e^{iHt} B e^{-iHt} |psi> = conj(<psi| e^{iHt} B e^{-iHt} A |psi>)
    Hc = Hs.tocsc()
    for t, c in zip(ts, c_ba):
        ket = spla.expm_multiply(-1j * t * Hc, psi)
        bt_psi = spla.expm_multiply(1j * t * Hc, B @ ket)  # B(t)|psi> in Schrodinger pieces
        rev = np.vdot(A @ psi, bt_psi)  # <psi|A B(t)|psi>
        assert np.isclose(rev, np.conj(c), atol=1e-9)
