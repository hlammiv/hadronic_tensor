"""M2 validation: Trotter order + seam trick in circuits, Clifford-point
stabilizer agreement, and all three measurement protocols vs exact ED.

All at ns=4-6 (8-13 qubits) with generic couplings.
"""

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Clifford, Statevector

from htensor import Z2Lattice, exact
from htensor import hamiltonian as ham
from htensor import currents as cur
from htensor import trotter, measure
from htensor.clifford import to_clifford_circuit

M0, G2, ETA = 0.7, 1.1, 1.3


def circuit_state(psi0: np.ndarray, qc: QuantumCircuit) -> np.ndarray:
    return np.asarray(Statevector(psi0).evolve(qc))


@pytest.fixture(scope="module")
def lat4():
    return Z2Lattice(4, pbc=True)


@pytest.fixture(scope="module")
def vac4(lat4):
    _, vecs = exact.lowest_physical_states(lat4, M0, G2, ETA, k=1)
    return vecs[:, 0]


# ------------------------------------------------------------------ Trotter
def test_trotter_is_second_order(lat4):
    """Global error at fixed t scales as dt^2 for the palindromic step,
    against exact evolution of the FULL (string-dressed) Hamiltonian."""
    psi0 = np.asarray(Statevector.from_instruction(
        trotter.strong_coupling_vacuum_circuit(lat4)))
    H = exact.to_sparse(ham.build_hamiltonian(lat4, M0, G2, ETA))
    t = 1.0
    psi_exact = exact.evolve(H, psi0, t)
    errs = []
    for n in (8, 16, 32):
        psi_n = circuit_state(psi0, trotter.trotter_circuit(lat4, M0, G2, ETA, t, n))
        errs.append(np.linalg.norm(psi_n - psi_exact))
    assert 3.3 < errs[0] / errs[1] < 4.7
    assert 3.3 < errs[1] / errs[2] < 4.7


def test_seam_trick_needs_physical_state(lat4):
    """Parity-form seam gate is exact on Gauss-law states; on a
    Gauss-violating state it must NOT reproduce string-H evolution."""
    psi_phys = np.asarray(Statevector.from_instruction(
        trotter.strong_coupling_vacuum_circuit(lat4)))
    # add one unpaired fermion: X on matter qubit 0 flips fermion parity
    bad = QuantumCircuit(lat4.n_qubits)
    bad.x(lat4.site_qubit(0))
    psi_bad = circuit_state(psi_phys, bad)

    H = exact.to_sparse(ham.build_hamiltonian(lat4, M0, G2, ETA))
    t, n = 1.0, 64
    qc = trotter.trotter_circuit(lat4, M0, G2, ETA, t, n)

    err_phys = np.linalg.norm(circuit_state(psi_phys, qc) - exact.evolve(H, psi_phys, t))
    err_bad = np.linalg.norm(circuit_state(psi_bad, qc) - exact.evolve(H, psi_bad, t))
    assert err_phys < 5e-3
    assert err_bad > 0.05  # wrong parity sector -> wrong seam sign -> O(1) error


@pytest.mark.parametrize("ns", [4, 6])
def test_trotter_preserves_gauss_sector(ns):
    lat = Z2Lattice(ns, pbc=True)
    psi0 = np.asarray(Statevector.from_instruction(
        trotter.strong_coupling_vacuum_circuit(lat)))
    psi_t = circuit_state(psi0, trotter.trotter_circuit(lat, M0, G2, ETA, 0.9, 6))
    for n in range(ns):
        g = exact.expectation(ham.gauss_operator(lat, n), psi_t).real
        assert np.isclose(g, 1.0, atol=1e-10)


# ------------------------------------------------------------------ Clifford
def test_clifford_point_stabilizer_agreement(lat4):
    """Same generator code at a Clifford point: stabilizer == statevector."""
    from qiskit_aer import AerSimulator

    m0, g2, eta, dt = 1.0, 1.0, 2.0, np.pi
    qc = trotter.strong_coupling_vacuum_circuit(lat4)
    qc.compose(trotter.trotter_circuit(lat4, m0, g2, eta, 2 * dt, 2), inplace=True)
    cliff = to_clifford_circuit(qc)
    Clifford(cliff)  # must be representable

    obs = [ham.gauss_operator(lat4, n) for n in range(4)] + \
          [cur.charge_density(lat4, v) for v in range(4)]
    sv = Statevector.from_instruction(qc)
    sim = AerSimulator(method="stabilizer", max_parallel_threads=4)
    for op in obs:
        run = cliff.copy()
        run.save_expectation_value(op, list(range(lat4.n_qubits)))
        res = sim.run(run).result().data()["expectation_value"]
        assert np.isclose(res, complex(sv.expectation_value(op)).real, atol=1e-8)


def test_clifford_rejects_generic_angle(lat4):
    qc = trotter.trotter_circuit(lat4, M0, G2, ETA, 1.0, 1)
    with pytest.raises(ValueError):
        to_clifford_circuit(qc)


# ------------------------------------------------------------------ protocols
def _exact_correlator(lat, psi, later_op, earlier_op, times):
    H = exact.to_sparse(ham.build_hamiltonian(lat, M0, G2, ETA))
    return exact.two_current_correlator(
        H, exact.to_sparse(later_op), exact.to_sparse(earlier_op), psi, np.asarray(times))


@pytest.mark.parametrize("mu,nu", [(0, 0), (1, 0), (0, 1), (1, 1)])
def test_hadamard_protocol_vs_ed(lat4, vac4, mu, nu):
    times = [0.0, 0.6, 1.3]
    probes = ([cur.charge_density(lat4, v) for v in range(4)] if mu == 0
              else [cur.bond_current(lat4, b, ETA) for b in range(4)])
    insert = cur.charge_density(lat4, 0) if nu == 0 else cur.bond_current(lat4, 1, ETA)
    factory = trotter.make_evolution_factory(lat4, M0, G2, ETA, kind="exact")

    data = measure.hadamard_correlator_sv(vac4, insert, probes, factory, times)
    for j, probe in enumerate(probes):
        c_ed = _exact_correlator(lat4, vac4, probe, insert, times)
        assert np.allclose(data.correlator[:, j], c_ed, atol=1e-9), (mu, nu, j)


def test_hadamard_protocol_with_trotter_evolution(lat4, vac4):
    """End-to-end pipeline with real Trotter circuits (finite-dt tolerance)."""
    times = [0.5, 1.0]
    probes = [cur.charge_density(lat4, v) for v in range(4)]
    insert = cur.charge_density(lat4, 0)
    factory = trotter.make_evolution_factory(lat4, M0, G2, ETA, "trotter", dt_target=0.02)
    data = measure.hadamard_correlator_sv(vac4, insert, probes, factory, times)
    for j, probe in enumerate(probes):
        c_ed = _exact_correlator(lat4, vac4, probe, insert, times)
        assert np.allclose(data.correlator[:, j], c_ed, atol=3e-3), j


@pytest.mark.parametrize("nu", [0, 1])
def test_mf_protocol_vs_ed(lat4, vac4, nu):
    times = [0.0, 0.8]
    probes = [cur.charge_density(lat4, v) for v in range(4)]
    insert = cur.charge_density(lat4, 2) if nu == 0 else cur.bond_current(lat4, 0, ETA)
    factory = trotter.make_evolution_factory(lat4, M0, G2, ETA, kind="exact")
    data = measure.mf_correlator_sv(vac4, insert, probes, factory, times)
    for j, probe in enumerate(probes):
        c_ed = _exact_correlator(lat4, vac4, probe, insert, times)
        assert np.allclose(data.correlator[:, j], c_ed, atol=1e-9), (nu, j)


def test_protocols_agree_hadamard_vs_mf(lat4, vac4):
    times = [0.4, 0.9]
    probes = [cur.bond_current(lat4, b, ETA) for b in range(4)]
    insert = cur.bond_current(lat4, 2, ETA)
    factory = trotter.make_evolution_factory(lat4, M0, G2, ETA, kind="exact")
    h = measure.hadamard_correlator_sv(vac4, insert, probes, factory, times)
    m = measure.mf_correlator_sv(vac4, insert, probes, factory, times)
    assert np.allclose(h.correlator, m.correlator, atol=1e-9)
    assert np.allclose(h.connected, m.connected, atol=1e-9)


def test_lly_stencil_vs_connected(lat4, vac4):
    t, eps = 0.7, 0.01
    H = exact.to_sparse(ham.build_hamiltonian(lat4, M0, G2, ETA))
    later, earlier = cur.charge_density(lat4, 1), cur.charge_density(lat4, 0)
    val = measure.lly_connected_re(vac4, H, later, earlier, t, eps)

    c = _exact_correlator(lat4, vac4, later, earlier, [t])[0]
    a = exact.expectation(
        exact.to_sparse(later), exact.evolve(H, vac4, t))
    b = exact.expectation(exact.to_sparse(earlier), vac4)
    expected = (c - a * b).real
    assert np.isclose(val, expected, atol=5e-3)
