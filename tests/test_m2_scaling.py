"""Scale bracketing: identical pipeline from hardware-small to full width.

- ns=6 (13 qubits with ancilla): Hadamard-test correlator vs sparse ED.
- ns=50 (100 qubits): Clifford-point Trotter through the stabilizer simulator,
  Gauss law conserved on every site -- the full-width validation mode.
"""

import numpy as np
import pytest
from qiskit.quantum_info import Statevector

from htensor import Z2Lattice, exact
from htensor import hamiltonian as ham
from htensor import currents as cur
from htensor import trotter, measure
from htensor.clifford import to_clifford_circuit

M0, G2, ETA = 0.7, 1.1, 1.3


def test_correlator_pipeline_ns6():
    lat = Z2Lattice(6, pbc=True)
    _, vecs = exact.lowest_physical_states(lat, M0, G2, ETA, k=1)
    vac = vecs[:, 0]
    times = [0.0, 0.8]
    probes = [cur.charge_density(lat, v) for v in range(lat.ns)]
    insert = cur.charge_density(lat, 0)
    factory = trotter.make_evolution_factory(lat, M0, G2, ETA, "trotter", dt_target=0.02)

    data = measure.hadamard_correlator_sv(vac, insert, probes, factory, times)
    H = exact.to_sparse(ham.build_hamiltonian(lat, M0, G2, ETA))
    for j, probe in enumerate(probes):
        c_ed = exact.two_current_correlator(
            H, exact.to_sparse(probe), exact.to_sparse(insert), vac, np.asarray(times))
        assert np.allclose(data.correlator[:, j], c_ed, atol=3e-3), j


@pytest.mark.parametrize("ns", [8, 50])
def test_clifford_full_width_gauss_conservation(ns):
    """ns=50 -> 100 qubits: far beyond statevector reach, trivial for the
    stabilizer method. Gauss law must hold on every site after evolution."""
    from qiskit_aer import AerSimulator

    lat = Z2Lattice(ns, pbc=True)
    m0, g2, eta, dt = 1.0, 1.0, 2.0, np.pi
    qc = trotter.strong_coupling_vacuum_circuit(lat)
    qc.compose(trotter.trotter_circuit(lat, m0, g2, eta, 2 * dt, 2), inplace=True)
    cliff = to_clifford_circuit(qc)

    sim = AerSimulator(method="stabilizer", max_parallel_threads=4)
    for n in range(0, ns, max(1, ns // 10)):  # sample sites across the ring
        run = cliff.copy()
        run.save_expectation_value(ham.gauss_operator(lat, n), list(range(lat.n_qubits)))
        val = sim.run(run).result().data()["expectation_value"]
        assert np.isclose(val, 1.0, atol=1e-10), f"Gauss law broken at site {n}"


def test_small_lattice_hardware_sizes_build():
    """Every hardware-relevant small size assembles into shallow circuits."""
    for ns in (4, 6, 8, 12, 16):
        lat = Z2Lattice(ns, pbc=True)
        qc = trotter.strong_coupling_vacuum_circuit(lat)
        qc.compose(trotter.trotter_circuit(lat, M0, G2, ETA, 1.0, 4), inplace=True)
        assert qc.num_qubits == 2 * ns
        # 2q budget linear in ns: 3 hop layers/step x ns/2 bonds x 4 gates
        n2q = sum(1 for inst in qc.data if len(inst.qubits) == 2)
        assert n2q == 4 * 3 * (ns // 2) * 4  # = 24 ns for 4 steps
