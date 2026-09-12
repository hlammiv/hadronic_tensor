"""M5: four-component tensor machinery for the next hardware campaign.

  * parity-ladder controlled-Pauli gadget == naive controlled string
  * Hadamard protocol with the ladder gadget vs ED for all (mu, nu)
  * seam-bond J^1: JW string == seam_sign x 3-local form on physical states
  * two-basis (XYA/XYB) bond-current readout estimator from sampled bits
  * parametric Trotter segment: assigned == numeric, eps == identity,
    physics/mirror skeleton identical after transpilation (O1/O3, cz/rzz)
  * tensor assembly at ns=6 from ED: continuity, parity, windowed Ward
"""

import numpy as np
import pytest
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Operator, Statevector
from qiskit.transpiler import CouplingMap

from htensor import Z2Lattice, exact, trotter, measure, readout, tensor, analysis
from htensor import hamiltonian as ham
from htensor import currents as cur

M0, G2, ETA = 0.7, 1.1, 1.3


@pytest.fixture(scope="module")
def lat4():
    return Z2Lattice(4, pbc=True)


@pytest.fixture(scope="module")
def vac4(lat4):
    _, vecs = exact.lowest_physical_states(lat4, M0, G2, ETA, k=1)
    return vecs[:, 0]


def _ops_term(lat, bond, term):
    a, l, b = lat.site_qubit(bond), lat.link_qubit(bond), lat.site_qubit(bond + 1)
    return {a: "Y", l: "Z", b: "X"} if term == 1 else {a: "X", l: "Z", b: "Y"}


# ------------------------------------------------------------------ gadget
@pytest.mark.parametrize("term", [1, 2])
@pytest.mark.parametrize("accumulate", ["site", "link", "pendant"])
def test_parity_ladder_equals_controlled_pauli(lat4, term, accumulate):
    n = lat4.n_qubits
    ops = _ops_term(lat4, 1, term)
    ref = QuantumCircuit(n + 1)
    measure.controlled_pauli(ref, n, ops)
    lad = QuantumCircuit(n + 1)
    measure.j1_gadget(lad, n, ops, accumulate)
    assert Operator(ref).equiv(Operator(lad))
    n2q = sum(1 for inst in lad.data if inst.operation.num_qubits == 2)
    assert n2q == 5


def test_ladder_rejects_bad_path(lat4):
    ops = _ops_term(lat4, 1, 1)
    qc = QuantumCircuit(lat4.n_qubits + 1)
    with pytest.raises(ValueError):
        measure.parity_ladder_controlled_pauli(qc, lat4.n_qubits, ops, [(0, 1)])


def _exact_correlator(lat, psi, later_op, earlier_op, times):
    H = exact.to_sparse(ham.build_hamiltonian(lat, M0, G2, ETA))
    return exact.two_current_correlator(
        H, exact.to_sparse(later_op), exact.to_sparse(earlier_op), psi, np.asarray(times))


@pytest.mark.parametrize("mu,nu", [(0, 0), (1, 0), (0, 1), (1, 1)])
def test_hadamard_ladder_gadget_vs_ed(lat4, vac4, mu, nu):
    times = [0.0, 0.6, 1.3]
    probes = ([cur.charge_density(lat4, v) for v in range(4)] if mu == 0
              else [cur.bond_current(lat4, b, ETA) for b in range(4)])
    insert = cur.charge_density(lat4, 0) if nu == 0 else cur.bond_current(lat4, 1, ETA)
    factory = trotter.make_evolution_factory(lat4, M0, G2, ETA, kind="exact")
    data = measure.hadamard_correlator_sv(vac4, insert, probes, factory, times,
                                         gadget=measure.make_ladder_gadget("site"))
    for j, probe in enumerate(probes):
        c_ed = _exact_correlator(lat4, vac4, probe, insert, times)
        assert np.allclose(data.correlator[:, j], c_ed, atol=1e-9), (mu, nu, j)


def test_seam_current_parity_replacement(lat4, vac4):
    """On the Hadamard-test state (both branches physical, definite fermion
    parity) the seam bond current with its JW string equals seam_sign times
    the bulk 3-local form -- the identity the hardware readout uses."""
    n = lat4.n_qubits
    seam = lat4.ns - 1
    a, l, b = lat4.site_qubit(seam), lat4.link_qubit(seam), lat4.site_qubit(0)
    qc = measure.hadamard_test_circuit(n, _ops_term(lat4, 1, 1),
                                       trotter.trotter_circuit(lat4, M0, G2, ETA, 0.7, 2))
    sv = Statevector(np.kron([1, 0], vac4)).evolve(qc)
    full = cur.bond_current(lat4, seam, ETA)
    from htensor.pauli import pauli_sum
    local = pauli_sum(n, [({a: "Y", b: "X", l: "Z"}, ETA / 4),
                          ({a: "X", b: "Y", l: "Z"}, -ETA / 4)])
    s = trotter.seam_sign(lat4)
    for anc in ("I", "X", "Y"):
        e_full = sv.expectation_value(measure._with_ancilla(full, anc))
        e_loc = sv.expectation_value(measure._with_ancilla(local, anc))
        assert np.isclose(e_full, s * e_loc, atol=1e-12), anc


# ------------------------------------------------------------------ readout
def test_two_basis_probe_estimator_sampled(lat4):
    from qiskit_aer import AerSimulator
    n = lat4.n_qubits
    anc = n
    # a generic physical state: strong-coupling vacuum, J^0 insertion, 1 step
    base = measure.hadamard_test_circuit(n, {lat4.site_qubit(0): "Z"},
                                         trotter.trotter_circuit(lat4, M0, G2, ETA, 0.5, 1))
    prep = QuantumCircuit(n + 1)
    prep.compose(trotter.strong_coupling_vacuum_circuit(lat4), range(n), inplace=True)
    prep.compose(base, inplace=True)
    sv = Statevector(prep)
    shots = 200_000
    sim = AerSimulator(seed_simulator=11)
    results = {}
    for setting in readout.SETTINGS:
        qc = prep.copy()
        readout.apply_readout_rotations(qc, readout.basis_map(lat4, setting, anc, "X"))
        qc.measure_all()
        counts = sim.run(transpile(qc, sim), shots=shots).result().get_counts()
        keys = np.array(list(counts))
        reps = np.array(list(counts.values()))
        bits = np.array([[int(ch) for ch in k[::-1]] for k in keys])  # col q = qubit q
        bits = np.repeat(bits, reps, axis=0)
        results[setting] = readout.estimate_probes(bits, lat4, setting, anc, ETA)
    r = results["Z"]
    for v in range(4):
        ex = sv.expectation_value(measure._with_ancilla(cur.charge_density(lat4, v), "X")).real
        assert abs(r["xJ0"][v] - ex) < 5 * r["xJ0_err"][v] + 1e-3
    for setting in ("XYA", "XYB"):
        r = results[setting]
        for bnd in range(4):
            ops = _ops_term(lat4, bnd, r["term"][bnd])
            if lat4.is_seam(bnd):
                ops = {**ops, **{q: "Z" for q in lat4.seam_string_qubits()}}
            from htensor.pauli import pauli_term
            T = pauli_term(n, ops, 1.0)
            ex = sv.expectation_value(measure._with_ancilla(T, "X")).real
            assert abs(r["xT"][bnd] - ex) < 5 * r["xT_err"][bnd] + 1e-3, (setting, bnd)
    val, err = readout.combine_j1(results["XYA"], results["XYB"], ETA)
    for bnd in range(4):
        ex = sv.expectation_value(
            measure._with_ancilla(cur.bond_current(lat4, bnd, ETA), "X")).real
        assert abs(val[bnd] - ex) < 5 * err[bnd] + 1e-3, bnd


# ------------------------------------------------------------------ parametric step
def _skeleton(qc):
    return [(inst.operation.name, tuple(qc.find_bit(q).index for q in inst.qubits))
            for inst in qc.data if inst.operation.num_qubits == 2]


def test_parametric_segment_matches_numeric_and_mirror_is_identity(lat4):
    step, p = trotter.parametric_step(lat4, M0, G2, ETA)
    num = trotter.trotter_step(lat4, M0, G2, ETA, 0.5)
    assert Operator(step.assign_parameters({p: 0.5})).equiv(Operator(num))
    mir = Operator(step.assign_parameters({p: 1e-8})).data
    assert np.allclose(mir, np.eye(mir.shape[0]), atol=1e-5)


@pytest.mark.parametrize("basis", [["rz", "sx", "x", "cz"], ["rz", "sx", "x", "rzz", "cz"]])
@pytest.mark.parametrize("level", [1, 3])
def test_parametric_transpile_preserves_skeleton(lat4, basis, level):
    n = lat4.n_qubits
    ring = CouplingMap([[q, (q + 1) % n] for q in range(n)] + [[(q + 1) % n, q] for q in range(n)])
    step, p = trotter.parametric_step(lat4, M0, G2, ETA)
    isa = transpile(step, coupling_map=ring, basis_gates=basis, initial_layout=list(range(n)),
                    optimization_level=level, seed_transpiler=7)
    phys = isa.assign_parameters({p: 0.5})
    mirr = isa.assign_parameters({p: 1e-8})
    assert _skeleton(phys) == _skeleton(mirr) and len(_skeleton(phys)) > 0
    num = trotter.trotter_step(lat4, M0, G2, ETA, 0.5)
    assert Operator.from_circuit(phys).equiv(Operator(num))


# ------------------------------------------------------------------ tensor assembly (ED, ns=6)
def _ed_grids(lat, psi, vac, center, times):
    """Connected-minus-connected four-component grids from ED, plus raw."""
    H = exact.to_sparse(ham.build_hamiltonian(lat, M0, G2, ETA))
    ns = lat.ns
    probes = {0: [cur.charge_density(lat, v) for v in range(ns)],
              1: [cur.bond_current(lat, b, ETA) for b in range(ns)]}
    inserts = {0: cur.charge_density(lat, center), 1: cur.bond_current(lat, center, ETA)}
    g, raw = {}, {}
    for comp in tensor.COMPONENTS:
        mu, nu = int(comp[0]), int(comp[1])
        ins = exact.to_sparse(inserts[nu])
        rows = {}
        for name, state in (("wp", psi), ("vac", vac)):
            c = np.stack([exact.two_current_correlator(H, exact.to_sparse(pr), ins, state, times)
                          for pr in probes[mu]], axis=1)
            one_ins = exact.expectation(inserts[nu], state)
            one_pr = np.stack([[exact.expectation(pr, exact.evolve(H, state, t)) for pr in probes[mu]]
                               for t in times])
            rows[name] = analysis.subtract(c, None, one_pr, one_ins)
            if name == "wp":
                raw[comp] = c
        g[comp] = rows["wp"] - rows["vac"]
    x = {comp: tensor.x_grid(lat, comp, center) for comp in tensor.COMPONENTS}
    return tensor.TensorGrids(lat, center, 0.0, np.asarray(times, float), g, x, raw, {}, {})


@pytest.fixture(scope="module")
def grids6():
    lat = Z2Lattice(6, pbc=True)
    _, vecs = exact.lowest_physical_states(lat, M0, G2, ETA, k=2)
    vac, meson = vecs[:, 0], vecs[:, 1]          # k=0 meson: P (and PC) eigenstate
    times = np.arange(0.0, 1.0001, 0.02)
    return _ed_grids(lat, meson, vac, 2, times)


def test_tensor_continuity_and_parity(grids6):
    cont = tensor.continuity_residual_xt(grids6, use_raw=True, form="differential")
    assert cont["0"] < 1e-2 and cont["1"] < 1e-2, cont       # O(dt^2) central differences
    cint = tensor.continuity_residual_xt(grids6, use_raw=True, form="integral")
    assert cint["0"] < 1e-3 and cint["1"] < 1e-3, cint       # Simpson, exact evolution
    cconn = tensor.continuity_residual_xt(grids6, use_raw=False, form="integral")
    assert cconn["0"] < 1e-3 and cconn["1"] < 1e-3, cconn    # one-point terms cancel
    par = tensor.parity_residuals(grids6)
    assert par["01"] < 1e-8 and par["11"] < 1e-8, par        # exact: bond insertion on the centre
    assert np.isfinite(par["00"]) and np.isfinite(par["10"])


def test_tensor_assembly_and_ward_conversion(grids6):
    q0 = np.linspace(0, 6, 31)
    q1 = 2 * np.pi * np.arange(-1, 2) / grids6.lat.nx
    W = tensor.assemble_W(grids6, q0, q1)
    assert set(W) == set(tensor.COMPONENTS)
    conv = tensor.ward_converted_W11(W["00"][0], q0, q1)
    assert np.isnan(conv[:, 1]).all() and np.isfinite(conv[:, 0]).all()
