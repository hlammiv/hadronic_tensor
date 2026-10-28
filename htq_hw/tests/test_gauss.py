"""gauss-midcircuit preset: relay identity, Aer dry run at Ns=6, ladder embedding
with the spare-qubit ancilla, mirror skeleton, local-mode dynamic circuit."""

import warnings

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

from htq_hw import analyze as A
from htq_hw import campaign as CP
from htq_hw import circuits as C
from htq_hw import gauss as G
from htq_hw import sim as S
from htq_hw import target as T
from htq_hw.model import Lattice

warnings.simplefilter("ignore")


def test_relay_copy_identity():
    """a ^= z_site through chains of 0, 1, 2 sites; intermediates restored."""
    lat = Lattice(4)
    for chain in ([], [1], [1, 2]):
        qc = QuantumCircuit(lat.n_wires + 1)
        anc = lat.n_wires
        G.relay_copy(qc, lat, 0, chain, anc)
        ref = QuantumCircuit(lat.n_wires + 1)
        ref.cx(lat.site_qubit(0), anc)
        assert Operator(qc).equiv(Operator(ref)), chain
        assert qc.count_ops().get("cx") == {0: 1, 1: 4, 2: 10}[len(chain)]


def test_gauss_specs_and_plan():
    specs = CP.preset_specs("gauss-midcircuit")
    assert len(specs) == 8 * 2 + 1 and {s.family for s in specs} == {"j0"} and {s.readout for s in specs} == {"Z"}
    lat = Lattice(50)
    assert G.patch_sites(lat, 24) == [22, 23, 24, 25, 26] and G.gauss_ancilla_sites(lat, 24) == [25]
    plan = G.relay_plan(lat, 24)
    assert plan[24] == (25, [25]) and plan[22] == (25, [23, 24, 25]) and plan[25] == (25, [])


def test_dry_run_ns6_syndromes_trivial_and_postselection():
    card = C.load_card()
    lat, center = Lattice(6), 2
    sites = G.patch_sites(lat, center)
    qc = G.gauss_circuit(lat, card, 4, 2.0, center, sites=sites)
    full = C.readout_layer(qc, lat, list(range(lat.n_wires)), "Z")
    assert full.num_clbits == lat.n_wires + 2 * 5 and full.count_ops()["reset"] == 10
    sim = S.make_simulator(full.num_qubits)
    bits = S.sample_bits(sim, full, 100, seed=1)
    acc = G.acceptance(bits, lat, sites, G.n_rounds(4))
    assert np.all(acc["per_round"] == 1.0) and np.all(acc["flip_rate"] == 0.0)
    plain = C.base_circuit(lat, card, "J0", center=center, accumulate="ladder")
    plain.compose(C.trotter_block(lat, 4, 2.0, n_wires=lat.n_wires), inplace=True)
    ref = A.estimate_probes(S.sample_bits(sim, C.readout_layer(plain, lat, list(range(lat.n_wires)), "Z"), 20000, seed=2), lat, "Z")
    got = A.estimate_probes(G.postselect(bits, acc["accepted"], lat), lat, "Z")
    assert np.abs(got["xJ0"] - ref["xJ0"]).max() < 0.25                # 100 shots: 0.1 shot noise


def test_nighthawk_gauss_embedding_and_mirror_skeleton():
    be = T.resolve_backend("fake:nighthawk")
    g = T.Graph.from_backend(be)
    lat = Lattice(50)
    emb = T.choose_embedding(be, 50, 24, gauss_sites=G.gauss_ancilla_sites(lat, 24))
    emb.validate(g)
    assert emb.kind == "ladder" and emb.info["gauss_ancillas"] == {25: 101} and len(emb.layout) == 102
    assert g.has_edge(emb.layout[2 * 25], emb.layout[101])
    plain = T.choose_embedding(be, 50, 24)
    assert plain.kind == "ladder" and len(plain.layout) == 101              # default ladder unchanged
    card = C.load_card()
    phys = G.gauss_circuit(lat, card, 2, 1.0, 24)
    mir = G.gauss_circuit(lat, card, 2, 1.0, 24, mirror=True)
    isa_p, fl_p, _ = T.transpile_isa(phys, be, emb.initial_layout, 1, 7, None, None, "gp", routing="none")
    isa_m, fl_m, _ = T.transpile_isa(mir, be, emb.initial_layout, 1, 7, None, None, "gm", routing="none")
    assert fl_p == emb.layout and fl_m == emb.layout
    assert T.assert_skeleton_equal(isa_p, isa_m) > 0
    assert isa_p.count_ops()["measure"] == 5 and isa_p.count_ops()["reset"] == 5
    assert "reset" in be.target.operation_names and "measure" in be.target.operation_names


def test_local_mode_dynamic_circuit_small_grid():
    """Mid-circuit measure + reset through the SamplerV2 local testing mode
    (Aer MPS, shot by shot) with the ISA pub of a Ns=10 gauss ladder on a 6x8
    generic square lattice; the FakeNighthawk target accepts the same
    instructions but Aer cannot hold 103 qubits."""
    from qiskit_aer import AerSimulator
    from qiskit_ibm_runtime import SamplerV2
    lat, center = Lattice(10), 4
    be = T.resolve_backend("grid:6x8")
    emb = T.choose_embedding(be, 10, center, mode="ladder", gauss_sites=G.gauss_ancilla_sites(lat, center))
    emb.validate(T.Graph.from_backend(be))
    assert emb.info["gauss_ancillas"] == {5: 21}
    qc = G.gauss_circuit(lat, C.load_card(), 2, 1.0, center)
    isa, fl, _ = T.transpile_isa(qc, be, emb.initial_layout, 1, 7, None, None, "g", routing="none")
    assert fl == emb.layout and isa.count_ops()["reset"] == 5
    pub = C.readout_layer(isa, lat, fl, "Z")
    res = SamplerV2(mode=AerSimulator(method="matrix_product_state")).run([(pub, None, 40)]).result()
    arr = res[0].data.c.to_bool_array(order="little").astype(np.uint8)
    assert arr.shape == (40, lat.n_wires + 5)
    acc = G.acceptance(arr, lat, G.patch_sites(lat, center), 1)
    assert acc["per_round"][0] == 1.0


def test_analyze_drops_shots_with_a_nontrivial_midcircuit_syndrome(tmp_path):
    """gauss-midcircuit pubs carry the patch syndromes of every extraction
    round beyond the system columns.  Without --postselect-rounds the reducer
    ignores those columns; with it, shots with any non-trivial syndrome are
    dropped before the estimators and the slice records the acceptance per
    round.  30% of the shots get a flipped syndrome and a randomised ancilla:
    the raw t = 0 reference must land back on the clean reduction."""
    from htq_hw import campaign as CP
    card = C.load_card()
    NS6, CEN = 6, 2
    lat = Lattice(NS6)
    tpl = str(tmp_path / "ideal_{family}.npz")
    S.write_ideal_grids(card, NS6, CEN, ("j0",), [0.0, 1.0], tpl, threads=2)
    sim = S.make_simulator(lat.n_wires + 1)
    rng = np.random.default_rng(4)
    shots = 3000
    circuits = {CP.pub_name("j0", 0.0, False, "Z", "X"): G.gauss_circuit(lat, card, 0, 0.0, CEN),
                CP.pub_name("j0", 1.0, False, "Z", "X"): G.gauss_circuit(lat, card, 2, 1.0, CEN),
                CP.pub_name("j0", 1.0, True, "Z", "X"): G.gauss_circuit(lat, card, 2, 1.0, CEN, mirror=True)}
    clean, bad = {}, {}
    for name, qc in circuits.items():
        full = C.readout_layer(qc, lat, list(range(lat.n_wires)), "Z")
        arr = S.sample_bits(sim, full, shots, seed=int(rng.integers(2**31)))
        assert arr.shape[1] == lat.n_wires + 5                      # one syndrome round each
        clean[name] = arr
        b = arr.copy()
        rows = rng.random(shots) < 0.3
        b[rows, lat.n_wires + 1] ^= 1                                # one patch check flipped
        b[rows, lat.ancilla] = rng.integers(0, 2, int(rows.sum()))
        bad[name] = b

    def run(bits, tag, rounds):
        path = str(tmp_path / f"htq_bits_{tag}.npz")
        np.savez_compressed(path, job_id=tag, backend="synthetic", pub_names=np.array(list(bits)), **bits)
        return A.analyze([path], tpl, str(tmp_path / f"{tag}_{int(rounds)}_{{comp}}_t{{t:.1f}}.npz"),
                         NS6, CEN, components=("00",), log=None, rounds=rounds)

    ref, raw, ps = run(clean, "clean", False), run(bad, "bad", False), run(bad, "bad", True)
    for key in ref:
        zr, zb, zp = (np.load(d[key], allow_pickle=True) for d in (ref, raw, ps))
        assert "midcircuit_per_round" not in zb.files and int(zb["nshot"]) == shots
        assert bool(zp["postselect_rounds"]) and int(zp["postselect_window"]) == 0
        assert abs(float(zp["postselect_kept"]) - 0.7) < 0.03
        assert zp["midcircuit_per_round"].shape == (1,) and abs(float(zp["midcircuit_per_round"][0]) - 0.7) < 0.03
    zr, zb, zp = (np.load(d[("00", 0.0)], allow_pickle=True) for d in (ref, raw, ps))
    err = np.hypot(zr["C_err"][0], zp["C_err"][0])
    assert np.all(np.abs(zp["C_cal"][0] - zr["C_cal"][0]) < 4 * err)
    v = np.argmax(np.abs((zr["C_cal"][0] - zr["id_a"] * zr["id_b"]).real))
    assert abs(zb["C_cal"][0][v] - zr["C_cal"][0][v]) > 4 * err[v]


def test_rehearsal_samples_midcircuit_pubs_from_the_cached_state(tmp_path):
    """The rehearsal had no path for gauss-midcircuit pubs (KeyError on the
    missing bundle, 28 October 2026).  On a lattice above the statevector
    limit it now starts the steps and syndrome rounds from the cached
    prepared-and-inserted MPS, keeps the syndrome columns in the bits, and
    the analysis with the rounds post-selection reduces the slices with every
    syndrome trivial on the noiseless simulator."""
    import glob
    from htq_hw import campaign as CP
    card = C.load_card()
    NS12, CEN = 12, 5
    lat = Lattice(NS12)
    assert lat.n_wires > S.STATEVECTOR_MAX_WIRES
    be = T.resolve_backend("grid:6x8")
    emb = T.choose_embedding(be, NS12, CEN, mode="ladder", gauss_sites=G.gauss_ancilla_sites(lat, CEN))
    emb.validate(T.Graph.from_backend(be))
    tpl = str(tmp_path / "ideal_{family}.npz")
    S.write_ideal_grids(card, NS12, CEN, ("j0",), [0.0, 1.0], tpl, threads=2)
    specs = [s for s in CP.manifest(times=(1.0,), families=("j0",), readouts=("Z",),
                                    card=card["name"], preset="gauss-midcircuit")]
    assert {s.t for s in specs} == {0.0, 1.0} and all(s.preset == "gauss-midcircuit" for s in specs)
    shots = {s.name: 120 for s in specs}
    out = S.rehearse(be, lat, card, emb, specs, shots, tpl, str(tmp_path / "reh"), cache_dir=str(tmp_path / "cache"),
                     components=("00",), log=None, threads=2)
    z = np.load(out["bits"], allow_pickle=True)
    k = len(G.patch_sites(lat, CEN))
    for s in specs:
        arr = z[s.name]
        rounds = G.n_rounds(s.n_steps) if s.n_steps else 1
        assert arr.shape == (120, lat.n_wires + rounds * k), s.name
        if rounds:
            acc = G.acceptance(arr, lat, G.patch_sites(lat, CEN), rounds)
            assert np.all(acc["per_round"] == 1.0), s.name      # noiseless: every syndrome trivial
    # composed ('preset.card:') specs key the slices per card, as the gate expects
    sl = {key: np.load(p, allow_pickle=True) for key, p in out["slices"].items()}
    c = card["name"]
    assert (c, "00", 0.0) in sl and (c, "00", 1.0) in sl
    assert bool(sl[(c, "00", 1.0)]["postselect_rounds"]) and float(sl[(c, "00", 1.0)]["postselect_kept"]) == 1.0
    assert abs(float(sl[(c, "00", 1.0)]["kappa_v"][0][CEN]) - 1.0) < 0.2
