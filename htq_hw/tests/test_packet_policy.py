"""--packet: protect the qubits the wave packet occupies, not just the core.

Written after the 2026-11-10/11 ibm_chicago runs: the layout passed every cut
(5% readout, 2% two-qubit, core T2 >= 100 us) and still put a T2 = 23 us link
qubit and 1.4-4.5% couplers at sites 26-28, where the t = 0 Gauss syndromes
read 0.11-0.14 against 0.65 elsewhere.  The chooser only ever tried the first
eight placements of the insertion site on the first cycle, in one direction,
with whatever pendants the matching found first.
"""
from datetime import datetime, timezone

import pytest
from qiskit.circuit import Measure, Parameter
from qiskit.circuit.library import CZGate, RZZGate
from qiskit.transpiler import InstructionProperties, Target
from qiskit_ibm_runtime.models import BackendProperties

from htq_hw import target as T
from htq_hw.__main__ import _packet

NOW = datetime(2026, 11, 14, tzinfo=timezone.utc)


def _p(name, value, unit=""):
    return {"name": name, "value": value, "date": NOW, "unit": unit}


class Device:
    """FakeNighthawk's coupling map with uniform calibration (rzz/cz 0.3%,
    readout 1%, T2 200 us) except where ``t2_us`` / ``edge_error`` say."""

    def __init__(self, t2_us=None, edge_error=None, readout=None):
        nh = T.resolve_backend("fake:nighthawk", fractional=True)
        self.coupling_map, self.num_qubits, self.name = nh.coupling_map, nh.num_qubits, "packet_test"
        t2_us, edge_error, readout = t2_us or {}, edge_error or {}, readout or {}
        pairs = {tuple(sorted(e)) for e in nh.coupling_map.get_edges()}
        err = lambda e: edge_error.get(tuple(sorted(e)), 0.003)
        t = Target(num_qubits=self.num_qubits)
        directed = [(u, v) for u, v in pairs] + [(v, u) for u, v in pairs]
        t.add_instruction(RZZGate(Parameter("theta")), {e: InstructionProperties(error=err(e)) for e in directed})
        t.add_instruction(CZGate(), {e: InstructionProperties(error=err(e)) for e in directed})
        t.add_instruction(Measure(), {(q,): InstructionProperties(error=readout.get(q, 0.01))
                                      for q in range(self.num_qubits)})
        self.target = t
        self._props = BackendProperties.from_dict({
            "backend_name": "packet_test", "backend_version": "1", "last_update_date": NOW,
            "qubits": [[_p("T2", t2_us.get(q, 200.0), "us"), _p("readout_error", readout.get(q, 0.01))]
                       for q in range(self.num_qubits)],
            "gates": [], "general": []})

    def properties(self):
        return self._props


NS, CENTER, SITES = 10, 4, (2, 6)


def _choose(be, packet=None, min_t2=None):
    return T.choose_embedding(be, NS, CENTER, mode="ladder", graph=T.Graph.from_backend(be), basis="rzz",
                              log=None, packet=packet, min_core_t2_us=min_t2)


def test_policy_parses_and_names_the_packet_wires():
    p = T.PacketPolicy.parse("2-6", min_t2_us=50, max_2q_error=0.01)
    assert p.sites == (2, 6) and p.min_t2_us == 50 and p.max_2q_error == 0.01
    assert p.wires(NS) == [4, 6, 8, 10, 12, 5, 7, 9, 11, 20]       # sites 2..6, links 2..5, ancilla
    assert p.to_dict()["sites"] == [2, 6]
    for bad in ("6-2", "x", "3"):
        with pytest.raises(T.EmbeddingError):
            T.PacketPolicy.parse(bad)
    with pytest.raises(T.EmbeddingError):
        T.PacketPolicy((2, 6), min_t2_us=-1)
    with pytest.raises(T.EmbeddingError):
        T.PacketPolicy((2, 6), max_2q_error=1.5)
    with pytest.raises(T.EmbeddingError):
        T.PacketPolicy((2, 12)).wires(NS)


def test_cli_builds_the_policy_and_refuses_floors_without_it():
    import argparse
    ns = argparse.Namespace(packet="20-28", packet_min_t2=50.0, packet_max_2q_error=None,
                            packet_max_readout_error=0.03)
    p = _packet(ns)
    assert p.sites == (20, 28) and p.min_t2_us == 50 and p.max_readout_error == 0.03 and p.max_2q_error is None
    assert _packet(argparse.Namespace()) is None
    with pytest.raises(SystemExit):
        _packet(argparse.Namespace(packet=None, packet_min_t2=50.0))


def test_cost_weighted_matching_takes_the_better_pendant():
    # square 0-1-2-3; vertex 0 has two free neighbours, 4 (bad) and 5 (good)
    g = T.Graph(8, [(0, 1), (1, 2), (2, 3), (0, 3), (0, 4), (0, 5), (1, 6), (2, 7), (3, 4)])
    cyc = [0, 1, 2, 3]
    cost = lambda v, w, extra: 10.0 if w == 4 else 0.0
    pend, _ = T._pendant_matching(g, cyc, cost=cost)
    assert pend[0] == 5 and pend[3] == 4                 # 3's only free neighbour is 4
    assert T._pendant_matching(g, cyc, extra_for=0, cost=cost) is None     # 0 would need 4 and 5 and 3 needs 4
    unweighted, _ = T._pendant_matching(g, cyc)
    assert set(unweighted.values()) == {4, 5, 6, 7}


def test_default_chooser_is_unchanged_without_a_policy():
    be = Device()
    a, b = _choose(be), _choose(be)
    assert a.layout == b.layout and "packet" not in a.info and "packet_policy" not in a.info


def test_policy_moves_the_packet_off_a_planted_bad_qubit_and_coupler():
    clean = Device()
    base = _choose(clean)
    pol = T.PacketPolicy(SITES)
    wires = pol.wires(NS)
    # plant the 11 Nov failure inside the default layout's packet: a 23 us link
    # qubit and a 4.5% coupler between two packet sites
    link = base.layout[2 * (CENTER + 1) + 1]
    s1, s2 = base.layout[2 * (CENTER + 1)], base.layout[2 * (CENTER + 2)]
    be = Device(t2_us={link: 23.0}, edge_error={tuple(sorted((s1, s2))): 0.045})
    plain = _choose(be)
    q_plain = T.packet_quality(plain, be, be.properties(), pol, "rzz")
    assert q_plain["min_t2_us"] == pytest.approx(23.0) or q_plain["max_2q_error"] >= 0.09 - 1e-9
    prot = _choose(be, packet=pol)
    q = prot.info["packet"]
    packet_qubits = {prot.layout[w] for w in wires}
    assert link not in packet_qubits
    assert not ({s1, s2} <= packet_qubits and any(
        {prot.layout[a], prot.layout[b]} == {s1, s2} for a, b in prot.required_edges()))
    assert q["cost"] < q_plain["cost"] and q["min_t2_us"] >= 200 - 1e-9 and q["max_2q_error"] < 0.01
    assert prot.info["packet_policy"]["sites"] == list(SITES) and prot.info["packet_candidates"] > 8
    prot.validate(T.Graph.from_backend(be))


def test_floors_refuse_when_no_placement_passes_and_list_the_best():
    be = Device(t2_us={q: 40.0 for q in range(0, 120, 2)})      # every other qubit dephases
    with pytest.raises(T.EmbeddingError, match="packet region") as ex:
        _choose(be, packet=T.PacketPolicy(SITES, min_t2_us=100))
    assert ex.value.diagnostics["candidates"] and ex.value.diagnostics["packet"]["min_t2_us"] == 100
    e = _choose(be, packet=T.PacketPolicy(SITES, min_t2_us=30))  # a floor below the worst qubit passes
    assert e.info["packet"]["min_t2_us"] >= 30


def test_policy_composes_with_the_core_t2_floor():
    be = Device()
    centre_bad = _choose(be, packet=T.PacketPolicy(SITES)).layout[2 * CENTER]
    be2 = Device(t2_us={centre_bad: 60.0})
    e = _choose(be2, packet=T.PacketPolicy(SITES), min_t2=100)
    assert e.layout[2 * CENTER] != centre_bad and e.info["core_t2_us"]["min"] >= 100
