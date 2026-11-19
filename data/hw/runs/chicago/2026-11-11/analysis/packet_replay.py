"""Replay the layout chooser on saved ibm_chicago calibration snapshots with
and without the packet policy (htq_hw.target.PacketPolicy, --packet 20-28):
where site 24 lands and the calibration of the packet region (sites 20-28,
links 20-27, ancilla).  Production cuts: 5% readout, 2% two-qubit, rzz,
--min-t2 100.  Offline: recorded numbers only, no account.

usage: PYTHONPATH=. .venv/bin/python data/hw/runs/chicago/2026-11-11/analysis/packet_replay.py SNAPSHOT.json [...]
"""
import os, sys, time
sys.path.insert(0, os.getcwd()); sys.path.insert(0, "data/hw/runs/chicago/2026-11-08/analysis")
from replay_embedding import Standin
from htq_hw import target as T

POLICIES = [("production (no packet policy)", None),
            ("--packet 20-28", dict()),
            ("--packet 20-28 --packet-min-t2 50 --packet-max-2q-error 0.01", dict(min_t2_us=50, max_2q_error=0.01)),
            ("--packet 20-28 --packet-min-t2 80 --packet-max-2q-error 0.01 --packet-max-readout-error 0.03",
             dict(min_t2_us=80, max_2q_error=0.01, max_readout_error=0.03))]
for snap in sys.argv[1:]:
    be = Standin(snap); props = be.properties()
    g, rep = T.operational_graph(be, basis="rzz", max_readout_error=0.05, max_2q_error=0.02, log=None)
    print(f"== {be.name}: {len(rep['dropped_qubits'])} qubits, {len(rep['dropped_edges'])} couplers cut")
    for label, kw in POLICIES:
        pol = None if kw is None else T.PacketPolicy((20, 28), **kw)
        t0 = time.time()
        try:
            e = T.choose_embedding(be, 50, 24, mode="ladder", graph=g, basis="rzz", log=None,
                                   min_core_t2_us=100.0, packet=pol)
        except T.EmbeddingError as ex:
            print(f"  {label}: NONE ({str(ex)[:400]})"); continue
        q = e.info.get("packet") or T.packet_quality(e, be, props, T.PacketPolicy((20, 28)), "rzz")
        core = e.info["core_t2_us"]
        print(f"  {label}: {e.info.get('packet_candidates', e.info['n_candidates'])} candidates, "
              f"{time.time() - t0:.1f} s, global score {e.info['score']:.4f}, core min T2 {core['min']:.0f} us")
        print(f"      {T._packet_summary(q)}")
