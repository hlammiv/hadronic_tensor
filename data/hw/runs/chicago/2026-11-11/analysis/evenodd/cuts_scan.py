"""Replay the layout chooser on the 11 Nov ibm_chicago calibration under
stricter cuts and report the hardware quality of the packet region."""
import glob, json, os, sys
import numpy as np
sys.path.insert(0, os.getcwd()); sys.path.insert(0, "data/hw/runs/chicago/2026-11-08/analysis")
from replay_embedding import Standin
from htq_hw import target as T
from htq_hw.model import Lattice
snap = sorted(glob.glob("data/hw/runs/chicago/2026-11-11/calibration/backend_properties_*.json"))[-1]
be = Standin(snap); props = be.properties(); lat = Lattice(50)
d = json.load(open(snap)); G = {}
for g in d["gates"]:
    if len(g["qubits"]) == 2:
        for p in g["parameters"]:
            if p["name"] == "gate_error": G[tuple(sorted(g["qubits"]))] = p["value"]
med = np.median([v for v in G.values() if v < 1])
t2 = {q: T._t2_us(props, q) for q in range(be.num_qubits)}
region = [lat.site_qubit(n) for n in range(20, 29)] + [lat.link_qubit(n) for n in range(20, 28)]
def quality(L, graph_edges):
    qs = [L[w] for w in region]; used = set(L)
    worst = max(v for (a, b), v in G.items() if (a in qs or b in qs) and a in used and b in used and ((a, b) in graph_edges or (b, a) in graph_edges))
    return min(t2[q] for q in qs), worst / med, max(props.readout_error(q) for q in qs)
for ro in (0.05, 0.03):
    for e2 in (0.02, 0.015, 0.01, 0.0075, 0.005):
        g, rep = T.operational_graph(be, basis="rzz", max_readout_error=ro, max_2q_error=e2, log=None)
        try:
            emb = T.choose_embedding(be, 50, 24, mode="ladder", graph=g, basis="rzz", log=None, min_core_t2_us=100.0)
        except Exception as ex:
            print(f"readout cut {ro:.2f}, 2q cut {e2:.4f} ({e2/med:.0f}x median): {len(rep['dropped_qubits'])} qubits, {len(rep['dropped_edges'])} couplers cut -> NO ladder ({type(ex).__name__})")
            continue
        L = list(emb.layout)
        try: edges = set(map(tuple, g.edges()))
        except Exception: edges = {(a, b) for (a, b) in G}
        mt2, w, mro = quality(L, edges)
        print(f"readout cut {ro:.2f}, 2q cut {e2:.4f} ({e2/med:.0f}x median): {len(rep['dropped_qubits'])} qubits, {len(rep['dropped_edges'])} couplers cut -> "
              f"ladder, {emb.info.get('n_candidates')} candidates, site 24 q{L[lat.site_qubit(24)]}, ancilla q{L[lat.ancilla]}; "
              f"packet region: min T2 {mt2:.0f} us, worst coupler {w:.1f}x median, worst readout {mro:.3f}")
