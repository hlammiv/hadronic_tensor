"""Replay the layout choice on Declan's saved ibm_chicago calibration snapshot.

Builds an offline stand-in for ibm_chicago from
calibration/backend_properties_*.json (coupling map and rzz errors from the
gate list, readout errors and T2 from the qubit list), reproduces the 8 Nov
choice under the production cuts (5% readout, 2% two-qubit, basis rzz), and
then shows where the insertion site lands with the Hadamard-test core T2
floor (--min-t2).  The snapshot lists rzz gates only, so the one CZ coupler
the real run cut (115-116, cz error 0.0228) is excluded by hand; with it the
recorded 8 Nov layout is among the eight candidates.  No hardware, no
account: this is the chooser on recorded numbers.

usage: PYTHONPATH=. .venv/bin/python data/hw/runs/chicago/2026-11-08/analysis/replay_embedding.py [--floors 80,100,120,150]
"""
import argparse, glob, json, os, sys
import numpy as np
from qiskit.circuit import Measure, Parameter
from qiskit.circuit.library import RZZGate
from qiskit.transpiler import CouplingMap, InstructionProperties, Target
from qiskit_ibm_runtime.models import BackendProperties
sys.path.insert(0, os.getcwd())
from htq_hw import target as T
from htq_hw.model import Lattice

RUN = 'data/hw/runs/chicago/2026-11-08'
RUN6 = 'data/hw/runs/chicago/2026-11-06'


class Standin:
    """What operational_graph / choose_embedding / score_embedding read."""
    def __init__(self, snapshot):
        d = json.load(open(snapshot))
        self._props = BackendProperties.from_dict(d)
        n = len(d['qubits'])
        pairs = {}
        for g in d['gates']:
            if len(g['qubits']) == 2:
                e = [p['value'] for p in g['parameters'] if p['name'] == 'gate_error']
                pairs[tuple(g['qubits'])] = e[0] if e else None
        directed = set(pairs) | {(v, u) for u, v in pairs}
        self.coupling_map = CouplingMap(sorted(directed))
        t = Target(num_qubits=n)
        t.add_instruction(RZZGate(Parameter('theta')),
                          {(u, v): InstructionProperties(error=pairs.get((u, v), pairs.get((v, u))))
                           for (u, v) in directed})
        t.add_instruction(Measure(), {(q,): InstructionProperties(error=self._props.readout_error(q))
                                      for q in range(n)})
        self.target = t
        self.name = f"ibm_chicago snapshot {d['last_update_date']}"
        self.num_qubits = n

    @property
    def operation_names(self):
        return self.target.operation_names

    def properties(self):
        return self._props


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--floors', default='80,100,120,150')
    ap.add_argument('--cz-cuts', default='115-116', help='couplers the real run cut on CZ error, absent from the rzz-only snapshot')
    ap.add_argument('--ns', type=int, default=50)
    ap.add_argument('--center', type=int, default=24)
    args = ap.parse_args()
    snap = sorted(glob.glob(f'{RUN}/calibration/backend_properties_*.json'))[-1]
    be = Standin(snap)
    props = be.properties()
    lat = Lattice(args.ns)
    L8 = json.load(open(f'{RUN}/acceptance_mirror_diag_control.json'))['embedding']['layout']
    L6 = json.load(open(f'{RUN6}/acceptance_21-49-59_889395Z.json'))['embedding']['layout']
    t2 = {q: T._t2_us(props, q) for q in range(be.num_qubits)}
    print(f"snapshot {os.path.basename(snap)}: {be.num_qubits} qubits, {be.coupling_map.size()} qubits in map, "
          f"{len(be.coupling_map.get_edges()) // 2} couplers; T2 median {np.nanmedian([v for v in t2.values() if v]):.0f} us, "
          f"below 100 us: {sorted(q for q, v in t2.items() if v is not None and v < 100)}")
    print(f"recorded 8 Nov layout: site 24 on q{L8[lat.site_qubit(24)]}, ancilla q{L8[lat.ancilla]}; "
          f"6 Nov: site 24 on q{L6[lat.site_qubit(24)]}, ancilla q{L6[lat.ancilla]}")
    floors = [None] + [float(x) for x in args.floors.split(',') if x]
    cz_cuts = [tuple(int(q) for q in e.split('-')) for e in args.cz_cuts.split(',') if e]
    g, rep = T.operational_graph(be, basis='rzz', max_readout_error=0.05, max_2q_error=0.02,
                                 exclude_edges=cz_cuts, log=None)
    for floor in floors:
        try:
            emb = T.choose_embedding(be, args.ns, args.center, mode='auto', graph=g, basis='rzz',
                                     log=print, min_core_t2_us=floor)
        except T.EmbeddingError as e:
            print(f"\n--- min_t2 {floor}: {len(rep['dropped_qubits'])} qubits, {len(rep['dropped_edges'])} couplers "
                  f"excluded; NO EMBEDDING: {str(e)[:200]}")
            continue
        L = list(emb.layout)
        matter = [L[lat.site_qubit(v)] for v in range(args.ns)]
        t2m = [t2[q] for q in matter]
        same8 = L == L8
        same6 = L == L6
        rev = matter == [L6[lat.site_qubit(v)] for v in range(args.ns)]
        print(f"\n--- min_t2 {floor if floor else 'none'} us: {len(rep['dropped_qubits'])} qubits, "
              f"{len(rep['dropped_edges'])} couplers excluded; {emb.kind}, redundancy {emb.info.get('redundancy')}, "
              f"score {emb.info.get('score'):.3f}, candidates {emb.info.get('n_candidates')}")
        print(f"    site 24 on q{L[lat.site_qubit(24)]} (T2 {t2[L[lat.site_qubit(24)]]:.0f} us), ancilla q{L[lat.ancilla]} "
              f"(T2 {t2[L[lat.ancilla]]:.0f} us); matter T2 min {np.nanmin(t2m):.0f} us at site "
              f"{int(np.nanargmin(t2m))}; sites 21..27 on {[L[lat.site_qubit(v)] for v in range(21, 28)]}")
        print(f"    equals recorded 8 Nov layout: {same8}; equals 6 Nov layout: {same6}; "
              f"matter chain equals 6 Nov: {rev}; shares {len(set(L) & set(L6))} qubits with 6 Nov, "
              f"{len(set(L) & set(L8))} with 8 Nov")
        core = emb.info.get('core_t2_us')
        if core:
            print(f"    core T2: {T._core_summary(core)}; per qubit {dict((q, round(v)) for q, v in core['qubits'].items())}")


if __name__ == '__main__':
    main()
