"""Audit the fractional-RZZ (native rzz) transpilation of the t=2.0 and
t=3.0 forward slices against ibm_kingston, vs the CZ-basis baseline.

Checks: 2q counts (cz vs rzz), depth, estimated usage at 7e4 shots,
mirror-skeleton assertion (t3_transpile), and rzz angle range
(hardware requires theta in [0, pi/2]; out-of-range angles must be
folded or the submission is rejected).

  PYTHONPATH=. .venv/bin/python scripts/rzz_audit.py
"""
import sys
import numpy as np

sys.path.insert(0, "scripts")
import ibm_hardware as ih
from htensor import Z2Lattice, trotter
from qiskit_ibm_runtime import QiskitRuntimeService

SHOTS = 70_000
TS = [2.0, 3.0]
svc = QiskitRuntimeService(name="edu")
lat = Z2Lattice(ih.NS, pbc=True)
prep = ih.build_prep(lat)
base, _, meta = ih.t3_segments(lat, prep)
segs = []
for t in TS:
    n = int(round(t / ih.DT))
    segs.append((f"t{t:.1f}", t,
                 trotter.trotter_circuit(lat, ih.M0, ih.G2, ih.ETA, t, n)))
    segs.append((f"m{t:.1f}", t,
                 trotter.trotter_circuit(lat, ih.M0, ih.G2, ih.ETA,
                                         ih.MIRROR_EPS, n)))

for frac in (False, True):
    be = svc.backend("ibm_kingston", use_fractional_gates=frac)
    print(f"--- fractional_gates={frac} "
          f"(basis has rzz: {'rzz' in be.target.operation_names})")
    isa = ih.t3_transpile(base, segs, be)
    tot = 0.0
    for name, t, qc, layout in isa:
        if name == "t0.0":
            continue
        ops = qc.count_ops()
        n2 = sum(v for g, v in ops.items() if g in ("cz", "cx", "rzz"))
        nrzz = int(ops.get("rzz", 0))
        bad = sum(1 for inst in qc.data
                  if inst.operation.name == "rzz"
                  and not (-1e-9 <= float(inst.operation.params[0])
                           <= np.pi / 2 + 1e-9))
        dur = qc.depth() * 7e-8 + 2.5e-4
        tot += dur * SHOTS
        print(f"{name:5s}: {n2:5d} 2q ({nrzz} rzz, {bad} bad-angle)  "
              f"depth {qc.depth():4d}  {dur*SHOTS:5.1f} s @ {SHOTS}")
    print(f"TOTAL est. per pass: {tot:.0f} s = {tot/60:.1f} min")
