"""Hardware quality of the packet region (sites 20..28, links 20..27) on each
Chicago layout: per wire T2, readout error, and the worst rzz error among the
couplers the circuit can use (edges between wires of the layout)."""
import glob, json, sys
import numpy as np
sys.path.insert(0, ".")
from htq_hw.model import Lattice
lat = Lattice(50)
def cal(path):
    d = json.load(open(path)); Q, G = {}, {}
    for q, items in enumerate(d["qubits"]):
        Q[q] = {it["name"]: it["value"] * (1e6 if it.get("unit") == "s" else 1) for it in items}
    for g in d["gates"]:
        if len(g["qubits"]) == 2 and g["gate"] == "rzz":
            for p in g["parameters"]:
                if p["name"] == "gate_error": G[tuple(sorted(g["qubits"]))] = p["value"]
    return Q, G
RUNS = [("6 Nov (8 Nov calibration)", "data/hw/runs/chicago/2026-11-06/jobs/main_rzz", "data/hw/runs/chicago/2026-11-08"),
        ("8 Nov", "data/hw/runs/chicago/2026-11-08/jobs/mirror_diag_control", "data/hw/runs/chicago/2026-11-08"),
        ("10 Nov", "data/hw/runs/chicago/2026-11-10/jobs/mirror_diag_control2", "data/hw/runs/chicago/2026-11-10"),
        ("11 Nov", "data/hw/runs/chicago/2026-11-11/jobs/main_rzz_twirl", "data/hw/runs/chicago/2026-11-11")]
for lab, jobs, crun in RUNS:
    Q, G = cal(sorted(glob.glob(crun + "/calibration/backend_properties_*.json"))[0])
    jr = [json.load(open(p)) for p in sorted(glob.glob(jobs + "/htq_job_*.json"))]
    jr = next(j for j in jr if not j.get("cancelled"))
    lay = jr.get("initial_layout") or jr["embedding"]["layout"]
    used = set(lay); inv = {q: w for w, q in enumerate(lay)}
    def wire(w):
        q = lay[w]; e = [v for (a, b), v in G.items() if q in (a, b) and a in used and b in used]
        return q, Q[q].get("T2", 0), Q[q].get("readout_error", 0), max(e) if e else float("nan")
    med = np.median([v for v in G.values() if v < 1])
    print(f"\n=== {lab}: median rzz error {med:.4f}")
    print("        " + "".join(f"{n:>9d}" for n in range(20, 29)))
    for kind, f in (("site", lat.site_qubit), ("link", lat.link_qubit)):
        rows = [wire(f(n)) for n in range(20, 29)]
        print(f"{kind} q   " + "".join(f"{r[0]:>9d}" for r in rows))
        print(f"{kind} T2  " + "".join(f"{r[1]:>9.0f}" for r in rows))
        print(f"{kind} ro  " + "".join(f"{r[2]:>9.3f}" for r in rows))
        print(f"{kind} rzz " + "".join(f"{r[3]/med:>8.1f}x" for r in rows) + "   (worst usable coupler, in units of the median)")
    allw = [wire(w) for w in range(101)]
    print(f"whole layout: wires with a coupler above 4x median: {sum(1 for r in allw if r[3] > 4*med)}/101; T2 < 60 us: {sum(1 for r in allw if r[1] < 60)}; readout > 3%: {sum(1 for r in allw if r[2] > 0.03)}")
