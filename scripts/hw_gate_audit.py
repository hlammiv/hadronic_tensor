"""Audit the true 2q-gate cost of every hardware job we ran.

Unlike rzz_audit.py, which re-transpiles and reports what the transpiler
*would* produce today, this fetches the ISA circuits actually submitted to
IBM (job.inputs["pubs"]) and counts ops on them.  Also pulls job.metrics()
for the billed QPU time.

  PYTHONPATH=. .venv/bin/python scripts/hw_gate_audit.py [--backend kingston]

Writes data/hw/gate_audit.json with a per-circuit record so the paper can
quote measured, not estimated, gate counts.
"""
import glob
import json
import os
import sys

from qiskit_ibm_runtime import QiskitRuntimeService

TWO_Q = ("cz", "cx", "ecr", "rzz", "rzx", "swap", "iswap")
ONLY = None
for a in sys.argv[1:]:
    if a.startswith("--backend"):
        ONLY = a.split("=", 1)[1] if "=" in a else sys.argv[sys.argv.index(a) + 1]

# every job-id record we wrote, in submission order
jobs = []            # (label, job_id)
for path in sorted(glob.glob("data/hw/*.json")):
    if path.endswith("gate_audit.json"):
        continue
    try:
        d = json.load(open(path))
    except Exception:
        continue
    tag = os.path.basename(path).replace(".json", "")
    if "job_id" in d:
        jobs.append((tag, d["job_id"], d))
    for sub in d.get("jobs", []):                # loschmidt passes
        jobs.append((f"{tag}:{'+'.join(sub.get('names', []))}",
                     sub["job_id"], d))

svcs = []
for name in (None, "edu", "prev-2025"):
    try:
        svcs.append(QiskitRuntimeService(name=name) if name
                    else QiskitRuntimeService())
    except Exception as e:
        print(f"  (account {name}: {type(e).__name__})")


def fetch(jid):
    last = None
    for s in svcs:
        try:
            return s.job(jid)
        except Exception as e:
            last = e
    raise last


out = []
for tag, jid, meta in jobs:
    try:
        job = fetch(jid)
    except Exception as e:
        print(f"{tag:34s} {jid}  UNAVAILABLE ({type(e).__name__})")
        continue
    be = job.backend().name if job.backend() else meta.get("backend", "?")
    if ONLY and ONLY not in be:
        continue
    try:
        m = job.metrics()
        qpu_s = (m.get("usage", {}) or {}).get("quantum_seconds")
    except Exception:
        qpu_s = None
    try:
        pubs = job.inputs.get("pubs", [])
    except Exception as e:
        print(f"{tag:34s} {jid}  inputs unavailable ({type(e).__name__})")
        continue

    shots = meta.get("shots")
    print(f"\n=== {tag}   {jid}   {be}   {job.status()}"
          f"   QPU {qpu_s if qpu_s is None else round(qpu_s, 1)} s")
    for i, pub in enumerate(pubs):
        qc = pub[0]
        ops = dict(qc.count_ops())
        n2 = sum(v for g, v in ops.items() if g in TWO_Q)
        n1 = sum(v for g, v in ops.items()
                 if g not in TWO_Q and g not in ("barrier", "measure", "delay"))
        nq = sum(1 for q in range(qc.num_qubits)
                 if any(q in [qc.find_bit(b).index for b in ins.qubits]
                        for ins in qc.data))
        d2 = qc.depth(lambda ins: len(ins.qubits) == 2)
        name = (meta.get("circ_names") or meta.get("names")
                or [d.get("names") for d in meta.get("jobs", [])])
        try:
            cname = name[i] if isinstance(name, list) else None
        except Exception:
            cname = None
        rec = dict(tag=tag, job_id=jid, backend=be, pub=i, name=cname,
                   shots=shots, qpu_seconds=qpu_s, n2q=n2, n1q=n1,
                   depth=qc.depth(), depth2q=d2, num_qubits=qc.num_qubits,
                   active_qubits=nq, ops=ops)
        out.append(rec)
        detail = " ".join(f"{g}={ops[g]}" for g in TWO_Q if g in ops)
        print(f"  pub{i} {str(cname):6s}: {n2:6d} 2q [{detail}]  {n1:6d} 1q  "
              f"depth {qc.depth():5d} (2q {d2:4d})  {qc.num_qubits}q "
              f"({nq} active)")

json.dump(out, open("data/hw/gate_audit.json", "w"), indent=1)
tot2 = sum(r["n2q"] for r in out)
print(f"\nwrote data/hw/gate_audit.json  "
      f"({len(out)} circuits, {tot2} two-qubit gates total)")
