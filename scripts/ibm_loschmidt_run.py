"""Loschmidt forward-region run on ibm_kingston (approved 2026-07-31).

Pass-2 manifest (adapted after the pass-1 verdict: matched-depth theta->0
mirrors are the working calibration; the Loschmidt pubs are dropped --
that head-to-head is complete at 7e4 shots in pass 1):
  t0.0   prep + gadget                       (reference / tier-1 harvest)
  t0.5   physics U(t), 1 Trotter step
  m0.5   theta->0 mirror at t=0.5 depth      (the missing calibrator)
  t1.0   physics U(t), 2 steps
  m1.0   theta->0 mirror at t=1.0 depth
  vac    matched vacuum prep (dM/dm0 upgrade)
7e4 shots each; gate twirling + XY4 DD.

Modes:
  audit            build + ISA-transpile against ibm_kingston, print 2q
                   counts, durations, estimated usage.  NO submission.
  submit <pass>    submit one pass (7 pubs, one job).  Saves job id.
  fetch <jobid>    retrieve bits -> data/hw/losch_pass_<jobid>.npz,
                   report usage.

  PYTHONPATH=. .venv/bin/python scripts/ibm_loschmidt_run.py audit
"""
import json
import sys
import numpy as np
from qiskit import QuantumCircuit

sys.path.insert(0, "scripts")
import ibm_hardware as ih
from htensor import Z2Lattice, trotter

SHOTS = 70_000
TS = [0.5, 1.0]
H_SEQ = [("rz", np.pi / 2), ("sx", None), ("rz", np.pi / 2)]  # H in ISA


def manifest(lat, prep, pass3=False, rzz=False, rzz_times=(2.0, 3.0)):
    """pass3: the t=1.5 quadrature-gap pass [t1.5, m1.5] (2026-08-02).
    rzz (pass >= 5): the fractional-gate deep pass [t2.0, m2.0, t3.0,
    m3.0] -- native rzz transpilation (-17% 2q, -47% depth vs CZ) to
    push kappa above the masking cliff at t=2 and reach t=3
    (delta-q0 ~ 1.0).  t2.0 doubles as the fractional-mirror
    validation against the CZ-basis tier-3d slice."""
    base, _, meta = ih.t3_segments(lat, prep)
    if pass3 or rzz:
        segs = []
        for t in (list(rzz_times) if rzz else [1.5]):
            n = int(round(t / ih.DT))
            segs.append((f"t{t:.1f}", t,
                         trotter.trotter_circuit(lat, ih.M0, ih.G2,
                                                 ih.ETA, t, n)))
            segs.append((f"m{t:.1f}", t,
                         trotter.trotter_circuit(lat, ih.M0, ih.G2,
                                                 ih.ETA, ih.MIRROR_EPS,
                                                 n)))
        return base, segs, meta
    segs = [("t0.0", 0.0, None)]
    for t in TS:
        n = int(round(t / ih.DT))
        segs.append((f"t{t:.1f}", t,
                     trotter.trotter_circuit(lat, ih.M0, ih.G2, ih.ETA,
                                             t, n)))
        segs.append((f"m{t:.1f}", t,
                     trotter.trotter_circuit(lat, ih.M0, ih.G2, ih.ETA,
                                             ih.MIRROR_EPS, n)))
    return base, segs, meta


def add_readout(isa, layout, lat):
    """Append ISA-basis H on ancilla + links (via layout), measure all."""
    fl = layout.final_index_layout() if hasattr(layout, "final_index_layout") \
        else layout
    qc = isa.copy()
    targets = [fl[lat.n_qubits]] + [fl[lat.link_qubit(b)]
                                    for b in range(lat.n_links)]
    for q in targets:
        for g, a in H_SEQ:
            getattr(qc, g)(*([a, q] if a is not None else [q]))
    meas = [fl[i] for i in range(lat.n_qubits + 1)]
    creg_qc = QuantumCircuit(qc.num_qubits, len(meas))
    creg_qc.compose(qc, inplace=True)
    for i, q in enumerate(meas):
        creg_qc.measure(q, i)
    return creg_qc


def build(be, pass3=False, rzz=False, rzz_times=(2.0, 3.0)):
    lat = Z2Lattice(ih.NS, pbc=True)
    prep = ih.build_prep(lat)
    base, segs, meta = manifest(lat, prep, pass3=pass3, rzz=rzz,
                                rzz_times=rzz_times)
    isa = ih.t3_transpile(base, segs, be)
    pubs = []
    for name, t, circ, layout in isa:
        qc = add_readout(circ, layout, lat)
        if rzz:
            from qiskit.transpiler import PassManager
            from qiskit_ibm_runtime.transpiler.passes import FoldRzzAngle
            qc = PassManager([FoldRzzAngle()]).run(qc)
            bad = sum(1 for i in qc.data if i.operation.name == "rzz"
                      and not (-1e-9 <= float(i.operation.params[0])
                               <= np.pi / 2 + 1e-9))
            assert bad == 0, f"{name}: {bad} rzz angles out of range"
            # fold emits literal global_phase instructions; ISA rejects
            # them -- absorb into the circuit phase attribute instead
            gp = [i for i in qc.data if i.operation.name == "global_phase"]
            if gp:
                qc.global_phase += sum(float(i.operation.params[0])
                                       for i in gp)
                qc.data = [i for i in qc.data
                           if i.operation.name != "global_phase"]
        pubs.append((name, qc))
    if pass3 or rzz:
        return lat, pubs
    # matched vacuum: same ansatz thetas, no packet block, no gadget
    from htensor import stateprep
    TH = stateprep.optimize_vacuum(Z2Lattice(6, pbc=True), ih.M0, ih.G2,
                                   ih.ETA, n_layers=2,
                                   restarts=2)["thetas"]
    vac = stateprep.vacuum_ansatz(lat, TH)
    from qiskit.transpiler.preset_passmanagers import \
        generate_preset_pass_manager
    pm3 = generate_preset_pass_manager(backend=be, optimization_level=3,
                                       seed_transpiler=7)
    v101 = QuantumCircuit(lat.n_qubits + 1)
    v101.compose(vac, range(lat.n_qubits), inplace=True)
    iv = pm3.run(v101)
    pubs.append(("vac", add_readout(iv, iv.layout, lat)))
    return lat, pubs


def audit(pass3=False, rzz=False):
    from qiskit_ibm_runtime import QiskitRuntimeService
    svc = QiskitRuntimeService(name="edu")
    be = svc.backend("ibm_kingston", use_fractional_gates=rzz)
    lat, pubs = build(be, pass3=pass3, rzz=rzz)
    tot = 0.0
    for name, qc in pubs:
        n2q = sum(v for g, v in qc.count_ops().items()
                  if g in ("cz", "cx", "rzz"))
        dur = qc.depth() * 7e-8 + 2.5e-4
        tot += dur * SHOTS
        print(f"{name:5s}: {n2q:5d} 2q  depth {qc.depth():4d}  "
              f"~{dur*1e6:5.0f} us/shot  {dur*SHOTS:6.1f} s @ {SHOTS}")
    print(f"TOTAL est. usage per pass: {tot:.0f} s = {tot/60:.1f} min")


def submit(pass_id):
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    # pass 10 runs on the DEFAULT (FNAL) account: 5 s residual there
    svc = (QiskitRuntimeService() if pass_id == 10
           else QiskitRuntimeService(name="edu"))
    rzz = pass_id >= 5
    rzz_times = {7: (1.5,), 9: (1.0,), 10: (0.5,)}.get(pass_id,
                                                      (2.0, 3.0))
    shots = {9: 20_000, 10: 7_000}.get(pass_id, SHOTS)
    be = svc.backend("ibm_kingston", use_fractional_gates=rzz)
    lat, pubs = build(be, pass3=(pass_id in (3, 4)), rzz=rzz,
                      rzz_times=rzz_times)
    smp = SamplerV2(mode=be)
    smp.options.default_shots = shots
    # server rejects gate twirling with fractional gates (error 1519);
    # the mirror self-calibration + wing anchor absorb the untwirled
    # coherent residuals, and measure twirling/DD remain active
    smp.options.twirling.enable_gates = not rzz
    smp.options.twirling.enable_measure = True   # symmetrize readout
    smp.options.dynamical_decoupling.enable = True
    smp.options.dynamical_decoupling.sequence_type = "XY4"
    job = smp.run([qc for _, qc in pubs])
    names = [n for n, _ in pubs]
    with open(f"data/hw/losch_pass{pass_id}.json", "w") as f:
        json.dump({"job_id": job.job_id(), "names": names,
                   "shots": shots, "pass": pass_id}, f)
    print(f"submitted pass {pass_id}: job {job.job_id()}  pubs: {names}")


def submit6():
    """Pass 6: transpile-time-twirled fractional pass.  K=4 twirled
    copies per pub at SHOTS/K, split into two jobs by time slice to
    stay under the job payload limit.  The runtime's own gate twirling
    stays OFF (incompatible with rzz); these circuits are pre-twirled
    (rzz commutant 8-pair group, cz full 16-pair -- rzz_twirl.py,
    self-tested unitary-equivalent)."""
    import numpy as np
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    import rzz_twirl
    K = 4
    svc = QiskitRuntimeService(name="edu")
    be = svc.backend("ibm_kingston", use_fractional_gates=True)
    lat, pubs = build(be, rzz=True)
    rng = np.random.default_rng(2026)
    groups = [["t2.0", "m2.0"], ["t3.0", "m3.0"]]
    rec = {"jobs": [], "shots": SHOTS // K, "K": K, "pass": 6}
    for gi, gnames in enumerate(groups):
        gpubs, names = [], []
        for name, qc in pubs:
            if name not in gnames:
                continue
            for k in range(K):
                names.append(f"{name}#{k}")
                gpubs.append(rzz_twirl.twirl_circuit(qc, rng))
        smp = SamplerV2(mode=be)
        smp.options.default_shots = SHOTS // K
        smp.options.twirling.enable_gates = False
        smp.options.twirling.enable_measure = True
        smp.options.dynamical_decoupling.enable = True
        smp.options.dynamical_decoupling.sequence_type = "XY4"
        job = smp.run(gpubs)
        rec["jobs"].append({"job_id": job.job_id(), "names": names})
        print(f"submitted pass 6 job {gi}: {job.job_id()}  pubs: {names}")
    with open("data/hw/losch_pass6.json", "w") as f:
        json.dump(rec, f)


def _fold_rzz(qc):
    from qiskit.transpiler import PassManager
    from qiskit_ibm_runtime.transpiler.passes import FoldRzzAngle
    qc = PassManager([FoldRzzAngle()]).run(qc)
    bad = sum(1 for i in qc.data if i.operation.name == "rzz"
              and not (-1e-9 <= float(i.operation.params[0])
                       <= np.pi / 2 + 1e-9))
    assert bad == 0, f"{bad} rzz angles out of range"
    gp = [i for i in qc.data if i.operation.name == "global_phase"]
    if gp:
        qc.global_phase += sum(float(i.operation.params[0]) for i in gp)
        qc.data = [i for i in qc.data if i.operation.name != "global_phase"]
    return qc


def submit8():
    """Pass 8: vacuum-transfer bias calibration (drift-hardened).
    Job A (CZ, gate-twirled, matches the pooled t=0.5/1.0 sessions):
      v1.0   vacuum prep + gadget + U(1.0)
    Job B (fractional rzz, untwirled, matches passes 5/7):
      v2.0   vacuum prep + gadget + U(2.0)
      m2.0   packet mirror (in-session calibrator for both v2.0, t2.0)
      t2.0   packet physics re-run -> slice-level drift test vs pass 5
    Vacuum truth is exact at every site (energy eigenstate); the
    calibrated-vacuum residual is the transferable bias map, validated
    against the packet runs' wing anchors before application."""
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    from htensor import stateprep
    svc = QiskitRuntimeService(name="edu")
    lat = Z2Lattice(ih.NS, pbc=True)
    TH = stateprep.optimize_vacuum(Z2Lattice(6, pbc=True), ih.M0, ih.G2,
                                   ih.ETA, n_layers=2,
                                   restarts=2)["thetas"]
    vprep = stateprep.vacuum_ansatz(lat, TH)
    pprep = ih.build_prep(lat)

    def blk(t, mirror):
        n = int(round(t / ih.DT))
        return trotter.trotter_circuit(lat, ih.M0, ih.G2, ih.ETA,
                                       ih.MIRROR_EPS if mirror else t, n)

    rec = {"jobs": [], "shots": SHOTS, "pass": 8}
    spec = [("A", False, [("v", "v1.0", 1.0, False)]),
            ("B", True, [("v", "v2.0", 2.0, False),
                         ("p", "m2.0", 2.0, True),
                         ("p", "t2.0", 2.0, False)])]
    for tag, frac, entries in spec:
        be = svc.backend("ibm_kingston", use_fractional_gates=frac)
        pubs, names = [], []
        for src in ("v", "p"):
            segs = [(nm, t, blk(t, mir)) for s, nm, t, mir in entries
                    if s == src]
            if not segs:
                continue
            base, _, _ = ih.t3_segments(lat, vprep if src == "v" else pprep)
            isa = ih.t3_transpile(base, segs, be)
            for name, t, circ, layout in isa:
                if name == "t0.0":
                    continue
                qc = add_readout(circ, layout, lat)
                if frac:
                    qc = _fold_rzz(qc)
                names.append(name)
                pubs.append(qc)
        smp = SamplerV2(mode=be)
        smp.options.default_shots = SHOTS
        smp.options.twirling.enable_gates = not frac
        smp.options.twirling.enable_measure = True
        smp.options.dynamical_decoupling.enable = True
        smp.options.dynamical_decoupling.sequence_type = "XY4"
        job = smp.run(pubs)
        rec["jobs"].append({"job_id": job.job_id(), "names": names})
        print(f"submitted pass 8 job {tag}: {job.job_id()}  pubs: {names}")
    with open("data/hw/losch_pass8.json", "w") as f:
        json.dump(rec, f)


def fetch(job_id):
    from qiskit_ibm_runtime import QiskitRuntimeService
    try:
        svc = QiskitRuntimeService(name="edu")
        job = svc.job(job_id)
    except Exception:                    # pass-10 lives on FNAL
        svc = QiskitRuntimeService()
        job = svc.job(job_id)
    print("status:", job.status())
    if str(job.status()) not in ("DONE", "JobStatus.DONE"):
        return
    res = job.result()
    meta = None
    for f in sorted(__import__("glob").glob("data/hw/losch_pass*.json")):
        m_ = json.load(open(f))
        if "jobs" in m_:                       # pass-6 multi-job format
            for j in m_["jobs"]:
                if j["job_id"] == job_id:
                    meta = {"names": j["names"]}
        elif m_.get("job_id") == job_id or meta is None:
            meta = m_
    out = {}
    for name, pub in zip(meta["names"], res):
        arr = pub.data[list(pub.data.keys())[0]]
        out[name] = np.unpackbits(
            arr.array, axis=-1)[..., -arr.num_bits:].astype(np.uint8) \
            if hasattr(arr, "array") else np.array(arr.get_bitstrings())
    np.savez_compressed(f"data/hw/losch_bits_{job_id}.npz", **out)
    try:
        print("usage:", job.usage())
    except Exception:
        print("usage: (metrics unavailable)")
    print(f"saved data/hw/losch_bits_{job_id}.npz")


if __name__ == "__main__":
    m = sys.argv[1]
    if m == "audit":
        arg = sys.argv[2] if len(sys.argv) > 2 else ""
        audit(pass3=(arg == "3"), rzz=(arg == "rzz"))
    elif m == "submit":
        if int(sys.argv[2]) == 6:
            submit6()
        elif int(sys.argv[2]) == 8:
            submit8()
        else:
            submit(int(sys.argv[2]))
    elif m == "fetch":
        fetch(sys.argv[2])
    else:
        raise SystemExit("mode: audit | submit <pass> | fetch <jobid>")
