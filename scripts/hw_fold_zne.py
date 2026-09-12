"""ISA-level noise folding for the equal-time connected correlator.

The t=0 slice of the W00 integrand is bias-limited: prep-circuit noise
damps the connected screening cloud <Z_v Z_c>_conn ~3x while shot noise is
0.002.  Fold the TRANSPILED prep U -> U (U^dag U)^k  (k=0,1,2 -> noise
scale 1x,3x,5x) and extrapolate each pair to zero noise.  Folding happens
strictly at ISA level (exact gate-by-gate inversion, O0 basis translation
only) so no optimizer can cancel the fold -- same discipline as the tier-3
mirrors.  All three folds go in ONE sampler job (shared calibration/drift),
gate twirling ON so the noise ladder is Pauli-ized (exponential in k).

Modes:
  check [fez]        offline: build folds, assert 2q counts 1:3:5, MPS-
                     validate fold-3 == fold-1 == truth expectations.
  submit <backend> [shots]   one job, three pubs (default 50k each).
  analyze <jobfile>  per-pair exponential ZNE -> data/hw/zne_t0_*.npz.

  PYTHONPATH=. .venv/bin/python scripts/hw_fold_zne.py check
"""

import json
import os
import sys
import time

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, transpile

from htensor import Z2Lattice, stateprep, wavepacket

M0, G2, ETA = 0.7, 1.1, 1.3
NS, CENTER = 50, 24
K0TAG = "k1.26"
RO01, RO10 = 0.012, 0.028
FOLDS = [0, 1, 2]                  # noise scales 1, 3, 5
BASIS = ["cz", "rz", "sx", "x", "id"]
t0 = time.time()


def log(m):
    print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)


def build_prep():
    lat = Z2Lattice(NS, pbc=True)
    TH = stateprep.optimize_vacuum(Z2Lattice(6, pbc=True), M0, G2, ETA,
                                   n_layers=2, restarts=2)["thetas"]
    z = np.load(f"data/wp10reg_params_{K0TAG}_L3.npz", allow_pickle=True)
    params = wavepacket.params_from_vector(z["vec"], list(z["offsets"]),
                                           int(z["L"]))
    prep = stateprep.vacuum_ansatz(lat, TH)
    prep.compose(wavepacket.block_circuit(lat, CENTER, params), inplace=True)
    return lat, prep


def two_q(tc):
    return sum(v for k, v in tc.count_ops().items()
               if k in ("cz", "ecr", "cx"))


def isa_prep(lat, prep, be):
    from qiskit.transpiler.preset_passmanagers import \
        generate_preset_pass_manager
    pm = generate_preset_pass_manager(backend=be, optimization_level=3,
                                      seed_transpiler=7)
    tc = pm.run(prep)
    return tc, list(tc.layout.final_index_layout())


def fold_circuit(isa_P, k):
    """U (U^dag U)^k at ISA level; O0 = translation only, no cancellation."""
    inv = isa_P.inverse()
    qc = isa_P.copy()
    for _ in range(k):
        qc.barrier()
        qc.compose(inv, inplace=True)
        qc.barrier()
        qc.compose(isa_P, inplace=True)
    out = transpile(qc, basis_gates=BASIS, optimization_level=0)
    want = (2 * k + 1) * two_q(isa_P)
    got = two_q(out)
    if got != want:
        raise SystemExit(f"fold {k}: 2q count {got} != {want} -- "
                         "a pass cancelled fold gates")
    return out


def with_measure(tc, lat, fl):
    """H on links (X basis) + measure fl[q] -> clbit q: byte-compatible
    with the hw_sq_submit analyze convention."""
    qc = tc.copy()
    qc.add_register(ClassicalRegister(lat.n_qubits, "meas"))
    for n in range(lat.n_links):
        qc.h(fl[lat.link_qubit(n)])
    qc.barrier()
    for q in range(lat.n_qubits):
        qc.measure(fl[q], q)
    # translate the appended H's to the native basis (O0: no optimization)
    return transpile(qc, basis_gates=BASIS, optimization_level=0)


def check(backend_name="fez"):
    from qiskit.quantum_info import SparsePauliOp
    from qiskit_aer import AerSimulator
    from qiskit_ibm_runtime.fake_provider import FakeFez, FakeTorino
    be = FakeFez() if backend_name == "fez" else FakeTorino()
    lat, prep = build_prep()
    isa_P, fl = isa_prep(lat, prep, be)
    log(f"isa prep: {two_q(isa_P)} 2q gates, depth {isa_P.depth()}")
    folds = [fold_circuit(isa_P, k) for k in FOLDS]
    for k, f in zip(FOLDS, folds):
        log(f"fold {2*k+1}x: {two_q(f)} 2q gates, depth {f.depth()} OK")

    tru = np.load(f"data/w_meson_ns50_{K0TAG}_v3.npz")
    one = tru["one_pt_wp"][0, :NS].real
    corr = tru["corr_wp"][0, :NS].real           # <J0(x) J0(c)>, t=0
    probes = [CENTER + o for o in (-4, -2, 0, 2, 4)]
    sim = AerSimulator(method="matrix_product_state",
                       matrix_product_state_truncation_threshold=1e-8,
                       max_parallel_threads=4)
    ref = {}
    for tag, circ in (("1x", folds[0]), ("3x", folds[1])):
        qc = circ.copy()
        nq = circ.num_qubits
        for v in probes:
            lab = ["I"] * nq
            lab[nq - 1 - fl[lat.site_qubit(v)]] = "Z"
            lab[nq - 1 - fl[lat.site_qubit(CENTER)]] = "Z"
            if v == CENTER:
                lab[nq - 1 - fl[lat.site_qubit(v)]] = "I"  # Z.Z=I on same q
            qc.save_expectation_value(SparsePauliOp("".join(lab)),
                                      list(range(nq)), label=f"ZZ_{v}")
        d = sim.run(qc).result().data()
        vals = {v: float(np.real(d[f"ZZ_{v}"])) for v in probes}
        ref[tag] = vals
        log(f"[{tag}] " + " ".join(f"ZZ_{v}={vals[v]:+.4f}" for v in probes))
    worst = 0.0
    for v in probes:
        # truth <Z_v Z_c> from J0 grids: Z = (-1)^v - 2 J0
        sv, sc = (-1) ** v, (-1) ** CENTER
        zz_t = (sv * sc - 2 * sv * one[CENTER] - 2 * sc * one[v]
                + 4 * corr[v]) if v != CENTER else 1.0
        for tag in ("1x", "3x"):
            worst = max(worst, abs(ref[tag][v] - zz_t))
        if abs(ref["1x"][v] - ref["3x"][v]) > 2e-3:
            raise SystemExit(f"fold not identity at v={v}: "
                             f"{ref['1x'][v]:.4f} vs {ref['3x'][v]:.4f}")
    if worst > 5e-3:
        raise SystemExit(f"fold vs truth mismatch: {worst:.2e}")
    log(f"check PASSED: folds are identity (worst vs truth {worst:.1e})")


def submit(backend_name, shots=50000):
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    service = QiskitRuntimeService()
    be = service.backend(backend_name)
    lat, prep = build_prep()
    isa_P, fl = isa_prep(lat, prep, be)
    pubs = []
    for k in FOLDS:
        qc = with_measure(fold_circuit(isa_P, k), lat, fl)
        pubs.append(qc)
        log(f"pub {2*k+1}x: {two_q(qc)} 2q gates, depth {qc.depth()}")
    sampler = SamplerV2(mode=be)
    sampler.options.default_shots = shots
    sampler.options.twirling.enable_measure = True
    sampler.options.twirling.enable_gates = True   # Pauli-ize: exp ladder
    sampler.options.dynamical_decoupling.enable = True
    sampler.options.dynamical_decoupling.sequence_type = "XY4"
    job = sampler.run(pubs)
    try:
        rem = service.usage()["usage_remaining_seconds"]
        ue = job.usage_estimation
        est = ue.get("quantum_seconds") if ue else None
        log(f"budget: estimated {est} s, remaining {rem} s")
        if est is not None and est > 0.85 * rem:
            job.cancel()
            log(f"CANCELLED {job.job_id()}: over budget guard")
            return
    except Exception as e:
        log(f"usage guard unavailable ({type(e).__name__})")
    os.makedirs("data/hw", exist_ok=True)
    meta = {"job_id": job.job_id(), "backend": be.name, "shots": shots,
            "folds": FOLDS, "creg": "meas"}
    path = f"data/hw/zne_job_{job.job_id()}.json"
    json.dump(meta, open(path, "w"), indent=1)
    log(f"SUBMITTED {job.job_id()} -> {path}")


def analyze(jobfile):
    from qiskit_ibm_runtime import QiskitRuntimeService
    meta = json.load(open(jobfile))
    lat = Z2Lattice(NS, pbc=True)
    res = QiskitRuntimeService().job(meta["job_id"]).result()
    sites = np.array([lat.site_qubit(v) for v in range(NS)])
    sgn = np.array([(-1) ** v for v in range(NS)])
    g, ge, one = [], [], []
    for pub in res:
        ba = pub.data[meta["creg"]]
        cnt = ba.get_counts()
        bits = np.vstack([np.tile(np.frombuffer(bs[::-1].encode(), np.uint8)
                                  - ord("0"), (c, 1))
                          for bs, c in cnt.items()]).astype(np.int8)
        z = 1 - 2 * bits[:, sites]
        j0 = (sgn[None, :] - z) / 2
        o = j0.mean(0)
        prod = j0 * j0[:, [CENTER]]
        gk = prod.mean(0) - o * o[CENTER]
        g.append(gk)
        ge.append((prod - o[None, :] * o[CENTER]).std(0) / np.sqrt(len(j0)))
        one.append(o)
    g, ge, one = np.array(g), np.array(ge), np.array(one)
    scale = 2 * np.array(meta["folds"]) + 1              # 1, 3, 5
    # per-pair exponential ZNE: g_k = r * d^scale * g0 -> in log space a
    # straight line; readout factor r survives extrapolation and is removed
    # analytically with the validated asymmetric-readout constants.
    r_ro = (1 - RO01 - RO10) ** 2
    g0 = np.full(NS, np.nan)
    g0e = np.full(NS, np.nan)
    for v in range(NS):
        y = g[:, v]
        if np.abs(y[0]) < 3 * ge[0, v] or np.any(np.sign(y) != np.sign(y[0])):
            g0[v], g0e[v] = y[0] / r_ro, ge[0, v] / r_ro   # unextrapolable
            continue
        w = np.polyfit(scale, np.log(np.abs(y)), 1, w=np.abs(y) / ge[:, v])
        g0[v] = np.sign(y[0]) * np.exp(np.polyval(w, 0.0)) / r_ro
        # error: shot noise scaled by the extrapolation gain + fit residual
        gain = abs(g0[v] * r_ro / y[0])
        resid = np.abs(np.log(np.abs(y)) - np.polyval(w, scale)).max()
        g0e[v] = abs(g0[v]) * np.sqrt((ge[0, v] / abs(y[0])) ** 2
                                      + resid ** 2) * max(1.0, gain)
    tru = np.load(f"data/w_meson_ns50_{K0TAG}_v3.npz")
    one_t = tru["one_pt_wp"][0, :NS].real
    corr_t = tru["corr_wp"][0, :NS].real
    g_t = corr_t - one_t * one_t[CENTER]
    for v in range(CENTER - 4, CENTER + 5):
        log(f"v-c={v-CENTER:+2d}: raw {g[0, v]:+.4f} zne {g0[v]:+.4f} "
            f"+/-{g0e[v]:.4f}  truth {g_t[v]:+.4f}")
    cloud = np.abs(np.arange(NS) - CENTER) <= 4
    log(f"cloud |diff| raw {np.abs(g[0] - g_t)[cloud].mean():.4f} -> "
        f"zne {np.abs(g0 - g_t)[cloud].mean():.4f}")
    np.savez(f"data/hw/zne_t0_{K0TAG}.npz", g=g, ge=ge, one=one, g0=g0,
             g0e=g0e, scale=scale, r_ro=r_ro, backend=meta["backend"],
             nshot=meta["shots"])
    log(f"saved data/hw/zne_t0_{K0TAG}.npz")


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "check":
        check(sys.argv[2] if len(sys.argv) > 2 else "fez")
    elif mode == "submit":
        submit(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 50000)
    elif mode == "analyze":
        analyze(sys.argv[2])
    else:
        raise SystemExit(f"unknown mode {mode}")
