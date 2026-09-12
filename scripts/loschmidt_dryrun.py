"""Loschmidt-mirror gating study (task #21): does a forward-backward
U(t)U(-t) mirror cure the wing/forward bias that the theta->0 mirror
leaves, and what does gauge-patch post-selection buy?  All local (Aer
MPS + the validated Pauli-trajectory noise model of hw_t3_dryrun.py);
NO hardware jobs.

Modes:
  ideal            noiseless reference C(t,x) at the wing-extended probe
                   set -> data/losch_ideal.npz  (also prints transpiled
                   CZ counts / est. durations for the QPU budget check)
  traj <s0> <s1>   noise trajectories for seeds s0..s1-1 (one shared
                   prep MPS) -> data/tmp_losch_<seed>.npz
  assemble         average trajectories; report kappa per mirror kind,
                   wing-bias of calibrated physics per kind, uninvertible
                   counts, and the patch post-selection (k = 0,1,3,5)
                   bias/keep table -> data/losch_summary.npz

  PYTHONPATH=. .venv/bin/python scripts/loschmidt_dryrun.py ideal
"""
import glob
import sys
import time

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

sys.path.insert(0, "scripts")
import hw_t3_dryrun as base
from hw_t3_dryrun import build_prep, insertion, anc_obs, log
from htensor import backends, trotter
from htensor.measure import controlled_pauli

M0, G2, ETA = base.M0, base.G2, base.ETA
CENTER, DT, P2, P1 = base.CENTER, base.DT, base.P2, base.P1
PROBES = [CENTER + o for o in (-16, -12, -8, -4, 0, 4, 8, 12, 16)]
TS = [0.5, 1.0]
KINDS = ("phys", "mir0", "losch")
CAP = 256


def variant_circuit(lat, t, ins_ops, kind):
    n_sys = lat.n_qubits
    qc = QuantumCircuit(n_sys + 1)
    qc.h(n_sys)
    controlled_pauli(qc, n_sys, ins_ops)
    n = int(round(t / DT))
    if kind == "phys":
        blocks = [trotter.trotter_circuit(lat, M0, G2, ETA, t, n)]
    elif kind == "mir0":                    # zero-angle skeleton, 1x depth
        blocks = [trotter.trotter_circuit(lat, 1e-9, 1e-9, 1e-9, t, n)]
    else:                                   # losch: U(t)U(-t), 2x depth
        blocks = [trotter.trotter_circuit(lat, M0, G2, ETA, t, n),
                  trotter.trotter_circuit(lat, M0, G2, ETA, -t, n)]
    for b in blocks:
        qc.compose(b, qubits=range(n_sys), inplace=True)
    return qc


def noise_with_record(circ, rng, p2, p1):
    """base.noise_transform + record of hit qubit indices (chain frame)."""
    out = QuantumCircuit(circ.num_qubits)
    hits = []
    for inst in circ.data:
        qs = [circ.find_bit(b).index for b in inst.qubits]
        out.append(inst.operation, qs)
        nn = inst.operation.num_qubits
        if nn == 2 and rng.random() < p2:
            for q in qs:
                p = rng.integers(0, 4)
                if p:
                    (out.x if p == 1 else out.y if p == 2 else out.z)(q)
                    hits.append(q)
        elif nn == 1 and rng.random() < p1:
            p = rng.integers(1, 4)
            (out.x if p == 1 else out.y if p == 2 else out.z)(qs[0])
            hits.append(qs[0])
    return out, sorted(set(hits))


def circuits_and_obs():
    lat, prep = build_prep()
    n_tot = lat.n_qubits + 1
    mps, perm = backends.prepare_state_mps(lat, prep, CENTER * 2, cap=CAP,
                                           trunc=1e-10, max_threads=1,
                                           sim_opts={"mps_lapack": True})
    log(f"prep MPS ready (cap={CAP})")
    _, ins_ops, _ = insertion(lat)
    obs = anc_obs(lat, perm, PROBES)
    tqcs = {}
    for t in TS + [0.0]:
        for kind in (KINDS if t > 0 else ("phys",)):
            qcp = backends.permute_circuit(
                variant_circuit(lat, t, ins_ops, kind), perm, n_tot)
            tqcs[(t, kind)] = transpile(qcp, basis_gates=backends._AER_BASIS,
                                        optimization_level=1)
    return lat, mps, perm, obs, tqcs


def run_one(sim, mps, n_tot, tqc, obs, rng):
    if rng is not None:
        tqc, hits = noise_with_record(tqc, rng, P2, P1)
    else:
        hits = []
    full = QuantumCircuit(n_tot)
    full.set_matrix_product_state(mps)
    full.compose(tqc, inplace=True)
    for lbl, op in obs.items():
        full.save_expectation_value(op, list(range(n_tot)), label=lbl)
    try:
        d = sim.run(full).result().data()
    except Exception:                       # lapack gesvd failure: retry
        safe = AerSimulator(method="matrix_product_state",
                            matrix_product_state_max_bond_dimension=CAP,
                            matrix_product_state_truncation_threshold=1e-8,
                            max_parallel_threads=1)
        d = safe.run(full).result().data()
    return {k: float(np.real(d[k])) for k in obs}, hits


def make_sim():
    return AerSimulator(method="matrix_product_state",
                        matrix_product_state_max_bond_dimension=CAP,
                        matrix_product_state_truncation_threshold=1e-8,
                        max_parallel_threads=1, mps_lapack=True)


def ideal():
    lat, mps, perm, obs, tqcs = circuits_and_obs()
    n_tot = lat.n_qubits + 1
    sim = make_sim()
    out = {}
    for (t, kind), tqc in sorted(tqcs.items()):
        n2q = sum(v for g, v in tqc.count_ops().items()
                  if g in ("cx", "cz"))
        # rough Heron duration: 68 ns/2q gate serialized over depth
        dur = tqc.depth() * 8e-8 + 2.5e-4
        log(f"t={t} {kind}: {n2q} 2q gates, depth {tqc.depth()}, "
            f"~{dur*1e6:.0f} us/shot -> {dur*7e4/60:.1f} min @ 7e4 shots")
        r, _ = run_one(sim, mps, n_tot, tqc, obs, None)
        for k, v in r.items():
            out[f"{t}_{kind}_{k}"] = v
        log(f"  done")
    np.savez("data/losch_ideal.npz", **out,
             probes=np.array(PROBES), ts=np.array(TS))
    log("saved data/losch_ideal.npz")


def traj(s0, s1):
    lat, mps, perm, obs, tqcs = circuits_and_obs()
    n_tot = lat.n_qubits + 1
    sim = make_sim()
    inv = {c: l for l, c in perm.items()}    # chain -> logical
    for seed in range(s0, s1):
        rng = np.random.default_rng(seed)
        rec = {}
        for (t, kind), tqc in sorted(tqcs.items()):
            if t == 0.0 and kind == "phys" and seed != s0:
                pass                          # still rerun: cheap, per-seed
            r, hits = run_one(sim, mps, n_tot, tqc, obs, rng)
            for k, v in r.items():
                rec[f"{t}_{kind}_{k}"] = v
            rec[f"{t}_{kind}_hits"] = np.array(
                [inv.get(h, -1) for h in hits], dtype=int)
        np.savez(f"data/tmp_losch_{seed}.npz", **rec)
        log(f"seed {seed} saved")


def assemble():
    ideal_d = np.load("data/losch_ideal.npz")
    files = sorted(glob.glob("data/tmp_losch_*.npz"))
    log(f"{len(files)} trajectories")
    trajs = [np.load(f) for f in files]
    XB = lambda t, kind, v, d: d[f"{t}_{kind}_XB_{v}"]
    rep = {}
    for t in TS:
        i0 = {v: float(ideal_d[f"0.0_phys_XB_{v}"]) for v in PROBES}
        ci = {v: float(ideal_d[f"{t}_phys_XB_{v}"]) for v in PROBES}
        for kind in ("mir0", "losch"):
            kap = {v: np.mean([XB(t, kind, v, d) for d in trajs]) / i0[v]
                   for v in PROBES}
            phys = {v: np.mean([XB(t, "phys", v, d) for d in trajs])
                    for v in PROBES}
            cal = {v: phys[v] / kap[v] if abs(kap[v]) > 0.05 else np.nan
                   for v in PROBES}
            wings = [v for v in PROBES if abs(v - CENTER) >= 8]
            wb = np.nanmean([abs(cal[v] / ci[v]) for v in wings
                             if abs(ci[v]) > 1e-4])
            ninv = sum(1 for v in PROBES if abs(kap[v]) <= 0.05)
            rep[f"{t}_{kind}"] = (wb, ninv)
            log(f"t={t} {kind}: wing |cal/ideal| = {wb:.2f}  "
                f"uninvertible {ninv}/{len(PROBES)}  "
                f"kappa(center) = {kap[CENTER]:.3f}")
        # patch post-selection on the physics run, Loschmidt calibration
        for k in (1, 3, 5):
            kept, biases = [], []
            for v in PROBES:
                lo, hi = 2 * (v - k) - 2, 2 * (v + k) + 2
                sel = [d for d in trajs
                       if not np.any((d[f"{t}_phys_hits"] >= lo)
                                     & (d[f"{t}_phys_hits"] <= hi))]
                kept.append(len(sel) / len(trajs))
                if sel and abs(ci[v]) > 1e-4:
                    m = np.mean([XB(t, "phys", v, d) for d in sel])
                    biases.append(abs(m / ci[v]))
            log(f"t={t} postsel k={k}: keep = {np.mean(kept):.2f}, "
                f"raw |phys/ideal| = {np.nanmean(biases):.2f} "
                f"(unselected: "
                f"{np.nanmean([abs(np.mean([XB(t, 'phys', v, d) for d in trajs]) / ci[v]) for v in PROBES if abs(ci[v]) > 1e-4]):.2f})")
    np.savez("data/losch_summary.npz",
             **{f"{k}_wb": v[0] for k, v in rep.items()},
             **{f"{k}_ninv": v[1] for k, v in rep.items()},
             ntraj=len(files))
    log("saved data/losch_summary.npz")


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "ideal":
        ideal()
    elif mode == "traj":
        traj(int(sys.argv[2]), int(sys.argv[3]))
    elif mode == "assemble":
        assemble()
    else:
        raise SystemExit(f"unknown mode {mode}")
