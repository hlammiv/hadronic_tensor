"""Submission circuit + classical estimator for the static structure factor
S(q^1) on 101 qubits -- the prep-only, one-basis, all-momenta hardware
observable.  Built and validated here; NOT submitted.

Pipeline (exactly what a hardware run does):
  circuit  = prep (vacuum ansatz + wavepacket block); H on every link qubit
             (X basis for the Gauss stabilizers); measure all 101 qubits.
  estimator= per shot, z_v = 1-2 bit(site v); the momentum-projected
             magnetization rho(q) = sum_v e^{i q x_v} z_v is a complex scalar,
             and S(q) = [<|rho(q)|^2> - |<rho(q)>|^2]/N_x is its shot
             variance.  Gauss witnesses G_n from the same bitstrings post-
             select the physical sector.

Modes:
  audit                  transpile vs a fake Heron target; depth / 2q gates /
                         shots and QPU-seconds for an 11 s budget.
  simval <ntraj> <shots> noisy Pauli-trajectory sim WITH bitstring sampling
                         and readout error -> S(q), mitigation, recovery vs
                         the exact curve (the real end-to-end validation).
  submit / analyze       hardware hooks (defined, never auto-invoked).

  PYTHONPATH=. .venv/bin/python scripts/hw_sq_submit.py audit
"""

import sys
import time

import numpy as np
from qiskit import QuantumCircuit, transpile

from htensor import Z2Lattice, stateprep, wavepacket, backends
from htensor import currents as cur
from htensor.measure import split_current

M0, G2, ETA = 0.7, 1.1, 1.3
NS, CENTER = 50, 24
K0TAG = "k1.26"
P2, P1 = 0.005, 3e-4
RO01, RO10 = 0.012, 0.028      # asymmetric readout (amplitude-damping bias)
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


def measured_circuit(lat, prep):
    """prep + H on link qubits (X basis) + measure all."""
    qc = prep.copy()
    for n in range(lat.n_links):
        qc.h(lat.link_qubit(n))
    qc.measure_all()
    return qc


def estimator(bitstrings, lat, QS, ro=True):
    """counts -> S(q) and Gauss witnesses.  bitstrings: (Nshot, nq) uint8,
    bit=1 means |1> (Z=-1).  Qubit index i in the string is qubit i."""
    if ro:                        # apply asymmetric readout error to samples
        rng = np.random.default_rng(7)
        b = bitstrings.copy()
        flip0 = (b == 0) & (rng.random(b.shape) < RO01)
        flip1 = (b == 1) & (rng.random(b.shape) < RO10)
        b = b ^ (flip0 | flip1)
    else:
        b = bitstrings
    sites = np.array([lat.site_qubit(v) for v in range(lat.ns)])
    zv = 1 - 2 * b[:, sites]                      # (Nshot, ns) in +-1
    xv = 2 * np.pi * (np.arange(lat.ns) // 2) / lat.nx
    S = []
    for q in QS:
        rho = zv @ np.exp(1j * q * xv)            # (Nshot,) complex
        S.append((np.mean(np.abs(rho) ** 2) - np.abs(np.mean(rho)) ** 2)
                 / lat.nx)
    # Gauss witness G_n = (-1)^n Z_n X_{n-1} X_n (links already in X basis)
    G = []
    for n in range(lat.ns):
        val = (-1) ** n * (1 - 2 * b[:, lat.site_qubit(n)])
        val = val * (1 - 2 * b[:, lat.link_qubit(n - 1)])
        val = val * (1 - 2 * b[:, lat.link_qubit(n)])
        G.append(np.mean(val))
    return np.array(S).real, np.array(G)


def noise_transform(circ, rng):
    out = QuantumCircuit(circ.num_qubits)
    for inst in circ.data:
        qs = [circ.find_bit(b).index for b in inst.qubits]
        out.append(inst.operation, qs)
        nn = inst.operation.num_qubits
        if nn == 2 and rng.random() < P2:
            for q in qs:
                p = rng.integers(0, 4)
                (out.x if p == 1 else out.y if p == 2 else out.z
                 if p == 3 else (lambda _: None))(q)
        elif nn == 1 and rng.random() < P1:
            p = rng.integers(1, 4)
            (out.x if p == 1 else out.y if p == 2 else out.z)(qs[0])
    return out


mode = sys.argv[1]
lat, prep = build_prep()
QS = 2 * np.pi * np.arange(1, lat.nx // 2 + 1) / lat.nx

if mode == "audit":
    from qiskit_ibm_runtime.fake_provider import FakeFez
    from qiskit.transpiler.preset_passmanagers import \
        generate_preset_pass_manager
    be = FakeFez()
    qc = measured_circuit(lat, prep)
    pm = generate_preset_pass_manager(backend=be, optimization_level=3,
                                      seed_transpiler=7)
    tc = pm.run(qc)
    two_q = sum(v for k, v in tc.count_ops().items()
                if k in ("cz", "ecr", "cx"))
    log(f"target {be.name}: depth {tc.depth()}, 2q gates {two_q}, "
        f"2q depth {tc.depth(lambda i: i.operation.num_qubits == 2)}")
    for shots in (2e4, 4e4, 1e5):
        log(f"  {int(shots):>6} shots -> ~{shots*250e-6:.1f} s QPU "
            f"(single circuit, all {len(QS)} momenta)")

elif mode == "simval":
    from qiskit_aer import AerSimulator
    ntraj, shots = int(sys.argv[2]), int(sys.argv[3])
    per = shots // ntraj
    anc = min(split_current(cur.charge_density(lat, CENTER))[1][0][0])
    allbits = []
    ss = np.random.SeedSequence(2024)
    for t in range(ntraj):
        rng = np.random.default_rng(ss.spawn(1)[0])
        mps, perm = backends.prepare_state_mps(
            lat, prep, anc, cap=256, trunc=1e-8,
            circuit_transform=lambda c: noise_transform(c, rng))
        n = lat.n_qubits + 1
        qc = QuantumCircuit(n, lat.n_qubits)
        qc.set_matrix_product_state(mps)
        for nn in range(lat.n_links):        # links to X basis (permuted)
            qc.h(perm[lat.link_qubit(nn)])
        for v in range(lat.n_qubits):
            qc.measure(perm[v], v)
        sim = AerSimulator(method="matrix_product_state",
                           matrix_product_state_truncation_threshold=1e-8)
        cnt = sim.run(qc, shots=per).result().get_counts()
        for bs, c in cnt.items():
            bits = np.frombuffer(bs[::-1].encode(), np.uint8) - ord("0")
            allbits.append(np.tile(bits, (c, 1)))
        if (t + 1) % 8 == 0:
            log(f"traj {t+1}/{ntraj}, {sum(len(a) for a in allbits)} shots")
    B = np.vstack(allbits)
    S, G = estimator(B, lat, QS)
    Si = np.load(f"data/hwsf_ideal_ns{NS}_{K0TAG}.npz")["S"]
    A = np.vstack([Si, np.ones_like(Si)]).T
    (f, c), *_ = np.linalg.lstsq(A, S, rcond=None)
    Smit = (S - c) / f
    res = (Smit - Si) / Si
    rec = Si > c
    log(f"sampled {len(B)} shots; Gauss mean {G.mean():.3f}")
    log(f"noise fit f={f:.3f} c={c:.3f}; "
        f"recovered {int(np.sum(rec & (np.abs(res) < 0.15)))}/{len(QS)} "
        f"<15%, {int(np.sum(np.abs(res) < 0.01))} <1%")
    np.savez(f"data/hwsq_simval_{K0TAG}.npz", q=QS, S_raw=S, S_mit=Smit,
             S_ideal=Si, G=G, f=f, c=c, nshot=len(B))
    for i, q in enumerate(QS):
        log(f"  q={q:.2f}: exact {Si[i]:.3f} raw {S[i]:.3f} "
            f"mit {Smit[i]:.3f} ({res[i]:+.1%})")
