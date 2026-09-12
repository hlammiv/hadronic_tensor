"""Tier-3 (Hadamard-test W00 integrand) dry run at the k=1.26 production
point, BEFORE spending QPU time.  Answers three questions:

  1. Trotter gap: the MPS truth (w_meson_ns50_k1.26_v3.npz) used dt=0.1;
     the hardware circuit uses dt=0.5.  `ideal` computes the exact
     expectation of the ACTUAL dt=0.5 circuits so device noise and Trotter
     error can be separated in the figure.
  2. Noise damping: `onetraj`/`assemble` push the same circuits through the
     calibrated g*=1 Pauli-trajectory model (p2=0.005/2q, p1=3e-4/1q --
     the scale validated end-to-end by the S(q) self-consistency study)
     and report the raw-ancilla damping per (t, x).
  3. t=0 calibration: `assemble` checks whether rescaling by the measured
     t=0 central-probe damping recovers the wing probes.

Modes:
  ideal              noiseless dt=0.5 correlator at the tier-3 (t,x) grid,
                     + <H> certification of the k=1.26 prep
  ideal50            same, all 50 probe sites (figure-quality reference
                     for the all-site hardware job) -> hw_t3_ideal50_*.npz
  onetraj <seed>     one noise trajectory (3 ancilla circuits only)
                     -> data/tmp_t3traj_<seed>.npz
  assemble           average all tmp_t3traj_*.npz, print the damping /
                     Trotter-gap / calibration report, save summary npz

  PYTHONPATH=. .venv/bin/python scripts/hw_t3_dryrun.py ideal
"""

import glob
import sys
import time

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

from htensor import Z2Lattice, stateprep, wavepacket, backends, trotter
from htensor import currents as cur
from htensor import hamiltonian as ham
from htensor.measure import split_current, controlled_pauli

M0, G2, ETA = 0.7, 1.1, 1.3
NS, CENTER = 50, 24
K0TAG = "k1.26"
TIMES = [0.0, 0.5, 1.0]
DT = 0.5
PROBES = [CENTER + o for o in (-4, -2, 0, 2, 4)]
P2, P1 = 0.005, 3e-4               # g*=1: nominal per-gate depolarizing
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


def noise_transform(circ, rng, p2, p1):
    out = QuantumCircuit(circ.num_qubits)
    for inst in circ.data:
        qs = [circ.find_bit(b).index for b in inst.qubits]
        out.append(inst.operation, qs)
        nn = inst.operation.num_qubits
        if nn == 2 and rng.random() < p2:
            for q in qs:
                p = rng.integers(0, 4)
                (out.x if p == 1 else out.y if p == 2 else out.z
                 if p == 3 else (lambda _: None))(q)
        elif nn == 1 and rng.random() < p1:
            p = rng.integers(1, 4)
            (out.x if p == 1 else out.y if p == 2 else out.z)(qs[0])
    return out


def insertion(lat):
    id_a, terms = split_current(cur.charge_density(lat, CENTER))
    (ins_ops, c_a), = terms
    return id_a, ins_ops, float(np.real(c_a))


def anc_obs(lat, perm, probes=PROBES):
    """Permuted XB/YB/B observables for every probe + bare anc X, Y."""
    from qiskit.quantum_info import SparsePauliOp
    n_tot = lat.n_qubits + 1
    obs = {}
    for v in probes:
        B = cur.charge_density(lat, v)
        for tag, ap in (("XB", "X"), ("YB", "Y"), ("B", "I")):
            op = SparsePauliOp([ap + l for l in B.paulis.to_labels()], B.coeffs)
            obs[f"{tag}_{v}"] = backends.permute_pauli(op, perm, n_tot)
    for ap in ("X", "Y"):
        op = SparsePauliOp([ap + "I" * lat.n_qubits])
        obs[ap] = backends.permute_pauli(op, perm, n_tot)
    return obs


def anc_circuit(lat, t, ins_ops):
    """Gadget + dt=0.5 evolution on the LOGICAL register (ancilla last) --
    byte-for-byte the observable path of ibm_hardware.py tier 3."""
    n_sys = lat.n_qubits
    qc = QuantumCircuit(n_sys + 1)
    anc = n_sys
    qc.h(anc)
    controlled_pauli(qc, anc, ins_ops)
    qc.compose(trotter.trotter_circuit(lat, M0, G2, ETA, t,
                                       int(round(t / DT))),
               qubits=range(n_sys), inplace=True)
    return qc


def run_anc_grid(lat, mps, perm, rng=None, cap=None, trunc=1e-8,
                 probes=PROBES):
    """The 3 ancilla circuits from a stored prep MPS; optional noise rng."""
    n_tot = lat.n_qubits + 1
    _, ins_ops, _ = insertion(lat)
    obs = anc_obs(lat, perm, probes)
    opts = {"method": "matrix_product_state",
            "matrix_product_state_truncation_threshold": trunc,
            "max_parallel_threads": 4}
    if cap is not None:
        opts["matrix_product_state_max_bond_dimension"] = cap
    sim = AerSimulator(**opts)
    rows = {k: [] for k in obs}
    for t in TIMES:
        qcp = backends.permute_circuit(anc_circuit(lat, t, ins_ops),
                                       perm, n_tot)
        tqc = transpile(qcp, basis_gates=backends._AER_BASIS,
                        optimization_level=1)
        if rng is not None:
            tqc = noise_transform(tqc, rng, P2, P1)
        full = QuantumCircuit(n_tot)
        full.set_matrix_product_state(mps)
        full.compose(tqc, inplace=True)
        for lbl, op in obs.items():
            full.save_expectation_value(op, list(range(n_tot)), label=lbl)
        d = sim.run(full).result().data()
        for k in obs:
            rows[k].append(np.real(d[k]))
        log(f"  t={t:3.1f} done  <XB_{CENTER}> = {rows[f'XB_{CENTER}'][-1]:.5f}")
    return {k: np.array(v) for k, v in rows.items()}


def ideal(probes=PROBES, tag=""):
    lat, prep = build_prep()
    id_a, ins_ops, c_a = insertion(lat)
    anc_site = min(ins_ops)
    log("preparing noiseless state MPS (cap 512)")
    mps, perm = backends.prepare_state_mps(lat, prep, anc_site,
                                           cap=512, trunc=1e-10)
    n_tot = lat.n_qubits + 1
    qc = QuantumCircuit(n_tot)
    qc.set_matrix_product_state(mps)
    H50 = ham.build_hamiltonian(lat, M0, G2, ETA)
    qc.save_expectation_value(backends.permute_pauli(H50, perm, n_tot),
                              list(range(n_tot)), label="H")
    sim = AerSimulator(method="matrix_product_state",
                       matrix_product_state_truncation_threshold=1e-8,
                       max_parallel_threads=4)
    Hval = float(np.real(sim.run(qc).result().data()["H"]))
    log(f"<H> of {K0TAG} prep = {Hval:.4f}  (tier-1 truth at this packet)")

    log("ancilla grid (noiseless, dt=0.5 hardware Trotterization)")
    raw = run_anc_grid(lat, mps, perm, rng=None, trunc=1e-8, probes=probes)
    # probe/insert one-point functions from the same stored state
    probe_ops = {f"P_{v}": backends.permute_pauli(
        cur.charge_density(lat, v), perm, n_tot) for v in probes}
    one_pt = np.empty((len(TIMES), len(probes)))
    for i, t in enumerate(TIMES):
        qct = backends.permute_circuit(
            QuantumCircuit(n_tot).compose(
                trotter.trotter_circuit(lat, M0, G2, ETA, t,
                                        int(round(t / DT))),
                qubits=range(lat.n_qubits)), perm, n_tot)
        tqc = transpile(qct, basis_gates=backends._AER_BASIS,
                        optimization_level=1)
        full = QuantumCircuit(n_tot)
        full.set_matrix_product_state(mps)
        full.compose(tqc, inplace=True)
        for lbl, op in probe_ops.items():
            full.save_expectation_value(op, list(range(n_tot)), label=lbl)
        d = sim.run(full).result().data()
        one_pt[i] = [np.real(d[f"P_{v}"]) for v in probes]
        log(f"  1pt t={t:3.1f} done")
    np.savez(f"data/hw_t3_ideal{tag}_{K0TAG}.npz",
             times=np.array(TIMES), probes=np.array(probes),
             id_a=id_a, c_a=c_a, H_prep=Hval, one_pt=one_pt,
             **{k: v for k, v in raw.items()})
    log(f"saved data/hw_t3_ideal{tag}_{K0TAG}.npz")


def onetraj(seed):
    rng = np.random.default_rng(seed)
    lat, prep = build_prep()
    _, ins_ops, _ = insertion(lat)
    anc_site = min(ins_ops)
    log(f"traj seed {seed}: noisy prep (g*=1)")
    mps, perm = backends.prepare_state_mps(
        lat, prep, anc_site, cap=256, trunc=1e-8,
        circuit_transform=lambda c: noise_transform(c, rng, P2, P1))
    raw = run_anc_grid(lat, mps, perm, rng=rng, cap=256, trunc=1e-8)
    np.savez(f"data/tmp_t3traj_{seed}.npz", seed=seed,
             **{k: v for k, v in raw.items()})
    log(f"traj seed {seed} OK")


def assemble():
    lat = Z2Lattice(NS, pbc=True)
    id_a, ins_ops, c_a = insertion(lat)
    idl = np.load(f"data/hw_t3_ideal_{K0TAG}.npz")
    tru = np.load(f"data/w_meson_ns50_{K0TAG}_v3.npz")
    ti = [int(np.argmin(np.abs(tru["times"] - t))) for t in TIMES]
    C_tru = tru["corr_wp"][np.ix_(ti, PROBES)]

    files = sorted(glob.glob("data/tmp_t3traj_*.npz"))
    trajs = [dict(np.load(f)) for f in files]
    log(f"{len(trajs)} trajectories")

    id_b = np.array([split_current(cur.charge_density(lat, v))[0]
                     for v in PROBES])
    A0_id = idl[f"B_{CENTER}"][0]                # <A(0)> ideal (t=0 circuit)

    def sub_id(xb, yb, x, y):
        """<X (x) B_v> = id_b <X_anc> + <X (x) c_b P_b>: the bare-ancilla
        X/Y observables subtract the identity part of the probe."""
        return xb - id_b[None, :] * x[:, None], \
            yb - id_b[None, :] * y[:, None]

    def assemble_C(sx, sy, b, A0):
        """sx, sy = identity-subtracted ancilla signals <X/Y (x) c_b P_b>."""
        C = np.empty((len(TIMES), len(PROBES)), dtype=complex)
        for j, v in enumerate(PROBES):
            C[:, j] = (c_a * (sx[:, j] + 1j * sy[:, j])
                       + id_a * (b[:, j] - id_b[j])
                       + id_b[j] * (A0 - id_a) + id_a * id_b[j])
        return C

    def grid(src, tag):
        return np.stack([src[f"{tag}_{v}"] for v in PROBES], axis=1)

    xb_i, yb_i, b_i = (grid(idl, t) for t in ("XB", "YB", "B"))
    sx_i, sy_i = sub_id(xb_i, yb_i, idl["X"], idl["Y"])
    C_ideal = assemble_C(sx_i, sy_i, b_i, A0_id)

    xb_n = np.mean([grid(d, "XB") for d in trajs], axis=0)
    yb_n = np.mean([grid(d, "YB") for d in trajs], axis=0)
    b_n = np.mean([grid(d, "B") for d in trajs], axis=0)
    x_n = np.mean([d["X"] for d in trajs], axis=0)
    y_n = np.mean([d["Y"] for d in trajs], axis=0)
    xb_sd = np.std([grid(d, "XB") for d in trajs], axis=0) \
        / max(1, np.sqrt(len(trajs)))
    sx_n, sy_n = sub_id(xb_n, yb_n, x_n, y_n)
    A0_n = b_n[0, PROBES.index(CENTER)]
    C_noisy = assemble_C(sx_n, sy_n, b_n, A0_n)

    # t=0 self-calibration.  The whole t=0 slice is known exactly, so the
    # calibration is per-probe and per-sector, fitted to NOTHING at t>0:
    #   kappa_v: ancilla-signal damping (center is structurally kappa=1 to
    #            system noise -- Z_c Z_c = 1 -- so it isolates the ancilla
    #            sector on hardware);
    #   beta_v : Z-sector damping of <B_v> toward its 0.5 fixed point.
    jc = PROBES.index(CENTER)
    kappa = sx_n[0, jc] / sx_i[0, jc]          # global (ancilla) factor
    kap_v = np.where(np.abs(sx_i[0]) > 0.02, sx_n[0] / sx_i[0], np.nan)
    kap_v = np.where(np.isnan(kap_v), np.nanmedian(kap_v), kap_v)
    beta_v = np.where(np.abs(b_i[0] - 0.5) > 0.02,
                      (b_n[0] - 0.5) / (b_i[0] - 0.5), 1.0)
    b_cal = 0.5 + (b_n - 0.5) / beta_v[None, :]
    C_cal = assemble_C(sx_n / kap_v[None, :], sy_n / kap_v[None, :],
                       b_cal, b_cal[0, jc])

    print(f"\n=== tier-3 dry run, {K0TAG}, g*=1 noise, {len(trajs)} traj ===")
    print(f"<H>(prep) = {float(idl['H_prep']):.4f}")
    print(f"ancilla-sector kappa (t=0 center) = {kappa:.3f}")
    print(f"per-probe kappa_v = {kap_v.round(3)}")
    print(f"per-probe beta_v  = {beta_v.round(3)}")
    print(f"{'t':>4} {'x-c':>4} | {'truth':>8} {'ideal.5':>8} {'noisy':>8} "
          f"{'calib':>8} | {'damp':>6} {'trajSE':>7}")
    for i, t in enumerate(TIMES):
        for j, v in enumerate(PROBES):
            dmp = sx_n[i, j] / sx_i[i, j] if abs(sx_i[i, j]) > 1e-4 else np.nan
            print(f"{t:4.1f} {v-CENTER:+4d} | {C_tru[i, j].real:8.4f} "
                  f"{C_ideal[i, j].real:8.4f} {C_noisy[i, j].real:8.4f} "
                  f"{C_cal[i, j].real:8.4f} | {dmp:6.3f} {xb_sd[i, j]:7.4f}")
    trot = np.abs(C_ideal.real - C_tru.real)
    print(f"\nTrotter gap |ideal(dt=.5)-truth(dt=.1)|: max {trot.max():.4f}, "
          f"central col {trot[:, PROBES.index(CENTER)].round(4)}")
    err_cal = np.abs(C_cal.real - C_tru.real)
    print(f"calibrated-vs-truth: max {err_cal.max():.4f}  "
          f"(shot noise at 100k shots ~ 0.003/kappa = "
          f"{0.003/kappa:.4f} per raw obs)")
    np.savez(f"data/hw_t3_dryrun_{K0TAG}.npz",
             times=np.array(TIMES), probes=np.array(PROBES),
             C_truth=C_tru, C_ideal=C_ideal, C_noisy=C_noisy, C_cal=C_cal,
             kappa=kappa, kap_v=kap_v, beta_v=beta_v, ntraj=len(trajs),
             sx_ideal=sx_i, sx_noisy=sx_n, xb_sd=xb_sd)
    log(f"saved data/hw_t3_dryrun_{K0TAG}.npz")


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "ideal":
        ideal()
    elif mode == "ideal50":
        TIMES = [0.0, 0.5, 1.0, 2.0]   # includes the part-d slice
        ideal(probes=list(range(NS)), tag="50")
    elif mode == "onetraj":
        onetraj(int(sys.argv[2]))
    elif mode == "assemble":
        assemble()
    else:
        raise SystemExit(f"unknown mode {mode}")
