"""Static structure factor S(q) = <rho(-q) rho(q)>_c on the prepared meson
state -- the equal-time (energy-integrated) hadronic tensor -- with a
genuine 101-qubit noisy simulation (Pauli-trajectory MPS + readout model).

This is the prep-only, certificate-depth observable proposed for hardware:
one circuit, one basis, all momenta.  It is built from SHORT-RANGE density
correlators, so it survives the global fidelity collapse (like the tier-1
Gauss witnesses) up to a near-global damping factor; the momentum SHAPE is
preserved, which the noisy simulation demonstrates via the flat
S_noisy/S_ideal ratio.

Modes:
  ideal <ns> [k0tag]        clean S(q) via MPS
  noisy <ns> <ntraj> [k0tag]  Pauli-trajectory noisy S(q) + mitigation
  plot <k0tag>              overlay figure from saved ideal/noisy npz

  PYTHONPATH=. .venv/bin/python scripts/hw_structure_factor.py noisy 50 64
"""

import sys
import time

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

from htensor import Z2Lattice, stateprep, wavepacket, backends
from htensor import currents as cur
from htensor.pauli import pauli_sum
from htensor.measure import split_current

M0, G2, ETA = 0.7, 1.1, 1.3
BASIS = ["cx", "cz", "rz", "sx", "x", "h", "ry"]
P2, P1 = 0.005, 3e-4          # Heron-representative 2q / 1q depolarizing
RO_ATT = 0.97                 # per-qubit readout attenuation (1 - p01 - p10)
t0 = time.time()


def log(m):
    print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)


def rho_parts(lat, q):
    """Hermitian pieces for rho_Z(q) = sum_v e^{iqx_v} Z_v:
    A = sum cos(qx_v)Z_v, B = sum sin(qx_v)Z_v (so <rho> = <A> + i<B>),
    and RR = rho_Z(-q)rho_Z(q) = rho_Z(q)^dag rho_Z(q) (Hermitian, PSD)."""
    A = pauli_sum(lat.n_qubits, [({lat.site_qubit(v): "Z"},
                                  np.cos(q * (v // 2))) for v in range(lat.ns)])
    B = pauli_sum(lat.n_qubits, [({lat.site_qubit(v): "Z"},
                                  np.sin(q * (v // 2))) for v in range(lat.ns)])
    rq = pauli_sum(lat.n_qubits, [({lat.site_qubit(v): "Z"},
                                   np.exp(1j * q * (v // 2)))
                                  for v in range(lat.ns)])
    RR = (rq.adjoint() @ rq).simplify()
    RR = RR.__class__(RR.paulis, np.real(RR.coeffs))       # force Hermitian
    return A.simplify(), B.simplify(), RR


def build(ns, k0tag):
    lat = Z2Lattice(ns, pbc=True)
    TH = stateprep.optimize_vacuum(Z2Lattice(6, pbc=True), M0, G2, ETA,
                                   n_layers=2, restarts=2)["thetas"]
    z = np.load(f"data/wp10reg_params_{k0tag}_L3.npz", allow_pickle=True)
    params = wavepacket.params_from_vector(z["vec"], list(z["offsets"]),
                                           int(z["L"]))
    c = 2 * (lat.nx // 2)                    # even center site
    prep = stateprep.vacuum_ansatz(lat, TH)
    prep.compose(wavepacket.block_circuit(lat, c, params), inplace=True)
    anc = min(split_current(cur.charge_density(lat, c))[1][0][0])
    return lat, prep, anc


def noise_transform(circ, rng):
    """Depolarizing unravelling: random Paulis after gates (per trajectory)."""
    out = QuantumCircuit(circ.num_qubits)
    for inst in circ.data:
        qs = [circ.find_bit(b).index for b in inst.qubits]
        out.append(inst.operation, qs)
        n = inst.operation.num_qubits
        if n == 2 and rng.random() < P2:
            for q in qs:
                p = rng.integers(0, 4)
                (out.x if p == 1 else out.y if p == 2 else out.z
                 if p == 3 else (lambda _: None))(q)
        elif n == 1 and rng.random() < P1:
            p = rng.integers(1, 4)
            (out.x if p == 1 else out.y if p == 2 else out.z)(qs[0])
    return out


def Sq_from_mps(lat, mps, perm, qops, ro=1.0):
    """S(q) = ro^2 <rho(-q)rho(q)> - |ro <rho(q)>|^2 on a stored MPS."""
    n = lat.n_qubits + 1
    qc = QuantumCircuit(n)
    qc.set_matrix_product_state(mps)
    for i, (A, B, RR) in enumerate(qops):
        for lbl, op in (("A", A), ("B", B), ("RR", RR)):
            qc.save_expectation_value(backends.permute_pauli(op, perm, n),
                                      list(range(n)), label=f"{lbl}{i}")
    sim = AerSimulator(method="matrix_product_state",
                       matrix_product_state_truncation_threshold=1e-10,
                       max_parallel_threads=4)
    d = sim.run(qc).result().data()
    S = []
    for i in range(len(qops)):
        r = float(np.real(d[f"A{i}"])) + 1j * float(np.real(d[f"B{i}"]))
        rr = float(np.real(d[f"RR{i}"]))
        S.append((ro ** 2 * rr - abs(ro * r) ** 2) / lat.nx)
    return np.array(S)


mode = sys.argv[1]
if mode == "plot":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    k0tag = sys.argv[2] if len(sys.argv) > 2 else "k1.26"
    di = np.load(f"data/hwsf_ideal_ns50_{k0tag}.npz")
    dn = np.load(f"data/hwsf_noisy_ns50_{k0tag}.npz")
    q, Si, Sn = di["q"], di["S"], dn["S"]
    ntraj = int(dn["ntraj"])
    # 2-parameter noise model S_noisy = f*S_ideal + c  (damping + floor)
    Amat = np.vstack([Si, np.ones_like(Si)]).T
    (f, c), *_ = np.linalg.lstsq(Amat, Sn, rcond=None)
    Smit = (Sn - c) / f
    err = np.abs(Sn) / np.sqrt(ntraj) / f + 0.02 * np.abs(Si)
    rec = Si > c                                 # signal above the floor
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 3.0),
                                 constrained_layout=True)
    a1.plot(q, Si, "k-", lw=1.4, label="exact (MPS)")
    a1.plot(q, Sn, "x", color="0.65", ms=6, label="noisy sim, raw")
    a1.errorbar(q[rec], Smit[rec], yerr=err[rec], fmt="rs", ms=6, capsize=2,
                label="noisy sim, mitigated")
    a1.errorbar(q[~rec], Smit[~rec], yerr=err[~rec], fmt="s", ms=6,
                mfc="none", mec="r", capsize=2)
    a1.axhline(c, color="C0", ls=":", lw=0.8)
    a1.text(q[0], c + 0.02, "noise floor", fontsize=6.5, color="C0")
    a1.set_xlabel("$q^1$"); a1.set_ylabel(r"$S(q^1)$")
    a1.set_title("(a) structure factor, 101 qubits", fontsize=9)
    a1.legend(fontsize=7)
    res = (Smit - Si) / Si
    a2.axhspan(-0.05, 0.05, color="0.9")
    a2.plot(q[rec], res[rec], "rs", ms=6)
    a2.plot(q[~rec], res[~rec], "s", ms=6, mfc="none", mec="r")
    a2.axhline(0, color="0.5", lw=0.6)
    a2.set_xlabel("$q^1$"); a2.set_ylabel(r"$(S_{\rm mit}-S_{\rm exact})/S_{\rm exact}$")
    a2.set_title("(b) recovery vs momentum", fontsize=9)
    a2.set_ylim(-0.6, 0.6)
    fig.savefig("data/hwsf_figure.pdf", dpi=200)
    ng = int(np.sum(rec & (np.abs(res) < 0.15)))
    print(f"noise model: damping f={f:.3f}, floor c={c:.3f} "
          f"({ntraj} trajectories)")
    print(f"recovered: {ng}/{len(q)} points to <15%, "
          f"{int(np.sum(np.abs(res)<0.01))} to <1%")
    print("wrote data/hwsf_figure.pdf")
    sys.exit()

ns = int(sys.argv[2])
k0tag = sys.argv[-1] if sys.argv[-1].startswith("k") else "k1.26"
lat, prep, anc = build(ns, k0tag)
QS = 2 * np.pi * np.arange(1, lat.nx // 2 + 1) / lat.nx
qops = [rho_parts(lat, q) for q in QS]

if mode == "ideal":
    mps, perm = backends.prepare_state_mps(lat, prep, anc, cap=512, trunc=1e-10)
    S = Sq_from_mps(lat, mps, perm, qops)
    np.savez(f"data/hwsf_ideal_ns{ns}_{k0tag}.npz", q=QS, S=S)
    log(f"ideal S(q) = {np.round(S, 4)}")

elif mode == "noisy":
    ntraj = int(sys.argv[3])
    Sacc = np.zeros(len(QS))
    ss = np.random.SeedSequence(12345)
    for t in range(ntraj):
        rng = np.random.default_rng(ss.spawn(1)[0])
        mps, perm = backends.prepare_state_mps(
            lat, prep, anc, cap=256, trunc=1e-8,
            circuit_transform=lambda c: noise_transform(c, rng))
        Sacc += Sq_from_mps(lat, mps, perm, qops, ro=RO_ATT)
        if (t + 1) % 8 == 0:
            log(f"traj {t+1}/{ntraj}: running mean S = "
                f"{np.round(Sacc/(t+1), 4)}")
    S = Sacc / ntraj
    np.savez(f"data/hwsf_noisy_ns{ns}_{k0tag}.npz", q=QS, S=S, ntraj=ntraj,
             p2=P2, p1=P1, ro=RO_ATT)
    log(f"noisy S(q) = {np.round(S, 4)}")
