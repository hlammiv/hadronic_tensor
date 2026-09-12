"""Vacuum charge-profile certificate -- the missing half of the meson
sigma-term.  Same tier-1 observables as the packet certificate (Gauss
witnesses G_n, charge profile J0_v, energy H) but on the VACUUM prep (no
wavepacket block).  Subtracting a same-device vacuum profile from the
existing packet certificate cancels the common (readout + shared-prep)
staggered bias, so the staggered sum

    sigma = sum_v (-1)^v [ <J0(v)>_packet - <J0(v)>_vacuum ] = dM/dm0

is recoverable (target 1.88; MPS packet gives 1.86).

  PYTHONPATH=. .venv/bin/python scripts/hw_vacuum_cert.py test          # MPS validation
  PYTHONPATH=. .venv/bin/python scripts/hw_vacuum_cert.py audit fez     # offline transpile/shot/depth
  PYTHONPATH=. .venv/bin/python scripts/hw_vacuum_cert.py submit fez [shots]   # hardware (budget-guarded)

Only J0_v (all diagonal, single Z basis) is submitted by default -- the
minimal, cheapest set for the sigma-term.  Add 'full' to also take G_n + H.
"""
import sys
import os
import json
import time

import numpy as np

from htensor import Z2Lattice, stateprep, backends
from htensor import currents as cur
from htensor import hamiltonian as ham

NS, M0, G2, ETA = 50, 0.7, 1.1, 1.3
CENTER = 24                  # production packet center (matches the packet cert)
TRUNC = 1e-10
DEFAULT_SHOTS = 8192
BUDGET_GUARD = 0.85          # cancel before running if estimate > 85% of remaining s
PACKET_NPZ = "data/w_meson_ns50_k0.00_v3.npz"   # rest packet, for the sigma cross-check


def log(m):
    print(f"[vac-cert] {m}", flush=True)


def build_vac_prep(lat, matched=False):
    """Vacuum variational ansatz.  matched=True appends the wavepacket block
    at ZERO angle -- physically identity, but the same gate skeleton/depth as
    the packet prep, so the packet's in-circuit damping cancels in the
    connected <J0>_packet - <J0>_vacuum difference (cf. the tier-3 mirror)."""
    TH = stateprep.optimize_vacuum(Z2Lattice(6, pbc=True), M0, G2, ETA,
                                   n_layers=2, restarts=2)["thetas"]
    prep = stateprep.vacuum_ansatz(lat, TH)
    if matched:
        from htensor import wavepacket
        z = np.load("data/wp10reg_params_k0.00_L3.npz", allow_pickle=True)
        params = wavepacket.params_from_vector(np.zeros_like(z["vec"]),
                                               list(z["offsets"]), int(z["L"]))
        prep.compose(wavepacket.block_circuit(lat, CENTER, params), inplace=True)
    return prep


def observables(lat, full):
    obs = [(f"J0_{v}", cur.charge_density(lat, v)) for v in range(NS)]
    if full:
        obs = [(f"G{n}", ham.gauss_operator(lat, n)) for n in range(NS)] + obs
        obs += [("H", ham.build_hamiltonian(lat, M0, G2, ETA))]
    return obs


# ---------------------------------------------------------------- MPS test
def test_mps(full=True, matched=False):
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator
    lat = Z2Lattice(NS, pbc=True)
    t0 = time.time()
    prep = build_vac_prep(lat, matched)
    log(f"prep: {'depth-matched (zero-angle mirror)' if matched else 'bare vacuum'}, "
        f"2q depth {prep.depth(lambda i: i.operation.num_qubits == 2)}")
    obs = observables(lat, full)
    qc = prep.copy()
    for name, op in obs:
        qc.save_expectation_value(op, list(range(lat.n_qubits)), label=name)
    sim = AerSimulator(method="matrix_product_state",
                       matrix_product_state_truncation_threshold=TRUNC,
                       max_parallel_threads=4)
    res = sim.run(qc).result().data()
    log(f"MPS vacuum prep + {len(obs)} observables in {time.time()-t0:.0f}s")

    sgn = (-1.0) ** np.arange(NS)
    J = np.array([np.real(res[f"J0_{v}"]) for v in range(NS)])
    sig_vac = float(np.sum(sgn * J))
    rho = float(np.mean(sgn * J))          # <J0(v)> = (-1)^v rho
    log(f"vacuum profile: <J0(v)> = (-1)^v * rho, rho = {rho:+.4f} "
        f"(expect ~0.0868);  staggered sum sum(-1)^v<J0>_vac = {sig_vac:+.4f} (expect ~4.34)")
    if full:
        G = np.array([np.real(res[f"G{n}"]) for n in range(NS)])
        log(f"Gauss witnesses <G_n>: mean {G.mean():+.4f}, min {G.min():+.4f} "
            f"(ideal 1.0);  <H> = {np.real(res['H']):+.4f} (expect ~-52.22)")

    # cross-check against the stored production vacuum profile
    d = np.load(PACKET_NPZ)
    J_vac_ref = np.asarray(d["one_pt_vac"])[0, :NS].real
    J_wp = np.asarray(d["one_pt_wp"])[0, :NS].real
    log(f"max|this vacuum - stored one_pt_vac| = {np.max(np.abs(J - J_vac_ref)):.2e} "
        f"(should be ~0 -> same state)")
    sigma_here = float(np.sum(sgn * (J_wp - J)))
    sigma_ref = float(np.sum(sgn * (J_wp - J_vac_ref)))
    log(f"connected sigma-term  sum(-1)^v[<J0>_packet - <J0>_vac]:")
    log(f"    using THIS vacuum  = {sigma_here:+.4f}")
    log(f"    using stored vac   = {sigma_ref:+.4f}   (target dM/dm0 = 1.880, MPS = 1.86)")
    return J


# -------------------------------------------------------- offline audit
def fake_backend(name):
    from qiskit_ibm_runtime.fake_provider import FakeFez, FakeTorino
    return FakeFez() if name == "fez" else FakeTorino()


def isa_transpile(circ, backend, opt=3):
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    pm = generate_preset_pass_manager(backend=backend, optimization_level=opt,
                                      seed_transpiler=7)
    return pm.run(circ)


def audit(backend_name="fez", full=False, matched=False):
    be = fake_backend(backend_name)
    lat = Z2Lattice(NS, pbc=True)
    prep = build_vac_prep(lat, matched)
    obs = observables(lat, full)
    tc = isa_transpile(prep, be, opt=1 if matched else 3)
    two_q = sum(v for k, v in tc.count_ops().items() if k in ("cz", "ecr", "cx"))
    log(f"audit {be.name}: prep depth {tc.depth()}, 2q gates {two_q}, "
        f"2q depth {tc.depth(lambda i: i.operation.num_qubits == 2)}, "
        f"{len(obs)} observables ({'G+J0+H' if full else 'J0 only'})")


# -------------------------------------------------------- hardware submit
def submit(backend_name="fez", shots=DEFAULT_SHOTS, full=False, matched=False):
    from qiskit_ibm_runtime import EstimatorV2, QiskitRuntimeService
    service = QiskitRuntimeService()
    if not backend_name.startswith('ibm_'):
        backend_name = 'ibm_' + backend_name
    be = service.backend(backend_name)
    lat = Z2Lattice(NS, pbc=True)
    prep = build_vac_prep(lat, matched)
    obs = observables(lat, full)
    tc = isa_transpile(prep, be, opt=1 if matched else 3)
    layout_obs = [o.apply_layout(tc.layout) for _, o in obs]
    log(f"backend {be.name}: {len(obs)} observables, {shots} shots, "
        f"2q depth {tc.depth(lambda i: i.operation.num_qubits == 2)}")

    est = EstimatorV2(mode=be)
    est.options.default_shots = shots
    est.options.dynamical_decoupling.enable = True
    est.options.dynamical_decoupling.sequence_type = "XY4"
    est.options.twirling.enable_gates = True
    est.options.twirling.enable_measure = True
    est.options.resilience.measure_mitigation = True
    job = est.run([(tc, layout_obs)])

    # HARD budget guard -- cancel (free) before execution if the platform's
    # own estimate would eat more than BUDGET_GUARD of the remaining seconds.
    try:
        rem = service.usage().get("usage_remaining_seconds")
    except Exception:
        rem = None
    est_s = None
    try:
        ue = job.usage_estimation
        est_s = ue.get("quantum_seconds") if ue else None
    except Exception as e:
        log(f"usage_estimation unavailable ({type(e).__name__})")
    log(f"budget: estimated {est_s} s, remaining {rem} s")
    if est_s is not None and rem is not None and est_s > BUDGET_GUARD * rem:
        job.cancel()
        log(f"CANCELLED {job.job_id()}: estimate {est_s} s > "
            f"{BUDGET_GUARD:.0%} of remaining {rem} s. Lower shots and retry.")
        return

    os.makedirs("data/hw", exist_ok=True)
    meta = {"job_id": job.job_id(), "kind": "vacuum_cert", "backend": be.name,
            "shots": int(shots), "obs_names": [n for n, _ in obs]}
    path = f"data/hw/job_vaccert_{job.job_id()}.json"
    with open(path, "w") as f:
        json.dump(meta, f, indent=1)
    log(f"submitted {job.job_id()} -> {path}")



# ------------------------------------------------ estimate probe (free)
def probe(backend_name="kingston", matched=True, full=False,
          shots_list=(2048, 4096, 8192)):
    """Submit -> read platform usage_estimation -> cancel (free while queued).
    Use a long-queue device so nothing executes before the cancel."""
    from qiskit_ibm_runtime import EstimatorV2, QiskitRuntimeService
    import time as _t
    service = QiskitRuntimeService()
    if not backend_name.startswith('ibm_'):
        backend_name = 'ibm_' + backend_name
    be = service.backend(backend_name)
    lat = Z2Lattice(NS, pbc=True)
    prep = build_vac_prep(lat, matched)
    obs = observables(lat, full)
    tc = isa_transpile(prep, be, opt=1 if matched else 3)
    lo = [o.apply_layout(tc.layout) for _, o in obs]
    try:
        rem = service.usage().get("usage_remaining_seconds")
    except Exception:
        rem = None
    log(f"probe {be.name}: pending {be.status().pending_jobs}, remaining {rem}s, "
        f"{len(obs)} obs, 2q depth {tc.depth(lambda i: i.operation.num_qubits==2)}, "
        f"matched={matched}")
    for shots in shots_list:
        est = EstimatorV2(mode=be)
        est.options.default_shots = shots
        est.options.twirling.enable_gates = True
        est.options.twirling.enable_measure = True
        est.options.resilience.measure_mitigation = True
        est.options.dynamical_decoupling.enable = True
        job = est.run([(tc, lo)])
        es = None
        for _ in range(10):
            try:
                ue = job.usage_estimation
                es = ue.get("quantum_seconds") if ue else None
            except Exception:
                es = None
            if es is not None:
                break
            _t.sleep(3)
        job.cancel()
        log(f"  shots {shots}: estimated {es} s   (cancelled {job.job_id()})")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "test"
    full = "full" in sys.argv
    matched = "matched" in sys.argv
    if cmd == "test":
        test_mps(full=True, matched=matched)
    elif cmd == "audit":
        audit(sys.argv[2] if len(sys.argv) > 2 else "fez", full=full, matched=matched)
    elif cmd == "probe":
        probe(sys.argv[2] if len(sys.argv) > 2 else "kingston", matched=matched)
    elif cmd == "submit":
        bk = sys.argv[2] if len(sys.argv) > 2 else "fez"
        sh = int([a for a in sys.argv[3:] if a.isdigit()][0]) \
            if any(a.isdigit() for a in sys.argv[3:]) else DEFAULT_SHOTS
        submit(bk, sh, full=full, matched=matched)
    else:
        raise SystemExit(__doc__)
