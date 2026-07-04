"""Warm-start L2-regularized retrain of the wavepacket blocks: slide along
the degenerate solution manifold to small-angle (MPS-friendly) parameters,
then re-certify (F at ns=10, sigma_E + gap at ns=12) and time the ns=50
capped-MPS prep.

  PYTHONPATH=. .venv/bin/python scripts/regularize_wp_params.py
"""

import time

import numpy as np
from qiskit import transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator

from htensor import Z2Lattice, exact, stateprep, spectroscopy, wavepacket, \
    block_engine, backends
from htensor import hamiltonian as ham
from htensor.pauli import pauli_term

M0, G2, ETA = 0.7, 1.1, 1.3
SIGMA = 0.75
t0 = time.time()


def log(m):
    print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)


TH = stateprep.optimize_vacuum(Z2Lattice(6, pbc=True), M0, G2, ETA,
                               n_layers=2, restarts=2)["thetas"]
lat10 = Z2Lattice(10, pbc=True)
band10 = spectroscopy.meson_band(lat10, M0, G2, ETA, n_states=14,
                                 matrix_free=True)
vac10 = np.asarray(Statevector.from_instruction(
    stateprep.vacuum_ansatz(lat10, TH)))

lat12 = Z2Lattice(12, pbc=True)
H12 = ham.build_hamiltonian(lat12, M0, G2, ETA)
vac12_sv = Statevector.from_instruction(stateprep.vacuum_ansatz(lat12, TH))
v12 = np.asarray(vac12_sv)
e_vac12 = np.real(np.vdot(v12, exact.apply_pauli_sum(H12, v12)))

lat50 = Z2Lattice(50, pbc=True)
prep50_vac = stateprep.vacuum_ansatz(lat50, TH)
perm50 = backends._chain_perm(backends.ring_chain_order(lat50))

for k0 in (0.0, 2 * np.pi / 5):
    z = np.load(f"data/wp10_params_k{k0:.2f}_L3.npz", allow_pickle=True)
    vec0, offsets, L = z["vec"], list(z["offsets"]), int(z["L"])
    log(f"k0={k0:.3f}: |theta| max {np.abs(vec0).max():.2f}, "
        f"norm {np.linalg.norm(vec0):.2f}, F0 = {float(z['F']):.4f}")
    mix = spectroscopy.optimize_interpolator(lat10, band10, k0=k0,
                                             sigma_x=SIGMA, x0=2)
    target, _ = spectroscopy.meson_wavepacket(lat10, band10, k0=k0,
                                              sigma_x=SIGMA, x0=2,
                                              mix=mix["mix"])
    r = block_engine.train_adjoint(lat10, vac10, target, center=4,
                                   n_layers=L, maxiter=400,
                                   max_offset=max(abs(o) for o in offsets),
                                   l2=5e-4, inits=[np.asarray(vec0)])
    # refine at tiny l2 to recover fidelity lost to the penalty
    r = block_engine.train_adjoint(lat10, vac10, target, center=4,
                                   n_layers=L, maxiter=200,
                                   max_offset=max(abs(o) for o in offsets),
                                   l2=5e-6, inits=[r["vector"]])
    log(f"  regularized: F = {r['fidelity']:.4f}, "
        f"|theta| max {np.abs(r['vector']).max():.2f}, "
        f"norm {np.linalg.norm(r['vector']):.2f}")

    params = wavepacket.params_from_vector(r["vector"], r["offsets"], L)
    psi12 = np.asarray(vac12_sv.evolve(wavepacket.block_circuit(lat12, 6, params)))
    Hpsi = exact.apply_pauli_sum(H12, psi12)
    e = np.real(np.vdot(psi12, Hpsi))
    sig = np.sqrt(max(np.real(np.vdot(Hpsi, Hpsi)) - e**2, 0))
    t2ph = np.angle(np.vdot(psi12, spectroscopy.translate(psi12, lat12)))
    log(f"  ns=12: gap = {e - e_vac12:.4f}, sigma_E = {sig:.4f}, "
        f"arg<T2> = {t2ph:.3f}")

    prep = prep50_vac.copy()
    prep.compose(wavepacket.block_circuit(lat50, 24, params), inplace=True)
    tqc = transpile(backends.permute_circuit(prep, perm50, lat50.n_qubits),
                    basis_gates=backends._AER_BASIS, optimization_level=1)
    for cap in (64, 128, 256):
        sim = AerSimulator(method="matrix_product_state",
                           matrix_product_state_truncation_threshold=1e-8,
                           matrix_product_state_max_bond_dimension=cap,
                           max_parallel_threads=4)
        qc = tqc.copy()
        qc.save_expectation_value(
            backends.permute_pauli(pauli_term(lat50.n_qubits, {48: "Z"}, 1.0),
                                   perm50, lat50.n_qubits),
            list(range(lat50.n_qubits)), label="Z24")
        t1 = time.time()
        d = sim.run(qc).result().data()
        log(f"  ns=50 cap={cap}: {time.time()-t1:5.1f}s  "
            f"Z24 = {np.real(d['Z24']):+.6f}")
    np.savez(f"data/wp10reg_params_k{k0:.2f}_L3.npz", vec=r["vector"],
             offsets=r["offsets"], L=L, F=r["fidelity"], sigma=SIGMA, k0=k0)
    log(f"  saved data/wp10reg_params_k{k0:.2f}_L3.npz")
log("done")
