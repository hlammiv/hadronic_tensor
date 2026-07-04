"""M4c v2: train the wavepacket block at ns=10 (packet fits: sigma_x=0.75,
window +-4 staggered offsets) and certify volume transfer against exact
band-projected targets at ns=12 (matrix-free ED at 24 qubits).

Rest (k0=0) and boosted (k0=2*pi/5 = 1.257, on the momentum grids of BOTH
ns=10 and the ns=50 production volume).

  PYTHONPATH=. .venv/bin/python scripts/train_and_certify_v2.py
"""

import time

import numpy as np
from qiskit.quantum_info import Statevector

from htensor import Z2Lattice, exact, stateprep, spectroscopy, wavepacket, \
    block_engine
from htensor import hamiltonian as ham

M0, G2, ETA = 0.7, 1.1, 1.3
SIGMA = 0.75
K0S = (0.0, 2 * np.pi / 5)
t0 = time.time()


def log(msg):
    print(f"[{time.time()-t0:6.0f}s] {msg}", flush=True)


TH = stateprep.optimize_vacuum(Z2Lattice(6, pbc=True), M0, G2, ETA,
                               n_layers=2, restarts=2)["thetas"]
log("vacuum angles ready")

# ---------------- train at ns=10
lat10 = Z2Lattice(10, pbc=True)
band10 = spectroscopy.meson_band(lat10, M0, G2, ETA, n_states=14,
                                 matrix_free=True)
log(f"ns=10 band: E(k)={np.round(band10['energy'],4)} k={np.round(band10['k'],3)}")
vac10 = np.asarray(Statevector.from_instruction(
    stateprep.vacuum_ansatz(lat10, TH)))

trained = {}
for k0 in K0S:
    mix = spectroscopy.optimize_interpolator(lat10, band10, k0=k0, sigma_x=SIGMA,
                                             x0=2)
    log(f"k0={k0:.3f}: interpolator band fraction {mix['band_fraction']:.4f}")
    target, _ = spectroscopy.meson_wavepacket(lat10, band10, k0=k0,
                                              sigma_x=SIGMA, x0=2,
                                              mix=mix["mix"])
    r = block_engine.train_adjoint(lat10, vac10, target, center=4, n_layers=3,
                                   maxiter=800, max_offset=4)
    trained[k0] = r
    np.savez(f"data/wp10_params_k{k0:.2f}_L3.npz", vec=r["vector"],
             offsets=r["offsets"], L=3, F=r["fidelity"], sigma=SIGMA, k0=k0)
    log(f"k0={k0:.3f}: trained at ns=10, F = {r['fidelity']:.4f}")

# ---------------- certify at ns=12
lat12 = Z2Lattice(12, pbc=True)
try:
    band12 = spectroscopy.meson_band(lat12, M0, G2, ETA, n_states=10,
                                     matrix_free=True, ncv=17)
except MemoryError:
    log("ns=12 k=10 hit MemoryError; retrying k=8, ncv=13")
    band12 = spectroscopy.meson_band(lat12, M0, G2, ETA, n_states=8,
                                     matrix_free=True, ncv=13)
log(f"ns=12 band: E(k)={np.round(band12['energy'],4)} k={np.round(band12['k'],3)}")

H12 = ham.build_hamiltonian(lat12, M0, G2, ETA)
vac12_sv = Statevector.from_instruction(stateprep.vacuum_ansatz(lat12, TH))
e_vac12 = np.real(np.vdot(np.asarray(vac12_sv),
                          exact.apply_pauli_sum(H12, np.asarray(vac12_sv))))

for k0 in K0S:
    r = trained[k0]
    params = wavepacket.params_from_vector(r["vector"], r["offsets"], 3)
    mix12 = spectroscopy.optimize_interpolator(lat12, band12, k0=k0,
                                               sigma_x=SIGMA, x0=3)
    target12, _ = spectroscopy.meson_wavepacket(lat12, band12, k0=k0,
                                                sigma_x=SIGMA, x0=3,
                                                mix=mix12["mix"])
    psi = np.asarray(vac12_sv.evolve(wavepacket.block_circuit(lat12, 6, params)))
    F = abs(np.vdot(target12, psi)) ** 2
    e_pk = np.real(np.vdot(psi, exact.apply_pauli_sum(H12, psi)))
    t2ph = np.angle(np.vdot(psi, spectroscopy.translate(psi, lat12)))
    log(f"k0={k0:.3f}: TRANSFER 10->12 F = {F:.4f}; "
        f"E gap = {e_pk - e_vac12:.4f} (band {band12['energy'].min():.3f}-"
        f"{band12['energy'].max():.3f}); arg<T2> = {t2ph:.3f}")
log("done")
