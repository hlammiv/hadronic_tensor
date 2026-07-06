"""Pilot for the synthesis-error study at ns=8 (17 qubits, statevector):
full snapped pipeline vs exact, physics observables vs grid spacing delta.

For each delta in {pi/128 ... pi/2} and both rounding modes:
  - snap the transpiled prep circuit (vacuum ansatz + wavepacket block)
  - snap the transpiled Trotter evolution
  - measure: <H> gap of the snapped packet, and the W^{00}(q0) ridge at
    q1 = pi/2 through the snapped Hadamard-test pipeline
  - compare against the unsnapped pipeline (delta -> 0 limit)

  PYTHONPATH=. .venv/bin/python scripts/synthesis_pilot_ns8.py
"""

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector

from htensor import Z2Lattice, exact, stateprep, spectroscopy, wavepacket, \
    block_engine, measure, analysis, synthesis
from htensor import hamiltonian as ham
from htensor import currents as cur
from htensor import trotter
from scripts_helpers_ridge import onesided_ft

M0, G2, ETA = 0.7, 1.1, 1.3
BASIS = ["cx", "cz", "rz", "sx", "x", "h"]
DELTAS = np.pi / np.array([128, 64, 32, 16, 8, 4, 2])
TIMES = np.arange(0.0, 8.01, 0.5)
Q0 = np.arange(-0.6, 1.601, 0.04)
Q1 = np.array([np.pi / 2])

lat = Z2Lattice(8, pbc=True)
vc = 4
TH = stateprep.optimize_vacuum(Z2Lattice(6, pbc=True), M0, G2, ETA,
                               n_layers=2, restarts=2)["thetas"]
band = spectroscopy.meson_band(lat, M0, G2, ETA)
mix = spectroscopy.optimize_interpolator(lat, band, k0=0.0, sigma_x=1.0)
target, _ = spectroscopy.meson_wavepacket(lat, band, k0=0.0, sigma_x=1.0,
                                          mix=mix["mix"])
vac_state = np.asarray(Statevector.from_instruction(
    stateprep.vacuum_ansatz(lat, TH)))
r = block_engine.train_adjoint(lat, vac_state, target, 4, n_layers=3)
params = wavepacket.params_from_vector(r["vector"], r["offsets"], 3)
prep = stateprep.vacuum_ansatz(lat, TH)
prep.compose(wavepacket.block_circuit(lat, 4, params), inplace=True)
prep_t = transpile(prep, basis_gates=BASIS, optimization_level=1)
n_nc, n_rot = synthesis.rotation_census(prep_t)
print(f"prep circuit: {n_rot} rotations ({n_nc} non-Clifford); wp F = "
      f"{r['fidelity']:.4f}", flush=True)

H_op = ham.build_hamiltonian(lat, M0, G2, ETA)
Hs = exact.to_sparse(H_op)
e_vac_ex = float(np.real(np.vdot(vac_state, Hs @ vac_state)))
probes = [cur.charge_density(lat, v) for v in range(lat.ns)]
insert = cur.charge_density(lat, vc)
x = analysis.ring_fold((np.arange(lat.ns) - vc) / 2, lat.nx)


def pipeline(delta=None, mode="round", seed=0):
    """Snapped (or exact for delta=None) pipeline -> (<H> gap, W ridge)."""
    def prep_state():
        c = prep_t if delta is None else synthesis.snap_angles(
            prep_t, delta, mode, seed)
        return np.asarray(Statevector.from_instruction(c))

    def evo_factory(t):
        n = max(1, int(np.ceil(abs(t) / 0.1))) if t else 0
        c = trotter.trotter_circuit(lat, M0, G2, ETA, t, n)
        c = transpile(c, basis_gates=BASIS, optimization_level=1)
        if delta is not None:
            c = synthesis.snap_angles(c, delta, mode, seed)
        return c.to_instruction()

    psi = prep_state()
    gap = float(np.real(np.vdot(psi, Hs @ psi))) - e_vac_ex
    d = measure.hadamard_correlator_sv(psi, insert, probes, evo_factory, TIMES)
    G = d.connected
    W = onesided_ft(TIMES, x, G, Q0, Q1, 8 / 3, lat.nx / 6, 0.5, 0.5)
    return gap, W[:, 0]


gap0, W0 = pipeline(None)
print(f"exact: gap = {gap0:.4f}, ridge max = {W0.max():.4f}", flush=True)
rows = []
for delta in DELTAS:
    for mode, seeds in (("round", [0]), ("stochastic", [1, 2])):
        for s in seeds:
            gap, W = pipeline(delta, mode, s)
            err = float(np.sqrt(np.mean((W - W0) ** 2)) / np.abs(W0).max())
            rows.append((delta, mode, s, gap, err))
            print(f"delta=pi/{np.pi/delta:>5.0f} {mode:10s} seed={s}: "
                  f"gap = {gap:+.4f} (exact {gap0:.4f}), "
                  f"ridge rms err = {err:.4f}", flush=True)
np.savez("data/synthesis_pilot_ns8.npz",
         rows=np.array([(r[0], 0 if r[1] == 'round' else 1, r[2], r[3], r[4])
                        for r in rows]),
         gap0=gap0, W0=W0, q0=Q0, n_rot=n_rot, n_nc=n_nc)
print("saved data/synthesis_pilot_ns8.npz")
