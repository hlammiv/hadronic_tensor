"""M4c validation: interpolator optimization, block engine, adjoint training.

Calibrated 2026-07-04 (ns=8, m0=0.7, g2=1.1, eta=1.3):
  optimized interpolator band fraction 0.995 (bare 0.67);
  adjoint training F: L=1 0.6677 (displacement-leakage ceiling, a physics
  feature), L=2 0.971, L=3 0.9988 (rest) / 0.9976 (k0=pi/2).
Full L=3 training (~15 min) lives in scripts/, not here.
"""

import numpy as np
import pytest
from qiskit.quantum_info import Statevector

from htensor import Z2Lattice, stateprep, spectroscopy, wavepacket, block_engine

M0, G2, ETA = 0.7, 1.1, 1.3


@pytest.fixture(scope="module")
def setup8():
    lat6 = Z2Lattice(6, pbc=True)
    th = stateprep.optimize_vacuum(lat6, M0, G2, ETA, n_layers=2,
                                   restarts=2)["thetas"]
    lat = Z2Lattice(8, pbc=True)
    band = spectroscopy.meson_band(lat, M0, G2, ETA)
    vac = np.asarray(Statevector.from_instruction(
        stateprep.vacuum_ansatz(lat, th)))
    return lat, th, band, vac


def test_optimized_interpolator_band_fraction(setup8):
    lat, _, band, _ = setup8
    opt = spectroscopy.optimize_interpolator(lat, band, k0=0.0, sigma_x=1.0)
    assert opt["band_fraction"] > 0.98  # bare cur-even sits at ~0.67
    wp, frac = spectroscopy.meson_wavepacket(lat, band, k0=0.0, sigma_x=1.0,
                                             mix=opt["mix"])
    assert frac > 0.99


def test_engine_matches_circuit(setup8):
    lat, _, _, vac = setup8
    eng = block_engine.BlockEngine(lat, 4, n_layers=2)
    rng = np.random.default_rng(5)
    vec = 0.4 * rng.standard_normal(len(eng.gates))
    psi_eng = eng.state(vac, vec)
    p = wavepacket.params_from_vector(vec, eng.offsets, 2)
    psi_qk = np.asarray(Statevector(vac).evolve(
        wavepacket.block_circuit(lat, 4, p)))
    assert np.abs(psi_eng - psi_qk).max() < 1e-12


def test_adjoint_gradient_correct(setup8):
    lat, _, band, vac = setup8
    wp, _ = spectroscopy.meson_wavepacket(lat, band, k0=0.0, sigma_x=1.0)
    eng = block_engine.BlockEngine(lat, 4, n_layers=1)
    rng = np.random.default_rng(7)
    vec = 0.3 * rng.standard_normal(len(eng.gates))
    f, g = eng.fidelity_and_grad(vac, wp, vec)
    for i in (0, 5, 17):
        eps = 1e-6
        vp = vec.copy()
        vp[i] += eps
        fp, _ = eng.fidelity_and_grad(vac, wp, vp)
        assert np.isclose(g[i], (fp - f) / eps, rtol=1e-3, atol=1e-8)


def test_block_preserves_gauss_sector(setup8):
    lat, _, _, vac = setup8
    from htensor import hamiltonian as ham
    from htensor import exact
    eng = block_engine.BlockEngine(lat, 4, n_layers=2)
    vec = 0.5 * np.random.default_rng(9).standard_normal(len(eng.gates))
    psi = eng.state(vac, vec)
    for n in range(lat.ns):
        assert np.isclose(exact.expectation(ham.gauss_operator(lat, n), psi).real,
                          1.0, atol=1e-10)


def test_short_training_beats_displacement_ceiling(setup8):
    """L=2 with adjoint gradients must clear the L=1 leakage ceiling (~0.67)."""
    lat, _, band, vac = setup8
    opt = spectroscopy.optimize_interpolator(lat, band, k0=0.0, sigma_x=1.0)
    wp, _ = spectroscopy.meson_wavepacket(lat, band, k0=0.0, sigma_x=1.0,
                                          mix=opt["mix"])
    r = block_engine.train_adjoint(lat, vac, wp, 4, n_layers=2, maxiter=400)
    assert r["fidelity"] > 0.9
