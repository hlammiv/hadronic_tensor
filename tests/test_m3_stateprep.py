"""M3 validation: variational vacuum + volume transfer, meson band, wavepacket.

Thresholds calibrated on actual runs (2026-07-03):
  ns=6 L=2 fidelity 0.999977; transfer 6->8 fidelity 0.999970;
  E/ns identical to 1e-6 across ns=6,8,10; band overlap of J^1 interpolator 0.67.
"""

import numpy as np
import pytest

from htensor import Z2Lattice, exact, stateprep, spectroscopy
from htensor import hamiltonian as ham

M0, G2, ETA = 0.7, 1.1, 1.3


@pytest.fixture(scope="module")
def opt6():
    return stateprep.optimize_vacuum(Z2Lattice(6, pbc=True), M0, G2, ETA,
                                     n_layers=2, restarts=2)


@pytest.fixture(scope="module")
def band8():
    return spectroscopy.meson_band(Z2Lattice(8, pbc=True), M0, G2, ETA)


# ------------------------------------------------------------------ vacuum
def test_vacuum_ansatz_fidelity(opt6):
    assert opt6["fidelity"] > 0.999
    assert opt6["energy"] < opt6["exact_energy"] + 5e-3


def test_vacuum_ansatz_stays_physical(opt6):
    lat = Z2Lattice(6, pbc=True)
    psi = stateprep.ansatz_state(lat, opt6["thetas"])
    for n in range(lat.ns):
        assert np.isclose(exact.expectation(ham.gauss_operator(lat, n), psi).real,
                          1.0, atol=1e-10)


def test_volume_transfer_6_to_8(opt6):
    """The scalable-circuits property: small-volume angles reused at larger
    volume with no reoptimization stay at >0.999 fidelity and identical
    energy density (gapped, confining theory)."""
    lat8 = Z2Lattice(8, pbc=True)
    psi8 = stateprep.ansatz_state(lat8, opt6["thetas"])
    _, v8 = exact.lowest_physical_states(lat8, M0, G2, ETA, k=1)
    assert abs(np.vdot(v8[:, 0], psi8)) ** 2 > 0.999

    H8 = exact.to_sparse(ham.build_hamiltonian(lat8, M0, G2, ETA))
    e_density_8 = np.real(np.vdot(psi8, H8 @ psi8)) / 8
    assert np.isclose(e_density_8, opt6["energy"] / 6, atol=1e-4)


# ------------------------------------------------------------------ translation
@pytest.mark.parametrize("ns", [4, 6, 8])
def test_translation_operator_properties(ns):
    """T2 (with the ns = 0 mod 4 string twist) is a symmetry on the PHYSICAL
    sector: project a random state onto all G_n = +1 first.  On the full JW
    space it is not a symmetry -- that is the seam-string pinning effect."""
    lat = Z2Lattice(ns, pbc=True)
    rng = np.random.default_rng(3)
    psi = rng.standard_normal(2**lat.n_qubits) + 1j * rng.standard_normal(2**lat.n_qubits)
    for n in range(ns):  # project onto the Gauss sector
        G = exact.to_sparse(ham.gauss_operator(lat, n))
        psi = 0.5 * (psi + G @ psi)
    psi /= np.linalg.norm(psi)

    # T2^Nx = 1 on the sector
    out = psi.copy()
    for _ in range(lat.nx):
        out = spectroscopy.translate(out, lat)
    assert np.allclose(out, psi, atol=1e-10)

    # [T2, H] = 0 on the sector
    H = exact.to_sparse(ham.build_hamiltonian(lat, M0, G2, ETA))
    lhs = H @ spectroscopy.translate(psi, lat)
    rhs = spectroscopy.translate(H @ psi, lat)
    assert np.allclose(lhs, rhs, atol=1e-9)

    # strong-coupling vacuum is translation invariant
    vac = exact.strong_coupling_vacuum(lat)
    assert np.allclose(spectroscopy.translate(vac, lat), vac)


# ------------------------------------------------------------------ band
def test_meson_band_structure(band8):
    ks = sorted(np.round(band8["k"], 6))
    assert np.allclose(ks, sorted(np.round(
        [0.0, np.pi / 2, -np.pi / 2, np.pi], 6)))
    # E(k) = E(-k)
    e = {round(k, 6): en for k, en in zip(band8["k"], band8["energy"])}
    assert np.isclose(e[round(np.pi / 2, 6)], e[round(-np.pi / 2, 6)], atol=1e-6)
    assert all(band8["energy"] > 0)


# ------------------------------------------------------------------ wavepacket
def test_wavepacket_at_rest(band8):
    lat = Z2Lattice(8, pbc=True)
    wp, frac = spectroscopy.meson_wavepacket(lat, band8, k0=0.0, sigma_x=1.0)
    assert np.isclose(np.linalg.norm(wp), 1.0)
    assert frac > 0.3  # J^1 interpolator carries O(1) meson-band weight
    # orthogonal to vacuum, energy inside the band envelope (exact for a
    # band-projected state)
    assert abs(np.vdot(band8["vacuum"], wp)) < 1e-10
    H = exact.to_sparse(ham.build_hamiltonian(lat, M0, G2, ETA))
    gap = np.real(np.vdot(wp, H @ wp)) - band8["e0"]
    assert band8["energy"].min() - 1e-8 <= gap <= band8["energy"].max() + 1e-8


def test_wavepacket_boosted(band8):
    lat = Z2Lattice(8, pbc=True)
    k0 = np.pi / 2
    wp, _ = spectroscopy.meson_wavepacket(lat, band8, k0=k0, sigma_x=1.0)
    t2 = spectroscopy.translate(wp, lat)
    phase = np.angle(np.vdot(wp, t2))
    # mean momentum ~ +k0 (finite width on Nx=4 shifts it by O(0.05))
    assert abs(np.angle(np.exp(1j * (phase - k0)))) < 0.2
    # C-conjugate wavepacket: k0 -> -k0 flips the phase
    wm, _ = spectroscopy.meson_wavepacket(lat, band8, k0=-k0, sigma_x=1.0)
    pm = np.angle(np.vdot(wm, spectroscopy.translate(wm, lat)))
    assert abs(np.angle(np.exp(1j * (pm + k0)))) < 0.2
    # rest packet sits at k = 0
    w0, _ = spectroscopy.meson_wavepacket(lat, band8, k0=0.0, sigma_x=1.0)
    p0 = np.angle(np.vdot(w0, spectroscopy.translate(w0, lat)))
    assert abs(p0) < 1e-6
