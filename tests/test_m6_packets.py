"""M6 (WS2-B): gauge-fixed engine for wide-packet training.

Every gf_engine ingredient is checked against the full-space code through
the embedding isometry V (gf basis -> 2^(2 ns) product states) at ns = 6
and 8: Hamiltonian and bond generators (1e-12), T2 translation, vacuum
ansatz (both link references), block state vs BlockEngine, adjoint
gradient vs BlockEngine and finite differences, band energies vs
spectroscopy.meson_band, packet targets vs spectroscopy.meson_wavepacket,
and the saved-vacuum file convention.  Calibrated 2026-09-02.
"""

import os

import numpy as np
import pytest

from htensor import Z2Lattice, exact, stateprep, spectroscopy, block_engine
from htensor import hamiltonian as ham
from htensor.currents import charge_density
from htensor.wavepacket import _bond_generator_full
from htensor import gf_engine as gfe

M0, G2, ETA = 0.7, 1.1, 1.3
VAC_FILE = "data/vac_prod.npz"


@pytest.fixture(scope="module")
def thetas():
    if os.path.exists(VAC_FILE):
        return stateprep.load_vacuum(VAC_FILE)["thetas"]
    return stateprep.optimize_vacuum(Z2Lattice(6, pbc=True), M0, G2, ETA,
                                     n_layers=2, restarts=2)["thetas"]


@pytest.fixture(scope="module", params=[6, 8])
def setup(request):
    lat = Z2Lattice(request.param, pbc=True)
    space = gfe.GFSpace(lat)
    V = gfe.embed_isometry(space)
    return lat, space, V


def _matrix(space, apply):
    return np.column_stack([apply(np.eye(space.dim)[:, j].astype(complex))
                            for j in range(space.dim)])


# ------------------------------------------------------------ operators
def test_isometry_and_dimension(setup):
    lat, space, V = setup
    from math import comb
    assert space.dim == 2 * comb(lat.ns, lat.ns // 2)
    assert np.abs(V.T @ V - np.eye(space.dim)).max() < 1e-12


def test_hamiltonian_matches_full_space(setup):
    lat, space, V = setup
    Hfull = exact.to_sparse(ham.build_hamiltonian(lat, M0, G2, ETA))
    Hg = gfe.GFHamiltonian(space, M0, G2, ETA)
    ref = V.T @ (Hfull @ V)
    assert np.abs(ref - _matrix(space, Hg.apply)).max() < 1e-12
    Hop = ham.build_hamiltonian(lat, M0, G2, ETA)
    assert np.abs(ref - _matrix(space, lambda v: space.apply(Hop, v))).max() < 1e-12


@pytest.mark.parametrize("kind", ["hop", "cur"])
def test_bond_generators_match_full_space(setup, kind):
    lat, space, V = setup
    for b in (0, 1, lat.ns - 1):          # bulk even, bulk odd, seam
        Gf = exact.to_sparse(_bond_generator_full(lat, b, kind))
        Gg = _matrix(space, lambda v: space.apply_generator(kind, b, v))
        assert np.abs(V.T @ (Gf @ V) - Gg).max() < 1e-12
        # (2G)^2 = projector: idempotent and Hermitian
        M2 = (2 * Gg) @ (2 * Gg)
        assert np.abs(M2 @ M2 - M2).max() < 1e-12


def test_translation_matches_spectroscopy(setup):
    lat, space, V = setup
    Tfull = np.column_stack([spectroscopy.translate(V[:, j], lat)
                             for j in range(space.dim)])
    Tg = _matrix(space, space.translate)
    assert np.abs(V.T @ Tfull - Tg).max() < 1e-12
    # T2^nx = 1 on the sector; orbits consistent
    rep, m, chi, ell, Phi = space.orbits()
    assert np.all(ell > 0) and np.all(ell <= lat.nx)


# ------------------------------------------------------------ circuits
@pytest.mark.parametrize("link_ref", ["+", "-"])
def test_vacuum_ansatz_matches_statevector(setup, thetas, link_ref):
    lat, space, V = setup
    a = stateprep.ansatz_state(lat, thetas, link_ref=link_ref)
    g = gfe.gf_vacuum_state(space, thetas, link_ref)
    assert np.abs(V @ g - a).max() < 1e-12
    assert np.abs(gfe.embed_vector(space, g) - a).max() < 1e-12
    back = gfe.project_vector(space, a)
    assert np.abs(back - g).max() < 1e-12


def test_block_state_and_gradient_match_block_engine(setup, thetas):
    lat, space, V = setup
    vacg = gfe.gf_vacuum_state(space, thetas)
    vacf = V @ vacg
    rng = np.random.default_rng(5)
    center = 2 * (lat.ns // 4)
    for L, mo in ((2, None), (1, 2)):
        eng = block_engine.BlockEngine(lat, center, L, max_offset=mo)
        geng = gfe.GFEngine(space, center, L, max_offset=mo)
        assert eng.keys == geng.keys
        vec = 0.4 * rng.standard_normal(len(eng.keys))
        assert np.abs(V @ geng.state(vacg, vec) - eng.state(vacf, vec)).max() < 1e-12
        tgt = rng.standard_normal(space.dim) + 1j * rng.standard_normal(space.dim)
        tgt /= np.linalg.norm(tgt)
        f, g = geng.fidelity_and_grad(vacg, tgt, vec)
        f2, g2 = eng.fidelity_and_grad(vacf, V @ tgt, vec)
        assert abs(f - f2) < 1e-12 and np.abs(g - g2).max() < 1e-12
        for i in (0, 5, len(vec) - 1):        # finite differences
            eps = 1e-6
            vp = vec.copy()
            vp[i] += eps
            fp, _ = geng.fidelity_and_grad(vacg, tgt, vp)
            assert np.isclose(g[i], (fp - f) / eps, rtol=1e-4, atol=1e-8)


def test_block_stays_normalized_and_in_sector(setup, thetas):
    lat, space, V = setup
    vacg = gfe.gf_vacuum_state(space, thetas)
    geng = gfe.GFEngine(space, 2, 2)
    vec = 0.6 * np.random.default_rng(9).standard_normal(len(geng.keys))
    psi = geng.state(vacg, vec)
    assert abs(np.linalg.norm(psi) - 1) < 1e-12
    full = V @ psi
    for n in range(lat.ns):
        assert np.isclose(exact.expectation(ham.gauss_operator(lat, n), full).real,
                          1.0, atol=1e-10)
    j0 = [exact.expectation(charge_density(lat, v), full).real for v in range(lat.ns)]
    assert np.abs(space.j0_profile(psi) - j0).max() < 1e-12


# ------------------------------------------------------------ band
def test_band_matches_meson_band(setup):
    import warnings
    lat, space, V = setup
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bsv = spectroscopy.meson_band(lat, M0, G2, ETA)
    bg = gfe.gf_band(space, M0, G2, ETA)
    assert np.allclose(np.sort(bg["k"]), np.sort(bsv["k"]))
    assert abs(bg["e0"] - bsv["e0"]) < 1e-8
    for k, e, s, mult in zip(bg["k"], bg["energy"], bg["states"], bg["multiplets"]):
        i = int(np.argmin(np.abs(bsv["k"] - k)))
        assert abs(e - bsv["energy"][i]) < 1e-8
        # gf state is a T2 eigenstate with the meson_band phase convention
        t2 = space.t2_expect(s)
        assert abs(t2) > 1 - 1e-10 and abs(np.angle(t2 * np.exp(-1j * k))) < 1e-8
        # the eigsh state lies in the gf multiplet (k = pi is a Kramers pair)
        w = sum(abs(np.vdot(V @ m, bsv["states"][i])) ** 2 for m in mult)
        assert w > 1 - 1e-4
    assert abs(np.vdot(V @ bg["vacuum"], bsv["vacuum"])) > 1 - 1e-8


def test_packet_target_matches_spectroscopy(thetas):
    """ns = 6 (no k = pi on the grid, so the two codes see the same band)."""
    lat = Z2Lattice(6, pbc=True)
    space = gfe.GFSpace(lat)
    V = gfe.embed_isometry(space)
    bsv = spectroscopy.meson_band(lat, M0, G2, ETA)
    bg = gfe.gf_band(space, M0, G2, ETA)
    for k0 in (0.0, 2 * np.pi / 3):
        mix_sv = spectroscopy.optimize_interpolator(lat, bsv, k0=k0, sigma_x=1.0)
        mix_g = gfe.gf_optimize_interpolator(space, bg, k0=k0, sigma_x=1.0)
        assert abs(mix_sv["band_fraction"] - mix_g["band_fraction"]) < 1e-3
        wsv, _ = spectroscopy.meson_wavepacket(lat, bsv, k0=k0, sigma_x=1.0,
                                               mix=mix_sv["mix"])
        wg, _ = gfe.gf_packet_target(space, bg, k0, 1.0, mix=mix_sv["mix"])
        assert abs(np.vdot(wsv, V @ wg)) > 1 - 1e-5
        # clean target: exact Gaussian weights and exact T2 phase
        wc, _ = gfe.gf_packet_target(space, bg, k0, 1.0, mix=mix_g["mix"], clean=True)
        p, pv = gfe.band_weights(bg, wc)
        w = np.exp(-2 * 1.0 ** 2 * gfe._wrap(bg["k"] - k0) ** 2)
        assert np.allclose(p, w / w.sum(), atol=1e-10) and pv < 1e-20
        assert abs(np.angle(space.t2_expect(wc) * np.exp(-1j * k0))) < 1e-10


def test_tune_k0_env_identity_on_grid():
    k = gfe.GFSpace(Z2Lattice(10, pbc=True)).momentum_grid()
    K = 2 * np.pi / 5
    assert abs(gfe.tune_k0_env(k, K, 0.75) - K) < 1e-9
    # off-grid K: the tuned envelope restores arg<T2> = K on the grid
    Koff = 0.9
    ke = gfe.tune_k0_env(k, Koff, 0.75)
    w2 = np.exp(-2 * 0.75 ** 2 * gfe._wrap(k - ke) ** 2)
    assert abs(np.angle(np.sum(w2 * np.exp(1j * k))) - Koff) < 1e-5


# ------------------------------------------------------------ vacuum files
def test_save_load_vacuum_roundtrip(tmp_path, thetas):
    fn = tmp_path / "vac_test.npz"
    stateprep.save_vacuum(fn, thetas, 2, "-", M0, G2, ETA, extra={"ns_opt": 6})
    d = stateprep.load_vacuum(fn)
    assert np.allclose(d["thetas"], thetas) and d["n_layers"] == 2
    assert d["link_ref"] == "-" and d["m0"] == M0 and d["ns_opt"] == 6
    with pytest.raises(ValueError):
        stateprep.save_vacuum(fn, thetas, 3, "+")
    with pytest.raises(ValueError):
        stateprep.save_vacuum(fn, thetas, 2, "x")


def test_link_ref_reference_states():
    lat = Z2Lattice(6, pbc=True)
    for ref in "+-":
        v = exact.strong_coupling_vacuum(lat, link_ref=ref)
        from qiskit.quantum_info import Statevector
        from htensor.trotter import strong_coupling_vacuum_circuit
        c = np.asarray(Statevector.from_instruction(
            strong_coupling_vacuum_circuit(lat, link_ref=ref)))
        assert np.abs(v - c).max() < 1e-12
        for n in range(lat.ns):
            assert np.isclose(exact.expectation(ham.gauss_operator(lat, n), v).real, 1.0)
        # links carry sigma^x = +-1
        x = exact.expectation(ham.gauge_term(lat, 2.0), v).real / lat.n_links
        assert np.isclose(x, 1.0 if ref == "+" else -1.0)
    assert np.abs(exact.strong_coupling_vacuum(lat)
                  - exact.strong_coupling_vacuum(lat, link_ref="+")).max() == 0
