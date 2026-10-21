"""M4 validation: windowed-FT analysis chain and Aer execution backends.

Calibrated on 2026-07-03 probe: backend agreement 4e-15 (statevector) /
8e-9 (MPS); ns=8 wavepacket W real to 0.14%, elastic peak at q0 ~ 0.3.
"""

import numpy as np
import pytest
from qiskit.quantum_info import Statevector

from htensor import Z2Lattice, exact, spectroscopy, measure, backends, analysis
from htensor import hamiltonian as ham
from htensor import currents as cur
from htensor import trotter

M0, G2, ETA = 0.7, 1.1, 1.3


# ------------------------------------------------------------------ FT unit
def test_windowed_ft_locates_synthetic_peak():
    t = np.arange(-8, 8.01, 0.2)
    x = np.arange(-4, 4.0, 0.5)
    w0, k1 = 1.7, np.pi / 2
    c = np.exp(-1j * w0 * t)[:, None] * np.exp(1j * k1 * x)[None, :]
    q0 = np.arange(0, 4.001, 0.02)
    W = analysis.windowed_ft(t, x, c, q0, [k1], sigma_t=3.0, sigma_x=2.0)
    assert np.isclose(q0[np.argmax(W.real[:, 0])], w0, atol=0.03)
    # off-momentum response strongly suppressed
    W_off = analysis.windowed_ft(t, x, c, [w0], [-k1], 3.0, 2.0)
    assert abs(W_off[0, 0]) < 0.05 * W.real.max()


def test_ring_fold():
    assert np.allclose(analysis.ring_fold(np.array([-2.0, -1.5, 0, 1.5, 2.0]), 4),
                       [-2.0, -1.5, 0, 1.5, -2.0])


# ------------------------------------------------------------------ backends
@pytest.fixture(scope="module")
def lat6_setup():
    lat = Z2Lattice(6, pbc=True)
    prep = trotter.strong_coupling_vacuum_circuit(lat)
    psi0 = np.asarray(Statevector.from_instruction(prep))
    probes = [cur.charge_density(lat, v) for v in range(6)]
    ins = cur.bond_current(lat, 2, ETA)
    times = [0.0, 0.5]
    fac = trotter.make_evolution_factory(lat, M0, G2, ETA, "trotter", dt_target=0.05)
    ref = measure.hadamard_correlator_sv(psi0, ins, probes, fac, times)
    return lat, prep, ins, probes, times, ref


@pytest.mark.parametrize("method,tol", [("statevector", 1e-12),
                                        ("matrix_product_state", 5e-6)])
def test_aer_backend_matches_reference(lat6_setup, method, tol):
    lat, prep, ins, probes, times, ref = lat6_setup
    d = backends.hadamard_correlator_aer(lat, prep, ins, probes, M0, G2, ETA,
                                         times, 0.05, method=method)
    assert np.abs(d.correlator - ref.correlator).max() < tol
    assert np.abs(d.probe_expect - ref.probe_expect).max() < tol


def test_initial_mps_path_matches_direct(lat6_setup):
    """prepare_state_mps + set_matrix_product_state reproduces the direct
    path (prep re-simulated per circuit) exactly at high cap."""
    lat, prep, ins, probes, times, ref = lat6_setup
    from htensor.measure import split_current
    anc_site = min(split_current(ins)[1][0][0])
    mps, perm = backends.prepare_state_mps(lat, prep, anc_site,
                                           cap=4096, trunc=1e-15)
    d = backends.hadamard_correlator_aer(
        lat, prep, ins, probes, 0.7, 1.1, 1.3, times, 0.05,
        method="matrix_product_state", initial_mps=mps, initial_perm=perm)
    assert np.abs(d.correlator - ref.correlator).max() < 5e-6
    assert np.abs(d.probe_expect - ref.probe_expect).max() < 5e-6


def test_mps_backend_moderate_size():
    """ns=10 (21 qubits with ancilla): MPS vs statevector method of Aer."""
    lat = Z2Lattice(10, pbc=True)
    prep = trotter.strong_coupling_vacuum_circuit(lat)
    probes = [cur.charge_density(lat, v) for v in (3, 5, 7)]
    ins = cur.charge_density(lat, 5)
    kw = dict(dt_target=0.1, mps_trunc=1e-10)
    sv = backends.hadamard_correlator_aer(lat, prep, ins, probes, M0, G2, ETA,
                                          [0.4], method="statevector", **kw)
    mps = backends.hadamard_correlator_aer(lat, prep, ins, probes, M0, G2, ETA,
                                           [0.4], method="matrix_product_state", **kw)
    assert np.abs(sv.correlator - mps.correlator).max() < 1e-6


# ------------------------------------------------------------------ W pipeline
def test_w_assembly_end_to_end_ns6():
    """Wavepacket -> exact C(x,t) -> subtractions -> time completion ->
    windowed FT: W real, supported at physical q0, peak at the band
    transition energies."""
    lat = Z2Lattice(6, pbc=True)
    band = spectroscopy.meson_band(lat, M0, G2, ETA)
    wp, _ = spectroscopy.meson_wavepacket(lat, band, k0=0.0, sigma_x=1.0)
    H = exact.to_sparse(ham.build_hamiltonian(lat, M0, G2, ETA))
    vc = 2  # even site nearest the wavepacket center x0 = nx//2 = 1 -> site 2
    ts = np.arange(0, 4.51, 0.3)
    ins = exact.to_sparse(cur.charge_density(lat, vc))
    c_wp = np.empty((len(ts), lat.ns), complex)
    c_vac = np.empty_like(c_wp)
    one_pt = np.empty_like(c_wp)
    for v in range(lat.ns):
        B = exact.to_sparse(cur.charge_density(lat, v))
        c_wp[:, v] = exact.two_current_correlator(H, B, ins, wp, ts)
        c_vac[:, v] = exact.two_current_correlator(H, B, ins, band["vacuum"], ts)
        psi_t, tprev = wp.copy(), 0.0
        for i, t in enumerate(ts):
            if t != tprev:
                psi_t = exact.evolve(H, psi_t, t - tprev)
                tprev = t
            one_pt[i, v] = np.vdot(psi_t, B @ psi_t)
    c_sub = analysis.subtract(c_wp, c_vac, one_pt, np.vdot(wp, ins @ wp))
    x = analysis.ring_fold((np.arange(lat.ns) - vc) / 2, lat.nx)
    t_full, c_full = analysis.complete_time(analysis.CorrelatorGrid(ts, x, c_sub))

    # hermiticity of the completed grid (construction check)
    assert np.isclose(t_full[0], -ts[-1]) and np.isclose(t_full[-1], ts[-1])

    q0 = np.arange(-2, 5.001, 0.05)
    q1 = np.array([0.0, 2 * np.pi / 3])
    W, spread = analysis.window_scan(t_full, x, c_full, q0, q1,
                                     sigma_t=1.5, sigma_x=1.0)
    # real up to wavepacket-width corrections
    assert np.abs(W.imag).max() < 0.02 * np.abs(W.real).max()
    # dominant support in the physical region: elastic band transitions
    ipk = np.unravel_index(np.argmax(W.real), W.real.shape)
    assert -0.2 < q0[ipk[0]] < 1.0
    # unphysical far-negative q0 strongly suppressed
    neg = np.abs(W.real[q0 < -1.0, :]).max()
    assert neg < 0.2 * np.abs(W.real).max()
    # window systematic is finite and was actually scanned
    assert np.all(spread >= 0) and spread.max() > 0


def test_g_error_reweights_the_probe_one_point_term():
    """G = C_cal - b_cal A0 sees b_cal only through (id_a - A0), so a C_err
    used unchanged over-weights that part by id_a / (id_a - A0)."""
    id_a, A0 = 0.5, 0.522674                     # published production point
    swap = abs(id_a / (id_a - A0))
    assert 21 < swap < 23

    # a slice whose error is nothing but the wing-anchor scatter
    s_N = np.full(5, 0.2525)
    pure = analysis.g_error(id_a * s_N, s_N, id_a, A0)
    assert np.allclose(pure, abs(id_a - A0) * s_N)
    assert np.allclose(id_a * s_N / pure, swap)

    # a slice with an additional, independent per-site term: only the b_cal
    # part moves, the rest is carried through in quadrature
    other = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
    c_err = np.hypot(id_a * s_N, other)
    got = analysis.g_error(c_err, s_N, id_a, A0)
    assert np.allclose(got, np.hypot(abs(id_a - A0) * s_N, other))
    assert np.all(got < c_err)

    # no measured one-point was subtracted -> nothing cancels
    assert np.allclose(analysis.g_error(c_err, None, id_a, A0), c_err)

    # a b_err inconsistent with C_err (different merge weights) can only ever
    # floor the result at the term that survives, never go imaginary
    floored = analysis.g_error(0.5 * other, 10 * s_N, id_a, A0)
    assert np.all(np.isfinite(floored))
    assert np.allclose(floored, abs(id_a - A0) * 10 * s_N)

    # the relA point cancels less completely, so it gains less
    assert (analysis.g_error(id_a * s_N, s_N, id_a, 0.568391)[0]
            > analysis.g_error(id_a * s_N, s_N, id_a, A0)[0])


def test_time_window_defaults_to_the_published_gaussian():
    t = np.arange(0.0, 6.001, 0.5)
    assert np.array_equal(analysis.time_window(t, 3.0, "gauss"),
                          np.exp(-t ** 2 / (2 * 3.0 ** 2)))
    assert np.array_equal(analysis.time_window(t, 0.0, "rect"), np.ones(len(t)))

    # every kind starts at 1 on t = 0 and decreases, so one width knob orders
    # them all and the t = 0 half-weight stays the caller's business
    for kind, p in (("gauss", 3.0), ("kaiser", 3.0), ("dpss", 1.5)):
        w = analysis.time_window(t, p, kind)
        assert w.shape == t.shape
        assert np.isclose(w[0], 1.0) and np.all(np.diff(w) < 0) and w[-1] > 0

    # a wider window keeps more of the late slices, for every family
    for kind, narrow, wide in (("gauss", 1.5, 4.0), ("kaiser", 6.0, 1.0), ("dpss", 3.0, 1.0)):
        assert (analysis.time_window(t, wide, kind).sum()
                > analysis.time_window(t, narrow, kind).sum())

    # the default argument leaves onesided_ft exactly as it was
    rng = np.random.default_rng(3)
    x = np.linspace(-6, 6, 13)
    G = rng.standard_normal((len(t), len(x)))
    q0, q1 = np.linspace(-1, 5, 17), np.linspace(-2, 2, 9)
    assert np.array_equal(analysis.onesided_ft(t, x, G, q0, q1, 3.0, 6.0, 0.5, 0.5),
                          analysis.onesided_ft(t, x, G, q0, q1, 3.0, 6.0, 0.5, 0.5,
                                               window="gauss"))
    with pytest.raises(ValueError):
        analysis.time_window(t, 1.0, "hann")
