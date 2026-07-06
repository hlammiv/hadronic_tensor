"""VALIDATED two-band parameter-free model of the low-omega ridge (2026-07-06).

Model space: both meson bands (10 states at ns=10, gaps 2.745-3.285).  The
ridge in wavepacket W^{00} = axial-channel transitions within this system:
intra-band packet beats (omega in [0, 0.33]) + interband M -> M' transitions
(omega in [0.16, 0.54]).  Closure vs exact ns=10 (one-sided windowed
transform, core window (-0.5, 1.0)):

    time-domain max|Gm-Gx|/max|Gx| = 0.134   (residual = higher sectors)
    core-window max rel dev        = 0.035
    W1(q0=0.30): model 0.4421/0.8695 vs exact 0.4264/0.8410  (q1=2pi/5, 4pi/5)

Everything from ED: states, energies, local MEs <k'|J0(v)|k>, packet
overlaps.  No fitted parameters.  ns=50 extension: interpolate the two
dispersions + the factorized ME amplitudes (A ~ 1/N_x), one-sided transform
convention throughout (the hermiticity completion is NOT valid for
wavepacket states at this precision -- it scatters higher-sector content
into the low-omega window; production analysis must adopt the one-sided
convention for ridge-region physics).

  PYTHONPATH=. .venv/bin/python scripts/ridge_model_v2.py
"""

import numpy as np

from htensor import Z2Lattice, exact, spectroscopy, analysis
from htensor import hamiltonian as ham
from htensor import currents as cur

M0, G2, ETA = 0.7, 1.1, 1.3


def two_band_space(lat, k_states=14):
    e_all, v_all = exact.lowest_physical_states(lat, M0, G2, ETA, k=k_states,
                                                matrix_free=True)
    gaps = e_all - e_all[0]
    keep = [i for i in range(k_states) if 2.0 < gaps[i] < 3.5]
    return [v_all[:, i] for i in keep], e_all[keep], v_all[:, 0], e_all[0]


def model_correlator(lat, sts, Ek, f, vc, times):
    Mv = np.array([[[np.vdot(sp, exact.apply_pauli_sum(
        cur.charge_density(lat, v), s)) for s in sts] for sp in sts]
        for v in range(lat.ns)])
    G = np.empty((len(times), lat.ns), complex)
    one0 = np.conj(f) @ Mv[vc] @ f
    for i, t in enumerate(times):
        a = np.exp(-1j * Ek * t) * f
        for v in range(lat.ns):
            two = np.conj(a) @ (Mv[v] @ (np.exp(-1j * Ek * t) * (Mv[vc] @ f)))
            G[i, v] = two - (np.conj(a) @ Mv[v] @ a) * one0
    return G, Mv


def onesided_ft(times, x, G, q0, q1, sigma_t, sigma_x, dt, dx):
    wt = np.exp(-times**2 / (2 * sigma_t**2)).copy()
    wt[0] *= 0.5
    wx = np.exp(-x**2 / (2 * sigma_x**2))
    et = np.exp(1j * np.outer(q0, times))
    ex = np.exp(-1j * np.outer(x, q1))
    return 2 * np.real(dt * dx * (et @ ((wt[:, None] * wx[None, :]) * G) @ ex))


if __name__ == "__main__":
    lat = Z2Lattice(10, pbc=True)
    sts, Ek, vac, e0 = two_band_space(lat)
    print(f"model space: {len(sts)} states, gaps {np.round(Ek - e0, 3)}")
    band1 = spectroscopy.meson_band(lat, M0, G2, ETA, n_states=14,
                                    matrix_free=True)
    for k0 in (0.0, 2 * np.pi / 5):
        mix = spectroscopy.optimize_interpolator(band=band1, lat=lat, k0=k0,
                                                 sigma_x=0.75, x0=2)
        wp, _ = spectroscopy.meson_wavepacket(lat, band1, k0=k0, sigma_x=0.75,
                                              x0=2, mix=mix["mix"])
        f = np.array([np.vdot(s, wp) for s in sts])
        TIMES = np.arange(0.0, 8.01, 0.5)
        Gm, _ = model_correlator(lat, sts, Ek, f, 4, TIMES)
        np.savez(f"data/ridge_model2_k{k0:.2f}.npz", G=Gm, times=TIMES,
                 Ek=Ek, f=f, e0=e0)
        print(f"k0={k0:.2f}: model correlator saved")
