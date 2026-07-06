"""Parameter-free band-model prediction of the low-omega (axial-beat) ridge.

Ingredients, all exact from the ns=10 band (volume-converged): complex local
MEs M_v[k',k] = <k'|J0(v)|k>, energies E_k, and packet overlaps f_k computed
from the same states (phase-consistent by construction).  Model:

  C_band(v,t)  = sum_{k',kap} f*_{k'} e^{i(E_{k'}-E_kap)t} M_v[k',kap] (M_vc f)_kap
  C_conn       = C_band - <J(v,t)>_band <J(vc)>_band

pushed through the IDENTICAL windowed FT as the data.  Validated against the
exact ns=10 connected correlator, then overlaid on the 101-qubit data at the
shared momenta q1 = +-2pi/5, +-4pi/5.  No fitted parameters, no normalization.

  PYTHONPATH=. .venv/bin/python scripts/ridge_model.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from htensor import Z2Lattice, exact, spectroscopy, analysis
from htensor import hamiltonian as ham
from htensor import currents as cur

OI = ["#0072B2", "#D55E00", "#009E73", "#E69F00"]
plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "stix",
    "font.size": 11, "axes.labelsize": 12,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
})

M0, G2, ETA = 0.7, 1.1, 1.3
SIGMA_T = 8.0 / 3.0
lat = Z2Lattice(10, pbc=True)
band = spectroscopy.meson_band(lat, M0, G2, ETA, n_states=14, matrix_free=True)
sts, Ek = band["states"], band["energy"] + band["e0"]
vc = 4
TIMES = np.arange(0.0, 8.01, 0.5)

Mv = []
for v in range(lat.ns):
    op = cur.charge_density(lat, v)
    Mv.append(np.array([[np.vdot(sp, exact.apply_pauli_sum(op, s))
                         for s in sts] for sp in sts]))
Mv = np.array(Mv)  # (ns, 5, 5)


def model_G(f):
    """Connected band-model correlator on the (t, v) grid."""
    G = np.empty((len(TIMES), lat.ns), dtype=complex)
    one0 = np.conj(f) @ Mv[vc] @ f
    for i, t in enumerate(TIMES):
        a = np.exp(-1j * Ek * t) * f
        for v in range(lat.ns):
            two = np.conj(f * np.exp(-1j * Ek * t)) @ (
                Mv[v] @ (np.exp(-1j * Ek * t) * (Mv[vc] @ f)))
            one_t = np.conj(a) @ Mv[v] @ a
            G[i, v] = two - one_t * one0
    return G


def windowed(G, sig_x):
    x = analysis.ring_fold((np.arange(lat.ns) - vc) / 2, lat.nx)
    tf, cf = analysis.complete_time(analysis.CorrelatorGrid(TIMES, x, G))
    q0 = np.arange(-1.0, 2.001, 0.04)
    q1 = np.array([2 * np.pi / 5, -2 * np.pi / 5, 4 * np.pi / 5, -4 * np.pi / 5])
    return q0, q1, analysis.windowed_ft(tf, x, cf, q0, q1, SIGMA_T, sig_x)


# ---------------- validation vs exact ns=10 connected correlator
H = exact.to_sparse(ham.build_hamiltonian(lat, M0, G2, ETA))
ins = exact.to_sparse(cur.charge_density(lat, vc))
results = {}
for k0 in (0.0, 2 * np.pi / 5):
    mix = spectroscopy.optimize_interpolator(lat, band, k0=k0, sigma_x=0.75, x0=2)
    wp, _ = spectroscopy.meson_wavepacket(lat, band, k0=k0, sigma_x=0.75, x0=2,
                                          mix=mix["mix"])
    f = np.array([np.vdot(s, wp) for s in sts])
    Gm = model_G(f)

    c_wp = np.empty((len(TIMES), lat.ns), complex)
    one_t = np.empty_like(c_wp)
    for v in range(lat.ns):
        B = exact.to_sparse(cur.charge_density(lat, v))
        c_wp[:, v] = exact.two_current_correlator(H, B, ins, wp, TIMES)
        psi_t, tp = wp.copy(), 0.0
        for i, t in enumerate(TIMES):
            if t != tp:
                psi_t = exact.evolve(H, psi_t, t - tp)
                tp = t
            one_t[i, v] = np.vdot(psi_t, B @ psi_t)
    Gx = analysis.subtract(c_wp, None, one_t, np.vdot(wp, ins @ wp))

    q0, q1s, Wm = windowed(Gm, lat.nx / 6)
    _, _, Wx = windowed(Gx, lat.nx / 6)
    r = np.abs(Wm.real - Wx.real).max() / np.abs(Wx.real).max()
    print(f"k0={k0:.3f}: band-model vs exact ns=10 windowed ridge: "
          f"max rel dev = {r:.3f}")
    results[k0] = (q0, q1s, Wm)

# ---------------- structural checks enabling the volume extension
# Ansatz: M_v[k',k] = (-1)^v e^{i(k-k')x_v} A(k',k) with A intensive
# (A ~ 1/N_x) and, in the gauge where the rest-packet overlaps f_k are real
# positive, A real symmetric and smooth.  If verified, the ns=50 prediction
# is parameter-free: interpolate A, rescale by 5/25, sum over the fine grid.
mix0 = spectroscopy.optimize_interpolator(lat, band, k0=0.0, sigma_x=0.75, x0=2)
wp0, _ = spectroscopy.meson_wavepacket(lat, band, k0=0.0, sigma_x=0.75, x0=2,
                                       mix=mix0["mix"])
f0 = np.array([np.vdot(s, wp0) for s in sts])
chi = np.angle(f0)
U = np.exp(1j * chi)  # |k> -> e^{-i chi_k}|k>: M -> diag(U*) M diag(U)... fix
ks = band["k"]
x_v = (np.arange(lat.ns) - vc) / 2
# A_v[k',k] = (-1)^v e^{-i(k-k')x_v} M_gauged[k',k]
Amats = []
for v in range(lat.ns):
    ph = np.exp(-1j * (ks[None, :] - ks[:, None]) * x_v[v])
    M_gauged = np.conj(U)[:, None] * Mv[v] * U[None, :]
    Amats.append((-1) ** v * ph * M_gauged)
Amats = np.array(Amats)
Amean = Amats.mean(axis=0)
scatter = np.abs(Amats - Amean).max() / np.abs(Amean).max()
imfrac = np.abs(Amean.imag).max() / np.abs(Amean.real).max()
sym = np.abs(Amean - Amean.T).max() / np.abs(Amean).max()
print(f"\nA-structure checks: v-scatter {scatter:.4f}, Im/Re {imfrac:.4f}, "
      f"asymmetry {sym:.4f}")
print("A(k',k) real part:")
for row in Amean.real:
    print("   " + "  ".join(f"{r:+.4f}" for r in row))
np.savez("data/ridge_model_ns10.npz", A=Amean, ks=ks, Ek=Ek,
         f0=np.abs(f0), scatter=scatter)
print("saved data/ridge_model_ns10.npz")
