"""Task 4: elastic meson-meson phase shifts from finite-volume two-meson
levels via the 1+1d quantization condition  p*Nx + 2*delta(p) = 2*pi*n,
with the measured dispersion E(k)^2 = M^2 + B sin^2(k/2) (monotonic to pi).

Cross-volume consistency at ns = 6, 8, 10 is the certificate; Wigner time
delays 2 d(delta)/dE follow.  These are the prediction-table inputs for
comparison with wavepacket-collision simulations.

  PYTHONPATH=. .venv/bin/python scripts/phase_shifts.py
"""

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq

from htensor import Z2Lattice, exact, spectroscopy
from htensor import hamiltonian as ham

M0, G2, ETA = 0.7, 1.1, 1.3

# ---------------- dispersion: exact spline through volume-converged points
# (a parametric sin^2(k/2) fit leaves 0.05 residuals and biases M by 0.025,
# which near threshold corrupts the momentum inversion)
k_ed = np.array([0.0, 1.2566, 1.5708, 2.0944, 2.5133, np.pi])
e_ed = np.array([2.7451, 2.8188, 2.8560, 2.9275, 2.9886, 3.0778])
_spl = CubicSpline(np.concatenate([-k_ed[:0:-1], k_ed]),
                   np.concatenate([e_ed[:0:-1], e_ed]))


def disp(k, *_):
    return _spl(np.abs(np.asarray(k, float)))


Mfit, Bfit = e_ed[0], np.nan
print(f"dispersion: cubic spline through measured band, M = {Mfit:.4f}")

# ---------------- levels per volume
print("\nvolume sweeps:")
results = []
for ns, nst in ((6, 20), (8, 26), (10, 30)):
    lat = Z2Lattice(ns, pbc=True)
    mf = ns >= 10
    e, v = exact.lowest_physical_states(lat, M0, G2, ETA, k=nst,
                                        matrix_free=mf, ncv=48 if mf else None)
    states = [v[:, i] for i in range(v.shape[1])]
    resolved, phases = spectroscopy._t2_phases(states, e, lat)
    H_op = ham.build_hamiltonian(lat, M0, G2, ETA)
    if mf:
        e_res = np.array([np.real(np.vdot(s, exact.apply_pauli_sum(H_op, s)))
                          for s in resolved])
    else:
        Hs = exact.to_sparse(H_op)
        e_res = np.array([np.real(np.vdot(s, Hs @ s)) for s in resolved])
    order = np.argsort(e_res)
    e_res, phases = e_res[order], phases[order]
    gaps = e_res - e_res[0]
    L = ns // 2
    thresh = 2 * Mfit
    # P_tot = 0, above two-meson threshold, excluding the flat M* cluster
    cands = [g for g, ph in zip(gaps[1:], phases[1:])
             if abs(np.angle(np.exp(1j * ph))) < 1e-4
             and thresh - 0.05 < g < 2 * disp(np.pi, Mfit, Bfit) + 0.3
             and not (4.8 < g < 5.15)]
    free = [float(2 * disp(2 * np.pi * n / L)) for n in range(0, L // 2 + 1)]
    print(f"  ns={ns} (L={L}): P=0 two-meson candidates {np.round(cands, 4)}; "
          f"free levels 2E(2pi n/L) = {np.round(free, 4)}")
    for E2 in cands:
        pmax = np.pi * 0.999
        if E2 >= 2 * disp(pmax, Mfit, Bfit):
            continue
        p = brentq(lambda q: 2 * disp(q, Mfit, Bfit) - E2, 1e-9, pmax)
        # free levels p_n = 2 pi n / L; assign n = nearest free level
        n = max(1, int(round(p * L / (2 * np.pi))))
        for ntry in {n, n + 1, max(1, n - 1)}:
            delta = (2 * np.pi * ntry - p * L) / 2
            if -np.pi / 2 < delta <= np.pi:
                results.append((ns, E2, p, ntry, delta))
                print(f"     E2={E2:.4f}  p={p:.4f}  n={ntry}  "
                      f"delta = {delta:+.4f} rad ({np.degrees(delta):+6.1f} deg)")

# cross-volume view: delta(E) should collapse onto one curve for the right n
print("\ndelta(E) collapse (all volumes, all n-assignments kept):")
for ns, E2, p, n, delta in sorted(results, key=lambda r: r[1]):
    print(f"  E = {E2:.4f}  p = {p:.4f}  delta = {delta:+.4f}  "
          f"[ns={ns}, n={n}]")
np.savez("data/phase_shifts_ed.npz",
         results=np.array([(r[0], r[1], r[2], r[3], r[4]) for r in results]),
         M=Mfit, B=Bfit)
print("\nsaved data/phase_shifts_ed.npz")
