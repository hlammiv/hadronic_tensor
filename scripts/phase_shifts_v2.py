"""Phase shifts v2 -- incorporates the adversarial-review fixes (notes/
adversarial_review_2026-07-06.txt):

  B-F1: single-channel extraction restricted to E < M + M' (= 5.900), the
        MM' inelastic threshold; the 6.0-6.2 cluster is reported as
        multi-channel and NOT converted to a phase shift.
  B-F2: n assigned by counting interacting levels up from the free tower
        (n = 0 allowed); symmetric branch delta in (-pi/2, pi/2]; deduped.
  B-F4: uses the deep 60-state spectra (data/deep_levels_ns{8,10}.npz).
  B-F5: dispersion spline clamped E'(0) = E'(pi) = 0.
  B-F3: ns=6 excluded with the explicit statement that L=3 sits below the
        asymptotic-validity range (exponential corrections; its levels sit
        ABOVE free counterparts, in tension with ns=8/10 -- reported).

  PYTHONPATH=. .venv/bin/python scripts/phase_shifts_v2.py
"""

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq

k_ed = np.array([0.0, 1.2566, 1.5708, 2.0944, 2.5133, np.pi])
e_ed = np.array([2.7451, 2.8188, 2.8560, 2.9275, 2.9886, 3.0778])
E = CubicSpline(k_ed, e_ed, bc_type=((1, 0.0), (1, 0.0)))
M = e_ed[0]
MPRIME = 3.155          # band-2 minimum gap (at k = pi: the band is inverted)
# P=0 MM' threshold is min_p [E1(p) + E2(p)] = E1(0) + E2(0) = 6.0304, NOT
# M + min(E2) = 5.900: with the inverted band-2 dispersion the pair must
# carry opposite momenta, so the naive threshold overquarantined the
# 5.90-6.03 levels (they are elastic).  Derived in scripts/two_channel.py.
E_INEL = 6.0304
print(f"dispersion: clamped spline, E'(pi) = {float(E(np.pi, 1)):.4f}; "
      f"single-channel window: 2M = {2*M:.4f} .. M+M' = {E_INEL:.4f}")

rows = []  # (ns, E2, p, n, delta)
for ns in (8, 10, 12, 14, 16, 18, 20):
    d = np.load(f"data/deep_levels_ns{ns}.npz")
    gaps, phases = d["gaps"], d["phases"]
    L = ns // 2
    refl = d["refl"] * d["refl"][0] if "refl" in d else np.ones_like(gaps)
    # (vacuum-normalized: stored labels carry a volume-alternating Fock
    # reordering sign; refl[0] is the vacuum)
    # reflection parity R = +1 required: identical-boson MM levels are
    # parity-even (the k=0 meson is R = -1, squared).  R = -1 levels in the
    # elastic window are odd-channel MM' states, not MM (two such
    # misassignments caught at ns=12 and ns=20).
    p0 = [g for g, ph, r in zip(gaps, phases, refl)
          if abs(np.angle(np.exp(1j * ph))) < 1e-4 and r > 0.99]
    # interacting MM candidates in the single-channel window
    cands = sorted(g for g in p0 if 2 * M - 0.06 < g < E_INEL)
    multi = sorted(g for g in p0 if E_INEL <= g < 2 * E(np.pi) + 0.2)
    # free tower (n = 0, 1, ...)
    free = [2 * float(E(2 * np.pi * n / L)) for n in range(L // 2 + 1)]
    print(f"\nns={ns} (L={L}): free MM tower {np.round(free, 4)}")
    print(f"  single-channel candidates: {np.round(cands, 4)}")
    print(f"  multi-channel window (excluded, MM/MM' mixed): {np.round(multi, 4)}")
    for i, E2 in enumerate(cands):
        n = i  # level counting up from the free tower, n = 0 allowed
        if E2 <= 2 * M + 1e-9:
            print(f"    E2 = {E2:.4f}: below threshold (bound-state candidate)")
            continue
        p = brentq(lambda q: 2 * E(q) - E2, 1e-9, np.pi)
        delta = (2 * np.pi * n - p * L) / 2
        delta = (delta + np.pi / 2) % np.pi - np.pi / 2  # branch (-pi/2, pi/2]
        rows.append((ns, E2, p, n, delta))
        print(f"    E2 = {E2:.4f}  p = {p:.4f}  n = {n}  "
              f"delta = {delta:+.4f} rad ({np.degrees(delta):+6.1f} deg)")

np.savez("data/phase_shifts_6vol.npz", rows=np.array(rows),
         columns="ns,E2,p,n,delta", M=M, MPRIME=MPRIME)
print(f"\nsaved data/phase_shifts_6vol.npz ({len(rows)} points)")
