# Elastic phase shifts from the real-time hadronic-tensor correlators

Script: `scripts/rt_levels_luscher.py` (post-processing only, ~2 s).
Outputs: `data/rt_levels_luscher.{npz,json,pdf}`. Inputs: the matrix-pencil
poles of `data/rt_levels_ns{12,16,20}.npz` (from `scripts/rt_levels_ns*.py`,
T = 40) and the ED phase-shift rows of `data/phase_shifts_6vol.npz`
(the points behind `figures/phase_shift_collapse.pdf`, Sec. VI.1).
Production couplings (m0, g2, eta) = (0.7, 1.1, 1.3), M = 2.7451,
elastic window 2M = 5.490 < E2 < M + M'(0) = 6.030.

## What was computed

1. **P = 0.** The poles of C11(t) = <M|J1(t) J1(0)|M> (total spatial
   current on the k = 0 meson) in the elastic window were inverted with the
   paper's quantization condition (eq. luscher, code copied from
   `scripts/phase_shifts_v2.py`): p N_x + 2 delta(p) = 2 pi n, E2 = 2E(p),
   with the same clamped-spline dispersion E(k) through the six ED
   single-meson levels (M = 2.7451 ... E(pi) = 3.0778), n counted up from the
   free tower (n = 0 allowed), branch (-pi/2, pi/2]. Every point was compared
   with the stored ED point of the same volume and n (the stored rows were
   re-inverted here as a check: agreement to < 1e-6 in p and delta).
2. **P = 2 pi/N_x.** The poles of C00_q(t) = <M|J0_q(t) J0_{-q}(0)|M>
   (charge density at the smallest lattice momentum q) were inverted with the
   1+1d Bethe-Yang condition of `scripts/moving_frame_luscher.py`
   (`solve_k1`/`fold`/`extract` reused with the production dispersion):
   E2 = E(k1) + E(P - k1), k1 N_x + 2 delta = 2 pi n1, n1 = 1 + j for the j-th
   interacting level of the m = 1 sector, E_cm = 2E(k1 - P/2). The sector
   MM' threshold thr(P) = min_k [E(k) + E'(P - k)] was built from the ED
   band-2 (M') levels of ns = 20 plus the stored band minimum 3.155 at k = pi.
3. The ERE curve delta = pi/2 - a p + r p^3 was refitted to the 16 stored ED
   points exactly as in `scripts/phase_shift_figure.py` (a = 1.0045,
   r = 0.0244) and used as the overlay curve in the figure.

## Levels reproduced / not reproduced

**P = 0, C11 (identical-boson MM channel, R = +1).** All nine ED elastic
levels at N_s = 12, 16, 20 appear in C11 and none is missed:

| N_s | N_x | n | E2 (real time) | E2 (ED) | E2_rt - E2_ED | rel. weight |
|---|---|---|---|---|---|---|
| 12 | 6 | 0 | 5.549840 | 5.549840 | +3.6e-10 | 1.0 |
| 12 | 6 | 1 | 5.896133 | 5.896133 | -1.9e-09 | 0.097 |
| 16 | 8 | 0 | 5.517451 | 5.517451 | -4.3e-08 | 1.0 |
| 16 | 8 | 1 | 5.705067 | 5.705068 | -1.9e-06 | 0.096 |
| 16 | 8 | 2 | 5.975273 | 5.975250 | +2.4e-05 | 0.035 |
| 20 | 10 | 0 | 5.505706 | 5.505706 | -5.1e-08 | 1.0 |
| 20 | 10 | 1 | 5.618983 | 5.618984 | -5.3e-07 | 0.100 |
| 20 | 10 | 2 | 5.806715 | 5.806742 | -2.7e-05 | 0.033 |
| 20 | 10 | 3 | 6.017606 | 6.017185 | +4.2e-04 | 0.018 |

Nothing that the current can couple to is absent. ED levels in or just above
the window that do NOT appear (exact sector weight ~1e-26 or below, checked
in the rt_levels_ns* runs) are the C-odd MM' states (e.g. 6.0356 at N_s = 16,
the 6.03-6.16 cluster at N_s = 20): J1 is C-odd and the meson is C-odd, so
J1|M> is C-even and cannot reach them; at P = 0 the ED reflection tag gives
the same exclusion (they are the R = -1 states the paper's channel assignment
removes). The current therefore performs the paper's parity/channel
selection automatically: only identical-boson MM levels enter C11 in the
window. The single-meson elastic piece <M|J|M> vanishes (C-odd operator
between C-odd states), so no zero-frequency line has to be subtracted.

**P = 2 pi/N_x, C00_q.** Reproduced: 5.71548 (N_s = 12); 5.60449, 5.84140
(N_s = 16); 5.55827, 5.71150, 5.91351 (N_s = 20), all to < 4e-5. Not
reproduced: 5.99595 (N_s = 12), 6.01452 and 6.02887 (N_s = 16), 6.02144
(N_s = 20), all with exact weight < 1e-25. These sit at the moving-frame
MM' threshold M(0) + M'(q) (5.989 / 6.012 / 6.018) and are C-odd MM'
(vector x scalar) levels, again forbidden from the C-even ket J0_{-q}|M>
(verified at N_s = 16 by coupling the C-even scalar density instead, which
reaches exactly those two levels and not the MM pair). Reflection is not a
quantum number in the moving frame, so C-parity is the only selection rule
there, and it is exactly what the ED-level bookkeeping needs.

## delta(p) comparison with the ED curve (P = 0)

| N_s | n | p | delta_rt | delta_ED | delta_rt - delta_ED |
|---|---|---|---|---|---|
| 12 | 0 | 0.77665 | +0.81165 | +0.81165 | -7.5e-09 |
| 12 | 1 | 2.24724 | -0.45853 | -0.45853 | +2.1e-08 |
| 16 | 0 | 0.51722 | +1.07270 | +1.07269 | +1.7e-06 |
| 16 | 1 | 1.54367 | +0.10852 | +0.10849 | +2.9e-05 |
| 16 | 2 | 2.50780 | -0.60644 | -0.60617 | -2.7e-04 |
| 20 | 0 | 0.38733 | +1.20497 | +1.20496 | +3.3e-06 |
| 20 | 1 | 1.16821 | +0.44212 | +0.44211 | +1.3e-05 |
| 20 | 2 | 1.91905 | -0.17047 | -0.17096 | +4.9e-04 |
| 20 | 3 | 2.62220 | -0.54464 | -0.53916 | -5.5e-03 |

RMS(delta_rt - delta_ED) over the 9 points = **1.8e-3 rad**, max
5.5e-3 rad, to be compared with the 0.03 rad scatter of the ED collapse
about the ERE fit. The RMS is entirely the N_s = 20, n = 3 level
(6.0176 vs 6.0172 at T = 40; the stored T = 80 rerun
`data/rt_levels_ns20_T80.npz` gives 6.017185, i.e. it reproduces the ED
level, and the real-time delta then agrees to < 1e-4). Without that point
the RMS is 2.0e-4 rad. The dominant (n = 0) line of every volume gives
delta to 1e-6 or better.

Moving frame (supplement, not in the RMS): the six Bethe-Yang points
(p_rel, delta) = (1.529, +0.126), (1.037, +0.563), (2.041, -0.310),
(0.780, +0.814), (1.547, +0.120), (2.281, -0.407) agree with the ED-level
values from the same sector to < 5e-4 (level differences only) and lie
+0.004 to +0.024 rad above the P = 0 ERE curve (-0.037 for the N_s = 16
n1 = 2 point), i.e. within the ERE scatter at these small boosts
(P <= 1.05). No stored production-coupling moving-frame curve exists to
compare with (`data/moving_frame_delta.npz` is CGK-A), and at CGK-A the
moving-frame phase was found to be frame-dependent, so these points are
reported as level comparisons plus a consistency check only.

## Caveats

- The levels come from the tensor's own correlators (the currents J1 and
  J0_q acting on the single meson), noise-free and to T = 40 with a
  matrix pencil; on hardware data the Fourier limit 2 pi/T ~ 0.16 applies
  and the sub-dominant lines (relative weight 0.02-0.1) would need
  T >~ 40-80 or a Prony/pencil fit with error bars. The precision here is a
  statement about the exact correlator, not about a measurement.
- Only positions were used; the spectral amplitudes |<n|J|M>|^2 (which the
  pencil also returns and which match the exact sector weights) were not
  converted into anything. No Lellouch-Luscher style amplitude analysis was
  attempted.
- The channel selection by C-parity is what makes the correlator levels a
  clean MM set; it is a property of these operators (C-odd current on a
  C-odd meson) and does not by itself tag R at P = 0 (the R = -1 MM'
  levels happen to also be C-odd).
- Dispersion, window, n counting, branch and ERE fit are the paper's,
  reused unchanged, so the comparison isolates the level source alone.
- The N_s = 20, n = 3 level at 6.0176 lies 0.013 below the MM' threshold
  6.030; the paper already treats it as elastic (it is in the stored set).
