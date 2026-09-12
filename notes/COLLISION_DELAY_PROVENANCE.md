# Collision-delay provenance concern

Status: tabled for later resolution. This note records the read-only provenance
audit performed on 2026-09-04. It does not alter the manuscript or analysis.

## Numbers at issue

The manuscript quotes, for the broad resonant packet at the CGK-A couplings,

\[
\tau_{j=3}=+0.4\pm1.9_{\rm stat}\pm3.0_{\rm sys},
\qquad
\tau_{\rm folded}=+2.6.
\]

## PRV-001: folded prediction `+2.6`

Historical notes identify `+2.6` as the nominal average of the Wigner delay
`2 d delta/dE` over the broad `j=3` packet, whose momentum width is
approximately `sigma_k = 0.67`.

This is not evidently a unique folding prescription. The same record reports
that restricting the average to the faster components followed by the
trajectory estimator gives approximately `+5.3` for `j=3` and `+5.0` for the
control packet. It also reports that 18--32% of the packet weight lies above
the elastic window, where the smooth elastic Wigner-delay treatment is not
applicable.

No surviving script or saved numerical output was found that independently
regenerates exactly `+2.6`. Before using it as a quantitative prediction, the
paper should define the momentum/energy weighting, normalization, treatment of
group velocity, and treatment of weight outside the elastic region.

## PRV-002: measured result `+0.4 +/- 1.9 +/- 3.0`

The first recovered record of this result calls it the final `N_s = 40`
straight-leg trajectory fit:

- central value: average over packet and fit-window combinations;
- `+/- 1.9`: propagated uncertainty from the straight-line fits;
- `+/- 3.0`: spread under the trajectory-window scan.

The surviving script described as the production implementation is
`scripts/stageB_delay_ns40.py`. Its recorded first execution on 2026-08-03 did
not reproduce the quoted result. It returned

\[
\tau_{j=3}=+77.38\pm44.89_{\rm stat}\pm62.21_{\rm window},
\]

with a window/packet range from `+5.67` to `+159.53`. Re-running the current
script produces the same result.

The historical session immediately diagnosed that the two-Gaussian tracker
collapsed onto the same weak outgoing feature after the collision. The
periodic ring also permits additional encounters before a clean asymptotic
outgoing region is established.

## PRV-003: failed cross-checks

Later solo-packet-reference analyses did not validate the leg-fit value. The
recovered resonant-channel estimates were approximately

- `+6.8 +/- 1.1` for one mover;
- `+2.7 +/- 1.6` for the mirror-related mover;
- approximately `+4.8` when combined.

The two movers are physically equivalent apart from reflection, so their
4.1-unit disagreement is an internal-consistency failure. The result also
changed substantially with the correlation-window width.

The historical record therefore contains three incompatible values:

| Method | Delay |
|---|---:|
| straight-leg fit | `+0.4` |
| solo-packet reference | approximately `+4.8` |
| nominal packet-folded prediction | `+2.6` |

Each method has an identified systematic: nonlinear/dispersive free
trajectories for the leg fit, partner-packet and periodic-image contamination
for the solo reference, and an ambiguous weighting plus nonelastic packet
support for the folded prediction.

## PRV-004: present recommendation

Until an older missing analysis artifact is recovered or the calculation is
redone:

1. Do not describe `+0.4 +/- 1.9 +/- 3.0` as a reproducible numerical
   measurement.
2. Do not describe `+2.6` as a unique packet-folded prediction without an
   explicit folding definition.
3. Prefer reporting the broad-packet collision delay as unresolved, or remove
   its numerical row while retaining the qualitative lesson about packet
   reshaping and resolution.
4. Resolve the issue with a larger ring and momentum-narrow packets. This
   reduces the energy fold and provides a longer interval of asymptotic
   separation before a periodic-image collision.

## Recovered provenance sources

- `scripts/stageB_delay_ns40.py`
- `scripts/stageB_ctrl_delay.py`
- `scripts/stageB_collision_density.py`
- Claude file-history snapshot `0d5761d26063c584@v6`, titled
  `project-collision-delay-lessons`
- Claude session `fb5de832-5f2a-4241-8c29-2ea520bb7131`, especially the
  collision-analysis records from 2026-07-23--24 and 2026-08-03

