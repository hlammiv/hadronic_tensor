# Mirror-understatement systematic (adopted 2026-11-19)

The mirror's survival kappa is what every hardware slice is divided by.  In
the empty-lattice region far from the packet the physics circuit loses more
than the mirror implies, on both chips, so the mitigation under-corrects
the late times.  Choices (Henry, 19 Nov): measure the ratio from the far-site
Z profile, one ratio per t per run (per-site and per-parity values recorded
for later checks), and quote the full shift of each asymmetry under
kappa -> kappa r(t) as the systematic, central value unchanged.

## Files

- `measure_mirror_ratio.py OUT.npz BITS_GLOB`: r(t) = median over the far
  sites (|n - 24| >= 8) of [physics Z(t)/Z(0) over ideal Z(t)/Z(0)] / [mirror
  Z(t)/Z(0)]; also records r per site, per parity and the centre-correlator
  cross-check (`<tag>_r`, `_r_even`, `_r_odd`, `_r_site`, `_r_centre_corr`).
  Beyond the relA ideal grid (t > 3) the far sites take B(0) plus the vacuum
  surrogate's shift, the rule of `analyze.wing_target`.
- `mirror_ratio_{phoenix_2026-10-06,chicago_2026-11-11}.{npz,txt}`: measured.
- `../analyze_debias_kscale.py`: the analyze wrapper that multiplies every
  mirror kappa by r(t) (`HTQ_KAPPA_SCALE=<npz>`), recording the unscaled
  kappa so the map masks the same sites.
- `run_mirror_systematic.sh`, `shift_table.py` -> `shift_table.txt`.

## Measured ratio r(t), far sites

| t | Phoenix slow | Chicago slow | Phoenix fast | Chicago fast |
|---:|---:|---:|---:|---:|
| 1 | 0.91 | 0.87 | 0.81 | 0.90 |
| 2 | 0.80 | 0.71 | 0.56 | 0.63 |
| 4 | 0.68 | 0.69 | 0.44 | 0.61 |
| 6 | 0.56 | 0.60 | 0.30 | 0.37 |

Even and odd sites agree to a few hundredths.  The fast-particle card
(stronger coupling, larger vacuum breathing) shows the larger
understatement.  Early fast-particle values above 1 (1.05-1.16 at
t <= 0.5) mean the hardware vacuum breathes less than the ideal one there,
which this measurement cannot separate from damping.  The centre-site
correlator check is noisier and sits between 0.5 and 1.

## The systematic (wedge [0, 2.5], t_max 6)

| run | particle | W | nominal | shot | window | with kappa x r | systematic |
|---|---|---|---:|---:|---:|---:|---:|
| Phoenix | slow | 00 | 0.76 | 0.03 | 0.15 | 0.96 | 0.19 |
| Phoenix | slow | 10 | 0.94 | 0.07 | 0.14 | 1.12 | 0.18 |
| Phoenix | fast | 00 | 0.83 | 0.08 | 0.03 | 1.06 | 0.23 |
| Phoenix | fast | 10 | 1.39 | 0.06 | 0.10 | 1.61 | 0.22 |
| Phoenix | fast | 01 | 1.45 | 0.08 | 0.09 | 1.76 | 0.30 |
| Phoenix | fast | 11 | 1.08 | 0.12 | 0.02 | 1.36 | 0.27 |
| Chicago 11 Nov | slow | 00 | 1.10 | 0.04 | 0.23 | 1.53 | 0.43 |
| Chicago 11 Nov | slow | 10 | 3.38 | 0.55 | 0.20 | 3.93 | 0.55 |
| Chicago 11 Nov | fast | 00 | 0.09 | 0.11 | 0.05 | 0.19 | 0.10 |
| Chicago 11 Nov | fast | 10 | 1.89 | 0.10 | 0.11 | 2.04 | 0.15 |
| Chicago 11 Nov | fast | 01 | 1.31 | 0.06 | 0.04 | 1.34 | 0.03 |
| Chicago 11 Nov | fast | 11 | 1.20 | 0.41 | 0.03 | 1.24 | 0.03 |

The Phoenix nominal reduction through `reduce_run.sh` reproduces the
published values (0.76, 0.83, 1.45, 1.39).  Every shift is upward: scaling
kappa down boosts the late slices, which carry the asymmetry.  On Phoenix
the systematic is 0.18-0.30, comparable to or larger than the window error,
so it becomes the leading error of every Phoenix asymmetry.  It is one-sided
in fact; we quote it symmetric, as chosen.
