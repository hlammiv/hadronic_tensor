# Two-meson levels from the WAVEPACKET real-time current correlator

Scripts: `scripts/rt_packet_levels_ns12.py`, `scripts/rt_packet_levels_ns16.py`
(the latter imports `run` from the former). Outputs:
`data/rt_packet_levels_ns{12,16}.{npz,json}`, `data/rt_packet_levels.pdf`.
Wall time 119 s (N_s = 12) and 335 s (N_s = 16), one machine, no remote.
Production couplings (m0, g2, eta) = (0.7, 1.1, 1.3), M = 2.7452, elastic
window 5.490 < E_n - E_vac < 6.030. Everything is exact in the gauge-fixed
Q = 0 basis (`htensor/gf_engine.py`: dims 1848 and 25740).

## What was computed

**Packet.** The production target of `scripts/train_packets.py`: band-projected
Gaussian meson packet at rest, sigma_x = 0.75, x0 = N_x//2 (mid-ring), optimized
cur/hop mix, "clean" momentum weights (`gf_packet_target(..., clean=True)`).
Momentum content |Phi(k)|^2 on k = 2 pi j / N_x (the raw band-projected
interpolator packet, `clean=False`, in brackets):

| k/q | N_s = 12 | N_s = 16 |
|---|---|---|
| 0 | 0.6262 (0.6065) | 0.4700 (0.4596) |
| +-1 | 0.1824 (0.1919) | 0.2348 (0.2371) |
| +-2 | 0.0045 (0.0048) | 0.0293 (0.0320) |
| +-3 | -- | 0.0009 (0.0011) |
| pi | 1e-5 (1e-4) | 1e-8 (1e-8) |

Clean weights equal exp(-2 sigma_x^2 k^2) to 4 digits. Packet energy
<H> - E_vac = 2.7658 (N_s = 12), 2.7661 (N_s = 16), sigma_E = 0.029, 0.028;
vacuum weight < 1e-26; band fraction of the raw interpolator packet 0.9999;
<T2> = 0.804, 0.801 (a packet at rest is not a T2 eigenstate).

**Correlators** C(t) = <Phi| J(t) J(0) |Phi>, T = 200, dt = 0.1 (Nyquist
pi/dt = 31.4 vs the largest occurring |omega|: 17.8 at N_s = 12, 24.7 at
N_s = 16, so no aliasing; the ns = 16 full bandwidth E_max - E_vac is 27.4):

* (a) J = J1 = sum_b bond_current(b, eta): translation invariant, so every
  packet component |k_j> scatters into P = k_j.
* (b) J = J0'(c) = -Z_c/2 at the packet-centre site c = 2 x0 (c = 6, 8): the
  charge density minus its c-number (-1)^c/2. The c-number only adds an
  exactly known band-beat term (1/4) sum_jj' c_j* c_j' e^{-i(E_j' - E_j)t}
  (every cross term vanishes), so it was dropped. All P mix.
* The same two insertions on the momentum eigenstate |k=0> at the same T
  (the `rt_levels_ns*.py` correlators C11 and a local-density variant), for
  the line-density comparison.

Method: |Phi> = sum_j c_j |S_j> over the band multiplet states (7 states at
N_s = 12, 9 at N_s = 16, the k = pi level being a Kramers pair);
psi(t) = e^{-iHt} J|Phi> is projected onto the T2 sectors
(`_sector_hamiltonian`, sector dims 300-3240) and each sector is evolved with
`expm_multiply` for all 2001 samples; C(t) = sum_j c_j* e^{iE_j t}
<S_j|J|psi(t)>. Checks per correlator: sector completeness of J|Phi>
(residual < 2e-13), unitarity max |d||psi(t)||^2| < 1e-9, and
C(t) = sum_{n,P,j} A_{nPj} e^{-i(E_n(P) - E_j)t} with
A_{nPj} = c_j* <S_j|J|n,P><n,P|J|Phi> from a dense eigh of every sector:
max |C(t) - sum A e^{-i omega t}| < 6e-11 over the 2001 samples in all 8
correlators. C(t) is therefore a finite sum of exponentials with frequencies
omega = E_n(P) - E(k_j); in case (b) the amplitudes are complex and there is
one line per bra momentum k_j for each level (n, P).

**Extraction.** Hankel-SVD / ESPRIT matrix pencil as in `rt_levels_ns16.py`
(L = N/2, rank cut 1e-10 relative), with one necessary addition: poles with
||z| - 1| > 1e-3 (spurious, from near-noise singular vectors) are discarded
and the survivors put on the unit circle before the amplitude least-squares.
Without this the |z|^N Vandermonde columns of the spurious poles swamp the
amplitude fit for the dense local-insertion series (all amplitudes came out
~1e-30). Lines above 1e-4 of the largest |A| are kept. Each line is assigned
to the nearest exact merged line (degenerate (n, +-P), (+-k_j) partners and
Kramers pairs summed): status `ok` if |d omega| < 1e-3 and the pencil weight
is within a factor 3 of the exact one, `weight-mismatch` if only the
frequency agrees, `unmatched` otherwise; `E-ambiguous` flags lines for which
the energy-only enumeration of (n, P, k_j) with |omega - (E_n(P) - E_j)| < 1e-3
(P = k_j in case a, any P in case b) has more candidates than the exact
amplitudes single out. ED cross-reference: `data/deep_levels_ns{12,16}.npz`
(energies agree with the sector eigh to < 1e-6; the refl tag is quoted). n is
the 0-based level index inside the T2 sector P (n = 0 at P = 0 is the vacuum,
n = 1 the meson).

## N_s = 12 (N_x = 6, q = 1.0472)

**(a) J1 on the packet: 28 lines above 1e-4 (17 above 1e-3), 28 exact, all
`ok`, max |d omega| 8e-6, max ||z| - 1| 9e-6.** The eigenstate correlator
has 11 (10). Elastic-window lines (relative pencil weight; the exact
merged-line weight is the same to < 1%):

| omega | M + omega | (n, P/q) | E_n - E_vac | k_j/q | rel. weight | ED (refl) | d omega |
|---|---|---|---|---|---|---|---|
| 2.80465 | 5.5498 | (5, 0) | 5.5498 | 0 | 1.0 | 5.5498 (-1) | 7e-14 |
| 3.15094 | 5.8961 | (6, 0) | 5.8961 | 0 | 0.097 | 5.8961 (-1) | 9e-13 |
| 2.91817 | 5.6634 | (4, +-1) | 5.7155 | +-1 | 0.134 | 5.7155 | 1e-12 |
| 2.71061 | 5.4558 | (4, +-2) | 5.6381 | +-2 | 6.5e-3 | 5.6381 | 8e-11 |
| 2.98857 | 5.7338 | (6, +-2) | 5.9161 | +-2 | 8.1e-4 | 5.9161 | 3e-8 |

So the packet correlator exposes 5 distinct elastic levels in three frames
(P = 0, +-q, +-2q), against 2 (P = 0 only) for the eigenstate. Note that
omega = E_n(P) - E(k_j) with E(k_j) the boosted meson energy, so the line does
not sit at E_n - E_vac - M when k_j != 0 (5.6634 vs 5.7155 for P = +-q).
Not exposed: 5.8661, 5.9766 (P = +-2q) and 5.9959 (P = +-q): exact weight
0 (< 1e-20): these are the C-odd MM' levels (J1 is C-odd on a C-odd meson,
the same selection rule as in the eigenstate study); 5.8208 (P = pi, Kramers
pair): weight 5e-6, below the 1e-4 cut, because the packet has 1e-5 of its
norm at k = pi (momentum content, not a symmetry).

**(b) J0'(c) on the packet, T = 200: 135 exact lines above 1e-4 (84 above
1e-3), 1766 distinct frequencies, largest lines are the intra-band beats
omega = E(k') - E(k) (0.13, 0.18, 0.05, ...) and the vacuum lines
omega = -E(k_j).** The pencil returns 93 lines but only 18 `ok` (1
`degenerate`, 23 `weight-mismatch`, 51 `unmatched`, fit residual 2.5e-2):
the spectrum is too dense for T = 200. No elastic-window level is exposed
with a reliable amplitude at T = 200. The window levels do carry weight,
all of them, including the C-odd MM' levels and the P = 0 levels the
eigenstate cannot reach (relative to the largest line): 5.9766 (+-2q) 8e-2,
5.8208 (pi) 6e-2, 5.6381 (+-2q) 2e-2, 5.9161 (+-2q) 2e-2, 5.9959 (+-q) 2e-2,
5.8661 (+-2q) 9e-3, 5.7155 (+-q) 7e-3, 5.5498 (P = 0) 2e-3, 5.8961 (P = 0)
5e-4. The local insertion lifts the C-parity rule (Z_c alone is not a C
eigen-operator: C for staggered fermions includes the one-site shift) but
pays with a spectrum 5x denser than J1 on the packet and 8x denser than J1
on the eigenstate. A P = 0 elastic level is reached only through the
packet's k = +-q components (the k = 0 component gives exactly zero: the
site-centred density is reflection-even about c, and the k = 0 meson and the
P = 0 MM levels reached by J1 have opposite reflection parity about a site).

Pencil vs T on the exact spectral series (validated against the evolved series
to 7e-12; dt = 0.2, still above Nyquist for all lines that carry weight;
"ok" = |d omega| < 1e-3 and weight within a factor 3):

| T | J1 packet, lines ok | J0'(c) packet, lines ok | J0'(c) packet, window lines ok (of 22) | J0'(c) eigenstate, lines ok |
|---|---|---|---|---|
| 200 | 28/28 | 16 % | 0 | 66 % |
| 400 | 28/28 | 49 % | 12 | 95 % |
| 800 | 28/28 | 66 % | 17 | 95 % |
| 1600 | 28/28 | 94 % | 22 | 98 % |

**Resolution.** Smallest gap between adjacent J1-packet lines above 1e-4:
0.0016 (the pair 3.75688 / 3.75848 = (11, +-q) at E = 6.5542 seen from
k = +-q and (13, 0) at 6.5037 from k = 0; resolved by the pencil at T = 200
to 2e-7 and 3e-7, with the correct 8.0e-3 / 2.4e-2 weights). A plain
Fourier read-off needs T ~ 2 pi / 0.0016 = 3900 (twice that with a Hann
window) for that pair; for the elastic-window lines the nearest other line
above 1e-4 is 0.070 away (T_F ~ 90), against 0.35 for the eigenstate
correlator (T_F ~ 18, which is why it was Fourier-readable at T = 40 in
`rt_levels_ns12`). Local insertion: adjacent exact lines above
1e-3 come as close as 6e-5 (near-coincidences of E_n(P) - E_j for different
(n, P, k_j)), so a Fourier read-off is out of the question and even the
pencil needs T ~ 1600.

## N_s = 16 (N_x = 8, q = 0.7854)

**(a) J1 on the packet: 38 exact lines above 1e-4 (27 above 1e-3); pencil at
T = 200 returns 37 (26), 28 `ok`, 9 `unmatched`** (all nine are lines above
M + omega = 6.28 with relative weight < 2e-3, off by 1.7e-3 to 1.5e-2 at
T = 200; the T-scan gives 74 % ok at T = 200, 95 % at 400, 100 % at 800).
The eigenstate correlator has 13 (11). All 10 elastic-window lines are `ok`
at T = 200:

| omega | M + omega | (n, P/q) | E_n - E_vac | k_j/q | rel. weight | ED (refl) | d omega |
|---|---|---|---|---|---|---|---|
| 2.77226 | 5.5175 | (5, 0) | 5.5175 | 0 | 1.0 | 5.5175 (+1) | 5e-8 |
| 2.95988 | 5.7051 | (6, 0) | 5.7051 | 0 | 0.096 | 5.7051 (+1) | 9e-8 |
| 3.23006 | 5.9752 | (7, 0) | 5.9752 | 0 | 0.035 | 5.9752 (+1) | 1e-8 |
| 2.82938 | 5.5746 | (4, +-1) | 5.6045 | +-1 | 0.28 | 5.6045 | 8e-8 |
| 3.06629 | 5.8115 | (5, +-1) | 5.8414 | +-1 | 0.051 | 5.8414 | 6e-8 |
| 2.71697 | 5.4622 | (4, +-2) | 5.5730 | +-2 | 0.042 | 5.5730 | 3e-6 |
| 2.87903 | 5.6242 | (5, +-2) | 5.7350 | +-2 | 0.013 | 5.7350 | 4e-6 |
| 3.12184 | 5.8670 | (8, +-2) | 5.9778 | +-2 | 3.2e-3 | 5.9778 | 2e-5 |
| 2.72543 | 5.4706 | (4, +-3) | 5.6910 | +-3 | 2.1e-3 | 5.6910 | 4e-5 |
| 2.90739 | 5.6526 | (6, +-3) | 5.8724 | +-3 | 1.0e-4 | 5.8724 | 6e-4 |

Ten distinct elastic levels in four frames (P = 0, +-q, +-2q, +-3q) against
3 (P = 0) for the eigenstate; the +-3q pair is at the edge of the weight cut
(the packet has 9e-4 of its norm at k = 3q). Not exposed: zero weight (C-odd
MM'): 5.8454, 5.9340 (+-3q); 5.9321, 5.9492, 6.0125 (+-2q); 6.0145, 6.0289
(+-q). P = pi (Kramers pairs 5.7246, 5.8215, 5.9885): weight < 1e-5, packet
momentum content. `E-ambiguous` flags: 2 lines (5.7350 at +-2q also matches
an energy-only candidate within 1e-3; the exact amplitude selects (5, +-2q)).

**(b) J0'(c) on the packet: 265 exact lines above 1e-4 (155 above 1e-3),
5534 distinct frequencies; pencil at T = 200: 17 `ok`, 23 `weight-mismatch`,
50 `unmatched`.** One window level (5.5730, +-2q, through the k = +-q bra,
pencil weight 8e-5 vs exact 4e-3: `weight-mismatch`, i.e. not usable).
Window weights are all non-zero (1.6e-4 ... 5.9e-2 relative; P = pi levels
now at 8e-3 ... 5e-2 because Z_c is local). Scan: T = 400 gives 6 of the 65
window lines, T = 800 22, T = 1600 42 (62 % of all lines): a 101-qubit
site-density correlator on the packet would need T well beyond 1600 to be
inverted line by line at this volume. The eigenstate version (61 lines above
1e-4) is converged at T = 800 (97 %, 15/15 window lines).

**Resolution.** Smallest gap between adjacent J1-packet lines above 1e-4:
0.0085 (2.71697 = (4, +-2q) from k = +-2q vs 2.72543 = (4, +-3q) from
k = +-3q, both in the window; resolved at T = 200 to 3e-6 and 4e-5).
Fourier read-off: T ~ 2 pi / 0.0085 = 740 (1500 Hann); the pencil does it at
T = 200. The k_j-dependence of the frames therefore produces window lines
that a T ~ 100 Fourier analysis would blend (the P = +-2q and +-3q ground
levels 0.0085 apart, the P = 0 and +-q levels 0.06 apart). Local insertion:
adjacent exact lines as close as 3.6e-5.

## Line density (lines above 1e-3 of the largest, exact / pencil at T = 200)

| | J1 eigenstate | J1 packet | J0'(c) eigenstate | J0'(c) packet |
|---|---|---|---|---|
| N_s = 12 | 10 / 10 | 17 / 17 | 31 / 29 | 84 / 64 |
| N_s = 16 | 11 / 11 | 27 / 26 | 41 / 32 | 155 / 71 |

(`rt_levels_ns12/16.py`, T = 40: C11 10 and 10 modes above 1e-3, C00_q 6
and 7, consistent with the eigenstate columns.) The packet multiplies the J1
line count by 1.7 (N_s = 12) to 2.5 (N_s = 16), one extra family per
populated frame; the local insertion multiplies it by 8 to 14 and adds the
band-beat lines below omega = 0.5, which dominate the weight.

## Summary of what the packet correlator exposes

* J1 (what a translation-invariant current insertion measures): every
  identical-boson MM level of every frame P = k_j the packet populates, with
  weight |c_{k_j}|^2 |<n,P|J1|k_j>|^2; C-odd MM' levels never, P = pi levels
  only at the 1e-5 level. Lines in the window are read off to < 1e-4 at
  T = 200 by the pencil at both volumes (< 1e-6 for the dominant ones).
* J0'(c) (a single-site density): all levels, both C-parities, all frames,
  but with a 5-14x denser spectrum dominated by intra-band beats; at T = 200
  none of its window lines is recoverable, at N_s = 12 T ~ 1600 recovers all
  22 of them, at N_s = 16 T = 1600 recovers 42 of 65.

## Caveats

* Everything is exact and noise-free; the T-scan is a statement about the
  pencil on an exact series, not about shot noise or hardware error. The
  T = 400 ... 1600 series in the scan were synthesized from the exact
  spectral decomposition (validated against the sparse evolution at T = 200
  to < 6e-11); the evolved series themselves are T = 200 as specified.
* The pencil's amplitude-consistency criterion (factor 3) and the 1e-3
  frequency criterion are choices; the JSON keeps every line with its
  deviation and weight ratio so other cuts can be applied.
* J0'(c) drops the c-number (-1)^c/2 of charge_density(c); with it the
  band-beat lines are a factor 1 + O(1) stronger and nothing else changes.
* At N_s = 16 nine weak J1-packet lines above the window (rel. weight
  < 2e-3) are off by up to 1.5e-2 at T = 200 (converged by T = 800); no
  window line is affected.
* The ED `refl` tag has opposite sign conventions in deep_levels_ns12 and
  ns16 (vacuum -1 vs +1); it is quoted raw and only used as a label.
* The k = pi Kramers pair enters through both multiplet members; its weight in
  the packet is set by the eigensolver's basis choice only at the 1e-8 level.
