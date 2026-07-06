# Data release: Hadronic tensor of 1+1d Z2 lattice gauge theory on 101 qubits

Companion data for *"The hadronic tensor from a quantum computer: meson
structure in 1+1d Z2 gauge theory"* (Fermilab, 2026). Every file was produced
by a script in `scripts/` of this repository (provenance noted per file);
figures (`*.pdf`) alongside the `.npz` files are the paper figures rendered
from exactly these arrays. Machine-readable CSV exports of the key tables
live in `data/csv/` (regenerate with
`PYTHONPATH=. .venv/bin/python scripts/export_tables.py`).

All array schemas below were verified by loading each `.npz` with NumPy.

## Physics conventions (apply to all files)

- **Model.** Z2 gauge theory with staggered fermions on a periodic ring of
  `ns = 50` staggered sites (`Nx = ns/2 = 25` spatial sites), explicit link
  qubits. Couplings for all headline files: `m0 = 0.7`, `g2 = 1.1`,
  `eta = 1.3` (stored redundantly in each production file; the
  `predict_*.npz` prediction sweeps use other couplings, stored in-file).
  All quantities are in lattice units (staggered-site spacing = 1/2
  spatial-site spacing; positions `x` are quoted in *spatial-site* units,
  `x = (v - v_c)/2`, minimal-image folded on the ring).
- **Circuits.** 101 qubits = 50 site qubits + 50 link qubits + 1 Hadamard-test
  ancilla, simulated with the Qiskit Aer matrix-product-state backend
  (truncation threshold `trunc = 1e-8`, Trotter step `dt_target = 0.1`
  unless noted). Correlators are ancilla-assisted (Hadamard-test)
  measurements `C^{0 nu}(v, t) = <psi| J^nu(v, t) J^0(v_c, 0) |psi>`.
- **Subtractions.** Connected correlators are formed as
  `C - <J^nu(v,t)><J^0(v_c)>` (the interacting vacuum carries a staggered
  one-point profile, so this matters); wavepacket runs additionally subtract
  the matched vacuum grid pointwise ("connected-minus-connected").
- **One-sided vs completed Fourier transform (important).** Two time-domain
  conventions appear:
  1. *Completed (hermiticity) transform*: data on `t in [0, T]` are extended
     to `t < 0` via `G(x, -t) = [G(-x, t)]*` (`htensor.analysis.complete_time`)
     before the Gaussian-windowed double FT. This is exact for the **vacuum**
     tensor and is used in `w00_vac_ns50_W.npz` and the
     `w_meson_*_v3_W.npz` dispersion-region grids.
  2. *One-sided transform*: FT over `t >= 0` only (t = 0 weighted 1/2). For
     **wavepacket** states the hermiticity completion is *not* valid at our
     precision — it scatters higher-sector content into the low-omega window —
     so all ridge-region (low q0) physics in the paper uses the one-sided
     convention (`scripts/final_ridge_overlay.py`, `scripts/ridge_model_v2.py`,
     `scripts/axial_from_existing.py`). Reproduce ridge-region numbers from
     the raw time-domain files with a one-sided transform; the
     `*_axial.npz` grids already are one-sided.
- **Windowed FT.** `W(q0, q1) = dt dx sum_{t,x} e^{i q0 t - i q1 x}
  e^{-t^2/2 sigma_t^2} e^{-x^2/2 sigma_x^2} G(x, t)`. Central window and
  `spread` = peak-to-peak of Re W over window rescalings {0.75, 1.0, 1.5}
  (the windowing systematic). Sign convention: `+i q0 t - i q1 x`.
- **Key spectral inputs** (small-volume ED, volume-converged): meson band
  `E(k) = 2.7451, 2.8188, 2.8560, 2.9275, 2.9886, 3.0778` at
  `k = 0, 2pi/5, pi/2, 2pi/3, 4pi/5, pi`; second-band minimum
  `M' = 3.155` (band-2 gaps 3.155–3.285); two-meson threshold `2M = 5.490`;
  inelastic MM' threshold `M + M' = 5.900`.

## Files

### `w00_vac_ns50.npz` — raw vacuum-polarization run (101 qubits)
Produced by `scripts/run_w00_vacuum_ns50.py`.
| key | shape / dtype | meaning |
|---|---|---|
| `times` | (13,) float64 | t = 0.0, 0.5, ..., 6.0 |
| `corr` | (13, 50) complex128 | raw `<J^0(v,t) J^0(v_c,0)>` on the ns=6-optimized variational vacuum transferred to ns=50; columns = staggered site v = 0..49 |
| `probe_1pt` | (13, 50) complex128 | `<J^0(v,t)>` one-point grid (for the disconnected subtraction) |
| `insert_1pt` | scalar complex128 | `<J^0(v_c)>` = -0.086734 |
| `ns`, `vc` | int64 | 50 staggered sites; insertion at v_c = 25 |
| `m0`, `g2`, `eta`, `dt_target`, `trunc` | float64 | 0.7, 1.1, 1.3, 0.1, 1e-8 |
| `thetas` | (8,) float64 | 2-layer variational vacuum angles (optimized at ns=6, volume-transferred) |

### `w00_vac_ns50_W.npz` — vacuum W^{00}(q0, q1)
Produced by `scripts/plot_w00_vacuum.py` (completed-time convention; exact for
the vacuum).
| key | shape | meaning |
|---|---|---|
| `q0` | (151,) float64 | -0.5 .. 5.5, step 0.04 |
| `q1` | (13,) float64 | 2 pi k / 25, k = 0..12 (vacuum W is q1-symmetric) |
| `W` | (151, 13) complex128 | windowed W^{00}; ridge along q0 = E(q1) gives the 101-qubit meson dispersion (~1% vs ED) |
| `spread` | (151, 13) float64 | window-scan systematic (ptp of Re W) |
| `sigma_t`, `sigma_x` | float64 | 2.0, 25/6 (central window) |

### `sigma_vac_ns50.npz` — optical conductivity (Ward-converted)
Produced by `scripts/kubo_conductivity.py`.
`Re sigma(w) = w W^{00}(w, q1)/q1^2` via the exact bond-current Ward identity;
excitonic-insulator gap at M; sum rule saturated to 0.07%.
| key | shape | meaning |
|---|---|---|
| `q0` | (276,) float64 | omega = 0.0 .. 5.5, step 0.02 |
| `q1` | (3,) float64 | 2 pi k/25, k = 1, 2, 3 (smallest momenta; q1 -> 0 limit) |
| `sigma` | (276, 3) float64 | `q0 * Re W^{00} / q1^2` |
| `spread` | (276, 3) float64 | window systematic on W^{00} (convert with the same factor `q0/q1^2` for a sigma band) |

### `w_meson_ns50_k{0.00,1.26}_v3.npz` — raw wavepacket hadronic-tensor runs (headline data)
Produced by `scripts/run_w_meson_ns50.py` (v3 = L2-regularized `wp10reg` block
parameters). `k1.26` = boosted packet, kbar = 2pi/5 = 1.2566 (measured group
velocity +0.093). Gaussian packet sigma_x = 0.75, built from L=3
gauge-invariant blocks on the volume-transferred vacuum, certified on-circuit
via `<H>` on both preparations.
| key | shape | meaning |
|---|---|---|
| `times` | (17,) float64 | t = 0.0, 0.5, ..., 8.0 |
| `corr_wp` | (17, 100) complex128 | packet-state correlator; **columns 0..49 = J^0 at staggered sites 0..49, columns 50..99 = J^1 at bonds 0..49** (both read from the same circuits; enables the lattice continuity/Ward check) |
| `one_pt_wp` | (17, 100) complex128 | packet one-point grid `<J^nu(v,t)>` |
| `insert_1pt_wp` | scalar complex128 | `<J^0(center)>` on the packet (0.5398 at k=0, 0.5228 at k=1.26) |
| `corr_vac`, `one_pt_vac`, `insert_1pt_vac` | same shapes | matched vacuum grids for the pointwise subtraction (`insert_1pt_vac` = +0.086734) |
| `k0` | float64 | packet momentum (0.0 or 1.2566370614359172) |
| `ns`, `center` | int64 | 50; insertion/packet center at staggered site 24 (spatial x0 = 12) |
| `m0`, `g2`, `eta`, `dt_target`, `trunc` | float64 | 0.7, 1.1, 1.3, 0.1, 1e-8 |
| `wp_fidelity_ns8` | float64 | packet-preparation fidelity certificate: F = 0.99463 (k=0), 0.99357 (k=2pi/5) |

`w_meson_ns50_k0.00.npz` is the earlier (un-regularized `wp_params` block,
F = 0.99882) run kept for cross-checks; the paper uses the `_v3` files.

### `w_meson_ns50_k{0.00,1.26}_v3_W.npz` — wavepacket W^{00}(q0, q1)
Produced by `scripts/analyze_w_meson.py` (**completed-time convention** —
suitable for the dispersion region; use one-sided transforms on the raw files
for the low-omega ridge, see conventions above).
| key | shape | meaning |
|---|---|---|
| `q0` | (176,) float64 | -1.0 .. 6.0, step 0.04 |
| `q1` | (25,) float64 | full BZ, 2 pi k / 25, k = -12..12 (signed: boosted packet is q1-asymmetric) |
| `W`, `spread` | (176, 25) complex128 / float64 | as above; windows sigma_t = 8/3, sigma_x = 25/6 |
| `k0` | float64 | packet momentum |

### `w_meson_ns50_k{0.00,1.26}_v3_axial.npz` — axial vs vector channel (one-sided)
Produced by `scripts/axial_from_existing.py` from the matching `_v3.npz` file
(no new circuits: the connected axial-axial correlator is the staggered sign
flip `G_ax(v,c;t) = (-1)^{v+c} G_vec(v,c;t)` of the measured vector
correlator). **One-sided transform** — these are the ridge-region grids: the
low-omega ridge is axial intra+interband transitions, matched by the
parameter-free two-band model at 0.7–2.2% RMS.
| key | shape | meaning |
|---|---|---|
| `q0` | (153,) float64 | -0.6 .. 5.48, step 0.04 |
| `q1` | (25,) float64 | full BZ, 2 pi k / 25, k = -12..12 |
| `W_ax` | (153, 25) float64 | one-sided windowed axial-channel W (sigma_t = 8/3, sigma_x = 25/6) |
| `W_vec` | (153, 25) float64 | matching one-sided vector-channel W |

### `phase_shifts_ed.npz` — meson-meson elastic phase shifts (finite-volume ED)
Produced by `scripts/phase_shifts.py`; **superseded interpretation** in
`scripts/phase_shifts_v2.py` (uses `deep_levels_ns{8,10}.npz`). 1+1d
quantization `p Nx + 2 delta(p) = 2 pi n` with the spline dispersion through
the measured band.
| key | shape | meaning |
|---|---|---|
| `results` | (6, 5) float64 | columns: `[ns, E2, p, n, delta_rad]`; all candidate n-assignments kept |
| `M` | float64 | 2.7451 (band minimum; threshold 2M = 5.490) |
| `B` | float64 | NaN (spline dispersion used; no sin^2 fit parameter) |

Physical single-channel branch (v2 analysis, restricted to
`E < M + M' = 5.900`): `delta = +0.567 (p = 1.03, ns = 10)` and
`+0.150 (p = 1.50, ns = 8)`. The point at E2 = 6.062 (delta = +0.672,
p = 2.81) lies above the MM' inelastic threshold and is quarantined as
multi-channel; ns = 6 rows sit below the asymptotic-validity range (L = 3).

### `deep_levels_ns{8,10}.npz` — deep 60-state ED spectra (headline couplings)
60 lowest levels at (m0, g2, eta) = (0.7, 1.1, 1.3); input to
`scripts/phase_shifts_v2.py`. Run log: `deep_levels.log`. (The producing
sweep script was an ad-hoc deep-ED run and is not retained in `scripts/`;
see Provenance caveats below.)
| key | shape | meaning |
|---|---|---|
| `gaps` | (60,) float64 | E - E_vac, ascending (gaps[0] = 0 is the vacuum) |
| `phases` | (60,) float64 | two-site translation (momentum) phase of each level, in {0, ±pi/2, pi} multiples of 2 pi / Nx; P = 0 states have phase 0 |

### `predict_{DHK,CGK_el,CGK_inA}_ns{8,10}.npz` — phase-shift prediction sweeps at literature couplings
Same schema as `deep_levels_*` plus the couplings; deep 60-state spectra at
other groups' parameter sets, for the predictions table
(log: `prediction_sweeps.log`).
| key | shape | meaning |
|---|---|---|
| `gaps`, `phases` | (60,) float64 | as in `deep_levels_*` |
| `m0`, `g2`, `eta` | float64 | DHK: (1.0, 0.6, 1.0); CGK_el: (0.1, 2.0, 1.0); CGK_inA: (0.1, 0.4, 1.0) |

### `ridge_model_ns10.npz` — parameter-free two-band ridge-model inputs
Produced by `scripts/ridge_model.py` (validated in `ridge_model_v2.py` /
`final_ridge_overlay.py`; reproduces the 101-qubit ridge at 0.7–2.2% RMS).
| key | shape | meaning |
|---|---|---|
| `A` | (5, 5) complex128 | intensive current matrix element `A(k',k)` in `<k'|J^0(v)|k> = (-1)^v e^{i(k-k')x_v} A(k',k)` (ns=10 gauge-fixed, real-symmetric to `scatter`); rescale by Nx = 5/25 for ns=50 |
| `ks` | (5,) float64 | ns=10 band momenta 2 pi n / 5, n = -2..2 |
| `Ek` | (5,) float64 | absolute band energies (subtract E_vac for gaps) |
| `f0` | (5,) float64 | rest-packet overlaps of the optimized interpolator with band states |
| `scatter` | float64 | 0.0140 = max site-scatter of A (factorization-quality certificate) |

### `synthesis_pilot_ns8.npz` — synthesis-error pilot (17 qubits, statevector)
Produced by `scripts/synthesis_pilot_ns8.py`. Full snapped pipeline
(prep + Trotter evolution rotations snapped to a grid of spacing delta,
post-transpile) vs exact, at ns = 8: on-circuit `<H>` gap of the snapped
packet and the one-sided W^{00}(q0) ridge at q1 = pi/2. Headline: physics
survives at ~20 T gates/rotation; stochastic rounding beats deterministic
by 2–6x.
| key | shape | meaning |
|---|---|---|
| `rows` | (21, 5) float64 | columns: `[delta, mode, seed, gap, ridge_rms_err]`; delta in {pi/128 .. pi/2}; mode 0 = deterministic round, 1 = stochastic (seeds 1, 2); `gap` = snapped `<H>` packet gap (exact `gap0`); `ridge_rms_err` = RMS(W - W0)/max|W0| over the q1 = pi/2 ridge |
| `gap0` | float64 | exact-pipeline gap = 2.7599 |
| `W0`, `q0` | (56,) float64 | exact one-sided ridge Re W^{00}(q0, pi/2), q0 = -0.6 .. 1.6, step 0.04 |
| `n_rot`, `n_nc` | int64 | 370 rotations (208 non-Clifford) in the transpiled prep circuit |

### `synthladder_exact.npz` — 101-qubit synthesis-ladder exact baseline
Produced by `scripts/synthesis_ladder_ns50.py` (exact rung; snapped rungs
`synthladder_<mode>_pi<n>.npz` share the schema). Reduced rest-packet run:
J^0 probes only, t = 0..6, `dt_target = 0.5`.
| key | shape | meaning |
|---|---|---|
| `times` | (13,) float64 | t = 0.0, 0.5, ..., 6.0 |
| `corr` | (13, 50) complex128 | raw packet-state `<J^0(v,t) J^0(v_c,0)>` |
| `one_pt` | (13, 50) complex128 | `<J^0(v,t)>` one-point grid |
| `insert_1pt` | scalar complex128 | 0.5398 |
| `H` | float64 | on-circuit `<H>` certificate = -49.4260 |
| `delta`, `mode`, `seed` | float64 / str / int64 | 0.0, "exact", 0 for this baseline |

### `wp10reg_params_k{0.00,1.26}_L3.npz` (and `wp10_params_*`, `wp_params_*`) — wavepacket block parameters
Produced by `scripts/regularize_wp_params.py` (`wp10reg`, used in production
v3 runs) and the earlier trainers. Variational parameters of the L=3
gauge-invariant wavepacket block.
| key | shape | meaning |
|---|---|---|
| `vec` | (108,) float64 | flattened block angles (`htensor.wavepacket.params_from_vector(vec, offsets, L)` reconstructs them); (96,) in the older `wp_params_k0.00_L3.npz` |
| `offsets` | (9,) int64 | block support, staggered-site offsets -4..+4 about the packet center; (8,) offsets -3..+4 in `wp_params_k0.00_L3.npz` |
| `L` | int64 | 3 layers |
| `F` | float64 | preparation fidelity vs the ED target packet: 0.99463 / 0.99357 (regularized k=0 / k=2pi/5); 0.99727 / 0.99573 (`wp10_params_*`); 0.99882 (`wp_params_*`) |
| `sigma`, `k0` | float64 | Gaussian width 0.75 (spatial sites); packet momentum (absent from `wp_params_k0.00_L3.npz`, which is k = 0) |

## CSV exports (`data/csv/`)

Generated by `scripts/export_tables.py`; each file carries `#`-prefixed
comment lines with couplings and source provenance:

- `w00_vac_ns50_W.csv`, `w_meson_ns50_k0.00_W.csv`, `w_meson_ns50_k1.26_W.csv`
  — long-format W^{00} grids (`q0, q1, ReW00, ImW00, window_spread`)
- `sigma_vac_ns50.csv` — `omega, q1, Re_sigma, window_band`
- `phase_shifts_ed.csv` — `ns, E2, p, n, delta_rad, delta_deg, channel_note`
- `meson_dispersion_ed.csv` — ED band `k, E`
- `synthesis_pilot_ns8.csv` — `delta, pi_over_delta, mode, seed, gap, ridge_rms_err`

## Provenance caveats

- `deep_levels_ns{8,10}.npz` and `predict_*.npz` were generated by an ad-hoc
  deep-ED sweep (60 eigenstates via `htensor.exact`); the driver script was
  not retained. The run logs are `deep_levels.log` and
  `prediction_sweeps.log`, and `scripts/phase_shifts_v2.py` documents the
  consuming analysis.
- The `*_v3_W.npz` wavepacket grids use the completed-time convention;
  ridge-region (low-omega) physics must use the one-sided convention (the
  `*_axial.npz` files, or a one-sided transform on the raw `_v3.npz` grids).

## License

Data: **CC-BY-4.0** (Creative Commons Attribution 4.0 International).
Code (`scripts/`, `htensor/`): see the repository license file.

## Citation

If you use these data, please cite:

> [AUTHORS], *The hadronic tensor from a quantum computer: meson structure in
> 1+1d Z2 lattice gauge theory*, [JOURNAL/arXiv:XXXX.XXXXX] (2026), and this
> dataset: doi:10.5281/zenodo.XXXXXXX.

FERMILAB-PUB-26-XXX-T.
