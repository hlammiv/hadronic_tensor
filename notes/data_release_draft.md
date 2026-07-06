===== KEY: v2:b25d072e60bc5707dfea51febe1a328403247c818fac7d23936a7c326da6173e =====
# Open-data release draft for github.com/hlammiv/hadronic_tensor

All array facts below were verified by loading every `.npz` in `/home/hlamm/Desktop/QC/hadronic_tensor/data/` and reading the producing scripts in `/home/hlamm/Desktop/QC/hadronic_tensor/scripts/` plus `/home/hlamm/Desktop/QC/hadronic_tensor/htensor/analysis.py` (read-only; nothing was written).

---

## Deliverable 1 — proposed content of `data/README.md`

```markdown
# Data release: Hadronic tensor of 1+1d Z2 lattice gauge theory on 101 qubits

Companion data for *"The hadronic tensor from a quantum computer: meson
structure in 1+1d Z2 gauge theory"* (Fermilab, 2026). Every file was produced
by a script in `scripts/` of this repository (provenance noted per file);
figures (`*.pdf`) alongside the `.npz` files are the paper figures rendered
from exactly these arrays.

## Physics conventions (apply to all files)

- **Model.** Z2 gauge theory with staggered fermions on a periodic ring of
  `ns = 50` staggered sites (`Nx = ns/2 = 25` spatial sites), explicit link
  qubits. Couplings throughout: `m0 = 0.7`, `g2 = 1.1`, `eta = 1.3`
  (stored redundantly in each production file). All quantities are in
  lattice units (staggered-site spacing = 1/2 spatial-site spacing; positions
  `x` are quoted in *spatial-site* units, `x = (v - v_c)/2`, minimal-image
  folded on the ring).
- **Circuits.** 101 qubits = 50 site qubits + 50 link qubits + 1 Hadamard-test
  ancilla, simulated with the Qiskit Aer matrix-product-state backend
  (truncation threshold `trunc = 1e-8`, Trotter step `dt_target = 0.1`).
  Correlators are ancilla-assisted (Hadamard-test) measurements
  `C^{0 nu}(v, t) = <psi| J^nu(v, t) J^0(v_c, 0) |psi>`.
- **Subtractions.** Connected correlators are formed as
  `C - <J^nu(v,t)><J^0(v_c)>` (the interacting vacuum carries a staggered
  one-point profile, so this matters); wavepacket runs additionally subtract
  the matched vacuum grid pointwise ("connected-minus-connected").
- **One-sided vs completed Fourier transform (important).** Two time-domain
  conventions appear:
  1. *Completed (hermiticity) transform*: data on `t in [0, T]` are extended
     to `t < 0` via `G(x, -t) = [G(-x, t)]*` (`htensor.analysis.complete_time`)
     before the Gaussian-windowed double FT. This is exact for the **vacuum**
     tensor and is used in all `*_W.npz` files here.
  2. *One-sided transform*: FT over `t >= 0` only (t = 0 weighted 1/2). For
     **wavepacket** states the hermiticity completion is *not* valid at our
     precision — it scatters higher-sector content into the low-omega window —
     so all ridge-region (low q0) physics in the paper uses the one-sided
     convention (`scripts/final_ridge_overlay.py`, `scripts/ridge_model_v2.py`).
     The wavepacket `*_W.npz` grids (completed convention) are provided for the
     dispersion-region overview figures; reproduce ridge-region numbers from
     the raw time-domain files with a one-sided transform.
- **Windowed FT.** `W(q0, q1) = dt dx sum_{t,x} e^{i q0 t - i q1 x}
  e^{-t^2/2 sigma_t^2} e^{-x^2/2 sigma_x^2} G(x, t)`. Central window and
  `spread` = peak-to-peak of Re W over window rescalings {0.75, 1.0, 1.5}
  (the windowing systematic). Sign convention: `+i q0 t - i q1 x`.
- **Key spectral inputs** (small-volume ED, volume-converged): meson band
  `E(k) = 2.7451, 2.8188, 2.8560, 2.9275, 2.9886, 3.0778` at
  `k = 0, 2pi/5, pi/2, 2pi/3, 4pi/5, pi`; second band gaps 3.155–3.285;
  two-meson threshold `2M = 5.490`.

## Files

### `w00_vac_ns50.npz` — raw vacuum-polarization run (101 qubits)
Produced by `scripts/run_w00_vacuum_ns50.py`.
| key | shape / dtype | meaning |
|---|---|---|
| `times` | (13,) float64 | t = 0.0, 0.5, ..., 6.0 |
| `corr` | (13, 50) complex128 | raw `<J^0(v,t) J^0(v_c,0)>` on the ns=6-optimized variational vacuum transferred to ns=50; columns = staggered site v = 0..49 |
| `probe_1pt` | (13, 50) complex128 | `<J^0(v,t)>` one-point grid (for the disconnected subtraction) |
| `insert_1pt` | scalar complex | `<J^0(v_c)>` = -0.086734 |
| `ns`, `vc` | int | 50 staggered sites; insertion at v_c = 25 |
| `m0`, `g2`, `eta`, `dt_target`, `trunc` | scalars | 0.7, 1.1, 1.3, 0.1, 1e-8 |
| `thetas` | (8,) float64 | 2-layer variational vacuum angles (optimized at ns=6, volume-transferred) |

### `w00_vac_ns50_W.npz` — vacuum W^{00}(q0, q1)
Produced by `scripts/plot_w00_vacuum.py` (completed-time convention; exact for the vacuum).
| key | shape | meaning |
|---|---|---|
| `q0` | (151,) | -0.5 .. 5.5, step 0.04 |
| `q1` | (13,) | 2 pi k / 25, k = 0..12 (vacuum W is q1-symmetric) |
| `W` | (151, 13) complex128 | windowed W^{00}; ridge along q0 = E(q1) gives the 101-qubit meson dispersion (~1% vs ED) |
| `spread` | (151, 13) float64 | window-scan systematic (ptp of Re W) |
| `sigma_t`, `sigma_x` | scalars | 2.0, 25/6 (central window) |

### `sigma_vac_ns50.npz` — optical conductivity (Ward-converted)
Produced by `scripts/kubo_conductivity.py`. `Re sigma(w) = w W^{00}(w, q1)/(q1)^2`
via the exact bond-current Ward identity; excitonic-insulator gap at M.
| key | shape | meaning |
|---|---|---|
| `q0` | (276,) | omega = 0.0 .. 5.5, step 0.02 |
| `q1` | (3,) | 2 pi k/25, k = 1, 2, 3 (smallest momenta; q1 -> 0 limit) |
| `sigma` | (276, 3) | `q0 * Re W^{00} / q1^2` |
| `spread` | (276, 3) | window systematic on W^{00} (convert with the same factor `q0/q1^2` for a sigma band) |

### `w_meson_ns50_k{0.00,1.26}_v3.npz` — raw wavepacket hadronic-tensor runs (headline data)
Produced by `scripts/run_w_meson_ns50.py` (v3 = L2-regularized `wp10reg` block
parameters). `k1.26` = boosted packet, kbar = 2pi/5 = 1.2566. Gaussian packet
sigma_x = 0.75, built from L=3 gauge-invariant blocks on the volume-transferred
vacuum, certified on-circuit via `<H>` on both preparations.
| key | shape | meaning |
|---|---|---|
| `times` | (17,) | t = 0.0, 0.5, ..., 8.0 |
| `corr_wp` | (17, 100) complex128 | packet-state correlator; **columns 0..49 = J^0 at staggered sites 0..49, columns 50..99 = J^1 at bonds 0..49** (both read from the same circuits; enables the lattice continuity/Ward check) |
| `one_pt_wp` | (17, 100) complex128 | packet one-point grid `<J^nu(v,t)>` |
| `insert_1pt_wp` | scalar | `<J^0(center)>` on the packet (0.5398 at k=0, 0.5228 at k=1.26) |
| `corr_vac`, `one_pt_vac`, `insert_1pt_vac` | same shapes | matched vacuum grids for the pointwise subtraction (`insert_1pt_vac` = +0.086734) |
| `k0` | scalar | packet momentum (0.0 or 1.2566370614359172) |
| `ns`, `center` | int | 50; insertion/packet center at staggered site 24 (spatial x0 = 12) |
| `m0`, `g2`, `eta`, `dt_target`, `trunc` | | 0.7, 1.1, 1.3, 0.1, 1e-8 |
| `wp_fidelity_ns8` | scalar | packet-preparation fidelity certificate: F = 0.99463 (k=0), 0.99357 (k=2pi/5) |

`w_meson_ns50_k0.00.npz` is the earlier (un-regularized, F = 0.99882) run kept
for cross-checks; the paper uses the `_v3` files.

### `w_meson_ns50_k{0.00,1.26}_v3_W.npz` — wavepacket W^{00}(q0, q1)
Produced by `scripts/analyze_w_meson.py` (**completed-time convention** —
suitable for the dispersion region; use one-sided transforms on the raw files
for the low-omega ridge, see conventions above).
| key | shape | meaning |
|---|---|---|
| `q0` | (176,) | -1.0 .. 6.0, step 0.04 |
| `q1` | (25,) | full BZ, 2 pi k / 25, k = -12..12 (signed: boosted packet is q1-asymmetric) |
| `W`, `spread` | (176, 25) | as above; windows sigma_t = 8/3, sigma_x = 25/6 |
| `k0` | scalar | packet momentum |

### `phase_shifts_ed.npz` — meson-meson elastic phase shifts (finite-volume ED)
Produced by `scripts/phase_shifts.py`. 1+1d quantization
`p Nx + 2 delta(p) = 2 pi n` with the spline dispersion through the measured
band; cross-volume collapse at ns = 6, 8, 10 is the certificate.
| key | shape | meaning |
|---|---|---|
| `results` | (6, 5) float64 | columns: `[ns, E2, p, n, delta_rad]`; all candidate n-assignments kept — the physical branch is `delta = +0.56 (p=1.03, ns=10), +0.15 (p=1.50, ns=8), +0.67 (p=2.81, ns=8)` |
| `M` | scalar | 2.7451 (band minimum used as threshold 2M) |
| `B` | scalar | NaN (spline dispersion used; no sin^2 fit parameter) |

### `ridge_model_ns10.npz` — parameter-free two-band ridge-model inputs
Produced by `scripts/ridge_model.py` (validated in `ridge_model_v2.py` /
`final_ridge_overlay.py`; reproduces the 101-qubit ridge at 0.7–2.2% RMS).
| key | shape | meaning |
|---|---|---|
| `A` | (5, 5) complex128 | intensive current matrix element `A(k',k)` in `<k'|J^0(v)|k> = (-1)^v e^{i(k-k')x_v} A(k',k)` (ns=10 gauge-fixed, real-symmetric to `scatter`); rescale by Nx=5/25 for ns=50 |
| `ks` | (5,) | ns=10 band momenta 2 pi n / 5, n = -2..2 |
| `Ek` | (5,) | absolute band energies (subtract E_vac for gaps) |
| `f0` | (5,) | rest-packet overlaps of the optimized interpolator with band states |
| `scatter` | scalar | 0.0140 = max site-scatter of A (factorization-quality certificate) |

### `wp10reg_params_k{0.00,1.26}_L3.npz` (and `wp10_params_*`, `wp_params_*`)
Produced by `scripts/regularize_wp_params.py` (`wp10reg`, used in production v3
runs) and the earlier trainers (`train_and_certify_v2.py`). Variational
parameters of the L=3 gauge-invariant wavepacket block.
| key | shape | meaning |
|---|---|---|
| `vec` | (108,) float64 | flattened block angles (`htensor.wavepacket.params_from_vector(vec, offsets, L)` reconstructs them) |
| `offsets` | (9,) int | block support, staggered-site offsets -4..+4 about the packet center |
| `L` | int | 3 layers |
| `F` | scalar | preparation fidelity vs the ED target packet (0.9946 / 0.9936 for the regularized k=0 / k=2pi/5 sets) |
| `sigma`, `k0` | scalars | Gaussian width 0.75 (spatial sites); packet momentum |

## License

Data: **CC-BY-4.0** (Creative Commons Attribution 4.0 International).
Code (`scripts/`, `htensor/`): see the repository license file (MIT suggested).

## Citation

If you use these data, please cite:

> [AUTHORS], *The hadronic tensor from a quantum computer: meson structure in
> 1+1d Z2 lattice gauge theory*, [JOURNAL/arXiv:XXXX.XXXXX] (2026), and this
> dataset: doi:10.5281/zenodo.XXXXXXX.

FERMILAB-PUB-26-XXX-T.
```

---

## Deliverable 2 — draft of `scripts/export_tables.py`

```python
"""Export the key .npz release files to plain-text CSV tables.

Outputs (written to data/csv/):
  w00_vac_ns50_W.csv           long-format vacuum W^00: q0, q1, ReW, ImW, spread
  w_meson_ns50_k0.00_W.csv     long-format wavepacket W^00 (rest packet)
  w_meson_ns50_k1.26_W.csv     long-format wavepacket W^00 (boosted, kbar=2pi/5)
  sigma_vac_ns50.csv           Re sigma(omega) per q1 with window band
  phase_shifts_ed.csv          finite-volume delta(E) table
  meson_dispersion_ed.csv      ED band E(k) used throughout

Conventions: all quantities in lattice units; W grids are the Gaussian-
windowed transforms stored in the *_W.npz files (completed-time convention --
exact for the vacuum tensor; for low-omega ridge physics use one-sided
transforms on the raw time-domain files, see data/README.md).

  PYTHONPATH=. .venv/bin/python scripts/export_tables.py
"""

import csv
import os

import numpy as np

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
OUT = os.path.join(DATA, "csv")
os.makedirs(OUT, exist_ok=True)


def write_csv(name, header, rows, comment_lines=()):
    path = os.path.join(OUT, name)
    with open(path, "w", newline="") as fh:
        for line in comment_lines:
            fh.write(f"# {line}\n")
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    print(f"wrote {path}  ({len(rows)} rows)")


COUPLINGS = "m0=0.7, g2=1.1, eta=1.3; ns=50 staggered sites (101 qubits), PBC"


# ---------------- W(q0, q1) grids -> long format
def export_w(npz_name, csv_name, extra_comments=()):
    d = np.load(os.path.join(DATA, npz_name))
    q0, q1, W, spread = d["q0"], d["q1"], d["W"], d["spread"]
    rows = [(f"{q0[i]:.4f}", f"{q1[j]:.6f}",
             f"{W[i, j].real:.6e}", f"{W[i, j].imag:.6e}",
             f"{spread[i, j]:.6e}")
            for i in range(len(q0)) for j in range(len(q1))]
    write_csv(csv_name,
              ["q0", "q1", "ReW00", "ImW00", "window_spread"],
              rows,
              comment_lines=(COUPLINGS,
                             f"source: data/{npz_name}",
                             "Gaussian-windowed FT, completed-time convention;"
                             " spread = ptp over window rescalings"
                             " {0.75, 1.0, 1.5} (systematic)")
              + tuple(extra_comments))


export_w("w00_vac_ns50_W.csv".replace(".csv", ".npz"), "w00_vac_ns50_W.csv",
         ("vacuum polarization: ridge along q0 = E(q1) is the meson"
          " dispersion",))
export_w("w_meson_ns50_k0.00_v3_W.npz", "w_meson_ns50_k0.00_W.csv",
         ("meson wavepacket at rest (k0=0); sigma_x=0.75; prep fidelity"
          " F=0.9946 (ns=8 certificate)",))
export_w("w_meson_ns50_k1.26_v3_W.npz", "w_meson_ns50_k1.26_W.csv",
         ("boosted meson wavepacket, k0=2pi/5=1.2566; prep fidelity"
          " F=0.9936; signed q1 grid (boost asymmetry)",))


# ---------------- optical conductivity
d = np.load(os.path.join(DATA, "sigma_vac_ns50.npz"))
q0, q1, sig, spr = d["q0"], d["q1"], d["sigma"], d["spread"]
rows = [(f"{q0[i]:.4f}", f"{q1[j]:.6f}", f"{sig[i, j]:.6e}",
         f"{q0[i] * spr[i, j] / q1[j] ** 2:.6e}")
        for i in range(len(q0)) for j in range(len(q1))]
write_csv("sigma_vac_ns50.csv",
          ["omega", "q1", "Re_sigma", "window_band"],
          rows,
          comment_lines=(COUPLINGS,
                         "source: data/sigma_vac_ns50.npz",
                         "Re sigma(omega) = omega * ReW00 / q1^2 (Ward"
                         " identity); band = omega*spread/q1^2",
                         "gap at meson mass M = 2.7451 (excitonic"
                         " insulator)"))


# ---------------- phase shifts
d = np.load(os.path.join(DATA, "phase_shifts_ed.npz"))
rows = [(int(ns), f"{E2:.4f}", f"{p:.4f}", int(n),
         f"{delta:+.4f}", f"{np.degrees(delta):+.1f}")
        for ns, E2, p, n, delta in d["results"]]
write_csv("phase_shifts_ed.csv",
          ["ns", "E2", "p", "n", "delta_rad", "delta_deg"],
          rows,
          comment_lines=(COUPLINGS.replace("ns=50", "ns=6,8,10 (ED)"),
                         "source: data/phase_shifts_ed.npz"
                         " (scripts/phase_shifts.py)",
                         "1+1d quantization p*Nx + 2*delta(p) = 2*pi*n;"
                         " all n-assignments kept, physical branch:"
                         " delta=+0.56 (p=1.03), +0.15 (p=1.50),"
                         " +0.67 (p=2.81)",
                         f"threshold 2M = {2 * float(d['M']):.4f}"))


# ---------------- ED meson dispersion (paper inputs)
k_ed = [0.0, 1.2566, 1.5708, 2.0944, 2.5133, np.pi]
e_ed = [2.7451, 2.8188, 2.8560, 2.9275, 2.9886, 3.0778]
write_csv("meson_dispersion_ed.csv", ["k", "E"],
          [(f"{k:.4f}", f"{e:.4f}") for k, e in zip(k_ed, e_ed)],
          comment_lines=(COUPLINGS.replace("ns=50", "volume-converged ED"),
                         "single-meson band E(k); k in [0, pi],"
                         " E(-k) = E(k)"))

print("done")
```

(Note the small idiom `export_w("w00_vac_ns50_W.csv".replace(...))` should simply be `export_w("w00_vac_ns50_W.npz", ...)` — cleaner form:
`export_w("w00_vac_ns50_W.npz", "w00_vac_ns50_W.csv", (...))`. Use that line when committing.)

---

## Deliverable 3 — data-availability paragraph for the paper

> **Data availability.** All data underlying the figures and quoted numbers are openly available under a CC-BY-4.0 license in the repository `github.com/hlammiv/hadronic_tensor` (archived at doi:10.5281/zenodo.XXXXXXX). The release contains the raw 101-qubit time-domain correlator grids for the vacuum-polarization and meson-wavepacket runs at (m0, g2, eta) = (0.7, 1.1, 1.3) — including the matched vacuum grids, one-point functions, on-circuit `<H>` certificates, and wavepacket-preparation fidelities (F >= 0.9936) — together with the Gaussian-windowed W^{00}(q0, q1) grids with window-scan systematics, the Ward-converted optical conductivity, the finite-volume meson–meson phase-shift table, the parameter-free two-band ridge-model inputs, and the variational wavepacket-block parameters. Machine-readable CSV exports of the key tables are provided alongside the binary (NumPy `.npz`) files, and each file's producing script is included in `scripts/`, so the full analysis chain — connected subtraction, time completion (vacuum) or one-sided transform (wavepacket ridge region), and windowed Fourier transform — is reproducible from the raw grids.

### Points the parent agent may want to double-check before release
- The `*_W.npz` wavepacket grids were made with the completed-time convention (`analyze_w_meson.py`), while `ridge_model_v2.py`/`final_ridge_overlay.py` explicitly state the one-sided convention is mandatory for ridge physics; the README flags this, but you may want to add one-sided ridge-region CSVs to the release.
- `w_meson_ns50_k0.00.npz` (non-v3, F=0.99882 ns=8 certificate from the older `wp_params` block) is retained; decide whether to ship or drop it.
- `phase_shifts_ed.npz` contains ambiguous n-assignment rows (ns=6 duplicates at E2=5.985 with n=1 and n=2); README/CSV note the physical branch (+0.56/+0.15/+0.67 at p=1.03/1.50/2.81).
- Zenodo DOI, arXiv number, FERMILAB-PUB report number, and author list are placeholders.
