"""Export the key .npz release files to plain-text CSV tables.

Outputs (written to data/csv/):
  w00_vac_ns50_W.csv           long-format vacuum W^00: q0, q1, ReW, ImW, spread
  w_meson_ns50_k0.00_W.csv     long-format wavepacket W^00 (rest packet)
  w_meson_ns50_k1.26_W.csv     long-format wavepacket W^00 (boosted, kbar=2pi/5)
  sigma_vac_ns50.csv           Re sigma(omega) per q1 with window band
  phase_shifts_ed.csv          finite-volume delta(E) table (with channel notes)
  meson_dispersion_ed.csv      ED band E(k) used throughout
  synthesis_pilot_ns8.csv      synthesis-error pilot: gap + ridge error vs delta

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


export_w("w00_vac_ns50_W.npz", "w00_vac_ns50_W.csv",
         ("vacuum polarization: ridge along q0 = E(q1) is the meson"
          " dispersion",))
export_w("w_meson_ns50_k0.00_v3_W.npz", "w_meson_ns50_k0.00_W.csv",
         ("meson wavepacket at rest (k0=0); sigma_x=0.75; prep fidelity"
          " F=0.9946 (ns=8 certificate)",
          "completed-time grid: dispersion region only -- reproduce the"
          " low-omega ridge with a one-sided FT on"
          " data/w_meson_ns50_k0.00_v3.npz (see data/README.md)"))
export_w("w_meson_ns50_k1.26_v3_W.npz", "w_meson_ns50_k1.26_W.csv",
         ("boosted meson wavepacket, k0=2pi/5=1.2566; prep fidelity"
          " F=0.9936; signed q1 grid (boost asymmetry)",
          "completed-time grid: dispersion region only -- reproduce the"
          " low-omega ridge with a one-sided FT on"
          " data/w_meson_ns50_k1.26_v3.npz (see data/README.md)"))


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
                         "gap at meson mass M = 2.7451 (excitonic insulator);"
                         " sum rule saturated to 0.07%"))


# ---------------- phase shifts
d = np.load(os.path.join(DATA, "phase_shifts_ed.npz"))
M = float(d["M"])
E_INEL = M + 3.155  # MM' inelastic threshold (M' = band-2 minimum)


def channel_note(ns, E2):
    if int(ns) == 6:
        return "ns=6: below asymptotic-validity range (L=3); excluded"
    if E2 >= E_INEL:
        return "above MM' threshold 5.900: multi-channel; quarantined"
    return "single-channel (physical branch)"


rows = [(int(ns), f"{E2:.4f}", f"{p:.4f}", int(n),
         f"{delta:+.4f}", f"{np.degrees(delta):+.1f}",
         channel_note(ns, E2))
        for ns, E2, p, n, delta in d["results"]]
write_csv("phase_shifts_ed.csv",
          ["ns", "E2", "p", "n", "delta_rad", "delta_deg", "channel_note"],
          rows,
          comment_lines=(COUPLINGS.replace(
                             "ns=50 staggered sites (101 qubits), PBC",
                             "ns=6,8,10 (finite-volume ED), PBC"),
                         "source: data/phase_shifts_ed.npz"
                         " (scripts/phase_shifts.py; interpretation per"
                         " scripts/phase_shifts_v2.py)",
                         "1+1d quantization p*Nx + 2*delta(p) = 2*pi*n;"
                         " all n-assignments kept",
                         "physical single-channel branch below M+M'=5.900:"
                         " delta=+0.567 (p=1.03, ns=10),"
                         " +0.150 (p=1.50, ns=8)",
                         f"threshold 2M = {2 * M:.4f};"
                         f" MM' threshold M+M' = {E_INEL:.4f}"))


# ---------------- ED meson dispersion (paper inputs)
k_ed = [0.0, 1.2566, 1.5708, 2.0944, 2.5133, np.pi]
e_ed = [2.7451, 2.8188, 2.8560, 2.9275, 2.9886, 3.0778]
write_csv("meson_dispersion_ed.csv", ["k", "E"],
          [(f"{k:.4f}", f"{e:.4f}") for k, e in zip(k_ed, e_ed)],
          comment_lines=(COUPLINGS.replace(
                             "ns=50 staggered sites (101 qubits), PBC",
                             "volume-converged ED"),
                         "single-meson band E(k); k in [0, pi],"
                         " E(-k) = E(k)"))


# ---------------- synthesis-error pilot
d = np.load(os.path.join(DATA, "synthesis_pilot_ns8.npz"))
MODES = {0: "round", 1: "stochastic"}
rows = [(f"{delta:.6f}", f"{np.pi / delta:.0f}", MODES[int(mode)], int(seed),
         f"{gap:.4f}", f"{err:.4f}")
        for delta, mode, seed, gap, err in d["rows"]]
write_csv("synthesis_pilot_ns8.csv",
          ["delta", "pi_over_delta", "mode", "seed", "gap", "ridge_rms_err"],
          rows,
          comment_lines=("m0=0.7, g2=1.1, eta=1.3; ns=8 (17 qubits,"
                         " statevector pilot)",
                         "source: data/synthesis_pilot_ns8.npz"
                         " (scripts/synthesis_pilot_ns8.py)",
                         "rotation angles snapped to grid spacing delta"
                         " (round = deterministic, stochastic = unbiased"
                         " random rounding) in prep + Trotter circuits",
                         f"exact-pipeline gap0 = {float(d['gap0']):.4f};"
                         " ridge_rms_err = RMS(W - W0)/max|W0| on the"
                         " one-sided W^00(q0, q1=pi/2) ridge",
                         f"prep circuit: {int(d['n_rot'])} rotations"
                         f" ({int(d['n_nc'])} non-Clifford)"))

print("done")
