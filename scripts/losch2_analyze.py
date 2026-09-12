"""Analyze stage of the bit-level damping-model Loschmidt gate.

Applies the two-sector readout split (sites nominal, links 3.7x) to the
sampled bits, builds the Hadamard-test estimators, and reports:
  (a) model validation: does theta->0-mirror calibration over-estimate
      the wings (the hardware factor-2 signature)?
  (b) theta->0 vs Loschmidt wing recovery under this model,
  (c) real per-shot syndrome post-selection (k = 1, 3) bias/keep table.

  PYTHONPATH=. .venv/bin/python scripts/losch2_analyze.py
"""
import glob
import numpy as np

RO01, RO10, LINK_SCALE = 0.012, 0.028, 3.7
NS, CENTER, NSYS = 50, 24, 100
PROBES = [CENTER + o for o in (-16, -12, -8, -4, 0, 4, 8, 12, 16)]
TS = ["0.5", "1.0"]
rng = np.random.default_rng(7)

ideal = np.load("data/losch_ideal.npz")
files = sorted(glob.glob("data/work/dryrun/tmp_losch2_*.npz"))
print(f"{len(files)} seed files")

# load + logical reorder + readout split, concatenated over seeds
def load_all(key):
    outs = []
    for f in files:
        d = np.load(f)
        bits = d[key]
        pl = d["perm_logical"]
        blog = bits[:, pl]                       # column q = logical q
        ro01 = np.full(NSYS + 1, RO01)
        ro10 = np.full(NSYS + 1, RO10)
        for b in range(NS):                       # link qubits = odd logical
            ro01[2 * b + 1] = min(RO01 * LINK_SCALE, 0.5)
            ro10[2 * b + 1] = min(RO10 * LINK_SCALE, 0.5)
        flip = (((blog == 0) & (rng.random(blog.shape) < ro01)) |
                ((blog == 1) & (rng.random(blog.shape) < ro10)))
        outs.append(blog ^ flip.astype(np.uint8))
    return np.concatenate(outs)                   # (shots, NSYS+1)

def estimators(blog):
    """Per-shot X_anc x J0(v) values and syndrome pass masks."""
    x_anc = 1.0 - 2.0 * blog[:, NSYS]
    z = 1.0 - 2.0 * blog[:, 0:NSYS:2]             # site qubits, index n
    xl = 1.0 - 2.0 * blog[:, 1:NSYS:2]            # link qubits, bond b
    # G_n = (-1)^n z_n x_{n-1} x_n (PBC on links)
    G = np.empty((blog.shape[0], NS))
    for n in range(NS):
        G[:, n] = (-1) ** n * z[:, n] * xl[:, (n - 1) % NS] * xl[:, n]
    xb = {v: x_anc * (((-1) ** v) / 2 - z[:, v] / 2) for v in PROBES}
    return xb, G

res = {}
for key in [f"{t}_{k}" for t in TS for k in ("phys", "mir0", "losch")] \
        + ["0.0_phys"]:
    res[key] = estimators(load_all(key))

wings = [v for v in PROBES if abs(v - CENTER) >= 8]
for t in TS:
    i0 = {v: float(ideal[f"0.0_phys_XB_{v}"]) for v in PROBES}
    ci = {v: float(ideal[f"{t}_phys_XB_{v}"]) for v in PROBES}
    phys, _ = res[f"{t}_phys"]
    for kind in ("mir0", "losch"):
        mir, _ = res[f"{t}_{kind}"]
        kap = {v: mir[v].mean() / i0[v] for v in PROBES}
        cal = {v: phys[v].mean() / kap[v] if abs(kap[v]) > 0.05 else np.nan
               for v in PROBES}
        wr = np.nanmean([abs(cal[v] / ci[v]) for v in wings
                         if abs(ci[v]) > 5e-4])
        ninv = sum(1 for v in PROBES if abs(kap[v]) <= 0.05)
        print(f"t={t} {kind:5s}: wing |cal/ideal| = {wr:.2f}  "
              f"uninvertible {ninv}/{len(PROBES)}  "
              f"kappa(c) = {kap[CENTER]:.3f}")
    # per-shot syndrome post-selection on the physics run
    _, G = res[f"{t}_phys"]
    for k in (1, 3):
        kept, biases = [], []
        for v in PROBES:
            wit = [(v // 1 + d) % NS for d in range(-k, k + 1)]
            m = np.all(G[:, wit] > 0, axis=1)
            kept.append(m.mean())
            if m.sum() > 50 and abs(ci[v]) > 5e-4:
                biases.append(abs(phys[v][m].mean() / ci[v]))
        raw = np.nanmean([abs(phys[v].mean() / ci[v]) for v in PROBES
                          if abs(ci[v]) > 5e-4])
        print(f"t={t} postsel k={k}: keep = {np.mean(kept):.2f}  "
              f"|phys/ideal| = {np.nanmean(biases):.2f} (raw {raw:.2f})")
