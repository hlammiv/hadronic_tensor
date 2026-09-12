"""CGK (arXiv:2505.21240) spectrum postdiction test: extract their L=30
QSE/MPS meson dispersions (Fig. 2, vector paths in the arXiv-source figure
PDF) and compare against OUR small-volume (ns<=20) levels at the translated
couplings.

Their momentum label: k = 2*pi*j/30 (staggered, full BZ, C-eigenvalue
labeling).  Physical-ring momentum K = 2k mod 2pi.  Hypotheses tested:
  H-glue : |k| <= pi/2 -> our band-1 at |K|=2|k|;
           |k| >  pi/2 -> our band-2 at |K|=2pi-2|k|   (staggered partner)
  H-nodouble : band-1 evaluated at |k| directly.
Scalar (red) points -> our band-2 at |K|=2|k|.

  PYTHONPATH=. .venv/bin/python scripts/cgk_spectrum_compare.py
"""
import numpy as np
from collections import defaultdict
import fitz

SRC = ("/tmp/claude-1000/-home-hlamm-Desktop-QC-hadronic-tensor/"
       "fb5de832-5f2a-4241-8c29-2ea520bb7131/scratchpad/cgk_src/figure/"
       "spectrum_N30_m0.1.pdf")

# ---- panel calibrations from the embedded text ----
# panel (a) eps=1.0: x: j=-10 @ 97.1, j=10 @ 277.2 ; y: 2.4 @ 203.7, 2.6 @ 118.0
# panel (c) eps=0.2: x: j=-5 @ 124.1, j=5 @ 259.2  ; y: 1.0 @ 455.1, 1.4 @ 353.5
CAL = {
    "a": dict(x0=187.15, dx=(277.2 - 97.1) / 20, y0=203.7, yv=2.4,
              dy=(118.0 - 203.7) / 0.2, ymin=25, ymax=235),
    "c": dict(x0=191.65, dx=(259.2 - 124.1) / 10, y0=455.1, yv=1.0,
              dy=(353.5 - 455.1) / 0.4, ymin=305, ymax=505),
}


def extract(color):
    """centers of filled circles of a given fill color, per panel."""
    doc = fitz.open(SRC)
    out = {"a": [], "c": []}
    for g in doc[0].get_drawings():
        if g["type"] != "fs" or g.get("fill") is None:
            continue
        if tuple(round(v, 1) for v in g["fill"]) != color:
            continue
        pts = [q for it in g["items"] for q in it[1:] if hasattr(q, "x")]
        cx = float(np.mean([q.x for q in pts]))
        cy = float(np.mean([q.y for q in pts]))
        for pn, c in CAL.items():
            if c["ymin"] < cy < c["ymax"] and cx < 340:      # left column
                j = (cx - c["x0"]) / c["dx"]
                E = c["yv"] + (cy - c["y0"]) / c["dy"]
                out[pn].append((round(j), E, j))
    return out


blue = extract((0.0, 0.0, 1.0))     # QSE vector
red = extract((1.0, 0.0, 0.0))      # QSE scalar
for pn in "ac":
    js = [b[0] for b in blue[pn]]
    print(f"panel {pn}: {len(blue[pn])} vector pts, j in [{min(js)},{max(js)}],"
          f" rounding err max {max(abs(b[2]-b[0]) for b in blue[pn]):.3f}")
if red["c"]:
    js = [r[0] for r in red["c"]]
    print(f"panel c scalar: {len(red['c'])} pts, j in [{min(js)},{max(js)}]")

# ---- our small-volume bands ----
def bands(tag):
    """band-1/band-2 per |K|, multiplicity-aware: interior |K| clusters hold
    +-K pairs (2 states per band), the K=0 and K=pi sectors one each -- at
    K=pi the two bands touch, so naive dedup would replace band-2 with a
    two-meson state."""
    b1, b2 = {}, {}
    for ns in (8, 10, 12, 14, 16, 18, 20):
        d = np.load(f"data/deep_levels_cgk{tag}_ns{ns}.npz")
        cl = defaultdict(list)
        for g, p in zip(d["gaps"], d["phases"]):
            if g > 0.05:
                cl[abs(round(float(p), 4))].append(float(g))
        for k, gs in cl.items():
            gs = sorted(gs)
            mult = 1 if (k < 1e-3 or abs(k - np.pi) < 1e-3) else 2
            b1[k] = (ns, gs[0])          # largest volume wins (converged)
            if len(gs) > mult:
                b2[k] = (ns, gs[mult])
    k1 = np.array(sorted(b1)); e1 = np.array([b1[k][1] for k in k1])
    k2 = np.array(sorted(b2)); e2 = np.array([b2[k][1] for k in k2])
    return (k1, e1), (k2, e2)


def compare(tag, pts, label):
    (k1, e1), (k2, e2) = bands(tag)
    print(f"\n=== {label}: band-1 [{e1.min():.3f},{e1.max():.3f}], "
          f"band-2 [{e2.min():.3f},{e2.max():.3f}]")
    rows = []
    for j, E, _ in sorted(pts):
        k = 2 * np.pi * abs(j) / 30
        if k <= np.pi / 2 + 1e-9:
            Kp, band, which = 2 * k, (k1, e1), "b1"
        else:
            Kp, band, which = 2 * np.pi - 2 * k, (k2, e2), "b2"
        ours = np.interp(Kp, band[0], band[1])
        nod = np.interp(min(k, np.pi), k1, e1)          # no-doubling hypothesis
        rows.append((j, E, ours, which, nod))
    err_g = np.array([r[1] - r[2] for r in rows])
    err_n = np.array([r[1] - r[4] for r in rows])
    print(f"{'j':>4} {'CGK':>7} {'ours(glue)':>10} {'band':>5} {'diff':>8}")
    for j, E, ours, w, _ in rows:
        print(f"{j:4d} {E:7.4f} {ours:10.4f} {w:>5} {E-ours:+8.4f}")
    print(f"H-glue    : rms {np.sqrt(np.mean(err_g**2)):.4f}, "
          f"max |diff| {np.abs(err_g).max():.4f}")
    print(f"H-nodouble: rms {np.sqrt(np.mean(err_n**2)):.4f}")
    return rows


rows_a = compare("el", blue["a"], "eps=1.0 (cgkel) vector")
rows_c = compare("inA", blue["c"], "eps=0.2 (cgkinA) vector")

if red["c"]:
    (k1, e1), (k2, e2) = bands("inA")
    print("\n=== eps=0.2 scalar (red) vs our band-2 at K=2k:")
    errs = []
    for j, E, _ in sorted(red["c"]):
        Kp = 2 * (2 * np.pi * abs(j) / 30)
        ours = np.interp(Kp, k2, e2)
        errs.append(E - ours)
        print(f"  j={j:+d}: CGK {E:.4f}  ours {ours:.4f}  diff {E-ours:+.4f}")
    print(f"scalar rms: {np.sqrt(np.mean(np.array(errs)**2)):.4f}")

np.savez("data/cgk_spectrum_compare.npz",
         panel_a=np.array([(j, E) for j, E, _ in sorted(blue['a'])]),
         panel_c=np.array([(j, E) for j, E, _ in sorted(blue['c'])]),
         scalar_c=np.array([(j, E) for j, E, _ in sorted(red['c'])]) if red['c'] else np.zeros((0, 2)))
print("\nsaved data/cgk_spectrum_compare.npz")
