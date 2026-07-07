"""Prediction tables at the comparison coupling points (tasks 4/15).

For each coupling set (tag -> data/deep_levels_<tag>_ns{8..20}.npz):
  - band-1/band-2 dispersions from the ns=20 spectrum (clamped splines),
    band-2 orientation (inverted or not), MM' threshold = min_p[E1+E2]
  - bound-state candidates: parity-even P=0 levels below 2M at every
    volume (reported per volume; last-two-volume extrapolate)
  - elastic MM phase shifts: parity-filtered P=0 levels in
    (2M, min(MM' threshold, 2 E1(pi))), level-counting n, branch
    (-pi/2, pi/2] -- identical logic to phase_shifts_v2
  - predicted finite-volume MM levels at ANY target L from a clamped
    spline of delta(p): solve p L + 2 delta(p) = 2 pi n
    (DHK N_P = 13 -> L = 13 is their production volume)

Outputs: printed tables, data/predict_tables.npz, data/csv/predict_<tag>.csv

  PYTHONPATH=. .venv/bin/python scripts/predict_tables.py
"""

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq

VOLS = (8, 10, 12, 14, 16, 18, 20)
POINTS = {
    "dhk":    dict(m0=1.0, g2=0.6, eta=1.0, target_L=[5, 13]),
    "cgkel":  dict(m0=0.1, g2=2.0, eta=1.0, target_L=[10]),
    "cgkinA": dict(m0=0.1, g2=0.4, eta=1.0, target_L=[10]),
    "cgkinB": dict(m0=0.2, g2=0.4, eta=1.0, target_L=[10]),
}
out = {}

for tag, meta in POINTS.items():
    print(f"\n{'='*70}\n{tag}: (m0, g2, eta) = "
          f"({meta['m0']}, {meta['g2']}, {meta['eta']})")
    d20 = np.load(f"data/deep_levels_{tag}_ns20.npz")
    g, ph = d20["gaps"], d20["phases"]

    # band 1: lowest level at each |k| below the two-particle region
    ks = np.unique(np.round(np.abs(ph), 6))
    band1 = []
    for kk in ks:
        m = np.isclose(np.abs(ph), kk) & (g > 0.1)
        band1.append((kk, g[m].min()))
    band1 = np.array(sorted(band1))
    M = band1[0, 1]
    # keep only the connected band (levels below 2M are single-particle)
    b1 = band1[band1[:, 1] < 2 * M - 1e-6]
    if len(b1) < len(ks):
        print(f"  band-1 spans {len(b1)}/{len(ks)} momenta below 2M")
    E1 = CubicSpline(b1[:, 0], b1[:, 1], bc_type=((1, 0.0), (1, 0.0)))
    # band 2: next state per |k| in (band1, 2M)
    b2 = []
    for kk, e1 in b1:
        m = np.isclose(np.abs(ph), kk) & (g > e1 + 1e-6) & (g < 2 * M - 1e-6)
        if m.any():
            b2.append((kk, g[m].min()))
    have_b2 = len(b2) >= len(b1) - 1
    if have_b2:
        b2 = np.array(sorted(b2))
        E2 = CubicSpline(b2[:, 0], b2[:, 1], bc_type=((1, 0.0), (1, 0.0)))
        pg = np.linspace(0, np.pi, 400)
        ssum = E1(pg) + E2(pg)
        thr = float(ssum.min())
        inv = "inverted" if b2[0, 1] > b2[-1, 1] else "normal"
        print(f"  M = {M:.4f}, band-2 [{b2[:,1].min():.4f}, "
              f"{b2[:,1].max():.4f}] ({inv}); 2M = {2*M:.4f}, "
              f"MM' threshold = {thr:.4f} at p = {pg[np.argmin(ssum)]:.3f}")
    else:
        thr = np.inf
        print(f"  M = {M:.4f}, 2M = {2*M:.4f}; no band 2 below 2M")
    e_top = min(thr, 2 * float(E1(np.pi)))

    # bound states + elastic points across volumes
    bound = {}
    pts = []
    for ns in VOLS:
        dd = np.load(f"data/deep_levels_{tag}_ns{ns}.npz")
        gg, pp = dd["gaps"], dd["phases"]
        rf = dd["refl"] * dd["refl"][0]
        p0 = (np.abs(np.angle(np.exp(1j * pp))) < 1e-4) & (rf > 0.99)
        # bound-state window: below 2M, above the single-meson band top
        bw = p0 & (gg > float(E1(np.pi)) + 0.05) & (gg < 2 * M - 1e-4)
        if have_b2:
            bw &= (gg > b2[:, 1].max() + 0.05)
        for e in gg[bw]:
            bound.setdefault(ns, []).append(float(e))
        cands = sorted(gg[p0 & (gg > 2 * M + 1e-9) & (gg < e_top - 2e-3)])
        L = ns // 2
        for n, E2lvl in enumerate(cands):
            p = brentq(lambda q: 2 * E1(q) - E2lvl, 1e-9, np.pi)
            dl = ((2 * np.pi * n - p * L) / 2 + np.pi / 2) % np.pi - np.pi / 2
            pts.append((ns, E2lvl, p, dl))
    if bound:
        for ns, es in sorted(bound.items()):
            print(f"  ns={ns}: below-2M candidates {np.round(es, 4)}")
        eb = [min(es) for ns, es in sorted(bound.items())][-2:]
        print(f"  bound state: E_B ~ {eb[-1]:.4f} "
              f"(last-two-volume drift {abs(eb[-1]-eb[0]):.4f}), "
              f"binding {2*M - eb[-1]:.4f}")
    pts = np.array(pts)
    print(f"  elastic MM points ({len(pts)}):")
    for ns, E, p, dl in pts:
        print(f"    ns={int(ns):2d}  E = {E:.4f}  p = {p:.4f}  "
              f"delta = {dl:+.4f}")

    # delta(p) spline -> predicted levels at target volumes
    o = np.argsort(pts[:, 2])
    ps, ds = pts[o, 2], pts[o, 3]
    # collapse duplicates in p (average) for spline sanity
    pu, du = [], []
    for p, dl in zip(ps, ds):
        if pu and abs(p - pu[-1]) < 1e-3:
            du[-1] = 0.5 * (du[-1] + dl)
        else:
            pu.append(p)
            du.append(dl)
    dspl = CubicSpline(pu, du)
    pred = {}
    for L in meta["target_L"]:
        lv = []
        for n in range(0, L):
            try:
                pr = brentq(lambda q: q * L + 2 * dspl(q) - 2 * np.pi * n,
                            max(1e-6, pu[0] - 0.15), min(np.pi, pu[-1] + 0.15))
                E = 2 * float(E1(pr))
                if E < e_top:
                    lv.append(E)
            except ValueError:
                continue
        pred[L] = lv
        print(f"  PREDICTED MM levels at L = {L}: {np.round(lv, 4)} "
              f"(valid window p in [{pu[0]:.2f}, {pu[-1]:.2f}])")
    out[tag] = dict(M=M, thr=thr, points=pts,
                    pred={str(L): v for L, v in pred.items()})
    with open(f"data/csv/predict_{tag}.csv", "w") as f:
        f.write("ns,E,p,delta\n")
        for ns, E, p, dl in pts:
            f.write(f"{int(ns)},{E:.6f},{p:.6f},{dl:.6f}\n")

np.savez("data/predict_tables.npz",
         **{f"{t}_{k}": v for t, dd in out.items()
            for k, v in dd.items() if k != "pred"})
print("\nsaved data/predict_tables.npz + per-point CSVs")
