"""Two-channel MM/MM' finite-volume analysis in the inelastic window
(task 16), using the 7-volume gauge-fixed spectra.

Window: M+M' < E < 2 E1(pi) (both channels open; MM closes at the lattice
band edge).  Sector content at P = 0:
  even:  MM and MM' coupled by a 2x2 unitary S(E) = (delta1, delta2, theta)
         quantization: cos[(A+B)/2] = cos(2 theta) cos[(A-B)/2],
         A = p1 L + 2 delta1,  B = p2 L + 2 delta2
         (theta = 0 decouples to  p_i L + 2 delta_i = 2 pi n)
  odd:   MM' only (identical-boson MM has no odd sector):
         p2 L + 2 delta_odd = 2 pi n
Dispersions E1(k), E2(k): clamped splines through the ns=20 single-particle
bands.  Energy dependence: delta1 quadratic, delta2/theta/delta_odd linear
in (E - 6.0).  Fit: symmetric assignment-free least squares (each observed
level matched to the nearest predicted level), multi-start; leave-one-
volume-out postdiction test on ns=10.

  PYTHONPATH=. .venv/bin/python scripts/two_channel.py
"""

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq, least_squares

VOLS = (8, 10, 12, 14, 16, 18, 20)
LOO = 10                                  # leave-one-out volume


def band_splines():
    d = np.load("data/deep_levels_ns20.npz")
    gaps, ph = d["gaps"], d["phases"]
    out = []
    for lo, hi in ((2.5, 3.10), (3.10, 3.45)):
        sel = (gaps > lo) & (gaps < hi)
        k = np.abs(ph[sel])
        e = gaps[sel]
        ks, es = [], []
        for kk in np.unique(np.round(k, 6)):
            m = np.isclose(k, kk)
            ks.append(kk)
            es.append(e[m].min())        # lowest level at each |k| = band
        out.append(CubicSpline(ks, es, bc_type=((1, 0.0), (1, 0.0))))
    return out


E1, E2 = band_splines()
M, M2 = float(E1(0)), float(E2(0))
THR = M + M2
EDGE = 2 * float(E1(np.pi))
print(f"bands: M = {M:.4f}, M' = {M2:.4f}; window ({THR:.4f}, {EDGE:.4f})")

p1 = lambda E: brentq(lambda p: 2 * E1(p) - E, 1e-9, np.pi)
p2 = lambda E: brentq(lambda p: E1(p) + E2(p) - E, 1e-9, np.pi)

levels = {}
for ns in VOLS:
    d = np.load(f"data/deep_levels_ns{ns}.npz")
    g, ph = d["gaps"], d["phases"]
    sel = (np.abs(np.angle(np.exp(1j * ph))) < 1e-4) & \
          (g > THR + 2e-3) & (g < EDGE - 2e-3)
    levels[ns] = np.sort(g[sel])
    print(f"ns={ns} (L={ns//2}): {len(levels[ns])} window levels "
          f"{np.round(levels[ns], 4)}")


def predicted(params, L, grid=np.linspace(THR + 3e-4, EDGE - 3e-4, 1500)):
    d10, d11, d12, d20, d21, t0, t1, o0, o1 = params
    x = grid - 6.0
    A = np.array([p1(E) for E in grid]) * L + 2 * (d10 + d11 * x + d12 * x**2)
    B = np.array([p2(E) for E in grid]) * L + 2 * (d20 + d21 * x)
    th = t0 + t1 * x
    F1 = np.cos((A + B) / 2) - np.cos(2 * th) * np.cos((A - B) / 2)
    F2 = np.sin((np.array([p2(E) for E in grid]) * L
                 + 2 * (o0 + o1 * x)) / 2)
    roots = []
    for F in (F1, F2):
        s = np.where(np.diff(np.sign(F)) != 0)[0]
        for i in s:
            e0, e1_, f0, f1_ = grid[i], grid[i + 1], F[i], F[i + 1]
            roots.append(e0 - f0 * (e1_ - e0) / (f1_ - f0))
    return np.sort(roots)


def residuals(params, exclude=None):
    r = []
    for ns in VOLS:
        if ns == exclude or not len(levels[ns]):
            continue
        pred = predicted(params, ns // 2)
        if not len(pred):
            r.extend([1.0] * len(levels[ns]))
            continue
        for E in levels[ns]:
            r.append(E - pred[np.argmin(np.abs(pred - E))])
    return np.array(r)


def fit(exclude=None):
    best = None
    for d10 in (-0.5, 0.0):
        for t0 in (0.1, 0.4):
            for o0 in (-0.3, 0.0, 0.3):
                x0 = [d10, -1.0, 0.0, 0.0, -1.0, t0, 0.0, o0, -1.0]
                try:
                    res = least_squares(residuals, x0, kwargs={
                        "exclude": exclude}, method="lm", max_nfev=400)
                except Exception:
                    continue
                if best is None or res.cost < best.cost:
                    best = res
    return best


res = fit()
p = res.x
r = residuals(p)
print(f"\nfull fit: {len(r)} levels, rms residual "
      f"{np.sqrt(np.mean(r**2)):.5f}, max |res| {np.abs(r).max():.5f}")
names = ["d1(6.0)", "d1'", "d1''", "d2(6.0)", "d2'", "theta(6.0)",
         "theta'", "dodd(6.0)", "dodd'"]
for n, v in zip(names, p):
    print(f"  {n:11s} = {v:+.4f}")
for E in (5.95, 6.00, 6.05, 6.10, 6.15):
    x = E - 6.0
    print(f"  E={E:.2f}: delta_MM = {p[0]+p[1]*x+p[2]*x*x:+.3f}, "
          f"delta_MM' = {p[3]+p[4]*x:+.3f}, theta = {p[5]+p[6]*x:+.3f} "
          f"(inelasticity 1-cos2th = {1-np.cos(2*(p[5]+p[6]*x)):.3f}), "
          f"delta_odd = {p[7]+p[8]*x:+.3f}")

res_loo = fit(exclude=LOO)
pred = predicted(res_loo.x, LOO // 2)
print(f"\nleave-out ns={LOO} postdiction:")
for E in levels[LOO]:
    q = pred[np.argmin(np.abs(pred - E))]
    print(f"  observed {E:.4f}  predicted {q:.4f}  diff {E-q:+.4f}")

np.savez("data/two_channel_fit.npz", params=p, names=names,
         rms=float(np.sqrt(np.mean(r**2))), M=M, M2=M2, thr=THR, edge=EDGE,
         loo_ns=LOO, loo_obs=levels[LOO],
         loo_pred=[pred[np.argmin(np.abs(pred - E))] for E in levels[LOO]])
print("\nsaved data/two_channel_fit.npz")
