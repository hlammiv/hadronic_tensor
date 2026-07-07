"""Two-channel MM/MM' finite-volume analysis, final (parity-tagged).

Sector assignment is now EXACT: every P=0 level carries a reflection
parity R (htensor.gaugefixed.PhysicalBasis.select_reflection; the k=0
meson is R = -1, so MM levels are R = +1).  In the open-open window
(min_p[E1+E2], 2 E1(pi)):

  R = +1 : MM and MM'-even coupled by a 2x2 unitary S(E) =
           (delta1, delta2, theta):
           cos[(A+B)/2] = cos(2 theta) cos[(A-B)/2],
           A = p1 L + 2 delta1,  B = p2 L + 2 delta2
  R = -1 : MM'-odd alone: p2 L + 2 delta_odd = 2 pi n, i.e. pointwise
           delta_odd = branch(-p2 L / 2 mod pi)  (no n needed mod pi)

Sub-threshold R = +1 levels (5.85 < E < thr) anchor delta1 through the
theta-decoupled MM condition.  Energy dependence: delta1 quadratic,
delta2 and theta linear in (E - 6.0); physical-branch bounds; multi-start
least squares; leave-one-volume-out postdiction on ns = 10.

  PYTHONPATH=. .venv/bin/python scripts/two_channel.py
"""

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq, least_squares

VOLS = (8, 10, 12, 14, 16, 18, 20)
LOO = 10


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
            es.append(e[m].min())
        out.append(CubicSpline(ks, es, bc_type=((1, 0.0), (1, 0.0))))
    return out


E1, E2 = band_splines()
M, M2 = float(E1(0)), float(E2(0))
THR = M + M2
EDGE = 2 * float(E1(np.pi))
print(f"bands: M = {M:.4f}, M'(0) = {M2:.4f}; window ({THR:.4f}, {EDGE:.4f})")

p1 = lambda E: brentq(lambda p: 2 * E1(p) - E, 1e-9, np.pi)
p2 = lambda E: brentq(lambda p: E1(p) + E2(p) - E, 1e-9, np.pi)

E_MM_LO = 5.85
ev_lv, odd_lv, mm_lv = {}, {}, {}
for ns in VOLS:
    d = np.load(f"data/deep_levels_ns{ns}.npz")
    g, ph = d["gaps"], d["phases"]
    rf = d["refl"] * d["refl"][0]      # vacuum-normalized parity
    p0 = np.abs(np.angle(np.exp(1j * ph))) < 1e-4
    win = p0 & (g > THR + 2e-3) & (g < EDGE - 2e-3)
    amb = win & (np.abs(rf) < 0.99)
    assert not amb.any(), f"ambiguous parity at ns={ns}: {g[amb]}"
    ev_lv[ns] = np.sort(g[win & (rf > 0.99)])
    odd_lv[ns] = np.sort(g[win & (rf < -0.99)])
    mm = p0 & (g > E_MM_LO) & (g < THR - 2e-3)
    sub_odd = mm & (rf < -0.99)
    if sub_odd.any():
        # parity-odd below the MM' threshold: identical-boson MM cannot be
        # odd, so these are near-threshold odd-channel states (resonance /
        # virtual-state candidates); they need analytic continuation, not
        # the real-momentum quantization condition -- quarantine.
        print(f"  ns={ns}: sub-threshold ODD level(s) "
              f"{np.round(g[sub_odd], 4)} quarantined (odd-channel pole?)")
    mm_lv[ns] = np.sort(g[mm & (rf > 0.99)])
    print(f"ns={ns} (L={ns//2}): {len(mm_lv[ns])} MM-anchor + "
          f"{len(ev_lv[ns])} even + {len(odd_lv[ns])} odd: "
          f"{np.round(mm_lv[ns], 4)} | {np.round(ev_lv[ns], 4)} | "
          f"{np.round(odd_lv[ns], 4)}")

# ---------------- odd sector: pointwise, assignment-free mod pi ----------
print("\nMM'-odd phase shifts (pointwise):")
odd_pts = []
for ns in VOLS:
    for E in odd_lv[ns]:
        dlt = (-p2(E) * (ns // 2) / 2 + np.pi / 2) % np.pi - np.pi / 2
        odd_pts.append((ns, E, dlt))
        print(f"  ns={ns}  E = {E:.4f}  delta_odd = {dlt:+.4f} rad")

# ---------------- even sector: coupled 2x2 fit ---------------------------
GRID = np.linspace(THR + 3e-4, EDGE - 3e-4, 1500)
P1G = np.array([p1(E) for E in GRID])
P2G = np.array([p2(E) for E in GRID])
GRID_MM = np.linspace(E_MM_LO + 3e-4, THR - 3e-4, 500)
P1_MM = np.array([p1(E) for E in GRID_MM])


def roots(F, grid):
    s = np.where(np.diff(np.sign(F)) != 0)[0]
    return np.array([grid[i] - F[i] * (grid[i + 1] - grid[i])
                     / (F[i + 1] - F[i]) for i in s])


def predicted(params, L):
    d10, d11, d12, d20, d21, t0, t1 = params
    x = GRID - 6.0
    A = P1G * L + 2 * (d10 + d11 * x + d12 * x**2)
    B = P2G * L + 2 * (d20 + d21 * x)
    th = t0 + t1 * x
    return np.sort(roots(np.cos((A + B) / 2)
                         - np.cos(2 * th) * np.cos((A - B) / 2), GRID))


def predicted_mm(params, L):
    d10, d11, d12 = params[:3]
    x = GRID_MM - 6.0
    return roots(np.sin((P1_MM * L
                         + 2 * (d10 + d11 * x + d12 * x**2)) / 2), GRID_MM)


def residuals(params, exclude=None):
    r = []
    for ns in VOLS:
        if ns == exclude:
            continue
        for obs, pred in ((ev_lv[ns], predicted(params, ns // 2)),
                          (mm_lv[ns], predicted_mm(params, ns // 2))):
            for E in obs:
                r.append(E - pred[np.argmin(np.abs(pred - E))]
                         if len(pred) else 1.0)
    return np.array(r)


LB = [-1.6, -30, -300, -1.6, -30, 0.0, -15]
UB = [+1.6, +30, +300, +1.6, +30, 0.785, +15]


def fit(exclude=None):
    best = None
    for d10 in (-0.6, -0.2, 0.3):
        for t0 in (0.05, 0.3, 0.6):
            for d20 in (-0.5, 0.0, 0.5):
                x0 = [d10, -1.0, 0.0, d20, -1.0, t0, 0.0]
                try:
                    res = least_squares(residuals, x0,
                                        kwargs={"exclude": exclude},
                                        bounds=(LB, UB), method="trf",
                                        max_nfev=800)
                except Exception:
                    continue
                if best is None or res.cost < best.cost:
                    best = res
    return best


res = fit()
p = res.x
r = residuals(p)
n_lv = len(r)
print(f"\neven-sector fit: {n_lv} levels, rms {np.sqrt(np.mean(r**2)):.5f},"
      f" max |res| {np.abs(r).max():.5f}")
for n, v in zip(["d1(6.0)", "d1'", "d1''", "d2(6.0)", "d2'",
                 "theta(6.0)", "theta'"], p):
    print(f"  {n:11s} = {v:+.4f}")
print("\n  E      delta_MM  delta_MM'  theta   eta=cos2th")
for E in (6.04, 6.06, 6.08, 6.10, 6.12, 6.14):
    x = E - 6.0
    th = p[5] + p[6] * x
    print(f"  {E:.2f}  {p[0]+p[1]*x+p[2]*x*x:+8.4f} {p[3]+p[4]*x:+8.4f}"
          f"  {th:+7.4f}  {np.cos(2*th):+.4f}")

res_loo = fit(exclude=LOO)
pred = predicted(res_loo.x, LOO // 2)
print(f"\nleave-out ns={LOO} even-sector postdiction:")
for E in ev_lv[LOO]:
    q = pred[np.argmin(np.abs(pred - E))]
    print(f"  observed {E:.4f}  predicted {q:.4f}  diff {E-q:+.4f}")

np.savez("data/two_channel_fit.npz", params=p,
         rms=float(np.sqrt(np.mean(r**2))), n_levels=n_lv,
         odd_points=np.array(odd_pts), M=M, M2=M2, thr=THR, edge=EDGE,
         loo_ns=LOO, loo_obs=ev_lv[LOO],
         loo_pred=[pred[np.argmin(np.abs(pred - E))] for E in ev_lv[LOO]])
print("\nsaved data/two_channel_fit.npz")
