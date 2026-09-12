"""Final Wigner-delay extraction from the ns=40 purpose-built collisions
(antipodal launch, t <= 90), with the window-scan systematic used for
tab:delays: two-Gaussian ring fits per slice -> packet trajectories ->
linear incoming/outgoing legs over a grid of window choices,

    tau = -(x_out(t_ref) - x_in(t_ref)) / v_in ,

quoted as mean over {windows x packets}, stat = per-window fit error
propagated, sys = spread over the window grid.

  PYTHONPATH=. .venv/bin/python scripts/stageB_delay_ns40.py <j> [chi]
"""
import sys

import numpy as np
from scipy.optimize import curve_fit

J = int(sys.argv[1]) if len(sys.argv) > 1 else 3
CHI = int(sys.argv[2]) if len(sys.argv) > 2 else 256

z = np.load(f"data/stageB_colldens_j{J}_chi{CHI}_ns40.npz")
dens, times = z["density"], z["times"]
ncell = int(z["ns"]) // 2
c_r, c_l = int(z["c1"]) // 2, int(z["c2"]) // 2
vguess = 0.11 if J == 3 else 0.10
t_coll = (c_l - c_r) / (2 * vguess)


def ring_gauss(x, a, x0, s):
    d = (x - x0 + ncell / 2) % ncell - ncell / 2
    return a * np.exp(-0.5 * (d / s) ** 2)


def fit_two(cell, g1, g2):
    x = np.arange(ncell, dtype=float)

    def model(x, a1, x1, s1, a2, x2, s2):
        return ring_gauss(x, a1, x1, s1) + ring_gauss(x, a2, x2, s2)

    p0 = [max(cell.max(), 1e-3), g1, 1.2, max(cell.max(), 1e-3), g2, 1.2]
    lo = [0, g1 - 3, 0.5, 0, g2 - 3, 0.5]
    hi = [1, g1 + 3, 4.0, 1, g2 + 3, 4.0]
    p, _ = curve_fit(model, x, cell, p0=p0, bounds=(lo, hi), maxfev=20000)
    return p[1], p[4]


def leg(sel, x1g, x2g, v1, v2):
    t0 = times[sel[0]]
    xs1, xs2 = [], []
    for it in sel:
        cell = dens[it].reshape(ncell, 2).sum(1)
        cell = cell - cell.mean()
        f1, f2 = fit_two(cell, x1g + v1 * (times[it] - t0),
                         x2g + v2 * (times[it] - t0))
        xs1.append(f1)
        xs2.append(f2)
    return (np.unwrap(np.array(xs1), period=ncell),
            np.unwrap(np.array(xs2), period=ncell))


IN_W = [(4.0, 24.0), (4.0, 28.0), (8.0, 28.0)]
OUT_W = [(58.0, 88.0), (62.0, 88.0), (58.0, 82.0)]

taus, errs = [], []
for (ta, tb) in IN_W:
    si = np.where((times >= ta) & (times <= tb))[0]
    xi1, xi2 = leg(si, c_r, c_l, +vguess, -vguess)
    for (tc, td) in OUT_W:
        so = np.where((times >= tc) & (times <= td))[0]
        g = (c_r + c_l) / 2
        xo1, xo2 = leg(so, g + vguess * (tc - t_coll),
                       g - vguess * (tc - t_coll), +vguess, -vguess)
        for xin, xout in ((xi1, xo1), (xi2, xo2)):
            (vi, bi), ci = np.polyfit(times[si], xin, 1, cov=True)
            (vo, bo), co = np.polyfit(times[so], xout, 1, cov=True)
            tref = td
            dx = (vo * tref + bo) - (vi * tref + bi)
            tau = -dx / vi
            # propagate the two straight-line fits' parameter errors
            var = (co[0, 0] * tref**2 + co[1, 1]
                   + ci[0, 0] * tref**2 + ci[1, 1]) / vi**2 \
                + (dx / vi**2) ** 2 * ci[0, 0]
            taus.append(tau)
            errs.append(np.sqrt(var))
taus, errs = np.array(taus), np.array(errs)
print(f"j={J} chi={CHI}: {len(taus)} window/packet combos")
print(f"  tau = {taus.mean():+.2f} +- {np.median(errs):.2f} (stat) "
      f"+- {taus.std():.2f} (window sys)")
print(f"  per-window range [{taus.min():+.2f}, {taus.max():+.2f}]")
