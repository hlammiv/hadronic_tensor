"""Wigner-delay extraction from the Stage-B collision density maps.

Each time slice's cell-density excess is fit by a two-Gaussian ring model;
the two centers, unwrapped in time, give the packet trajectories.  Linear
fits to the incoming leg (before the first collision) and the outgoing leg
(after it, but before the PBC wrap-around second collision at
t ~ ring/closing-speed) give per-packet time delays

    tau = -(x_out(t) - x_in(t)) / v_in .

Predictions (Wigner 2 ddelta/dE folded over the packet energy spread):
j=3 (K=+-1.2566, on-resonance): +4.2;  j=2 (K=+-0.8378, null): +0.2.

  PYTHONPATH=. python scripts/stageB_delay_extract.py <chi> [j ...]
"""
import sys

import numpy as np
from scipy.optimize import curve_fit

CHI = int(sys.argv[1]) if len(sys.argv) > 1 else 256
JS = [int(a) for a in sys.argv[2:]] or [3, 2]
# fit windows chosen from the collision kinematics: group velocity is
# dE/dK ~ 0.11 (j=3) / 0.10 (j=2) cells/unit, so the packets (7 cells
# apart, closing at 2v) first meet at t ~ 32/35 and the PBC wrap-around
# second collision sits beyond t ~ 100 -- outside the t <= 90 data
WINDOWS = {3: ((4.0, 22.0), (42.0, 88.0)), 2: ((4.0, 24.0), (46.0, 88.0))}


def ring_gauss(x, a, x0, s, ncell):
    d = (x - x0 + ncell / 2) % ncell - ncell / 2
    return a * np.exp(-0.5 * (d / s) ** 2)


def fit_two(cell, g1, g2, ncell):
    x = np.arange(ncell, dtype=float)

    def model(x, a1, x1, s1, a2, x2, s2):
        return (ring_gauss(x, a1, x1, s1, ncell)
                + ring_gauss(x, a2, x2, s2, ncell))

    p0 = [max(cell.max(), 1e-3), g1, 1.2, max(cell.max(), 1e-3), g2, 1.2]
    lo = [0, g1 - 3, 0.5, 0, g2 - 3, 0.5]
    hi = [1, g1 + 3, 4.0, 1, g2 + 3, 4.0]
    p, _ = curve_fit(model, x, cell, p0=p0, bounds=(lo, hi), maxfev=20000)
    return p[1], p[4]


def leg(dens, times, sel, ncell, x1g, x2g, v1, v2):
    """Fit both packet centers over `sel`; guesses advected linearly."""
    t0 = times[sel[0]]
    xs1, xs2 = [], []
    for it in sel:
        cell = dens[it].reshape(ncell, 2).sum(1)
        cell -= cell.mean()
        g1 = x1g + v1 * (times[it] - t0)
        g2 = x2g + v2 * (times[it] - t0)
        f1, f2 = fit_two(cell, g1, g2, ncell)
        xs1.append(f1)
        xs2.append(f2)
    xs1 = np.unwrap(np.array(xs1), period=ncell)
    xs2 = np.unwrap(np.array(xs2), period=ncell)
    return xs1, xs2


for J in JS:
    z = np.load(f"data/stageB_colldens_j{J}_chi{CHI}_ns30.npz")
    dens, times = z["density"], z["times"]
    ncell = int(z["ns"]) // 2
    c_r, c_l = int(z["c1"]) // 2, int(z["c2"]) // 2
    vguess = {3: 0.11, 2: 0.10}[J]
    t_coll = (c_l - c_r) / (2 * vguess)
    (ta, tb), (tc, td) = WINDOWS[J]
    print(f"\n=== j={J} chi={CHI} ===")
    out = {}
    for tag, (t0, t1), guesses in (
            ("in", (ta, tb), (c_r, c_l, +vguess, -vguess)),
            # after the collision the right-moving packet emerges near the
            # midpoint moving right (identical particles: track direction)
            ("out", (tc, td), ((c_r + c_l) / 2 + vguess * (tc - t_coll),
                               (c_r + c_l) / 2 - vguess * (tc - t_coll),
                               +vguess, -vguess))):
        sel = np.where((times >= t0) & (times <= t1))[0]
        x1g, x2g, v1, v2 = guesses
        xs1, xs2 = leg(dens, times, sel, ncell, x1g, x2g, v1, v2)
        res = {}
        for name, xs in (("R", xs1), ("L", xs2)):
            v, x0 = np.polyfit(times[sel], xs, 1)
            r = np.abs(xs - (v * times[sel] + x0)).max()
            res[name] = (v, x0, r)
            print(f"  {tag:3s} {name}: v = {v:+.4f}  resid {r:.3f}")
        out[tag] = res
    for name in ("R", "L"):
        vi, xi, _ = out["in"][name]
        vo, xo, _ = out["out"][name]
        tref = WINDOWS[J][1][1]
        dx = (vo * tref + xo) - (vi * tref + xi)
        tau = -dx / vi
        print(f"  {name}: dx = {dx:+.3f} cells, tau = {tau:+.2f} "
              f"(v_in {vi:+.3f} vs v_out {vo:+.3f})")
