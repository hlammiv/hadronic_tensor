"""Control-channel (j=2, ns=40) Wigner delay for tab:delays.

Robust tracking: iterated windowed centroid (center of mass of the
cell-density excess within +-W cells of the advected guess).  Two
extractions:
  (a) leg method -- outgoing centroid track vs the extrapolated
      incoming line (the resonant-row method), window-scanned;
  (b) solo-reference -- circular cross-correlation of the outgoing
      density against the SAME packet launched alone (j2L/j2R), which
      builds dispersion spreading into the reference.

Outgoing windows stop before the PBC wrap re-scatter at t ~ 56.

  PYTHONPATH=. .venv/bin/python scripts/stageB_ctrl_delay.py
"""
import numpy as np

NCELL = 20
X = np.arange(NCELL, dtype=float)


def cellrow(dens, it):
    c = dens[it].reshape(NCELL, 2).sum(1)
    return c - c.mean()


def centroid(cell, guess, W=4):
    """Iterated windowed centroid on the ring."""
    g = guess
    for _ in range(6):
        d = (X - g + NCELL / 2) % NCELL - NCELL / 2
        m = np.abs(d) <= W
        w = np.clip(cell[m], 0, None)
        if w.sum() <= 0:
            return np.nan
        g = g + (w * d[m]).sum() / w.sum()
    return g


def track(dens, times, sel, x0, v0):
    xs, g, tp = [], x0, times[sel[0]]
    for it in sel:
        g = g + v0 * (times[it] - tp)
        tp = times[it]
        c = centroid(cellrow(dens, it), g)
        if not np.isnan(c):
            g = c
        xs.append(g)
    return np.unwrap(np.array(xs), period=NCELL)


def xcorr_shift(cell, ref, guess, W=5):
    """Sub-cell circular shift of `cell` relative to `ref`, near guess."""
    shifts = np.arange(-W, W + 1)
    d0 = (X - guess + NCELL / 2) % NCELL - NCELL / 2
    m = np.abs(d0) <= W + 2
    cc = np.array([np.dot(np.roll(ref, s)[m], cell[m]) for s in shifts])
    k = int(np.argmax(cc))
    if k in (0, len(shifts) - 1):
        return float(shifts[k])
    a, b, c = cc[k - 1], cc[k], cc[k + 1]
    return float(shifts[k]) + 0.5 * (a - c) / (a - 2 * b + c)


zc = np.load("data/stageB_colldens_j2_chi256_ns40.npz")
zl = np.load("data/stageB_colldens_j2L_chi256_ns40.npz")
zr = np.load("data/stageB_colldens_j2R_chi256_ns40.npz")
dens, times = zc["density"], zc["times"]

# solo references: identify movers, fit their velocity
solos = {}
for zz in (zl, zr):
    dd = zz["density"]
    x0 = float(np.argmax(cellrow(dd, 0)))
    xs = track(dd, zz["times"], np.arange(0, 89), x0, 0.0)
    v = np.polyfit(zz["times"][:89], xs, 1)[0]
    solos["R" if v > 0 else "L"] = (dd, zz["times"], xs, v)
    print(f"solo: start {x0:.0f}, v = {v:+.4f}")

IN_W = [(2.0, 18.0), (2.0, 20.0), (4.0, 20.0)]
OUT_W = [(36.0, 52.0), (38.0, 54.0), (40.0, 56.0)]

taus_leg, taus_ref = [], []
for (ta, tb) in IN_W:
    si = np.where((times >= ta) & (times <= tb))[0]
    xiR = track(dens, times, si, 5.0, +0.20)
    xiL = track(dens, times, si, 15.0, -0.20)
    viR, biR = np.polyfit(times[si], xiR, 1)
    viL, biL = np.polyfit(times[si], xiL, 1)
    t_coll = (biL - biR) / (viR - viL)
    xm = viR * t_coll + biR
    for (tc, td) in OUT_W:
        so = np.where((times >= tc) & (times <= td))[0]
        for name, vi, bi in (("R", viR, biR), ("L", viL, biL)):
            xo = track(dens, times, so, xm + vi * (tc - t_coll), vi)
            vo, bo = np.polyfit(times[so], xo, 1)
            dx = (vo * td + bo) - (vi * td + bi)
            taus_leg.append(-dx / vi)
            # (b) cross-correlate against the solo run, frame by frame
            dd, ts, xs_solo, vs = solos[name]
            offs = []
            for it in so:
                js = int(np.argmin(np.abs(ts - times[it])))
                ref = np.clip(cellrow(dd, js), 0, None)
                cel = np.clip(cellrow(dens, it), 0, None)
                g = np.interp(times[it], ts[:89], xs_solo)
                s = xcorr_shift(cel, ref, g)
                offs.append(s)          # cell shift: collided vs solo
            taus_ref.append(-np.mean(offs) / vs)

for name, taus in (("leg (resonant-row method)", taus_leg),
                   ("solo-reference xcorr", taus_ref)):
    t = np.array(taus)
    tR, tL = t.reshape(-1, 2)[:, 0], t.reshape(-1, 2)[:, 1]
    print(f"{name}:")
    print(f"  R: {tR.mean():+.2f} +- {tR.std():.2f}   "
          f"L: {tL.mean():+.2f} +- {tL.std():.2f}")
    print(f"  combined tau = {t.mean():+.2f} +- {t.std():.2f} (window sys)"
          f"   range [{t.min():+.2f}, {t.max():+.2f}]")
