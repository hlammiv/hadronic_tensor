"""Wigner-delay extraction from the PUBLISHED elastic-collision density
figure of Chai et al. (arXiv:2505.21240, Fig. elastic_scattering (a)),
via the raster embedded in their PDF (exact simulation grid, no
screen digitization).

Pipeline: PNG -> Purples-colormap inversion -> per-slice two-Gaussian
ring fit (stageB conventions) -> linear incoming/outgoing legs ->
tau = -(x_out(t*) - x_in(t*)) / v_in per packet.  Compare with our
prediction 2 ddelta/dE at their collision energy E = 4.86(2), from the
cgkel phase-shift data, folded over their packet width.

  PYTHONPATH=. .venv/bin/python scripts/chai_elastic_delay.py <png>
"""
import sys

import numpy as np
from PIL import Image
from matplotlib import cm
from scipy.optimize import curve_fit

png = sys.argv[1]
img = np.asarray(Image.open(png).convert("RGB")) / 255.0
H, W = img.shape[:2]
# invert the Purples colormap by nearest neighbour
import matplotlib as mpl
lut = mpl.colormaps["Purples"](np.linspace(0, 1, 256))[:, :3]
flat = img.reshape(-1, 3)
d2 = ((flat[:, None, :] - lut[None, :, :]) ** 2).sum(-1)
val = (np.argmin(d2, axis=1) / 255.0).reshape(H, W) * 0.20
# orientation: PDF image row 0 = top = t_max; columns = n = 0..29
rho = val[::-1, :]                     # rho[t_pix, n], t increasing
T_MAX, NCELL = 80.0, rho.shape[1]
tgrid = np.linspace(0, T_MAX, rho.shape[0])
print(f"raster {rho.shape[0]} time rows x {NCELL} sites, "
      f"value range {rho.min():.3f}..{rho.max():.3f}")


def ring_two_gauss(x, a1, x1, s1, a2, x2, s2):
    out = 0.0
    for k in (-NCELL, 0, NCELL):
        out = out + a1 * np.exp(-((x - x1 + k) ** 2) / (2 * s1 ** 2)) \
                  + a2 * np.exp(-((x - x2 + k) ** 2) / (2 * s2 ** 2))
    return out


NSITES = 30.0
ncoord = (np.arange(rho.shape[1]) + 0.5) * NSITES / rho.shape[1]


def ring_two_gauss(x, a1, x1, s1, a2, x2, s2):
    out = 0.0
    for k in (-NSITES, 0, NSITES):
        out = out + a1 * np.exp(-((x - x1 + k) ** 2) / (2 * s1 ** 2)) \
                  + a2 * np.exp(-((x - x2 + k) ** 2) / (2 * s2 ** 2))
    return out


tracks = {"L": [], "R": [], "t": []}
V0, TC0 = 0.26, 36.0
for i in range(0, rho.shape[0], 4):
    t = tgrid[i]
    row = rho[i] - np.median(rho[i])
    if row.max() < 0.02:
        continue
    if t < TC0 - 6:
        g1, g2 = 8 + V0 * t, 23 - V0 * t
    elif t > TC0 + 6:
        g1, g2 = 15.5 - V0 * (t - TC0), 15.5 + V0 * (t - TC0)
    else:
        continue                        # skip the overlap region
    try:
        popt, _ = curve_fit(ring_two_gauss, ncoord, row,
                            p0=(row.max(), g1, 2.5, row.max(), g2, 2.5),
                            maxfev=8000)
        xs = sorted([popt[1] % NSITES, popt[4] % NSITES])
        tracks["L"].append(xs[0]); tracks["R"].append(xs[1])
        tracks["t"].append(t)
    except Exception:
        continue
tt = np.array(tracks["t"])
xl, xr = np.array(tracks["L"]), np.array(tracks["R"])

taus = []
for w_in, w_out in [((5, 28), (46, 68)), ((8, 30), (48, 72)),
                    ((5, 26), (44, 66))]:
    mi = (tt >= w_in[0]) & (tt <= w_in[1])
    mo = (tt >= w_out[0]) & (tt <= w_out[1])
    li = np.polyfit(tt[mi], xl[mi], 1)   # left incoming (+v)
    ri = np.polyfit(tt[mi], xr[mi], 1)   # right incoming (-v)
    lo = np.polyfit(tt[mo], xl[mo], 1)   # left outgoing (-v)
    ro = np.polyfit(tt[mo], xr[mo], 1)   # right outgoing (+v)
    t_c = (ri[1] - li[1]) / (li[0] - ri[0])   # incoming crossing
    v = 0.5 * (abs(li[0]) + abs(ri[0]))
    tau_L = (np.polyval(lo, t_c) - np.polyval(li, t_c)) / v
    tau_R = -(np.polyval(ro, t_c) - np.polyval(ri, t_c)) / v
    taus.extend([tau_L, tau_R])
    print(f"  windows {w_in}/{w_out}: t_c={t_c:.1f} v={v:.3f} "
          f"tau_L={tau_L:+.1f} tau_R={tau_R:+.1f}")
tau_obs, tau_err = np.mean(taus), np.std(taus)
print(f"tau_obs = {tau_obs:+.1f} +- {tau_err:.1f}")

# ---- our prediction at their kinematics ----
d = np.load("data/predict_tables.npz", allow_pickle=True)
pts = d["cgkel_points"]
dl = np.where((pts[:, 1] > 4.95) & (pts[:, 3] > 0),
              pts[:, 3] - np.pi, pts[:, 3])
sel = (pts[:, 1] > 4.78) & (pts[:, 1] < 4.95)
slope = np.polyfit(pts[sel, 1], dl[sel], 1)[0]
print(f"prediction: ddelta/dE = {slope:.2f} over E in [4.78,4.95] "
      f"-> tau_pred = 2 ddelta/dE = {2*slope:+.1f} at E = 4.86(2)")
