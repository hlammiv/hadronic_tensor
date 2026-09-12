"""Paper figure (fig:cgkspec): zero-parameter postdiction of the CGK
(arXiv:2505.21240) L=30 meson spectra from our ns<=20 volumes.

Their momentum label k=2*pi*j/30 spans the full staggered BZ; under the
physical mapping K=2k their vector tower is our band-1 glued to band-2
(bands touch at K=pi), and their scalar is band-2 at |K|=2|k|.  Curves =
our small-volume prediction; points = their published L=30 QSE/MPS values
(vector-extracted from their arXiv-source figure; data/cgk_spectrum_compare.npz).

  PYTHONPATH=. .venv/bin/python scripts/cgk_postdiction_figure.py
"""
import numpy as np
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "stix",
                     "font.size": 9, "axes.grid": True, "grid.alpha": 0.25,
                     "grid.linewidth": 0.6})
BLUE, ORANGE = "#0072B2", "#D55E00"


def bands(tag):
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
            b1[k] = gs[0]
            if len(gs) > mult:
                b2[k] = gs[mult]
    k1 = np.array(sorted(b1)); e1 = np.array([b1[k] for k in k1])
    k2 = np.array(sorted(b2)); e2 = np.array([b2[k] for k in k2])
    return (k1, e1), (k2, e2)


def glue_curve(tag, jj):
    (k1, e1), (k2, e2) = bands(tag)
    k = 2 * np.pi * np.abs(jj) / 30
    E = np.where(k <= np.pi / 2,
                 np.interp(2 * k, k1, e1),
                 np.interp(2 * np.pi - 2 * k, k2, e2))
    return E


cmp = np.load("data/cgk_spectrum_compare.npz")
jj = np.linspace(-13.5, 14.5, 400)

fig, (aT, aB) = plt.subplots(2, 1, figsize=(3.5, 4.3), constrained_layout=True,
                             sharex=True)
# ---- (a) eps=1.0 vector ----
aT.plot(jj, glue_curve("el", jj), "-", color=BLUE, lw=1.4,
        label=r"this work, $N_s \leq 20$")
pa = cmp["panel_a"]
aT.plot(pa[:, 0], pa[:, 1], "o", mfc="none", mec="0.2", ms=4.5, mew=1.0,
        label=r"Ref. $L{=}30$ (MPS/QSE)")
aT.set_ylabel(r"$\Delta E$")
aT.legend(fontsize=7, loc="upper center", framealpha=0.9)
aT.text(0.03, 0.06, r"$m{=}0.1,\ \varepsilon{=}1.0$ (vector)",
        transform=aT.transAxes, fontsize=8)
aT.text(0.97, 0.06, "(a)", transform=aT.transAxes, fontsize=9, ha="right")

# ---- (b) eps=0.2 vector + scalar ----
aB.plot(jj, glue_curve("inA", jj), "-", color=BLUE, lw=1.4)
pc = cmp["panel_c"]
aB.plot(pc[:, 0], pc[:, 1], "o", mfc="none", mec="0.2", ms=4.5, mew=1.0)
(k1, e1), (k2, e2) = bands("inA")
js = np.linspace(-7.4, 7.4, 200)      # extends past their scalar cutoff |j|<=4
aB.plot(js, np.interp(2 * (2 * np.pi * np.abs(js) / 30), k2, e2), "-",
        color=ORANGE, lw=1.4)
sc = cmp["scalar_c"]
aB.plot(sc[:, 0], sc[:, 1], "D", mfc="none", mec=ORANGE, ms=4.5, mew=1.0)
aB.text(5.2, 1.455, "scalar", color=ORANGE, fontsize=8)
aB.text(3.5, 1.02, "vector", color=BLUE, fontsize=8)
aB.set_xlabel(r"$k \times (L/2\pi)$")
aB.set_ylabel(r"$\Delta E$")
aB.text(0.03, 0.06, r"$m{=}0.1,\ \varepsilon{=}0.2$",
        transform=aB.transAxes, fontsize=8)
aB.text(0.97, 0.06, "(b)", transform=aB.transAxes, fontsize=9, ha="right")

fig.savefig("data/cgk_postdiction_fig.pdf", dpi=200)
# report the RMS numbers for the caption
for tag, key in (("el", "panel_a"), ("inA", "panel_c")):
    p = cmp[key]
    r = p[:, 1] - glue_curve(tag, p[:, 0])
    print(f"{key}: n={len(p)} rms={np.sqrt(np.mean(r**2)):.4f}")
sc_r = sc[:, 1] - np.interp(2 * (2 * np.pi * np.abs(sc[:, 0]) / 30), k2, e2)
print(f"scalar: n={len(sc)} rms={np.sqrt(np.mean(sc_r**2)):.4f}")
print("wrote data/cgk_postdiction_fig.pdf")
