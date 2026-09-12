"""Plot the ns=8 synthesis-error pilot (Fig.~\\ref{fig:synth}): ridge and
<H> error vs grid spacing, deterministic vs stochastic rounding.  Reads the
npz written by synthesis_pilot_ns8.py.  Formatting matched to the 101-qubit
ladder (fig:ladder): Okabe-Ito colours, log-log, integer T-gate top axis,
10^x decade labels on delta and the error.

  PYTHONPATH=. .venv/bin/python scripts/synthesis_pilot_plot.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullFormatter, NullLocator
import numpy as np

plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "stix",
                     "font.size": 9, "axes.grid": True, "grid.alpha": 0.25,
                     "grid.linewidth": 0.6})
DET, STOCH = "#0072B2", "#D55E00"          # Okabe-Ito blue / orange

# shared delta<->T-gate map (Ross-Selinger), used by both synthesis figures
def d2t(z):
    return 3 * np.log2(2 / np.clip(z, 1e-12, None))
def t2d(t):
    return 2 / 2 ** (np.asarray(t) / 3)

d = np.load("data/synthesis_pilot_ns8.npz")
rows = d["rows"]                            # [delta, mode(0 det,1 stoch), seed, gap, err]
gap0 = float(d["gap0"])
deltas = np.unique(rows[:, 0])


def series(mode, kind):
    m, s = [], []
    for dl in deltas:
        sel = rows[(rows[:, 0] == dl) & (rows[:, 1] == mode)]
        v = np.abs(sel[:, 3] - gap0) if kind == "H" else sel[:, 4]
        m.append(np.mean(v)); s.append(np.std(v) if len(v) > 1 else 0.0)
    return np.array(m), np.array(s)


fig, (aL, aR) = plt.subplots(1, 2, figsize=(6.6, 2.7), constrained_layout=True)
for ax, kind, ylab in ((aL, "ridge", r"ridge RMS error / peak"),
                       (aR, "H", r"$|\Delta\langle H\rangle|$ error")):
    for mode, name, c, mk in ((0, "deterministic", DET, "o"),
                              (1, "stochastic", STOCH, "s")):
        m, s = series(mode, kind)
        ax.errorbar(deltas, m, yerr=s, fmt=mk + "-", color=c, ms=4,
                    capsize=2.5, label=name)
    ax.axhline(0.05, color="0.5", ls=":", lw=0.9)      # tolerance guide
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"synthesis grid spacing $\delta$ (rad)")
    ax.set_ylabel(ylab)
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    secax = ax.secondary_xaxis("top", functions=(d2t, t2d))
    secax.set_xlabel(r"$\sim T$ gates per rotation (Ross--Selinger)")
    secax.set_xticks([4, 8, 12, 16])
    secax.xaxis.set_major_formatter(FuncFormatter(lambda t, _: f"{t:.0f}"))
    secax.xaxis.set_minor_locator(NullLocator())
aL.legend(fontsize=7.5, loc="lower right")
fig.suptitle(r"Physics vs. rotation-synthesis error ($N_s=8$ pilot)",
             fontsize=9.5)
fig.savefig("data/synthesis_pilot_ns8.pdf", dpi=200)
print("wrote data/synthesis_pilot_ns8.pdf; deltas:", np.round(deltas, 4))
