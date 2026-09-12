"""Connected equal-time correlator G(t=0,x) = <J0(x)J0(c)>_conn on
hardware, with gauge-sector post-selection.  The screening cloud around
the insertion is damped ~4x in the raw device data; post-selecting shots
with G_n=+1 on the stabilizers around the insertion projects onto the
gauge-invariant subspace and un-damps it from 0.25 to 0.60 of its exact
depth at win=2 -- a truth-independent mitigation from the existing S(q)
bitstrings, no new QPU.

Metric note: "depth" is measured over the CLOUD, 0 < |x-x0| <= 4, with the
self-correlation at x=0 excluded.  Including x=0 inflates every ratio (the
self term is well measured); the ~80% once quoted for this recovery was the
center-included RMS-error reduction, a different quantity.  See the ladder
in sec:gaugeqed of the paper: win 0/1/2/3 -> 0.25/0.46/0.60/0.67 at
30000/11653/6415/3613 shots kept.

  PYTHONPATH=. .venv/bin/python scripts/hw_cloud_figure.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from htensor import Z2Lattice

NS, VC = 50, 24
lat = Z2Lattice(NS, pbc=True)
bits = np.load("data/hw/sq_bits_kingston.npz")["bits"].astype(np.int8)
sq = np.array([lat.site_qubit(v) for v in range(NS)])
sgn = np.array([(-1) ** v for v in range(NS)])
tru = np.load("data/w_meson_ns50_k1.26_v3.npz")
one_t = tru["one_pt_wp"][0, :NS].real
g_t = tru["corr_wp"][0, :NS].real - one_t * one_t[VC]


def gcol(b, n):
    return ((-1) ** n * (1 - 2 * b[:, lat.site_qubit(n)])
            * (1 - 2 * b[:, lat.link_qubit(n - 1)])
            * (1 - 2 * b[:, lat.link_qubit(n)]))


def conn(b):
    z = 1 - 2 * b[:, sq]
    j = (sgn[None, :] - z) / 2
    o = j.mean(0)
    g = (j * j[:, [VC]]).mean(0) - o * o[VC]
    ge = (j * j[:, [VC]] - o[None, :] * o[VC]).std(0) / np.sqrt(len(b))
    return g, ge, len(b)


def postselect(win):
    sel = np.ones(len(bits), bool)
    for n in range(VC - win, VC + win + 1):
        if win > 0:
            sel &= gcol(bits, n) > 0
    return conn(bits[sel])


x = np.arange(NS) - VC
g0, e0, n0 = postselect(0)
g2, e2, n2 = postselect(2)
fig, a = plt.subplots(figsize=(5.0, 3.4), constrained_layout=True)
a.axhline(0, color="0.7", lw=0.6)
a.plot(x, g_t, "k-", lw=1.4, zorder=3, label="exact (MPS)")
a.errorbar(x, g0, yerr=e0, fmt="s", color="0.55", ms=3, lw=0,
           elinewidth=0.8, capsize=1.5, zorder=2,
           label=rf"raw device ({n0//1000}k shots)")
a.errorbar(x, g2, yerr=e2, fmt="o", color="C3", ms=4, lw=0,
           elinewidth=1.0, capsize=2, zorder=4,
           label=rf"gauge post-selected ({n2//1000}k kept)")
a.set_xlim(-9, 9)
a.set_ylim(-0.17, 0.30)
a.set_xlabel(r"$x - x_{\rm insertion}$  (site)")
a.set_ylabel(r"$\langle J^0(x)\,J^0(x_0)\rangle_{\rm conn}$")
a.set_title(r"Connected screening cloud, $t=0$ (ibm_kingston)",
            fontsize=9.5)
a.legend(fontsize=7.5, loc="lower right")
cl = np.abs(x) <= 4
cl[VC] = False          # drop the self-correlation at x=0 (was cl[VC-VC], a no-op)
r0 = np.sqrt(np.mean((g0 - g_t)[cl] ** 2))
r2 = np.sqrt(np.mean((g2 - g_t)[cl] ** 2))
a.text(0.03, 0.06, f"cloud RMS error\nraw {r0:.3f} $\\to$ PS {r2:.3f}",
       transform=a.transAxes, fontsize=7, va="bottom",
       bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", lw=0.6))
fig.savefig("data/hw_cloud.pdf", dpi=200)
fig.savefig("data/hw_cloud.png", dpi=200)
print(f"wrote data/hw_cloud.{{pdf,png}}; cloud RMS {r0:.4f} -> {r2:.4f} "
      f"(truth RMS {np.sqrt(np.mean(g_t[cl]**2)):.4f})")
