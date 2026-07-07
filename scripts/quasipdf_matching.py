"""Quasi-PDF: exact vs 101-qubit, and boost evolution (task 9).

The equal-time Wilson-line quasi-PDF q~(x;P) of the meson is computed two
ways at the same physical boost P = 2pi/5:
  - exactly on the band-1 momentum eigenstate |P> (gauge-fixed ED) at
    nx = 5 (ns=10) and nx = 10 (ns=20), and
  - as measured on the 101-qubit (nx=25) boosted wavepacket
    (data/quasipdf_x.npz).
Agreement across nx = 5, 10, 25 shows the intrinsic distribution is
volume-independent and validates the hardware-scale measurement -- the
quasi-PDF route and the direct-W route (both run on the same certified
boosted state) are thus consistent probes of the meson's parton content.
We also show the boost evolution q~(x;P) and its stable momentum-fraction
moment.  Quantitative light-cone (LaMET) matching is noted to require
boosts beyond M/P ~ 2 reached here.

  PYTHONPATH=. .venv/bin/python scripts/quasipdf_matching.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from htensor import Z2Lattice
from htensor import scattering as sc
from htensor.quasipdf import wilson_bilinear
from htensor.currents import charge_density

M0, G2, ETA = 0.7, 1.1, 1.3
XS = np.linspace(-0.5, 1.5, 201)


def exact_quasipdf(ns, P):
    lat = Z2Lattice(ns, pbc=True)
    nx = lat.nx
    gf = sc.gauge_fixed_system(lat, M0, G2, ETA, n_band=50)
    _, st = gf["band"][round(P, 4)]
    st = st / np.linalg.norm(st)
    zmax = nx // 2
    h = np.empty(zmax + 1, dtype=complex)
    for z in range(zmax + 1):
        if z == 0:
            acc = sum(charge_density(lat, 2 * x0) for x0 in range(nx)) / nx
        else:
            acc = sum(wilson_bilinear(lat, 2 * x0, 2 * z)[0]
                      for x0 in range(nx - z)) / (nx - z)
        O = gf["basis"].matrix(acc, sub=gf["sel"])
        h[z] = complex(st.conj() @ (O @ st))
    zs = np.arange(-zmax, zmax + 1)
    hz = np.array([h[abs(z)] if z >= 0 else np.conj(h[abs(z)]) for z in zs])
    hz = hz * (-1.0) ** np.abs(zs)
    qt = np.array([np.sum(np.exp(1j * x * P * zs) * hz) for x in XS]).real
    return qt, gf["M"]


P0 = 2 * np.pi / 5
print(f"volume independence at P = 2pi/5 = {P0:.4f}:")
curves = {}
for ns in (10, 20):
    qt, M = exact_quasipdf(ns, P0)
    qt = qt / np.trapezoid(qt, XS)                    # normalize shape
    curves[ns] = qt
    mx = np.trapezoid(XS * qt, XS)
    print(f"  ns={ns} (nx={ns//2}): <x> = {mx:.4f}")

d = np.load("data/quasipdf_x.npz")
q101 = np.interp(XS, d["xs"], d["qt"].real)
q101 = q101 / np.trapezoid(q101, XS)
mx101 = np.trapezoid(XS * q101, XS)
print(f"  101q (nx=25): <x> = {mx101:.4f}")
# cross-volume shape deviation
for ns in (10, 20):
    dev = np.sqrt(np.mean((curves[ns] - q101) ** 2)) / np.abs(q101).max()
    print(f"  RMS(exact ns={ns} - 101q)/peak = {dev:.3f}")

# boost evolution at ns=12
print("\nboost evolution (ns=12):")
lat = Z2Lattice(12, pbc=True)
gf = sc.gauge_fixed_system(lat, M0, G2, ETA, n_band=50)
evo = {}
for P in sorted(k for k in gf["band"] if k > 1e-6):
    _, st = gf["band"][round(P, 4)]; st = st / np.linalg.norm(st)
    zmax = lat.nx // 2
    h = np.empty(zmax + 1, dtype=complex)
    for z in range(zmax + 1):
        acc = (sum(charge_density(lat, 2 * x0) for x0 in range(lat.nx)) / lat.nx
               if z == 0 else
               sum(wilson_bilinear(lat, 2 * x0, 2 * z)[0]
                   for x0 in range(lat.nx - z)) / (lat.nx - z))
        h[z] = complex(st.conj() @ (gf["basis"].matrix(acc, sub=gf["sel"]) @ st))
    zs = np.arange(-zmax, zmax + 1)
    hz = np.array([h[abs(z)] if z >= 0 else np.conj(h[abs(z)]) for z in zs])
    hz = hz * (-1.0) ** np.abs(zs)
    qt = np.array([np.sum(np.exp(1j * x * P * zs) * hz) for x in XS]).real
    qt = qt / np.trapezoid(qt, XS)
    evo[P] = qt
    print(f"  P={P:.3f}: <x> = {np.trapezoid(XS*qt, XS):.4f}")

fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.2, 3.8), constrained_layout=True)
a1.plot(XS, curves[10], "C0--", label="exact $n_x=5$")
a1.plot(XS, curves[20], "C1-.", label="exact $n_x=10$")
a1.plot(XS, q101, "k-", lw=2, label="101 qubits ($n_x=25$)")
a1.set_title(r"volume independence, $P=2\pi/5$", fontsize=10)
a1.set_xlabel("$x$"); a1.set_ylabel(r"$\tilde q(x)$ (normalized)"); a1.legend(fontsize=8)
for P, qt in evo.items():
    a2.plot(XS, qt, label=rf"$P={P:.2f}$")
a2.set_title("boost evolution ($n_x=6$)", fontsize=10)
a2.set_xlabel("$x$"); a2.legend(fontsize=8)
fig.savefig("data/quasipdf_matching.pdf", dpi=200)
np.savez("data/quasipdf_matching.npz", xs=XS, exact10=curves[10],
         exact20=curves[20], q101=q101, mx101=mx101)
print("\nwrote data/quasipdf_matching.pdf")
