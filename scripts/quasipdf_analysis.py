"""Quasi-PDF x-space analysis (task 9) from the 101-qubit boosted data.

The complex bilinear is A(z) = <chi^dag(c+z) W chi(c)> = (C_R - i C_I)/2
(connected, vacuum-subtracted).  Even staggered separations z = 2m are
same-sublattice (the gamma^0-like component); the quasi-distribution is
its Fourier transform against the packet momentum P = k0 per spatial
site,

  qtilde(x) = (P / 2 pi) sum_m e^{+i x P m} h(m) w(m),

with w a gentle Gaussian window (support has died by |m| ~ 4 anyway).

  PYTHONPATH=. .venv/bin/python scripts/quasipdf_analysis.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

K0 = 2 * np.pi / 5
d = np.load("data/quasipdf_ns50_k1.26.npz")
zs = d["zs"]
A = 0.5 * ((d["wp_R"] - d["vac_R"]) - 1j * (d["wp_I"] - d["vac_I"]))

# same-sublattice component: even z = 2m -> spatial separation m
ev = zs % 2 == 0
ms = (zs[ev] // 2).astype(int)
h = A[ev]
# m = 0: the z=0 bilinear IS the local density chi^dag chi = J0 on the
# (even) center site -- take its connected value from the production
# one-point data at t=0
p = np.load("data/w_meson_ns50_k1.26_v3.npz")
h0 = complex(p["one_pt_wp"][0, int(d["center"])]
             - p["one_pt_vac"][0, int(d["center"])])
ms = np.concatenate([ms, [0]])
h = np.concatenate([h, [h0]])
order = np.argsort(ms)
ms, h = ms[order], h[order]
# staggered taste phase: the quark modes sit at the staggered zone edge,
# so the physical (taste-singlet) bilinear is (-1)^m h(2m); verified to
# move the FT support into the physical-x region (raw: boundary-peaked)
h = h * (-1.0) ** ms
w = np.exp(-ms**2 / (2 * 5.0**2))
xs = np.linspace(-1.0, 2.0, 301)
qt = np.array([(K0 / (2 * np.pi)) * np.sum(np.exp(1j * x * K0 * ms) * h * w)
               for x in xs])

print("h(m) (windowed, same-sublattice):")
for m, v in zip(ms, h):
    print(f"  m={m:+3d}: {v.real:+.5f} {v.imag:+.5f}j")
n = np.trapezoid(qt.real, xs)
print(f"\nqtilde: integral = {n:.5f}, peak at x = {xs[np.argmax(qt.real)]:.3f}, "
      f"peak value = {qt.real.max():.5f}")
print(f"support (|qt| > 10% peak): x in "
      f"[{xs[np.abs(qt.real) > 0.1*qt.real.max()][0]:.2f}, "
      f"{xs[np.abs(qt.real) > 0.1*qt.real.max()][-1]:.2f}]")

fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), constrained_layout=True)
axes[0].stem(ms, h.real, basefmt=" ", linefmt="C0-", markerfmt="C0o",
             label=r"$\mathrm{Re}\,h(m)$")
axes[0].stem(ms + 0.15, h.imag, basefmt=" ", linefmt="C1-", markerfmt="C1s",
             label=r"$\mathrm{Im}\,h(m)$")
axes[0].set_xlabel("spatial separation $m$")
axes[0].set_ylabel("Wilson-line bilinear $h(m)$")
axes[0].legend(fontsize=9)
axes[1].plot(xs, qt.real, "C0-", lw=1.8, label=r"$\mathrm{Re}\,\tilde q(x)$")
axes[1].plot(xs, qt.imag, "C1--", lw=1.2, label=r"$\mathrm{Im}\,\tilde q(x)$")
axes[1].axvline(0, color="0.6", lw=0.6)
axes[1].axvline(1, color="0.6", lw=0.6, ls=":")
axes[1].set_xlabel(r"$x = k / P$")
axes[1].set_ylabel(r"$\tilde q(x)$  ($P = 2\pi/5$)")
axes[1].legend(fontsize=9)
fig.suptitle("Quasi-PDF from the 101-qubit boosted meson (equal-time, no ancilla)",
             fontsize=10)
fig.savefig("data/quasipdf_x.pdf", dpi=200)
np.savez("data/quasipdf_x.npz", xs=xs, qt=qt, ms=ms, h=h, P=K0)
print("wrote data/quasipdf_x.pdf, data/quasipdf_x.npz")
