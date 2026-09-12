"""Final DHK N_P=13 R(t) comparison figure (honest synthesis).

After an exhaustive round of physically-motivated hypotheses (echo, packet
width, dispersion, momentum/vacuum projection, single-meson survival, a
factorized spectral map, staggered-momentum doubling, Hamiltonian
normalization, sub-band weighting, separation extrapolation), NONE
reproduces DHK's Fig. 10 R(t) with a first-principles, non-fit small-volume
construction.

Two independent facts settle it:
  (1) Our genuine fewer-qubit result -- the EXACT Krylov evolution of the
      faithful two-meson state at DHK's own N_P=13 packet parameters
      (data/dhk_Rt_NP13.npz) -- craters and revives (RMS ~0.45 to DHK),
      because a small periodic box has a discrete meson spectrum.
  (2) DHK's IDEAL curve is, to within digitization error, a single Gaussian
      dephasing R(t)=exp(-sigma_E^2 t^2) with sigma_E ~ 0.084 (RMS 0.048).
      R(t) depends only on the product sigma_E*t, so any construction that
      hits RMS<0.1 does so by SETTING sigma_E to DHK's width -- a
      one-parameter fit to their curve, not a derived quantity. The
      "factor-of-2" / "time-rescale" hypotheses are all degenerate
      relabelings of that single fit and were rejected in verification.

This figure states that honestly: the target, our exact small-volume
attempt, and the Gaussian-dephasing characterization of DHK's curve.

  PYTHONPATH=. .venv/bin/python scripts/dhk_reproduced.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "dejavuserif",
    "font.size": 11,
})

DHK_T = np.arange(0, 21)
DHK_IDEAL = np.array([1.000, 0.962, 0.903, 0.881, 0.886, 0.864, 0.779, 0.695,
                      0.643, 0.617, 0.559, 0.474, 0.397, 0.354, 0.321, 0.277,
                      0.219, 0.173, 0.151, 0.131, 0.108])


def rms(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


# (1) our genuine fewer-qubit result: exact Krylov, DHK N_P=13 packet params
r = np.load("data/dhk_Rt_NP13.npz")
R_exact = np.interp(DHK_T, r["times"], r["R"])
rms_exact = rms(R_exact, DHK_IDEAL)

# (2) DHK's curve IS single-Gaussian dephasing: fit sigma_E on the monotone
#     early window t=1..8, then evaluate everywhere. This is a FIT to DHK,
#     shown only to characterize their curve -- it is not a prediction.
t = DHK_T.astype(float)
a = np.linalg.lstsq((t[1:9] ** 2)[:, None], np.log(DHK_IDEAL[1:9]),
                    rcond=None)[0][0]
sigma_E = float(np.sqrt(-a))
R_gauss = np.exp(-sigma_E ** 2 * t ** 2)
rms_gauss = rms(R_gauss, DHK_IDEAL)

print(f"exact small-V Krylov (N_P=13) RMS to DHK  = {rms_exact:.4f}")
print(f"single-Gaussian dephasing sigma_E         = {sigma_E:.4f}")
print(f"  (fit to DHK t=1..8)  RMS to DHK          = {rms_gauss:.4f}")

fig, ax = plt.subplots(figsize=(6.4, 4.4), constrained_layout=True)

ax.plot(DHK_T, DHK_IDEAL, "s", color="0.15", ms=6, zorder=5,
        label="DHK $N_P{=}13$ MPS-ideal (27 qubits) [target]")

tt = np.linspace(0, 20, 400)
ax.plot(tt, np.exp(-sigma_E ** 2 * tt ** 2), "--", color="C3", lw=1.8,
        label=(r"single-Gaussian dephasing $e^{-\sigma_E^2 t^2}$"
               + f",  $\\sigma_E{{=}}{sigma_E:.3f}$ (fit)"
               + f"\n     RMS $= {rms_gauss:.3f}$"))

ax.plot(DHK_T, R_exact, "-o", color="C0", lw=2.0, ms=3.5,
        label=(r"this work: exact small-$V$ Krylov, $n_s{=}26$ packet params"
               + f"\n     RMS $= {rms_exact:.3f}$"))

ax.axhline(0, color="0.85", lw=0.6)
ax.set_xlabel(r"time $t$ (lattice units)")
ax.set_ylabel(r"return probability $R(t)=|\langle\Psi|U(t)|\Psi\rangle|^2$")
ax.set_title(r"Meson-meson collision $R(t)$: DHK vs. small-volume",
             fontsize=11)
ax.legend(fontsize=8.2, loc="upper right", framealpha=0.95)
ax.set_xlim(0, 20)
ax.set_ylim(-0.02, 1.03)

# honest caption inside the axes
ax.text(0.5, 0.06,
        "No first-principles small-volume construction reproduces the target:\n"
        r"the exact result revives (discrete spectrum); matching DHK requires"
        "\n"
        r"fitting the single width $\sigma_E$ to their own curve.",
        transform=ax.transAxes, fontsize=7.6, color="0.3", ha="center",
        va="bottom")

fig.savefig("data/dhk_reproduced.pdf", dpi=200)
print("wrote data/dhk_reproduced.pdf")
