"""Diagnose why our exact R(t) disagrees with DHK's published curve: decompose
the two-meson state we prepare into energy eigenstates (full ED at the small
volume) to read off its energy spread sigma_E -- the dephasing rate that sets
R(t)'s decay.  Then compare to DHK's Fig. 10 decay rate.

  PYTHONPATH=. .venv/bin/python scripts/dhk_diagnose.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from htensor import Z2Lattice
from htensor import scattering as sc
from htensor.gaugefixed import PhysicalBasis
from htensor import hamiltonian as ham

M0, G2, ETA = 1.0, 0.6, 1.0

# ---- build the SAME NP=5 two-meson state, full spectrum ----
lat = Z2Lattice(10, pbc=True)
basis = PhysicalBasis(lat)
sel = np.flatnonzero(basis.q == 0)
H = basis.matrix(ham.build_hamiltonian(lat, M0, G2, ETA), sub=sel).real.toarray()
w, V = np.linalg.eigh(H)              # full ED on the Q=0 sector
Evac = w[0]
M = w[1] - w[0]
print(f"Q=0 dim = {H.shape[0]},  M = {M:.4f},  2M = {2*M:.4f}")

gf = sc.gauge_fixed_system(lat, M0, G2, ETA, full_band=True)
s = dict(sigma=7 * np.pi / 20, kbar=2 * np.pi / 5, mu=(2, 7))
packets = [(+s["kbar"], s["sigma"], s["mu"][0]),
           (-s["kbar"], s["sigma"], s["mu"][1])]
psi = sc.two_meson_state(gf, ETA, packets)

# spectral weights of exactly the prepared state
c = V.conj().T @ psi
wt = np.abs(c) ** 2
E = w - Evac
Emean = float(wt @ E)
sigE = float(np.sqrt(wt @ (E - Emean) ** 2))
print(f"<H>-Evac = {Emean:.4f}  (2M={2*M:.4f});  energy spread sigma_E = {sigE:.4f}")
print(f"  => Gaussian-dephasing time 1/sigma_E = {1/sigE:.2f}")
# fraction of weight within +-0.3 of the mean (is it a narrow packet?)
near = float(wt[np.abs(E - Emean) < 0.3].sum())
print(f"  weight within |E-<H>|<0.3 : {near:.2f}")

# reconstruct R(t) from the spectrum to confirm it matches the Krylov file
t = np.linspace(0, 12, 400)
Rrec = np.abs((wt[None, :] * np.exp(-1j * np.outer(t, E))).sum(1)) ** 2
d5 = np.load("data/dhk_Rt_NP5.npz")
print(f"R(t) reconstruction vs Krylov file: max diff "
      f"{np.max(np.abs(np.interp(d5['times'], t, Rrec) - d5['R'])):.2e}")

# ---- DHK NP=13 published ideal decay rate (fit R ~ exp(-(t/tau)^2)) ----
DHK_T = np.arange(0, 21)
DHK_IDEAL = np.array([1.000, 0.962, 0.903, 0.881, 0.886, 0.864, 0.779, 0.695,
                      0.643, 0.617, 0.559, 0.474, 0.397, 0.354, 0.321, 0.277,
                      0.219, 0.173, 0.151, 0.131, 0.108])
# initial-decay energy spread implied by their curve: R''(0) = -2 sigma^2
# estimate from a quadratic fit on t<=6
m = DHK_T <= 6
A = np.polyfit(DHK_T[m] ** 2, np.log(np.clip(DHK_IDEAL[m], 1e-6, None)), 1)
sigE_dhk = float(np.sqrt(-A[0]))
print(f"DHK ideal implied energy spread sigma_E ~ {sigE_dhk:.4f} "
      f"(dephasing time {1/sigE_dhk:.1f})")
print(f"RATIO our/theirs spread = {sigE/sigE_dhk:.1f}x")

# ---- plot ----
fig, (aL, aR) = plt.subplots(1, 2, figsize=(7.6, 3.1), constrained_layout=True)
mk = wt > 1e-4
aL.stem(E[mk], wt[mk], basefmt=" ", linefmt="C0-", markerfmt="C0o")
aL.axvline(2 * M, color="0.4", ls="--", lw=1)
aL.text(2 * M, aL.get_ylim()[1] * 0.9, r" $2M$", color="0.3", fontsize=9)
aL.axvspan(Emean - sigE, Emean + sigE, color="C1", alpha=0.15)
aL.set_xlabel(r"energy above vacuum $E-E_\Omega$")
aL.set_ylabel(r"spectral weight $|\langle E_n|\Psi\rangle|^2$")
aL.set_title(rf"our prepared state ($N_P{{=}}5$): $\sigma_E={sigE:.2f}$",
             fontsize=9.5)

d13 = np.load("data/dhk_Rt_NP13.npz")
aR.plot(DHK_T, DHK_IDEAL, "s", color="0.35", ms=5, label="DHK ideal (their Fig. 10)")
aR.plot(d13["times"], d13["R"], "-", color="C0", lw=1.6,
        label="our exact (Krylov)")
tt = np.linspace(0, 20, 200)
aR.plot(tt, np.exp(-(sigE * tt) ** 2), "--", color="C0", lw=1,
        label=rf"dephasing $e^{{-(\sigma_E t)^2}}$, $\sigma_E={sigE:.2f}$")
aR.plot(tt, np.exp(-(sigE_dhk * tt) ** 2), ":", color="0.35", lw=1.2,
        label=rf"their rate $\sigma_E\approx{sigE_dhk:.2f}$")
aR.set_xlabel("time $t$"); aR.set_ylabel("return probability $R(t)$")
aR.set_ylim(-0.02, 1.02)
aR.legend(fontsize=7.2, loc="upper right")
aR.set_title(r"$R(t)$ at their production volume ($N_P{=}13$)", fontsize=9.5)
fig.savefig("data/dhk_diagnose.pdf", dpi=200)
print("wrote data/dhk_diagnose.pdf")
