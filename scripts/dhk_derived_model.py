"""DHK N_P=13 return probability from small-volume nonperturbative input --
with the momentum convention now DERIVED from their paper (2505.20408), not
fitted.

Convention proof (arXiv:2505.20408 source):
  * Eq. (18): wavepacket momenta live on Gamma-tilde = (pi/N_P){-N_P..N_P-1}
    cap [-pi/2, pi/2) -- STAGGERED-site momenta in the reduced BZ (packet
    centers mu=6,19 only exist on the 26-site staggered ring).  Translation
    by one physical site = two staggered sites => physical-ring COM momentum
    K = 2k, i.e. (kbar, sigma) -> (2 kbar, 2 sigma), center x = mu/2.
  * Their appendix tables (exact/DMRG) confirm it: meson mass 2.7488 = ours
    exactly, and their gaps E(k)-E(0) at k = j*pi/13 match our band-1
    dispersion at K = 2k to 4 decimals (j=2..6).
  * Couplings m_f=1.0, eps=-0.3  <->  our (m0,g2,eta)=(1.0,0.6,1.0).

Model: band-projected factorized evolution.  |Psi_i> = sum_k Psi_i(k)|k>
(their Eq. 18) evolves freely as A_i(t) = sum_k |Psi_i(k)|^2 e^{-iE(K=2k)t};
R(t) = |A_1 A_2|^2, with E(K) the single-meson dispersion MEASURED at small
volume (ns=12..20 <= 41 qubits, merged).  The only physics left out is the
meson-meson interaction during overlap.

Target: DHK's MPS-ideal (TDVP) curve extracted EXACTLY from their figure's
vector data (data/dhk_Rt_true.npz; digitization-free).

  PYTHONPATH=. .venv/bin/python scripts/dhk_derived_model.py
"""
import numpy as np
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "stix",
                     "font.size": 9})

NP, SIGMA, KBAR = 13, 3 * np.pi / 13, 2 * np.pi / 13     # their Table I

# ---- small-volume single-meson dispersion (the nonperturbative input) ----
kpts = {}
for ns in (12, 16, 20):
    d = np.load(f"data/deep_levels_dhk_ns{ns}.npz")
    b = defaultdict(list)
    for g, p in zip(d["gaps"], d["phases"]):
        if g > 0.1:
            b[abs(round(float(p), 4))].append(float(g))
    for k in b:
        kpts[k] = min(kpts.get(k, 99), min(b[k]))
ks = np.array(sorted(kpts)); Es = np.array([kpts[k] for k in ks])


def E_of_K(K):
    Kf = (np.asarray(K) + np.pi) % (2 * np.pi) - np.pi
    return np.interp(np.abs(Kf), ks, Es)


def factorized_R(times, kbar, sigma, doubling):
    """R(t)=|A1 A2|^2 on their momentum grid; doubling=2 is the derived
    convention (K=2k), doubling=1 the naive one."""
    k = np.pi / NP * np.arange(-NP, NP)
    k = k[(k >= -np.pi / 2 - 1e-9) & (k < np.pi / 2 - 1e-9)]   # Gamma-tilde
    w1 = np.exp(-((k - kbar) ** 2) / (2 * sigma ** 2))
    w2 = np.exp(-((k + kbar) ** 2) / (2 * sigma ** 2))
    w1, w2 = w1 / w1.sum(), w2 / w2.sum()
    E = E_of_K(doubling * k)
    A1 = np.array([(w1 * np.exp(-1j * E * t)).sum() for t in times])
    A2 = np.array([(w2 * np.exp(-1j * E * t)).sum() for t in times])
    Em = (w1 * E).sum()
    sE = np.sqrt(2) * np.sqrt((w1 * (E - Em) ** 2).sum())
    return np.abs(A1 * A2) ** 2, float(sE)


true = np.load("data/dhk_Rt_true.npz")
t, R_dhk = true["t"].astype(float), true["R_ideal"]

R2, sE2 = factorized_R(t, KBAR, SIGMA, doubling=2)   # derived
R1, sE1 = factorized_R(t, KBAR, SIGMA, doubling=1)   # naive (wrong)
rms2 = float(np.sqrt(np.mean((R2 - R_dhk) ** 2)))
rms1 = float(np.sqrt(np.mean((R1 - R_dhk) ** 2)))
print(f"derived K=2k : two-meson sigma_E = {sE2:.4f}, RMS vs true DHK = {rms2:.4f}")
print(f"naive   K=k  : two-meson sigma_E = {sE1:.4f}, RMS vs true DHK = {rms1:.4f}")

# our earlier full-Krylov local-operator state (broadband) for context
old = np.load("data/dhk_Rt_NP13.npz")
rms_old = float(np.sqrt(np.mean((np.interp(t, old["times"], old["R"]) - R_dhk) ** 2)))
print(f"local-operator exact Krylov (multi-band state): RMS = {rms_old:.4f}")

fig, ax = plt.subplots(figsize=(4.6, 3.1), constrained_layout=True)
ax.plot(t, R_dhk, "s", color="0.25", ms=5, label="DHK MPS-ideal (exact, from their Fig.)")
tt = np.linspace(0, 20, 300)
Rt2, _ = factorized_R(tt, KBAR, SIGMA, doubling=2)
ax.plot(tt, Rt2, "-", color="#0072B2", lw=1.6,
        label=rf"this work: small-volume $E(K)$ + factorization (RMS {rms2:.3f})")
Rt1, _ = factorized_R(tt, KBAR, SIGMA, doubling=1)
ax.plot(tt, Rt1, "--", color="#D55E00", lw=1.2,
        label=rf"same, without site-doubling $K{{=}}2k$ (RMS {rms1:.2f})")
ax.set_xlabel("time $t$"); ax.set_ylabel(r"$\mathcal{R}(t)$")
ax.set_xlim(0, 20); ax.set_ylim(0, 1.09)
ax.legend(fontsize=7, loc="lower left", framealpha=0.95)
fig.savefig("data/dhk_derived.pdf", dpi=200)
np.savez("data/dhk_derived_model.npz", t=t, R_dhk=R_dhk, R_derived=R2,
         R_naive=R1, rms_derived=rms2, rms_naive=rms1, sigmaE=sE2,
         kbar=KBAR, sigma=SIGMA, doubling=2)
print("wrote data/dhk_derived.pdf, data/dhk_derived_model.npz")
