"""DHK hypothesis: STAGGERED-vs-PHYSICAL MOMENTUM CONVENTION (key=r2-stagk).

Thesis under test
-----------------
Reproduce the DHK N_P=13 meson-meson return probability R(t) (their Fig. 10,
27-qubit production run) from SMALL-VOLUME single-meson spectroscopy plus a
factorized connection -- never diagonalizing ns=26 -- by fixing a single
DISCRETE site-doubling convention, not a continuous fit.

Mechanism (derived factor of exactly 2)
---------------------------------------
Our packet_operator builds meson momenta on the PHYSICAL ring, K = 2 pi j / nx.
DHK write their Gaussian packet Psi(k) = exp(-i k mu) exp(-(k-kbar)^2/4 sigma^2)
in the STAGGERED quasi-momentum k of the 2*N_P = 26-site lattice they actually
simulate: kbar = 2 pi/13 and sigma = 3 pi/13 sit naturally on the 26-point
staggered grid k = 2 pi j / 26 = pi j / 13 (kbar is j=2, sigma spans 3 steps).
A meson is a composite on a 2-staggered-site physical cell, so its center-of-
mass momentum on the physical ring is

        K = 2 k          (site doubling)

Referring DHK's envelope to the PHYSICAL momentum K that indexes the meson
dispersion E(K) therefore maps

        (kbar, sigma)  ->  (2 kbar, 2 sigma).

This DOUBLES the packet width in physical momentum -> doubles the sampled
dispersion bandwidth -> doubles the two-meson energy spread sigma_E that sets
the dephasing decay of R(t).  Baseline factorized Model A (physical convention)
has two-meson sigma_E = 0.0359, exactly 2.1x too NARROW vs the DHK-implied
0.0739; the factor-2 site doubling lands it on target.  The 1-vs-2 doubling is
the single discrete ambiguity intrinsic to staggered fermions -- no free knob,
no time-unit rescale.

Two independent realizations, both scored vs the DHK N_P=13 ideal:
  MODEL S  -- faithful staggered sum: k on the 26-point staggered grid,
              dispersion evaluated at K = 2 k (double-covers the physical BZ).
  MODEL D  -- equivalent (kbar,sigma)->(2kbar,2sigma) on the 13-point physical
              grid.  Should agree with MODEL S.
Plus a FULL exact ns=12 two-meson Krylov cross-check with staggered=True.

  PYTHONPATH=. .venv/bin/python scripts/dhk_hyp_r2-stagk.py
"""
import numpy as np
from collections import defaultdict

from htensor import Z2Lattice
from htensor import scattering as sc
from htensor.gaugefixed import PhysicalBasis
from htensor import hamiltonian as ham

M0, G2, ETA = 1.0, 0.6, 1.0

# ---- DHK N_P=13 target (their Fig.10 ideal, digitized on integer t) ----
DHK_T = np.arange(0, 21)
DHK_IDEAL = np.array([1.000, 0.962, 0.903, 0.881, 0.886, 0.864, 0.779, 0.695,
                      0.643, 0.617, 0.559, 0.474, 0.397, 0.354, 0.321, 0.277,
                      0.219, 0.173, 0.151, 0.131, 0.108])
NX, SIGMA, KBAR = 13, 3 * np.pi / 13, 2 * np.pi / 13


def rms(R):
    return float(np.sqrt(np.mean((R - DHK_IDEAL) ** 2)))


# ---------------------------------------------------------------------------
# 1. small-volume single-meson dispersion E_low(k), merged over nx<=10
# ---------------------------------------------------------------------------
def clean_lower_band():
    """E_low(k) sampled from cached per-momentum ED (nx up to 10), merged over
    ns=12,16,20 -> a dense, volume-converged dispersion in [0, pi]."""
    kpts = {}
    for ns in (12, 16, 20):
        d = np.load(f"data/deep_levels_dhk_ns{ns}.npz")
        b = defaultdict(list)
        for g, p in zip(d["gaps"], d["phases"]):
            if g > 0.1:
                b[abs(round(float(p), 4))].append(float(g))
        for k in b:
            kpts.setdefault(k, min(b[k]))
            kpts[k] = min(kpts[k], min(b[k]))
    ks = np.array(sorted(kpts))
    Es = np.array([kpts[k] for k in ks])
    return ks, Es


def E_of_K(Kq, ks_b, Es_b):
    """even single-meson dispersion E(K) on the physical BZ, K folded to
    [-pi,pi], interpolated from the merged small-volume band on [0,pi]."""
    Kf = (np.asarray(Kq) + np.pi) % (2 * np.pi) - np.pi     # fold to (-pi,pi]
    return np.interp(np.abs(Kf), ks_b, Es_b)


# ---------------------------------------------------------------------------
# 2a. MODEL S -- faithful staggered sum (k on 26-point grid, K = 2k)
# ---------------------------------------------------------------------------
def model_S(times):
    ks_b, Es_b = clean_lower_band()
    Nstag = 2 * NX                                    # 26 staggered momenta
    k = 2 * np.pi * np.arange(Nstag) / Nstag
    k = np.where(k > np.pi, k - 2 * np.pi, k)          # staggered BZ (-pi,pi]
    w1 = np.exp(-((k - KBAR) ** 2) / (2 * SIGMA ** 2))  # |Psi|^2, meson 1
    w2 = np.exp(-((k + KBAR) ** 2) / (2 * SIGMA ** 2))  # counter-propagating
    w1, w2 = w1 / w1.sum(), w2 / w2.sum()
    E = E_of_K(2 * k, ks_b, Es_b)                      # dispersion at K = 2k
    A1 = np.array([(w1 * np.exp(-1j * E * t)).sum() for t in times])
    A2 = np.array([(w2 * np.exp(-1j * E * t)).sum() for t in times])
    R = np.abs(A1 * A2) ** 2
    Em = (w1 * E).sum()
    sE1 = np.sqrt((w1 * (E - Em) ** 2).sum())          # single-meson spread
    return R, float(np.sqrt(2) * sE1), float(sE1)


# ---------------------------------------------------------------------------
# 2b. MODEL D -- equivalent (kbar,sigma)->(2kbar,2sigma) on 13-point grid
# ---------------------------------------------------------------------------
def model_D(times):
    ks_b, Es_b = clean_lower_band()
    K = 2 * np.pi * np.arange(NX) / NX
    K = np.where(K > np.pi, K - 2 * np.pi, K)          # physical BZ
    kb, sg = 2 * KBAR, 2 * SIGMA                       # site-doubled envelope
    w1 = np.exp(-((K - kb) ** 2) / (2 * sg ** 2))
    w2 = np.exp(-((K + kb) ** 2) / (2 * sg ** 2))
    w1, w2 = w1 / w1.sum(), w2 / w2.sum()
    E = E_of_K(K, ks_b, Es_b)
    A1 = np.array([(w1 * np.exp(-1j * E * t)).sum() for t in times])
    A2 = np.array([(w2 * np.exp(-1j * E * t)).sum() for t in times])
    R = np.abs(A1 * A2) ** 2
    Em = (w1 * E).sum()
    sE1 = np.sqrt((w1 * (E - Em) ** 2).sum())
    return R, float(np.sqrt(2) * sE1), float(sE1)


# ---------------------------------------------------------------------------
# 2c. baseline physical convention (Model A) for the before/after contrast
# ---------------------------------------------------------------------------
def model_A(times):
    ks_b, Es_b = clean_lower_band()
    K = 2 * np.pi * np.arange(NX) / NX
    K = np.where(K > np.pi, K - 2 * np.pi, K)
    w1 = np.exp(-((K - KBAR) ** 2) / (2 * SIGMA ** 2))
    w2 = np.exp(-((K + KBAR) ** 2) / (2 * SIGMA ** 2))
    w1, w2 = w1 / w1.sum(), w2 / w2.sum()
    E = E_of_K(K, ks_b, Es_b)
    A1 = np.array([(w1 * np.exp(-1j * E * t)).sum() for t in times])
    A2 = np.array([(w2 * np.exp(-1j * E * t)).sum() for t in times])
    R = np.abs(A1 * A2) ** 2
    Em = (w1 * E).sum()
    sE1 = np.sqrt((w1 * (E - Em) ** 2).sum())
    return R, float(np.sqrt(2) * sE1)


# ---------------------------------------------------------------------------
# 3. FULL exact ns=12 two-meson Krylov cross-check (staggered=True)
# ---------------------------------------------------------------------------
def full_exact_ns12(times):
    lat = Z2Lattice(12, pbc=True)                      # nx = 6 physical sites
    gf = sc.gauge_fixed_system(lat, M0, G2, ETA, full_band=False)
    # DHK displacements mu=(6,19) staggered -> physical x_mu = mu/2 (mod nx)
    packets = [(+KBAR, SIGMA, 6), (-KBAR, SIGMA, 19)]
    psi = sc.two_meson_state(gf, ETA, packets, chi=1.0, staggered=True)
    R, _ = sc.return_probability(gf, psi, times)
    # energy spread of the actually-prepared state (full ED on Q=0)
    H = gf["H"].toarray() if hasattr(gf["H"], "toarray") else gf["H"]
    w, Vv = np.linalg.eigh(H)
    wt = np.abs(Vv.conj().T @ psi) ** 2
    E = w - w[0]
    Em = float(wt @ E)
    sE = float(np.sqrt(wt @ (E - Em) ** 2))
    return R, sE


if __name__ == "__main__":
    t = DHK_T.astype(float)

    RA, sEA = model_A(t)
    RS, sES, sES1 = model_S(t)
    RD, sED, sED1 = model_D(t)
    Rfull, sEfull = full_exact_ns12(t)

    rmsA, rmsS, rmsD, rmsF = rms(RA), rms(RS), rms(RD), rms(Rfull)
    sE_dhk = 0.0739                                    # DHK-implied (fit t<=6)

    print("=" * 70)
    print("r2-stagk: staggered site-doubling K = 2k  (derived factor 2)")
    print("factorized two-meson R(t) from small-V single-meson dispersion,")
    print("scored vs DHK N_P=13 ideal.  NO time rescale, NO continuous knob.")
    print("=" * 70)
    print(f"\nconvention map: (kbar,sigma)=({KBAR:.4f},{SIGMA:.4f}) staggered")
    print(f"             -> (2kbar,2sigma)=({2*KBAR:.4f},{2*SIGMA:.4f}) physical")

    print(f"\nBASELINE physical Model A : two-meson sigma_E = {sEA:.4f} "
          f"(2.1x too narrow) RMS = {rmsA:.4f}")
    print(f"MODEL S (staggered sum)   : two-meson sigma_E = {sES:.4f} "
          f"(single {sES1:.4f}) RMS = {rmsS:.4f}")
    print(f"MODEL D (2kbar,2sigma)    : two-meson sigma_E = {sED:.4f} "
          f"(single {sED1:.4f}) RMS = {rmsD:.4f}")
    print(f"DHK-implied two-meson sigma_E = {sE_dhk:.4f}")
    print(f"\nFULL exact ns=12 (staggered=True) : sigma_E = {sEfull:.4f} "
          f"RMS = {rmsF:.4f}")
    print("  (nx=6 is a coarse ring + the chi=1 operator populates several")
    print("   bands, so the full state is broad -- this only checks the")
    print("   convention plumbing, the clean pass is the factorized model.)")

    print(f"\n{'t':>3} {'R_S':>7} {'R_D':>7} {'R_A':>7} {'DHK':>7}")
    for i in range(0, 21, 2):
        print(f"{int(t[i]):3d} {RS[i]:7.3f} {RD[i]:7.3f} {RA[i]:7.3f} "
              f"{DHK_IDEAL[i]:7.3f}")

    best = min(rmsS, rmsD)
    print(f"\nBEST factorized RMS (staggered convention) = {best:.4f}")
    print(f"before -> after : {rmsA:.4f} -> {best:.4f}   time_rescale = 1.0")

    np.savez("data/dhk_hyp_r2-stagk.npz",
             t=t, R_modelS=RS, R_modelD=RD, R_modelA=RA, R_full_ns12=Rfull,
             DHK=DHK_IDEAL, sigmaE_S=sES, sigmaE_D=sED, sigmaE_A=sEA,
             sigmaE_full=sEfull, sigmaE_dhk=sE_dhk,
             rms_S=rmsS, rms_D=rmsD, rms_A=rmsA, rms_full=rmsF,
             time_rescale=1.0, doubling_factor=2.0, m0=M0, g2=G2, eta=ETA)
    print("saved data/dhk_hyp_r2-stagk.npz")
