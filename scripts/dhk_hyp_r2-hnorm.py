"""DHK hypothesis: HAMILTONIAN TIME-UNIT / NORMALIZATION (key=r2-hnorm).

Thesis under test
-----------------
The factorized "spectral-map" reconstruction of DHK's N_P=13 return
probability (their Fig. 10) has the RIGHT SHAPE but the wrong time unit: a
near-miss fit needs a global rescale s in [2.06, 2.53]
    (s=2.06 -> RMS 0.096 ; s=2.53 -> RMS 0.039).
HYPOTHESIS: that s is not a free fit but a DOCUMENTED ratio between DHK's
Hamiltonian / energy normalization and ours -- e.g. a factor-2 kinetic
amplitude (their w=1/2 vs our 1/4) and/or a meson-mass ratio M_DHK/M_ours.
If s in [2.06,2.53] is derivable from a stated coefficient or mass ratio and
yields RMS<0.1, the hypothesis passes.

What the paper actually says (arXiv:2505.20408, read verbatim)
--------------------------------------------------------------
* Eq. (5), their Hamiltonian:
      a H = 1/2 sum_n (xi_n^dag  sigma~_n^x  xi_{n+a} + H.c.)
            + a m_f sum_n (-1)^{n/a} xi_n^dag xi_n
            + a eps  sum_n sigma~_n^z .
* Sec. IV, production couplings (verbatim, line "We further set..."):
      m_f = 1.0 ,  eps = -0.3   ("the parameters used in our previous work").
  (The m_f=0.1,0.3 numbers elsewhere are an *ansatz-fidelity scan*, Fig. 11,
   NOT the collision run.)
* Eq. (38), wavepacket:  Psi_i(k) = N exp(-i k mu_i) exp(-(k-kbar_i)^2/(4 sigma^2)).
* N_P=13 packet (Sec. IV / App.):  sigma = 3 pi/13 ,  kbar = 2 pi/13.
* Fig. 10 caption:  R(t)=|<Psi1,Psi2|U(t)|Psi1,Psi2>|^2 vs time t, Trotter
  step dt=0.25, integer t plotted.  No rescaled / non-standard time unit.

Term-by-term against our H (htensor/hamiltonian.py)
---------------------------------------------------
  kinetic : DHK  1/2 (xi^dag sigma~^x xi + H.c.)  = JW =>  (1/4)(XX+YY) sigma^z
            ours (eta/4)(XX+YY) sigma^z with eta=1 =>  (1/4)(XX+YY) sigma^z   [EQUAL]
  mass    : DHK  a m_f (-1)^n xi^dag xi  = -(m_f/2)(-1)^n Z + const
            ours -(m0/2)(-1)^n Z  with m0=1.0=m_f                            [EQUAL]
  gauge   : DHK  a eps sigma~^z  (eps=-0.3)   ours (g2/2) sigma^x (g2=0.6=>0.3)
            sigma^z<->sigma^x and sign(eps) are a basis choice; |coeff|=0.3   [EQUAL]

=> DHK's Hamiltonian, couplings, wavepacket profile, and time variable are
   coefficient-for-coefficient IDENTICAL to ours.  The energy-scale ratio is
   1, the meson-mass ratio M_DHK/M_ours is 1, so the ONLY time rescale that is
   physically derivable is  s = 1.0.  There is NO documented factor-2 (our
   1/4 already IS w=1/2 for the fermion bilinear) and no mass-ratio factor.

This script therefore (a) recomputes the meson mass + dispersion bandwidth in
DHK's (== our) units at ns<=12 to show M_DHK/M_ours=1 first-principles, and
(b) scores the factorized model-A R(t) at the DERIVED s=1.0 vs DHK.  The
s~2.5 that "works" is exhibited only to show it maps to no stated quantity.

  PYTHONPATH=. .venv/bin/python scripts/dhk_hyp_r2-hnorm.py
"""
import numpy as np
from collections import defaultdict
from scipy.optimize import minimize_scalar

from htensor import Z2Lattice
from htensor import scattering as sc
from htensor.gaugefixed import PhysicalBasis
from htensor import hamiltonian as ham

M0, G2, ETA = 1.0, 0.6, 1.0            # == DHK m_f=1.0, |eps|=0.3, w=1/2

# DHK N_P=13 target (Fig.10 ideal / TDVP, digitized on integer t)
DHK_T = np.arange(0, 21)
DHK_IDEAL = np.array([1.000, 0.962, 0.903, 0.881, 0.886, 0.864, 0.779, 0.695,
                      0.643, 0.617, 0.559, 0.474, 0.397, 0.354, 0.321, 0.277,
                      0.219, 0.173, 0.151, 0.131, 0.108])
NX, SIGMA, KBAR = 13, 3 * np.pi / 13, 2 * np.pi / 13


# --------------------------------------------------------------------------
# (a) meson mass + single-meson dispersion in DHK's (== our) units, ns<=12
# --------------------------------------------------------------------------
def dispersion_dhk_units():
    """Diagonalize our H (identical to DHK Eq.5) at ns=10,12 and read off the
    lightest-meson dispersion E(k) and the meson mass M = E(k=0), by PROJECTING
    the momentum-k meson operator O_k|vac> onto eigenstates.  This is the band
    in *DHK's units* because the coefficient map above is the identity."""
    out = {}
    for ns in (10, 12):
        lat = Z2Lattice(ns, pbc=True)
        nx = lat.nx
        basis = PhysicalBasis(lat)
        sel = np.flatnonzero(basis.q == 0)
        H = basis.matrix(ham.build_hamiltonian(lat, M0, G2, ETA),
                         sub=sel).real.toarray()
        w, V = np.linalg.eigh(H)
        E = w - w[0]
        vac = V[:, 0]
        M = float(E[1])                        # lightest meson = mass gap
        mops = sc.meson_operator(lat, basis, sel, ETA)
        kgrid = 2 * np.pi * np.arange(nx) / nx
        kgrid = np.where(kgrid > np.pi, kgrid - 2 * np.pi, kgrid)
        ks, Es = [], []
        for k in sorted({round(abs(k), 6) for k in kgrid}):
            Ok = sum(mops[x] * np.exp(1j * k * x) for x in range(nx))
            st = Ok @ vac
            st = st / np.linalg.norm(st)
            wt = np.abs(V.conj().T @ st) ** 2
            m = (E > 2.0) & (E < 3.3)          # single-meson window
            ee, ww = E[m], wt[m]
            ks.append(k)
            Es.append((ww * ee).sum() / ww.sum())   # weighted band energy at k
        out[ns] = (np.array(ks), np.array(Es), M)
    return out


def clean_lower_band():
    """Dense volume-converged E_low(k) on [0,pi] from cached small-nx (nx<=10)
    per-momentum spectroscopy -- the "fewer qubits" dispersion, no ns=26."""
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


# --------------------------------------------------------------------------
# (b) factorized model-A R(t) with a global time rescale s
# --------------------------------------------------------------------------
def packet_weights():
    ks = 2 * np.pi * np.arange(NX) / NX
    ks = np.where(ks > np.pi, ks - 2 * np.pi, ks)
    w1 = np.exp(-((ks - KBAR) ** 2) / (2 * SIGMA ** 2))
    w2 = np.exp(-((ks + KBAR) ** 2) / (2 * SIGMA ** 2))
    return ks, w1 / w1.sum(), w2 / w2.sum()


def model_A(times, s=1.0):
    ks_b, Es_b = clean_lower_band()
    ks, w1, w2 = packet_weights()
    E = np.interp(np.abs(ks), ks_b, Es_b)
    A1 = np.array([(w1 * np.exp(-1j * E * (s * t))).sum() for t in times])
    A2 = np.array([(w2 * np.exp(-1j * E * (s * t))).sum() for t in times])
    R = np.abs(A1 * A2) ** 2
    Em = (w1 * E).sum()
    sE1 = np.sqrt((w1 * (E - Em) ** 2).sum())          # single-meson spread
    return R, np.sqrt(2) * sE1                          # two-meson spread


def rms(R):
    return float(np.sqrt(np.mean((R - DHK_IDEAL) ** 2)))


if __name__ == "__main__":
    print("=" * 72)
    print("r2-hnorm: is the s~2.5 time rescale a DERIVED DHK/our normalization?")
    print("=" * 72)

    # ---- (a) meson mass & bandwidth in DHK's (==our) units -----------------
    disp = dispersion_dhk_units()
    M_ours = disp[12][2]
    ks12, Es12, _ = disp[12]
    bw = float(Es12.max() - Es12.min())
    print("\n[a] meson dispersion recomputed under DHK Eq.(5) coefficients")
    print(f"    (our H with eta=1 IS DHK's 1/2(xi^dag sigma xi+h.c.) after JW)")
    for ns in (10, 12):
        ks, Es, M = disp[ns]
        print(f"    ns={ns}: M=E(k=0) = {M:.4f} ; band E(k) in "
              f"[{Es.min():.4f}, {Es.max():.4f}]  (bandwidth {Es.max()-Es.min():.4f})")
    # DHK use identical couplings -> their meson mass is the SAME number
    M_DHK = M_ours
    print(f"    => M_DHK / M_ours = {M_DHK/M_ours:.4f}   (couplings identical)")
    print(f"    => energy-scale ratio (kinetic w: 1/2 vs 1/2)         = 1.0000")
    print(f"    => the ONLY first-principles time rescale is  s = 1.0")

    # ---- (b) model-A R(t) at the DERIVED s=1.0 -----------------------------
    t = DHK_T.astype(float)
    R1, sE2 = model_A(t, s=1.0)
    rms1 = rms(R1)
    print("\n[b] factorized model-A R(t) at the DERIVED s = 1.0")
    print(f"    two-meson energy spread sigma_E = {sE2:.4f} "
          f"(DHK-implied ~0.074 -> we are {0.0739/sE2:.1f}x too NARROW)")
    print(f"    RMS to DHK = {rms1:.4f}   (DERIVED, physically defensible)")

    # ---- what fit WOULD close the gap, and does it map to anything? ---------
    fit = minimize_scalar(
        lambda s: rms(model_A(t, s=s)[0]), bounds=(1.0, 4.0), method="bounded")
    s_fit = float(fit.x)
    Rfit, _ = model_A(t, s=s_fit)
    print("\n[c] free fit (NOT derived): best s and its provenance check")
    print(f"    best-fit s = {s_fit:.3f} -> RMS = {fit.fun:.4f}")
    print(f"    provenance: energy-scale ratio=1, mass ratio=1, kinetic w ratio=1")
    print(f"    -> s={s_fit:.2f} corresponds to NO documented DHK normalization.")
    print(f"       It would require DHK energies {s_fit:.2f}x ours, contradicted")
    print(f"       by the byte-identical Hamiltonian Eq.(5). => a FIT/fudge.")

    R2 = model_A(t, s=2.06)[0]                    # low end of the fit window
    print(f"    (s=2.06 -> RMS {rms(R2):.4f}; s=2.53 -> RMS {rms(model_A(t,2.53)[0]):.4f})")

    # ---- table -------------------------------------------------------------
    print(f"\n{'t':>3} {'R(s=1)':>8} {'R(s=fit)':>9} {'DHK':>7}")
    for i in range(0, 21, 4):
        print(f"{int(t[i]):3d} {R1[i]:8.3f} {Rfit[i]:9.3f} {DHK_IDEAL[i]:7.3f}")

    np.savez("data/dhk_hyp_r2-hnorm.npz",
             t=t, R_s1=R1, R_sfit=Rfit, DHK=DHK_IDEAL,
             s_derived=1.0, rms_derived=rms1,
             s_fit=s_fit, rms_fit=fit.fun,
             M_ours=M_ours, M_DHK=M_DHK, mass_ratio=M_DHK / M_ours,
             bandwidth_dhk_units=bw, sigmaE_two_meson=sE2,
             m0=M0, g2=G2, eta=ETA,
             note="DHK Eq.5 coeffs identical to ours; derived s=1; s~2.5 is a fit")
    print("\nsaved data/dhk_hyp_r2-hnorm.npz")

    print("\n" + "=" * 72)
    print("VERDICT: DHK's H/couplings/packet/time-unit are identical to ours")
    print(f"  (mf=1.0, eps=-0.3, w=1/2, dt=0.25).  Derived s=1.0 -> RMS {rms1:.3f}.")
    print(f"  The s~2.5 that reaches RMS {fit.fun:.3f} is a FIT with no documented")
    print("  provenance (energy & mass ratios are both 1).  HYPOTHESIS FAILS:")
    print("  the mismatch is NOT a Hamiltonian time-unit normalization.")
    print("=" * 72)
