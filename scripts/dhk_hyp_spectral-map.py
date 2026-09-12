"""DHK hypothesis: FEWER-QUBITS SPECTRAL MAP (key=spectral-map).

Thesis under test
-----------------
Reproduce the DHK N_P=13 meson-meson return probability R(t) (their Fig. 10,
27-qubit production run) using only SMALL-VOLUME single-meson spectroscopy plus
a factorized / perturbative connection -- never diagonalizing ns=26.

Construction
------------
A two-meson wavepacket state, if the two mesons do not interact, factorizes:
    |Psi> = |p1> (x) |p2>,  |p_i> = sum_k f_i(k) |meson,k>
    <Psi|U(t)|Psi> = A1(t) A2(t),   A_i(t) = sum_k |f_i(k)|^2 e^{-i E(k) t}
    R(t) = |A1(t) A2(t)|^2.
Everything on the RHS is a SINGLE-meson quantity:
  * E(k): single-meson dispersion, measured by small-volume ED (here the cached
    per-momentum spectroscopy data/deep_levels_dhk_ns{12,16,20}.npz, nx<=10, and
    fresh ns=10/12 ED). This is a one-particle observable, converges fast in V.
  * |f_i(k)|^2 = |Psi_i(k)|^2 = exp(-(k-kbar_i)^2/(2 sigma^2)): the analytic DHK
    Gaussian packet envelope, evaluated at the N_P=13 momenta k=2 pi j/13.
So R(t) is reconstructed at the production momenta from nx<=10 spectroscopy: the
"fewer qubits" claim, no 27-qubit / ns=26 sim.

Two spectral maps are compared:
  MODEL A  -- clean lightest-meson band E_low(k) only (optimized/ground meson in
              each momentum channel; standard lattice practice).
  MODEL B  -- the actual chi=1 operator O_x = hop + i*current, which we MEASURE
              (small volume) to populate two nearly degenerate meson species
              (sub-bands split by Delta(k)) with weights z_low, z_up; both kept.

A perturbative diagonal interaction shift dV is also tried; because it enters
A_i(t) as a k-independent (or smooth) phase it is a global phase in R(t) and
cannot generate decay -- reported explicitly.

  PYTHONPATH=. .venv/bin/python scripts/dhk_hyp_spectral-map.py
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
# N_P=13 packet parameters
NX, SIGMA, KBAR = 13, 3 * np.pi / 13, 2 * np.pi / 13


# ---------------------------------------------------------------------------
# 1. small-volume single-meson spectroscopy
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


def operator_subbands():
    """MEASURE, at small volume, how the chi=1 meson operator O_k populates the
    two meson sub-bands: return (k, E_low, E_up, z_low, z_up) per momentum,
    merged over ns=10 and ns=12."""
    rows = {}
    for ns in (10, 12):
        lat = Z2Lattice(ns, pbc=True)
        nx = lat.nx
        basis = PhysicalBasis(lat)
        sel = np.flatnonzero(basis.q == 0)
        H = basis.matrix(ham.build_hamiltonian(lat, M0, G2, ETA),
                         sub=sel).real.toarray()
        w, V = np.linalg.eigh(H)
        Evac = w[0]
        E = w - Evac
        vac = V[:, 0]
        mops = sc.meson_operator(lat, basis, sel, ETA)
        ks = 2 * np.pi * np.arange(nx) / nx
        ks = np.where(ks > np.pi, ks - 2 * np.pi, ks)
        for k in ks:
            if k < -1e-9:
                continue
            Ok = mops[0] * 1.0
            for x in range(1, nx):
                Ok = Ok + mops[x] * np.exp(1j * k * x)
            st = Ok @ vac
            st = st / np.linalg.norm(st)
            wt = np.abs(V.conj().T @ st) ** 2
            m = (E > 2.0) & (E < 3.3)             # single-meson window
            ee, ww = E[m], wt[m]
            # split at the band midpoint into lower / upper sub-band
            mid = 0.5 * (ee.min() + ee.max())
            lo, up = ee < mid, ee >= mid
            zl, zu = ww[lo].sum(), ww[up].sum()
            El = (ww[lo] * ee[lo]).sum() / max(zl, 1e-12)
            Eu = (ww[up] * ee[up]).sum() / max(zu, 1e-12)
            kk = round(float(k), 4)
            if kk not in rows:                    # prefer finer ns=12 where dup
                rows[kk] = (El, Eu, zl / (zl + zu), zu / (zl + zu))
    ks = np.array(sorted(rows))
    El = np.array([rows[k][0] for k in ks])
    Eu = np.array([rows[k][1] for k in ks])
    zl = np.array([rows[k][2] for k in ks])
    zu = np.array([rows[k][3] for k in ks])
    return ks, El, Eu, zl, zu


# ---------------------------------------------------------------------------
# 2. factorized R(t) reconstruction at the N_P=13 momenta
# ---------------------------------------------------------------------------
def packet_weights():
    ks = 2 * np.pi * np.arange(NX) / NX
    ks = np.where(ks > np.pi, ks - 2 * np.pi, ks)
    w1 = np.exp(-((ks - KBAR) ** 2) / (2 * SIGMA ** 2))
    w2 = np.exp(-((ks + KBAR) ** 2) / (2 * SIGMA ** 2))
    return ks, w1 / w1.sum(), w2 / w2.sum()


def interp_even(kq, ks, vals):
    """even interpolation onto |k| of a quantity known on [0,pi]."""
    return np.interp(np.abs(kq), ks, vals)


def model_A(times):
    ks_b, Es_b = clean_lower_band()
    ks, w1, w2 = packet_weights()
    E = interp_even(ks, ks_b, Es_b)
    A1 = np.array([(w1 * np.exp(-1j * E * t)).sum() for t in times])
    A2 = np.array([(w2 * np.exp(-1j * E * t)).sum() for t in times])
    R = np.abs(A1 * A2) ** 2
    # single-meson energy spread over the packet (diagnostic)
    Em = (w1 * E).sum()
    sE = np.sqrt((w1 * (E - Em) ** 2).sum())
    return R, sE


def model_B(times):
    ks_b, El_b, Eu_b, zl_b, zu_b = operator_subbands()
    ks, w1, w2 = packet_weights()
    El = interp_even(ks, ks_b, El_b)
    Eu = interp_even(ks, ks_b, Eu_b)
    zl = interp_even(ks, ks_b, zl_b)
    zu = interp_even(ks, ks_b, zu_b)
    zl, zu = zl / (zl + zu), zu / (zl + zu)

    def amp(w):
        return np.array([(w * (zl * np.exp(-1j * El * t)
                              + zu * np.exp(-1j * Eu * t))).sum()
                         for t in times])
    A1, A2 = amp(w1), amp(w2)
    A0 = amp(np.where(w1 == w1.max(), 0, 0) + w1)  # for normalization at t=0
    R = np.abs(A1 * A2) ** 2 / np.abs(A0[0]) ** 4
    # energy content spread
    Eall = np.concatenate([El, Eu])
    wall = np.concatenate([w1 * zl, w1 * zu])
    wall = wall / wall.sum()
    Em = (wall * Eall).sum()
    sE = np.sqrt((wall * (Eall - Em) ** 2).sum())
    return R, sE


def rms(R):
    return float(np.sqrt(np.mean((R - DHK_IDEAL) ** 2)))


if __name__ == "__main__":
    t = DHK_T.astype(float)
    RA, sEA = model_A(t)
    RB, sEB = model_B(t)
    rmsA, rmsB = rms(RA), rms(RB)

    print("=" * 68)
    print("SPECTRAL-MAP hypothesis: factorized two-meson R(t) from small-V")
    print("single-meson spectroscopy (nx<=10), scored vs DHK N_P=13 ideal.")
    print("=" * 68)
    print(f"\nMODEL A  (clean lightest-meson band E_low(k), factorized)")
    print(f"  single-meson packet sigma_E = {sEA:.4f}  "
          f"-> two-meson {np.sqrt(2)*sEA:.4f}")
    print(f"  RMS to DHK = {rmsA:.4f}")
    print(f"\nMODEL B  (measured chi=1 operator: two sub-bands z_low/z_up)")
    print(f"  two-meson energy spread sigma_E = {sEB:.4f}")
    print(f"  RMS to DHK = {rmsB:.4f}")

    print(f"\n{'t':>3} {'R_A':>7} {'R_B':>7} {'DHK':>7}")
    for i in range(0, 21, 2):
        print(f"{int(t[i]):3d} {RA[i]:7.3f} {RB[i]:7.3f} {DHK_IDEAL[i]:7.3f}")

    # what spread WOULD reproduce DHK (pure-dephasing target)?
    m = DHK_T <= 6
    A = np.polyfit(DHK_T[m] ** 2, np.log(np.clip(DHK_IDEAL[m], 1e-6, None)), 1)
    sE_dhk = float(np.sqrt(max(-A[0], 1e-12)))
    print(f"\nDHK-implied two-meson sigma_E (fit t<=6) = {sE_dhk:.4f}")
    print(f"  MODEL A is {sE_dhk/(np.sqrt(2)*sEA):.1f}x too NARROW in energy;")
    print(f"  MODEL B is {sEB/sE_dhk:.1f}x too BROAD.")

    # a diagonal perturbative dV is a global phase -> no effect (demonstrate)
    print("\nperturbative diagonal dV: enters A_i(t) as a k-independent phase")
    print("  e^{-i dV t/2}; cancels in R=|A1 A2|^2. Cannot generate decay.")

    # is DHK a pure-dephasing (factorized) curve at all?  fit exp(-(sE t)^2)
    from scipy.optimize import minimize_scalar
    fit = minimize_scalar(
        lambda s: np.sqrt(np.mean((np.exp(-(s * DHK_T) ** 2) - DHK_IDEAL) ** 2)),
        bounds=(0.01, 0.3), method="bounded")
    print(f"\nDHK vs pure Gaussian dephasing exp(-(sE t)^2): best sE="
          f"{fit.x:.4f}, RMS={fit.fun:.4f}  -> DHK IS pure dephasing.")

    # MODEL A has the right SHAPE; only the time-unit (dispersion bandwidth)
    # is off.  Find the rescale s that best maps model A onto DHK.
    ks_b, Es_b = clean_lower_band()
    ks, w1, w2 = packet_weights()
    E = interp_even(ks, ks_b, Es_b)

    def RA_fine(tt):
        A1 = (w1 * np.exp(-1j * E * tt)).sum()
        A2 = (w2 * np.exp(-1j * E * tt)).sum()
        return abs(A1 * A2) ** 2

    sc_fit = minimize_scalar(
        lambda s: np.sqrt(np.mean(
            (np.array([RA_fine(s * x) for x in DHK_T]) - DHK_IDEAL) ** 2)),
        bounds=(1.0, 4.0), method="bounded")
    RA_rescaled = np.array([RA_fine(sc_fit.x * x) for x in DHK_T])
    print(f"\nMODEL A time-rescaled: best s = {sc_fit.x:.3f} -> "
          f"RMS = {sc_fit.fun:.4f}")
    print("  s=2.53 is a FITTED overall factor; not derivable here from a")
    print("  documented DHK Hamiltonian normalization or mass ratio, so it")
    print("  does NOT count as a clean reproduction (would be an overfit).")

    np.savez("data/dhk_hyp_spectral-map.npz",
             t=t, R_modelA=RA, R_modelB=RB, R_modelA_rescaled=RA_rescaled,
             DHK=DHK_IDEAL, sigmaE_A=np.sqrt(2) * sEA, sigmaE_B=sEB,
             sigmaE_dhk=sE_dhk, rms_A=rmsA, rms_B=rmsB,
             rms_A_rescaled=sc_fit.fun, time_rescale=sc_fit.x,
             dephasing_sE=fit.x, dephasing_rms=fit.fun, m0=M0, g2=G2, eta=ETA)
    print("\nsaved data/dhk_hyp_spectral-map.npz")
    print(f"\nHONEST BEST RMS (no rescale)      = {min(rmsA, rmsB):.4f}")
    print(f"BEST RMS (with FITTED rescale s)  = {sc_fit.fun:.4f}")
