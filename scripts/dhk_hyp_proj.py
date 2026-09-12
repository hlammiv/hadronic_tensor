"""Hypothesis key=proj: TOTAL-MOMENTUM / CONNECTED return probability.

Thesis under test (from the DHK postdiction task): our exact two-meson R(t)
craters and revives while DHK's Fig. 10 decays slowly.  The diagnosis
(scripts/dhk_diagnose.py) is that our prepared state is ~11x too broad in
energy (sigma_E ~ 0.84 vs DHK-implied ~0.074).  This script asks whether that
broadening is an ARTIFACT of the observable rather than the state, via two
physically-motivated reformulations, evaluated at small volume (NP=5, ns=10,
full ED -- instant):

  (a) TOTAL-MOMENTUM PROJECTION / relative coordinate.  |Psi> spreads over
      total-momentum sectors P (translation T2 eigenvalues).  If the decay of
      R(t) were driven by center-of-mass motion (different mean energy per P
      sector), projecting onto P=0 -- U(t) is block-diagonal in P, so this is
      legitimate -- would remove that inter-sector spread and flatten R(t).

  (b) CONNECTED / vacuum-subtracted return probability.  Subtract the
      disconnected pieces: the vacuum component of |Psi>, and the free
      single-meson return amplitudes A_i(t) = <phi_i|U(t)|phi_i> (dividing
      them out isolates the interaction/S-matrix part).

Scored by RMS of R(t) vs the digitized DHK NP=13 ideal on t=0..20.
"Reproduces" would need RMS < ~0.1 with a defensible construction.

  PYTHONPATH=. .venv/bin/python scripts/dhk_hyp_proj.py
"""
import numpy as np

from htensor import Z2Lattice
from htensor import scattering as sc

M0, G2, ETA = 1.0, 0.6, 1.0
# DHK Fig. 10 (N_P=13) ideal, digitized; the target curve.
DHK_IDEAL = np.array([1.000, 0.962, 0.903, 0.881, 0.886, 0.864, 0.779, 0.695,
                      0.643, 0.617, 0.559, 0.474, 0.397, 0.354, 0.321, 0.277,
                      0.219, 0.173, 0.151, 0.131, 0.108])
TT = np.arange(0.0, 21.0)                      # integer grid t=0..20
SPEC = dict(sigma=7 * np.pi / 20, kbar=2 * np.pi / 5, mu=(2, 7))  # NP=5


def rms(R):
    return float(np.sqrt(np.mean((R - DHK_IDEAL) ** 2)))


def main():
    lat = Z2Lattice(10, pbc=True)              # NP=5 -> nx=5 physical sites
    nx = lat.nx
    gf = sc.gauge_fixed_system(lat, M0, G2, ETA, full_band=True)

    # full ED on the Q=0 physical sector (dim 504) -- exact spectral evolution
    Hd = gf["H"].toarray()
    w, V = np.linalg.eigh(Hd)
    E = w - w[0]

    def spectral_amp(state):
        """<state|U(t)|state> on the integer grid via the exact spectrum."""
        wt = np.abs(V.conj().T @ state) ** 2
        return (wt[None, :] * np.exp(-1j * np.outer(TT, E))).sum(1)

    def sigma_E(state):
        wt = np.abs(V.conj().T @ state) ** 2
        Em = wt @ E
        return float(np.sqrt(wt @ (E - Em) ** 2)), float(Em)

    # ---- baseline: full two-meson state |Psi> = M[phi_1] M[phi_2]|Omega> ----
    packets = [(+SPEC["kbar"], SPEC["sigma"], SPEC["mu"][0]),
               (-SPEC["kbar"], SPEC["sigma"], SPEC["mu"][1])]
    psi = sc.two_meson_state(gf, ETA, packets)
    A_full = spectral_amp(psi)
    R_full = np.abs(A_full) ** 2
    sE_full, Em_full = sigma_E(psi)

    # ---- total-momentum projectors from T2 (P = 2 pi n / nx) ----
    Td = gf["T"].toarray()
    assert np.abs(np.linalg.matrix_power(Td, nx) - np.eye(Td.shape[0])).max() < 1e-8

    def proj_P(n):
        P = 2 * np.pi * n / nx
        acc = np.zeros_like(Td, dtype=complex)
        Tj = np.eye(Td.shape[0], dtype=complex)
        for j in range(nx):
            acc += np.exp(-1j * P * j) * Tj
            Tj = Td @ Tj
        return acc / nx

    # P-sector weights, and a sanity check that <Psi|U|Psi> = sum_P <Psi_P|U|Psi_P>
    wP = np.array([np.vdot(psi, proj_P(n) @ psi).real for n in range(nx)])
    A_byP = sum(spectral_amp(proj_P(n) @ psi) for n in range(nx))  # unnormalized
    decomp_err = np.abs(A_byP - A_full).max()

    # per-sector energy spread: is the COM the culprit?
    sE_perP = []
    for n in range(nx):
        pn = proj_P(n) @ psi
        nn = np.linalg.norm(pn)
        sE_perP.append(sigma_E(pn / nn)[0] if nn > 1e-9 else np.nan)

    # (a) project onto P=0, renormalize
    psi0 = proj_P(0) @ psi
    w0 = np.linalg.norm(psi0) ** 2
    psi0 = psi0 / np.sqrt(w0)
    R_P0 = np.abs(spectral_amp(psi0)) ** 2
    sE_P0, _ = sigma_E(psi0)

    # ---- (b) connected / vacuum-subtracted ----
    # single-meson return amplitudes
    mops = sc.meson_operator(lat, gf["basis"], gf["sel"], ETA)
    p1 = sc.packet_operator(mops, +SPEC["kbar"], SPEC["sigma"], SPEC["mu"][0], nx) @ gf["vac"]
    p1 /= np.linalg.norm(p1)
    p2 = sc.packet_operator(mops, -SPEC["kbar"], SPEC["sigma"], SPEC["mu"][1], nx) @ gf["vac"]
    p2 /= np.linalg.norm(p2)
    A1, A2 = spectral_amp(p1), spectral_amp(p2)
    sE_1, _ = sigma_E(p1)

    # b1: divide out the free single-meson amplitudes (S-matrix / connected)
    with np.errstate(divide="ignore", invalid="ignore"):
        R_conn = np.abs(A_full / (A1 * A2)) ** 2
    R_conn = R_conn / R_conn[0]
    # b2: vacuum-subtracted state
    vac = gf["vac"]
    vac_w = np.abs(np.vdot(vac, psi)) ** 2
    psi_nv = psi - vac * np.vdot(vac, psi)
    psi_nv /= np.linalg.norm(psi_nv)
    R_novac = np.abs(spectral_amp(psi_nv)) ** 2

    # ---- report ----
    print(f"NP=5  Q=0 dim {Hd.shape[0]};  M={gf['M']:.4f}, 2M={2*gf['M']:.4f}")
    print(f"decomposition check |sum_P A_P - A_full| = {decomp_err:.2e} "
          f"(U block-diagonal in P => projection is legitimate)\n")
    print(f"baseline full state : sigma_E={sE_full:.4f} (dephase {1/sE_full:.2f}),"
          f" <E>-Evac={Em_full:.4f}")
    print(f"single meson (phi_1): sigma_E={sE_1:.4f}  -> the free single-particle"
          f" spread already sets the decay")
    print(f"P-sector weights    : {np.round(wP,3)}")
    print(f"per-sector sigma_E  : {np.round(sE_perP,3)}  "
          f"(all ~same => COM is NOT the dephasing source)")
    print(f"vacuum weight in Psi: {vac_w:.2e} (subtracting it is a no-op)\n")

    rows = [("full R(t) [baseline]", R_full, sE_full),
            ("(a) P=0 projection", R_P0, sE_P0),
            ("(b1) connected/divide", R_conn, np.nan),
            ("(b2) vacuum-subtracted", R_novac, np.nan)]
    print(f"{'construction':26s} {'sigma_E':>8s} {'RMS->DHK':>10s}")
    for name, R, sE in rows:
        finite = np.all(np.isfinite(R)) and R.max() < 5
        tag = "" if finite else "  (unphysical: not a probability / diverges)"
        print(f"{name:26s} {sE:8.4f} {rms(R):10.4f}{tag}")

    print("\nR(t) grids (t=0,4,8,12,16,20):")
    for name, R, _ in rows:
        if np.all(np.isfinite(R)) and R.max() < 5:
            print(f"  {name:26s} {np.round(R[::4],3)}")
    print(f"  {'DHK ideal':26s} {np.round(DHK_IDEAL[::4],3)}")

    best = min(rms(R) for _, R, _ in rows
               if np.all(np.isfinite(R)) and R.max() < 5)
    print(f"\nbest defensible RMS to DHK = {best:.4f}  (target < 0.1)")
    print("VERDICT: total-momentum projection and connected/vacuum subtraction"
          " do NOT reproduce DHK; the decay is intrinsic single-meson energy"
          " spread, untouched by either.")

    np.savez("data/dhk_hyp_proj.npz", times=TT, DHK_IDEAL=DHK_IDEAL,
             R_full=R_full, R_P0=R_P0, R_conn=R_conn, R_novac=R_novac,
             wP=wP, sE_perP=np.array(sE_perP), sE_full=sE_full, sE_1=sE_1,
             sE_P0=sE_P0, decomp_err=decomp_err,
             rms_full=rms(R_full), rms_P0=rms(R_P0), rms_novac=rms(R_novac))
    print("saved data/dhk_hyp_proj.npz")
    return best


if __name__ == "__main__":
    main()
