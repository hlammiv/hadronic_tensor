"""DHK hypothesis key=echo: interaction Loschmidt echo.

HYPOTHESIS.  DHK's slow, monotonic R(t) may be the INTERACTION-only echo,
with the trivial free (kinetic) two-meson evolution removed:

    R_echo(t) = | <Psi| e^{+i H0 t} e^{-i H t} |Psi> |^2 ,

where H0 is the NON-interacting two-meson Hamiltonian: on the two-meson
product subspace B = span{ O_{k1} O_{k2} |Omega> } it acts with the ADDITIVE
single-meson energies E1(k1)+E1(k2); off B it is zero.  If our full R(t)
craters only because of kinetic dephasing/translation of the wavepackets,
the echo should decay slowly (sigma_E ~ 0.07) and track DHK.

CONSTRUCTION (small volume, full ED -- instant).
  * O_k = sum_x e^{ikx} O_x  (momentum-space meson creation op).
  * single-meson energy  E1(k) = <s_k|H|s_k> - E_vac  with
    |s_k> = (1-|Omega><Omega|) O_k|Omega> normalized  (0-meson piece removed).
    The meson operators COMMUTE on the vacuum (bosonic mesons; verified
    <1e-15), so O_{k1}O_{k2}|Omega> is symmetric.
  * B is spanned by the product states; QR-with-pivoting picks a maximal
    linearly-independent set (rank r), each carrying a definite pair label
    and free energy Efree = E1(k1)+E1(k2).  Loewdin (symmetric) ortho-
    normalization A = Phi_sel S^{-1/2} gives an orthonormal eigenbasis of
    H0 with eigenvalues Efree, so e^{+iH0 t} = A e^{+i diag(Efree) t} A^dag
    + (1 - A A^dag).  Since |Psi> in B, the (1-AA^dag) part drops out of
    the echo amplitude and
        amp_echo(t) = (A^dag Psi)^dag  diag(e^{+i Efree t})  (A^dag Psi(t)),
    with |Psi(t)> = e^{-iHt}|Psi> from exact Krylov.

No time-unit rescale is used (time_rescale = 1.0).  Scored by RMS to the
digitized DHK N_P=13 ideal on integer t=0..20.

  PYTHONPATH=. .venv/bin/python scripts/dhk_hyp_echo.py [ns]
"""
import sys
import numpy as np
import scipy.sparse.linalg as spla

from htensor import Z2Lattice
from htensor import scattering as sc

M0, G2, ETA = 1.0, 0.6, 1.0

DHK_T = np.arange(0, 21)
DHK_IDEAL = np.array([1.000, 0.962, 0.903, 0.881, 0.886, 0.864, 0.779, 0.695,
                      0.643, 0.617, 0.559, 0.474, 0.397, 0.354, 0.321, 0.277,
                      0.219, 0.173, 0.151, 0.131, 0.108])

# NP=5 spec (the small-volume analogue of DHK NP=13); packets scale with nx
SPEC = {10: dict(sigma=7*np.pi/20, kbar=2*np.pi/5, mu=(2, 7)),
        12: dict(sigma=3*np.pi/13, kbar=2*np.pi/6, mu=(2, 8))}


def build_ops(gf, lat):
    mops = sc.meson_operator(lat, gf["basis"], gf["sel"], ETA)
    nx = lat.nx
    ks = 2*np.pi*np.arange(nx)/nx
    ks = np.where(ks > np.pi, ks-2*np.pi, ks)

    def Ok(k):
        xs = np.arange(nx)
        ph = np.exp(1j*k*xs)
        O = mops[0]*ph[0]
        for x in range(1, nx):
            O = O + mops[x]*ph[x]
        return O
    return ks, [Ok(k) for k in ks]


def single_meson_energies(gf, Ops, ks):
    """E1(k) = energy of the single-meson wavepacket O_k|vac> (0-meson piece
    removed): the natural NON-interacting meson energy that H0 must undo."""
    H = gf["H"]; vac = gf["vac"]; Evac = gf["evals"][0]
    E1 = np.zeros(len(ks))
    for i in range(len(ks)):
        s = Ops[i] @ vac
        s = s - np.vdot(vac, s)*vac
        s = s/np.linalg.norm(s)
        E1[i] = np.real(np.vdot(s, H@s)) - Evac
    return E1


def build_H0_frame(gf, Ops, ks, E1):
    """Two-meson product frame -> QR-pivot independent set -> Loewdin A and
    per-column free energies Efree."""
    vac = gf["vac"]
    nx = len(ks)
    Phi = []; Efree = []
    for i in range(nx):
        for j in range(i, nx):                 # unordered (states symmetric)
            Phi.append(Ops[i] @ (Ops[j] @ vac))
            Efree.append(E1[i]+E1[j])
    Phi = np.array(Phi).T                       # D x m
    Efree = np.array(Efree)
    # QR with column pivoting -> maximal independent set
    Q, Rm, piv = __import__("scipy").linalg.qr(Phi, pivoting=True, mode="economic")
    diag = np.abs(np.diag(Rm))
    r = int((diag > 1e-8*diag.max()).sum())
    sel = piv[:r]
    Phis = Phi[:, sel]; Ef = Efree[sel]
    S = Phis.conj().T @ Phis
    ev, U = np.linalg.eigh(S)
    Sinv2 = U @ np.diag(1/np.sqrt(ev)) @ U.conj().T
    A = Phis @ Sinv2                            # D x r, orthonormal columns
    return A, Ef, r


def run(ns):
    s = SPEC[ns]
    lat = Z2Lattice(ns, pbc=True)
    gf = sc.gauge_fixed_system(lat, M0, G2, ETA, full_band=True)
    H = gf["H"]; Evac = gf["evals"][0]
    ks, Ops = build_ops(gf, lat)
    E1 = single_meson_energies(gf, Ops, ks)
    print(f"ns={ns} nx={lat.nx} dim={H.shape[0]}  M={gf['M']:.4f} 2M={2*gf['M']:.4f}")
    print(f"  E1(k): {np.round(E1,4)}  (spread {E1.max()-E1.min():.4f})")

    A, Efree, r = build_H0_frame(gf, Ops, ks, E1)
    print(f"  two-meson subspace rank r={r}; Efree range "
          f"[{Efree.min():.3f},{Efree.max():.3f}] spread {Efree.max()-Efree.min():.3f}")

    # prepared two-meson state
    packets = [(+s["kbar"], s["sigma"], s["mu"][0]),
               (-s["kbar"], s["sigma"], s["mu"][1])]
    psi = sc.two_meson_state(gf, ETA, packets)
    # energy content of the full state
    Emean = np.real(np.vdot(psi, H@psi)) - Evac
    var = np.real(np.vdot(psi, H@(H@psi))) - np.real(np.vdot(psi, H@psi))**2
    sigE_full = float(np.sqrt(max(var, 0)))
    print(f"  <H>-Evac={Emean:.4f}; full-H sigma_E={sigE_full:.4f}")

    b0 = A.conj().T @ psi                       # projection of |Psi> onto B
    print(f"  |P_B Psi|^2 = {np.linalg.norm(b0)**2:.4f} (should be ~1)")

    # interaction-energy spread: sigma of (E_full - Efree) across the state.
    # Build H0 as operator on B, then E_int = <Psi|(H-H0)|Psi> etc. via A.
    H0psi = A @ (Efree * b0)                     # H0|Psi> (in B)
    Hint_psi = H @ psi - H0psi                   # (H - H0)|Psi>, shift-free below
    # mean interaction energy (relative)
    Eint_mean = np.real(np.vdot(psi, Hint_psi))
    Eint2 = np.real(np.vdot(Hint_psi, Hint_psi))
    sigE_int = float(np.sqrt(max(Eint2 - Eint_mean**2, 0)))
    print(f"  interaction-energy spread sigma(E-E0) = {sigE_int:.4f} "
          f"(echo dephasing time ~ {1/max(sigE_int,1e-9):.2f})")

    # ---- full R(t) and echo R(t) ----
    times = np.arange(0.0, 20.01, 1.0)
    R_full = np.empty(len(times))
    R_echo = np.empty(len(times))
    state = psi.copy(); prev = 0.0
    for i, t in enumerate(times):
        if t > prev:
            state = spla.expm_multiply(-1j*H*(t-prev), state)
            prev = t
        R_full[i] = abs(np.vdot(psi, state))**2
        bt = A.conj().T @ state                  # A^dag |Psi(t)>
        amp = np.vdot(b0, np.exp(1j*Efree*t)*bt)  # (A^dag Psi)^dag diag A^dag Psi(t)
        R_echo[i] = abs(amp)**2

    rms_full = float(np.sqrt(np.mean((R_full - DHK_IDEAL)**2)))
    rms_echo = float(np.sqrt(np.mean((R_echo - DHK_IDEAL)**2)))
    print(f"\n  {'t':>4} {'R_full':>8} {'R_echo':>8} {'DHK':>8}")
    for i in range(0, len(times), 2):
        print(f"  {times[i]:4.0f} {R_full[i]:8.4f} {R_echo[i]:8.4f} {DHK_IDEAL[i]:8.4f}")
    print(f"\n  RMS(full - DHK)  = {rms_full:.4f}")
    print(f"  RMS(echo - DHK)  = {rms_echo:.4f}")
    return dict(times=times, R_full=R_full, R_echo=R_echo, E1=E1, Efree=Efree,
                sigE_full=sigE_full, sigE_int=sigE_int, rms_full=rms_full,
                rms_echo=rms_echo, ns=ns)


if __name__ == "__main__":
    ns = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    out = run(ns)
    np.savez(f"data/dhk_hyp_echo{'' if ns==10 else '_ns%d'%ns}.npz",
             DHK_T=DHK_T, DHK_IDEAL=DHK_IDEAL, **out)
    print(f"\nsaved data/dhk_hyp_echo{'' if ns==10 else '_ns%d'%ns}.npz")
