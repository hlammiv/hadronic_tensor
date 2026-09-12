"""DHK hypothesis: SLOW-PACKET / SURVIVAL (key=survival).

Thesis under test: DHK's slow, monotonic R(t) (Fig. 10, N_P=13; decays 1 ->
0.11 over t=0..20) might arise because (a) their packets are prepared nearly
at rest (kbar -> 0), so the return probability is interaction/dispersion
limited rather than collision limited, OR (b) their observable is effectively
a SINGLE-meson survival amplitude rather than the full two-meson return.

We test both at small volume (N_P=5, ns=10, full ED -- instant) and score by
RMS to the digitized DHK ideal on integer t=0..20.  Everything is exact
Krylov / spectral evolution -- no Trotter, no noise, no fit knobs beyond the
physical packet parameters we are explicitly varying.

  PYTHONPATH=. .venv/bin/python scripts/dhk_hyp_survival.py

RESULT (run 2026-07): hypothesis FAILS.  Best case is single-meson survival
at the nominal DHK packet params (kbar=2pi/5, sigma=7pi/20): RMS=0.280 to the
DHK ideal.  It reproduces the INITIAL fall-off shape but the discrete
5-momentum spectrum at ns=10 forces a crater+revival (R=0.004 at t=12,
rebounds to 0.32 at t=20) whereas DHK decays monotonically to 0.11.  Making
the packets SLOW (kbar -> 0) makes agreement strictly WORSE (RMS 0.38 -> 0.43),
directly refuting the slow-packet idea -- at rest the state spreads over MORE
energy levels, not fewer.  DHK's smooth decay is a dense-spectrum (large-
volume) effect: their N_P=13 revival is pushed past t=20.  No physically
DERIVED time rescale rescues this (the ~7.6x factor needed matches no physical
quantity here), so we do NOT claim reproduction.
"""
import numpy as np

from htensor import Z2Lattice
from htensor import scattering as sc
from htensor.gaugefixed import PhysicalBasis
from htensor import hamiltonian as ham

M0, G2, ETA = 1.0, 0.6, 1.0

DHK_T = np.arange(0, 21)
DHK_IDEAL = np.array([1.000, 0.962, 0.903, 0.881, 0.886, 0.864, 0.779, 0.695,
                      0.643, 0.617, 0.559, 0.474, 0.397, 0.354, 0.321, 0.277,
                      0.219, 0.173, 0.151, 0.131, 0.108])

# NP=5 packet params (from CLAUDE facts / SPEC)
SIGMA = 7 * np.pi / 20
KBAR = 2 * np.pi / 5
MU = (2, 7)
NS = 10


def rms_to_dhk(times, R):
    Ri = np.interp(DHK_T, times, R)
    return float(np.sqrt(np.mean((Ri - DHK_IDEAL) ** 2)))


def spectral_R(w, V, psi, times, Evac=None):
    """Exact R(t) via full spectral decomposition + energy spread of psi."""
    c = V.conj().T @ psi
    wt = np.abs(c) ** 2
    E = w - (w[0] if Evac is None else Evac)
    Emean = float(wt @ E)
    sigE = float(np.sqrt(max(wt @ (E - Emean) ** 2, 0.0)))
    R = np.abs((wt[None, :] * np.exp(-1j * np.outer(times, E - Emean))).sum(1)) ** 2
    return R, Emean, sigE


def main():
    lat = Z2Lattice(NS, pbc=True)
    basis = PhysicalBasis(lat)
    sel = np.flatnonzero(basis.q == 0)
    H = basis.matrix(ham.build_hamiltonian(lat, M0, G2, ETA), sub=sel).real.toarray()
    w, V = np.linalg.eigh(H)
    Evac = w[0]
    gf = sc.gauge_fixed_system(lat, M0, G2, ETA, full_band=True)
    M = gf["M"]
    print(f"ns={NS}  Q=0 dim={H.shape[0]}  M={M:.4f}  2M={2*M:.4f}")
    print(f"band momenta/energies:")
    for k in sorted(gf["band"]):
        print(f"    k={k:+.4f}  E-Evac={gf['band'][k][0]:.4f}")

    times = np.arange(0.0, 20.01, 0.25)
    results = {}

    # ---------- (b) SINGLE-PACKET survival ----------
    mops = sc.meson_operator(lat, basis, sel, ETA, chi=1.0)
    print("\n=== (b) single-packet survival R(t)=|<phi|U|phi>|^2 ===")
    for kb, tag in [(KBAR, "kbar=2pi/5"), (KBAR / 2, "kbar=pi/5"), (0.0, "kbar=0")]:
        p = sc.packet_operator(mops, kb, SIGMA, MU[0], lat.nx) @ gf["vac"]
        p /= np.linalg.norm(p)
        R, Em, sigE = spectral_R(w, V, p, times, Evac)
        r = rms_to_dhk(times, R)
        results[f"single_{tag}"] = (R, Em, sigE, r)
        print(f"  {tag:12s}: <E>-Evac={Em:.4f} sigE={sigE:.4f} "
              f"(deph {1/max(sigE,1e-9):.1f})  RMS_DHK={r:.4f}")

    # ---------- (a) SLOW two-meson packets: scan kbar -> 0 ----------
    print("\n=== (a) slow two-meson R(t): scan kbar ===")
    for frac in [1.0, 0.5, 0.25, 0.0]:
        kb = KBAR * frac
        packets = [(+kb, SIGMA, MU[0]), (-kb, SIGMA, MU[1])]
        psi = sc.two_meson_state(gf, ETA, packets, chi=1.0)
        R, Em, sigE = spectral_R(w, V, psi, times, Evac)
        r = rms_to_dhk(times, R)
        results[f"two_kbar{frac}"] = (R, Em, sigE, r)
        print(f"  kbar={kb:+.4f} (frac {frac}): <E>-Evac={Em:.4f} "
              f"sigE={sigE:.4f} (deph {1/max(sigE,1e-9):.1f})  RMS_DHK={r:.4f}")

    # narrower packet in momentum (larger real-space packet) at rest, to
    # shrink sigma_E: does a slow, monochromatic single meson decay like DHK?
    print("\n=== (b') single packet, momentum-narrowed (sigma scan, at rest) ===")
    for sig in [SIGMA, SIGMA / 2, SIGMA / 4]:
        p = sc.packet_operator(mops, KBAR, sig, MU[0], lat.nx) @ gf["vac"]
        p /= np.linalg.norm(p)
        R, Em, sigE = spectral_R(w, V, p, times, Evac)
        r = rms_to_dhk(times, R)
        results[f"single_sig{sig:.3f}"] = (R, Em, sigE, r)
        print(f"  sigma={sig:.4f}: <E>-Evac={Em:.4f} sigE={sigE:.4f} "
              f"RMS_DHK={r:.4f}")

    # best
    best_key = min(results, key=lambda k: results[k][3])
    bR, bEm, bsigE, brms = results[best_key]
    print(f"\nBEST: {best_key}  RMS={brms:.4f}  sigE={bsigE:.4f}")
    print("R(t) at integer t for best:")
    Rint = np.interp(DHK_T, times, bR)
    for t in [0, 4, 8, 12, 16, 20]:
        print(f"  R({t:2d})={Rint[t]:.4f}  (DHK {DHK_IDEAL[t]:.4f})")

    np.savez("data/dhk_hyp_survival.npz",
             times=times,
             **{k: v[0] for k, v in results.items()},
             best_key=best_key, best_R=bR, best_rms=brms, best_sigE=bsigE,
             DHK_T=DHK_T, DHK_IDEAL=DHK_IDEAL)
    print("\nsaved data/dhk_hyp_survival.npz")
    return best_key, brms, bsigE, Rint


if __name__ == "__main__":
    main()
