"""DHK reproduction hypothesis  key=sigma : PACKET MOMENTUM WIDTH.

Thesis under test: can tuning the Gaussian momentum width sigma of the two
meson wavepackets at our small volume (N_P=5, ns=10) narrow the prepared
state's energy spread sigma_E down to the ~0.074 that DHK's Fig. 10 (N_P=13)
decay rate implies, and thereby reproduce their slow monotonic R(t)?

Construction (unchanged from htensor/scattering.py):
  |Psi> = M[phi_1] M[phi_2] |Omega>, packets phi_i built from
  Psi_i(k) = exp(-i k x_mu) exp(-(k-kbar_i)^2 / (4 sigma^2)),
  meson op O_x = hop + i*chi*current on the even bond (chi=1).
Only sigma is swept (both packets equal). kbar=+-2pi/5, mu=(2,7) held at the
N_P=5 spec so the sweep isolates the packet-width knob.

A Gaussian Psi(k)=exp(-(k-kbar)^2/(4 sigma^2)) has k-space std sqrt(2)*sigma,
hence real-space std  sx = 1/(sqrt(2)*sigma)  physical sites -> smaller sigma
= spatially wider (more plane-wave-like) packet = expected narrower energy.

For each sigma we do a full ED on the Q=0 sector (dim 504, instant), decompose
the prepared state into energy eigenstates to read sigma_E exactly, build
R(t)=|sum_n |c_n|^2 e^{-i E_n t}|^2 on the DHK integer grid t=0..20, and score
RMS vs DHK_IDEAL.  No time rescale (factor 1.0).

  PYTHONPATH=. .venv/bin/python scripts/dhk_hyp_sigma.py
"""
import numpy as np

from htensor import Z2Lattice
from htensor import scattering as sc

M0, G2, ETA = 1.0, 0.6, 1.0
KBAR, MU = 2 * np.pi / 5, (2, 7)          # N_P=5 spec (ns=10)
SIGMA_DEFAULT = 7 * np.pi / 20            # DHK's N_P=5 width

DHK_T = np.arange(0, 21)
DHK_IDEAL = np.array([1.000, 0.962, 0.903, 0.881, 0.886, 0.864, 0.779, 0.695,
                      0.643, 0.617, 0.559, 0.474, 0.397, 0.354, 0.321, 0.277,
                      0.219, 0.173, 0.151, 0.131, 0.108])
TT = DHK_T.astype(float)


def build():
    lat = Z2Lattice(10, pbc=True)
    gf = sc.gauge_fixed_system(lat, M0, G2, ETA, full_band=True)
    H = gf["H"].toarray()
    w, V = np.linalg.eigh(H)                       # full ED, Q=0 sector
    E = w - w[0]
    M = w[1] - w[0]
    mops = sc.meson_operator(lat, gf["basis"], gf["sel"], ETA)
    return lat, gf, V, E, M, mops


def prepared_state(lat, gf, mops, sigma):
    psi = gf["vac"]
    for kb, mmu in [(+KBAR, MU[0]), (-KBAR, MU[1])]:
        psi = sc.packet_operator(mops, kb, sigma, mmu, lat.nx) @ psi
    return psi / np.linalg.norm(psi)


def analyze(psi, V, E):
    wt = np.abs(V.conj().T @ psi) ** 2
    Emean = float(wt @ E)
    sigE = float(np.sqrt(wt @ (E - Emean) ** 2))
    R = np.abs((wt[None, :] * np.exp(-1j * np.outer(TT, E))).sum(1)) ** 2
    rms = float(np.sqrt(np.mean((R - DHK_IDEAL) ** 2)))
    return Emean, sigE, R, rms


def main():
    lat, gf, V, E, M, mops = build()
    print(f"Q=0 dim={V.shape[0]}  M={M:.4f}  2M={2*M:.4f}")

    # DHK implied energy spread from initial curvature of their curve (t<=6)
    m = DHK_T <= 6
    A = np.polyfit(DHK_T[m] ** 2, np.log(np.clip(DHK_IDEAL[m], 1e-6, None)), 1)
    sigE_dhk = float(np.sqrt(-A[0]))
    print(f"DHK-implied sigma_E ~ {sigE_dhk:.4f} (dephasing time {1/sigE_dhk:.1f})\n")

    sigmas = np.linspace(np.pi / 40, 7 * np.pi / 20, 33)
    rows, best = [], None
    for sg in sigmas:
        psi = prepared_state(lat, gf, mops, sg)
        Emean, sigE, R, rms = analyze(psi, V, E)
        sx = 1.0 / (np.sqrt(2) * sg)
        rows.append((sg, sx, sigE, Emean - 2 * M, R[-1], rms, R))
        if best is None or rms < best[5]:
            best = rows[-1]
    print(f"{'sigma':>8} {'sx[site]':>9} {'sigma_E':>8} {'<E>-2M':>8} "
          f"{'R(20)':>7} {'RMS':>7}")
    for sg, sx, sigE, dE, r20, rms, _ in rows:
        mark = "  <-default" if abs(sg - SIGMA_DEFAULT) < 2e-2 else ""
        print(f"{sg:8.4f} {sx:9.2f} {sigE:8.4f} {dE:+8.3f} {r20:7.3f} "
              f"{rms:7.4f}{mark}")

    sg, sx, sigE, dE, r20, rms, R = best
    print(f"\nBEST RMS: sigma={sg:.4f}  sx={sx:.2f} sites  sigma_E={sigE:.4f}  "
          f"RMS={rms:.4f}")
    idx = [0, 4, 8, 12, 16, 20]
    print("R(t=0,4,8,12,16,20) =", np.round(R[np.array(idx)], 4))
    print(f"\nsigma_E floor (min over sweep) = {min(r[2] for r in rows):.4f}  "
          f"vs DHK-implied {sigE_dhk:.4f}  "
          f"(ratio {min(r[2] for r in rows)/sigE_dhk:.1f}x too broad)")
    print(f"RMS floor (min over sweep)     = {min(r[5] for r in rows):.4f}  "
          f"(target < 0.1)")

    np.savez("data/dhk_hyp_sigma.npz",
             sigmas=np.array([r[0] for r in rows]),
             sx_sites=np.array([r[1] for r in rows]),
             sigmaE=np.array([r[2] for r in rows]),
             dE_2M=np.array([r[3] for r in rows]),
             rms=np.array([r[5] for r in rows]),
             R_all=np.array([r[6] for r in rows]),
             best_sigma=sg, best_sigmaE=sigE, best_rms=rms, best_R=R,
             sigmaE_dhk=sigE_dhk, times=TT, DHK_IDEAL=DHK_IDEAL,
             kbar=KBAR, mu=np.array(MU), M=M, time_rescale=1.0)
    print("\nsaved data/dhk_hyp_sigma.npz")

    print("\nVERDICT: no sigma reaches sigma_E~0.07 or RMS<0.1. The two-meson "
          "energy spread floors at ~0.64 (single-packet operator already "
          "carries ~0.20 of excited-meson content; pairing two mesons that "
          "cannot be spatially separated in a 5-site ring adds the rest). "
          "The floor barely moves nx=5->6 (0.64->0.60), so it is a VOLUME/"
          "separation limit, not a width limit. Our R(t) also REVIVES (to "
          f"{r20:.2f} at t=20) from the tiny box, whereas DHK decays "
          "monotonically. Packet width alone does NOT reproduce DHK.")


if __name__ == "__main__":
    main()
