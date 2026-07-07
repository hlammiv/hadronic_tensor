"""DHK return-probability postdiction (task 15).

  validate : N_P=5 (ns=10) -- build the two-meson state, check single-packet
             momentum/band content and R(0)=1, evolve R(t); cross-checks the
             gauge-fixed construction where a full statevector is still cheap.
  run <NP> : compute R(t) at their volume (NP=13 -> ns=26) and save
             data/dhk_Rt_NP{NP}.npz for overlay on their Fig. 10.

DHK 2505.20408 -> our (m0,g2,eta)=(1.0,0.6,1.0), PBC, Q=0 (half filling).
Packets Psi_i(k)=N e^{-ik mu_i} e^{-(k-kbar_i)^2/4 sigma^2}:
  N_P=5 : sigma=7pi/20, kbar=+-2pi/5, mu=(2,7)
  N_P=13: sigma=3pi/13, kbar=+-2pi/13, mu=(6,19)

  PYTHONPATH=. .venv/bin/python scripts/dhk_postdiction.py validate
"""

import sys
import time

import numpy as np

from htensor import Z2Lattice
from htensor import scattering as sc
from htensor.gaugefixed import PhysicalBasis

M0, G2, ETA = 1.0, 0.6, 1.0
SPEC = {5:  dict(ns=10, sigma=7 * np.pi / 20, kbar=2 * np.pi / 5, mu=(2, 7)),
        13: dict(ns=26, sigma=3 * np.pi / 13, kbar=2 * np.pi / 13, mu=(6, 19))}
t0 = time.time()


def log(m):
    print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)


def build(NP, n_band=40, full_band=True, ncv=None):
    s = SPEC[NP]
    lat = Z2Lattice(s["ns"], pbc=True)
    log(f"N_P={NP} (ns={s['ns']}): building gauge-fixed system")
    gf = sc.gauge_fixed_system(lat, M0, G2, ETA, n_band=n_band,
                               full_band=full_band, ncv=ncv)
    log(f"  meson mass M = {gf['M']:.4f}; band momenta "
        f"{sorted(round(k,3) for k in gf['band'])}")
    packets = [(+s["kbar"], s["sigma"], s["mu"][0]),
               (-s["kbar"], s["sigma"], s["mu"][1])]
    psi = sc.two_meson_state(gf, ETA, packets)
    return lat, gf, psi, s


if sys.argv[1] == "validate":
    lat, gf, psi, s = build(5)
    # single-packet diagnostics: momentum via T2, band-1 content
    mops = sc.meson_operator(lat, gf["basis"], gf["sel"], ETA)
    p1 = sc.packet_operator(mops, +s["kbar"], s["sigma"], s["mu"][0], lat.nx) @ gf["vac"]
    p1 /= np.linalg.norm(p1)
    Tval = np.vdot(p1, gf["T"] @ p1)
    log(f"single packet +kbar: <T2> arg = {np.angle(Tval):+.4f} "
        f"(target kbar = {s['kbar']:+.4f}), |<T2>| = {np.abs(Tval):.3f}")
    # band-1 content: overlap with the single-meson eigenstates
    b1 = sum(np.abs(np.vdot(bs, p1)) ** 2 for bs in gf["band_states"])
    log(f"single packet band content (band 1): {b1:.3f}; "
        f"M = {gf['M']:.4f}, 2M = {2*gf['M']:.4f}")
    # energy above 2M
    E1 = np.real(np.vdot(psi, gf["H"] @ psi)) - gf["evals"][0]
    log(f"two-meson <H> - E_vac = {E1:.4f}  (2M = {2*gf['M']:.4f})")
    times = np.arange(0.0, 12.01, 0.5)
    R, amp = sc.return_probability(gf, psi, times)
    log(f"R(0) = {R[0]:.5f} (must be 1)")
    for t, r in zip(times[::2], R[::2]):
        log(f"  R({t:4.1f}) = {r:.4f}")
    np.savez("data/dhk_Rt_NP5.npz", times=times, R=R, amp=amp,
             M=gf["M"], E2=E1)
    log("saved data/dhk_Rt_NP5.npz")

elif sys.argv[1] == "run":
    NP = int(sys.argv[2])
    ncv = int(sys.argv[3]) if len(sys.argv) > 3 else 24
    lat, gf, psi, s = build(NP, full_band=False, ncv=ncv)
    E1 = np.real(np.vdot(psi, gf["H"] @ psi)) - gf["evals"][0]
    log(f"two-meson <H> - E_vac = {E1:.4f} (2M = {2*gf['M']:.4f}); "
        f"physical dim {gf['H'].shape[0]}")
    times = np.arange(0.0, 20.01, 0.5)
    R, amp = sc.return_probability(gf, psi, times)
    np.savez(f"data/dhk_Rt_NP{NP}.npz", times=times, R=R, amp=amp,
             M=gf["M"], E2=E1, m0=M0, g2=G2, eta=ETA)
    log(f"R(0)={R[0]:.4f}; saved data/dhk_Rt_NP{NP}.npz")
    for t, r in zip(times[::4], R[::4]):
        log(f"  R({t:4.1f}) = {r:.4f}")
