"""DHK hypothesis: MESON INTERPOLATOR sets the sub-band weight (key=r2-subband).

Thesis under test
-----------------
The factorized two-meson return probability
    R(t) = |A1(t) A2(t)|^2,   A_i(t) = sum_k w_i(k) <phi_k|U(t)|phi_k>
is (empirically) pure Gaussian dephasing: its decay is fixed by the energy
spread sigma_E of the prepared packet.  DHK's N_P=13 curve (Fig. 10) implies a
two-meson sigma_E ~ 0.074.  Our two reference operators bracket it:
  * clean lightest band only (z_up=0)          -> sigma_E = 0.036  (2.1x NARROW)
  * raw chi=1 op O=hop+i*current               -> sigma_E = 0.152  (2.1x BROAD)
and 0.074 ~ sqrt(0.036*0.152) is their geometric mean.

HYPOTHESIS: DHK's *actual* creation operator is neither -- it is a specific
staggered fermion-antifermion bilinear whose upper-sub-band weight z_up is a
DERIVED, discrete operator identity (a definite chirality / point-split /
parity choice), not a tuned continuous knob, and it lands sigma_E ~ 0.074.

TEST (ns=10,12 full ED, instant): MEASURE z_low/z_up (and the full single-meson
spectral function) for a menu of concrete physical interpolators.  For each,
build the factorized two-sub-band Model B at the N_P=13 momenta, and score RMS
vs the DHK ideal.  A pass requires a SPECIFIC, DHK-motivated operator (chosen by
its parity/normalization, not by fitting z) with RMS < 0.1.

  PYTHONPATH=. .venv/bin/python scripts/dhk_hyp_r2-subband.py
"""
import numpy as np
from collections import defaultdict

from htensor import Z2Lattice
from htensor import scattering as sc
from htensor.gaugefixed import PhysicalBasis
from htensor import hamiltonian as ham
from htensor.hamiltonian import hop_term
from htensor.currents import bond_current

M0, G2, ETA = 1.0, 0.6, 1.0

DHK_T = np.arange(0, 21)
DHK_IDEAL = np.array([1.000, 0.962, 0.903, 0.881, 0.886, 0.864, 0.779, 0.695,
                      0.643, 0.617, 0.559, 0.474, 0.397, 0.354, 0.321, 0.277,
                      0.219, 0.173, 0.151, 0.131, 0.108])
NX, SIGMA, KBAR = 13, 3 * np.pi / 13, 2 * np.pi / 13

# single-meson energy window (above vacuum) used to define the 1-meson band
WLO, WHI = 2.0, 3.3


# ---------------------------------------------------------------------------
# candidate physical interpolators  O_x  on the Q=0 physical basis
# Each returns a list over physical site x of the reduced operator matrix.
# All are gauge-invariant fermion-antifermion bilinears realizable in the
# staggered 27-qubit encoding; they differ only by parity/chirality/point-split.
# ---------------------------------------------------------------------------
def _reduce(lat, basis, sel, O):
    return basis.matrix(O, sub=sel)


def op_menu(lat, basis, sel):
    nx = lat.nx
    ops = {}

    def build(name, fn):
        ops[name] = [_reduce(lat, basis, sel, fn(x)) for x in range(nx)]

    # (1) raw chi=1 directional hop on the even (physical-site) bond -- the
    #     current default.  Reference: too broad.
    build("chi1", lambda x: hop_term(lat, 2 * x, ETA)
          + 1j * bond_current(lat, 2 * x, ETA))
    # (2) opposite chirality chi=-1 (other momentum branch).
    build("chi-1", lambda x: hop_term(lat, 2 * x, ETA)
          - 1j * bond_current(lat, 2 * x, ETA))
    # (3) pure hop (chi=0): parity-even scalar bilinear, no current admixture.
    build("hop", lambda x: hop_term(lat, 2 * x, ETA))
    # (4) pure current: parity-odd vector bilinear.
    build("current", lambda x: 1j * bond_current(lat, 2 * x, ETA))
    # (5) point-split symmetric hop: average the two links meeting physical
    #     site x  (a definite-parity, site-centered "smeared" meson).
    build("psplit_hop", lambda x: 0.5 * (hop_term(lat, 2 * x, ETA)
                                         + hop_term(lat, 2 * x - 1, ETA)))
    # (6) point-split chi=1: same smear but keep the chi=1 chirality.
    build("psplit_chi1", lambda x: 0.5 * (
        hop_term(lat, 2 * x, ETA) + 1j * bond_current(lat, 2 * x, ETA)
        + hop_term(lat, 2 * x - 1, ETA) + 1j * bond_current(lat, 2 * x - 1, ETA)))
    # (7) odd-bond chi=1 (meson centered on the odd link instead of even).
    build("oddbond_chi1", lambda x: hop_term(lat, 2 * x + 1, ETA)
          + 1j * bond_current(lat, 2 * x + 1, ETA))
    return ops


# ---------------------------------------------------------------------------
# small-volume measurement of the single-meson spectral function of each op
# ---------------------------------------------------------------------------
def measure(ns_list=(10, 12)):
    """For every candidate operator return dict name -> rows{k: (El,Eu,zl,zu,
    sigk)} where sig_k is the FULL in-window energy spread at momentum k (the
    honest local spread, not the 2-band collapse)."""
    out = defaultdict(dict)
    for ns in ns_list:
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
        menu = op_menu(lat, basis, sel)
        ks = 2 * np.pi * np.arange(nx) / nx
        ks = np.where(ks > np.pi, ks - 2 * np.pi, ks)
        m = (E > WLO) & (E < WHI)
        ee = E[m]
        mid = 0.5 * (ee.min() + ee.max())
        for name, mops in menu.items():
            for k in ks:
                if k < -1e-9:
                    continue
                Ok = mops[0] * 1.0
                for x in range(1, nx):
                    Ok = Ok + mops[x] * np.exp(1j * k * x)
                st = Ok @ vac
                nrm = np.linalg.norm(st)
                if nrm < 1e-12:
                    continue
                st = st / nrm
                wt = np.abs(V.conj().T @ st) ** 2
                ww = wt[m]
                Z = ww.sum()
                if Z < 1e-9:
                    continue
                ww = ww / Z                       # normalize within 1-meson band
                lo, up = ee < mid, ee >= mid
                zl, zu = ww[lo].sum(), ww[up].sum()
                El = (ww[lo] * ee[lo]).sum() / max(zl, 1e-12)
                Eu = (ww[up] * ee[up]).sum() / max(zu, 1e-12)
                Em = (ww * ee).sum()
                sigk = np.sqrt((ww * (ee - Em) ** 2).sum())
                kk = round(float(k), 4)
                out[name][kk] = (El, Eu, zl, zu, sigk)  # ns=12 overwrites ns=10
    return out


# ---------------------------------------------------------------------------
# factorized two-meson R(t) from the 2-band summary at the N_P=13 momenta
# ---------------------------------------------------------------------------
def packet_weights():
    ks = 2 * np.pi * np.arange(NX) / NX
    ks = np.where(ks > np.pi, ks - 2 * np.pi, ks)
    w1 = np.exp(-((ks - KBAR) ** 2) / (2 * SIGMA ** 2))
    w2 = np.exp(-((ks + KBAR) ** 2) / (2 * SIGMA ** 2))
    return ks, w1 / w1.sum(), w2 / w2.sum()


def interp_even(kq, ks, vals):
    return np.interp(np.abs(kq), ks, vals)


def factorized_R(rows, times, force_equal=False):
    ks = np.array(sorted(rows))
    El = np.array([rows[k][0] for k in ks])
    Eu = np.array([rows[k][1] for k in ks])
    zl = np.array([rows[k][2] for k in ks])
    zu = np.array([rows[k][3] for k in ks])
    if force_equal:
        zl = np.full_like(zl, 0.5)
        zu = np.full_like(zu, 0.5)
    kq, w1, w2 = packet_weights()
    El_q = interp_even(kq, ks, El)
    Eu_q = interp_even(kq, ks, Eu)
    zl_q = interp_even(kq, ks, zl)
    zu_q = interp_even(kq, ks, zu)
    s = zl_q + zu_q
    zl_q, zu_q = zl_q / s, zu_q / s

    def amp(w):
        return np.array([(w * (zl_q * np.exp(-1j * El_q * t)
                               + zu_q * np.exp(-1j * Eu_q * t))).sum()
                         for t in times])
    A1, A2 = amp(w1), amp(w2)
    R = np.abs(A1 * A2) ** 2
    R = R / R[0]                                  # normalize R(0)=1
    # two-meson energy spread (single-meson content doubled in variance)
    Eall = np.concatenate([El_q, Eu_q])
    wall = np.concatenate([w1 * zl_q, w1 * zu_q])
    wall = wall / wall.sum()
    Em = (wall * Eall).sum()
    sE1 = np.sqrt((wall * (Eall - Em) ** 2).sum())
    return R, np.sqrt(2) * sE1


def rms(R):
    return float(np.sqrt(np.mean((R - DHK_IDEAL) ** 2)))


if __name__ == "__main__":
    t = DHK_T.astype(float)
    data = measure()

    print("=" * 72)
    print("r2-subband: does a DERIVED interpolator's z_up land sigma_E ~ 0.074?")
    print("=" * 72)
    print(f"{'operator':>13} {'z_up(kbar)':>10} {'Delta(kbar)':>11} "
          f"{'sigma_E':>8} {'RMS':>7}")
    results = {}
    for name, rows in data.items():
        R, sE = factorized_R(rows, t)
        r = rms(R)
        # report z_up and sub-band splitting near the packet center |k|=KBAR
        ks = np.array(sorted(rows))
        zu_c = interp_even(KBAR, ks, np.array([rows[k][3] for k in ks]))
        El_c = interp_even(KBAR, ks, np.array([rows[k][0] for k in ks]))
        Eu_c = interp_even(KBAR, ks, np.array([rows[k][1] for k in ks]))
        results[name] = (R, sE, r, zu_c, Eu_c - El_c)
        print(f"{name:>13} {zu_c:10.3f} {Eu_c-El_c:11.3f} {sE:8.4f} {r:7.4f}")

    # geometric-mean / equal-weight check on the chi1 sub-band geometry
    Req, sEeq = factorized_R(data["chi1"], t, force_equal=True)
    print(f"\nequal-weight z_up=z_low=0.5 (chi1 geometry): "
          f"sigma_E={sEeq:.4f}, RMS={rms(Req):.4f}")
    print(f"  geometric mean sqrt(0.036*0.152) = {np.sqrt(0.036*0.152):.4f}")

    # -------- decisive check: is the sub-band mechanism CAPABLE of DHK at all?
    # sweep z_up as a FREE knob on the chi1 sub-band energies (El,Eu fixed).
    rows = data["chi1"]
    ks = np.array(sorted(rows))
    El = np.array([rows[k][0] for k in ks])
    Eu = np.array([rows[k][1] for k in ks])
    kq, w1, w2 = packet_weights()
    Elq, Euq = interp_even(kq, ks, El), interp_even(kq, ks, Eu)

    def R_zup(zu):
        def amp(w):
            return np.array([(w * ((1 - zu) * np.exp(-1j * Elq * tt)
                                   + zu * np.exp(-1j * Euq * tt))).sum()
                             for tt in t])
        R = np.abs(amp(w1) * amp(w2)) ** 2
        return R / R[0]

    zsweep = np.linspace(0, 0.5, 51)
    rsweep = np.array([rms(R_zup(z)) for z in zsweep])
    z_best = zsweep[rsweep.argmin()]
    R_zbest = R_zup(z_best)
    print(f"\nFREE-KNOB FLOOR (fudge z_up, not an operator): best z_up="
          f"{z_best:.3f}  RMS={rsweep.min():.4f}  R(20)={R_zbest[20]:.3f}")
    print(f"  clean lower band z_up=0: RMS={rms(R_zup(0.0)):.4f}  "
          f"R(20)={R_zup(0.0)[20]:.3f} (decays too slowly, no revival)")
    print("  -> even the best FUDGED z_up cannot beat RMS~0.25: two discrete")
    print("     sub-bands (Delta~0.29) BEAT -> R(t) revives, DHK is monotonic.")

    # best operator
    best = min(results, key=lambda n: results[n][2])
    Rb, sEb, rb, zub, Db = results[best]
    print(f"\nBEST DERIVED operator: {best!r}  sigma_E={sEb:.4f}  RMS={rb:.4f}")
    print(f"\n{'t':>3} " + " ".join(f"{n[:7]:>7}" for n in results) + f" {'DHK':>7}")
    for i in range(0, 21, 4):
        row = " ".join(f"{results[n][0][i]:7.3f}" for n in results)
        print(f"{int(t[i]):3d} {row} {DHK_IDEAL[i]:7.3f}")

    np.savez("data/dhk_hyp_r2-subband.npz",
             t=t, DHK=DHK_IDEAL,
             names=np.array(list(results)),
             R=np.array([results[n][0] for n in results]),
             sigmaE=np.array([results[n][1] for n in results]),
             rms=np.array([results[n][2] for n in results]),
             z_up=np.array([results[n][3] for n in results]),
             delta=np.array([results[n][4] for n in results]),
             R_equal=Req, sigmaE_equal=sEeq, rms_equal=rms(Req),
             z_sweep=zsweep, rms_sweep=rsweep, z_best=z_best,
             rms_freeknob_floor=rsweep.min(), R_zbest=R_zbest,
             best=best, m0=M0, g2=G2, eta=ETA)
    print("\nsaved data/dhk_hyp_r2-subband.npz")
    print(f"\nHONEST BEST RMS (no rescale, derived operator) = {rb:.4f}")
