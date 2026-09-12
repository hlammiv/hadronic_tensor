"""Two-meson finite-volume levels from the real-time current correlator of
the meson WAVEPACKET (what the 101-qubit circuits measure), exactly, in the
gauge-fixed Q = 0 basis (gf_engine), N_s = 12 (this file) and N_s = 16
(scripts/rt_packet_levels_ns16.py, which imports `run` from here).

Packet: the production target of scripts/train_packets.py -- band-projected
Gaussian at rest, sigma_x = 0.75, x0 = N_x // 2 (mid-ring), optimized
cur/hop mix, "clean" momentum weights |Phi(k)|^2 ~ exp(-2 sigma_x^2 k^2)
(gf_engine.gf_packet_target, clean=True).

Correlators  C(t) = <Phi| J(t) J(0) |Phi>,  J(t) = e^{iHt} J e^{-iHt}:
  (a) J = J1 = sum_b bond_current(b, eta)     translation invariant
  (b) J = J0'(c) = -Z_c / 2 at c = 2 x0       local; J0'(c) = charge_density(c)
      minus its c-number (-1)^c/2.  The c-number only adds the exactly known
      band-beat term (1/4) sum_jj' c_j* c_j' e^{-i(E_j' - E_j)t} (C-parity
      kills every cross term), so it is dropped.
Same-insertion EIGENSTATE correlators (|Phi> -> |k=0>) are computed alongside
at the same T for the line-density comparison.

Method.  |Phi> = sum_j c_j |S_j> over the band multiplet states (momentum
k_j, energy E_j); psi(t) = e^{-iHt} J|Phi> is evolved sector by sector with
the sparse T2-sector Hamiltonians (expm_multiply), and
    C(t) = sum_j c_j* e^{iE_j t} <S_j| J |psi(t)>
        = sum_{n,P,j} A_{nPj} e^{-i(E_n(P) - E_j) t},
    A_{nPj} = c_j* <S_j|J|n,P> <n,P|J|Phi>          (complex in case b).
The exact line table {omega, A} is built from dense sector eigh and checked
against the time series; the matrix pencil (Hankel SVD, rank cut 1e-10,
ESPRIT) is then run on C(t) as in scripts/rt_levels_ns*.py and every
pencil line is assigned to (n, P, k_j) combinations of the exact sector
spectra (P = k_j for case a, any P for case b) and to data/deep_levels.

    PYTHONPATH=. python scripts/rt_packet_levels_ns12.py [--T 200 --dt 0.1]

Outputs: data/rt_packet_levels_ns12.{npz,json}, data/rt_packet_levels.pdf
(both volumes, whichever npz exist).
"""
import argparse
import json
import os
import time

os.environ.setdefault("OMP_NUM_THREADS", "8")

import numpy as np
import scipy.sparse.linalg as spla

from htensor.lattice import Z2Lattice
from htensor import gf_engine as gfe
from htensor.gf_engine import GFSpace, GFHamiltonian, gf_band, _sector_hamiltonian
from htensor import currents as cur
from htensor.pauli import pauli_sum

M0, G2, ETA = 0.7, 1.1, 1.3
SIGMA_X = 0.75
ELASTIC = (5.490, 6.030)          # 2M < gap < M + M'
SV_REL = 1e-10                    # pencil singular-value cut (relative)
AMP_REL = 1e-4                    # report lines above AMP_REL x max |A|
DENSITY_REL = 1e-3                # line-density count threshold
MATCH_TOL = 1e-3                  # omega vs E_n(P) - E_j assignment tolerance
Z_TOL = 1e-3                      # discard pencil poles with ||z| - 1| > Z_TOL
L_MAX = 3000                      # Hankel pencil parameter cap (SVD cost)
W_RATIO = 3.0                     # pencil/exact weight consistency factor


# ----------------------------------------------------------- sector maps
def gf_to_sector(space, psi, k, idx):
    """Adjoint of sector_to_gf (float64 sqrt, as in rt_levels_ns16)."""
    rep, mrep, chi, ell, _ = space.orbits()
    ci = idx[rep]
    m = ci >= 0
    v = np.zeros(int(idx.max()) + 1, dtype=complex)
    np.add.at(v, ci[m], psi[m] * np.exp(1j * k * mrep[m]) * chi[m]
              / np.sqrt(ell[m].astype(float)))
    return v


def sector_to_gf(space, v, k, idx):
    rep, mrep, chi, ell, _ = space.orbits()
    ci = idx[rep]
    m = ci >= 0
    psi = np.zeros(space.dim, dtype=complex)
    psi[m] = v[ci[m]] * np.exp(-1j * k * mrep[m]) * chi[m] / np.sqrt(ell[m].astype(float))
    return psi


# ---------------------------------------------------------- matrix pencil
def matrix_pencil(y, dt, rank=None, sv_rel=SV_REL, z_tol=Z_TOL, L=None):
    """y[n] = sum_j A_j z_j^n.  Hankel SVD, rank cut, ESPRIT on the right
    singular vectors (rt_levels_ns16 version), then: poles with ||z| - 1| >
    z_tol are discarded (spurious poles from near-noise singular vectors;
    their Vandermonde columns |z|^N otherwise swamp the amplitude fit, which
    is what broke the dense local-insertion series), the survivors are put
    on the unit circle and the amplitudes fitted by least squares.
    Returns omega (y ~ sum A e^{-i omega t}), A, |z| (before projection),
    singular values, rank, number of discarded poles."""
    N = len(y)
    L = min(N // 2, L_MAX) if L is None else L
    Y = np.array([y[i:i + L + 1] for i in range(N - L)])
    U, s, Vh = np.linalg.svd(Y, full_matrices=False)
    if rank is None:
        rank = int(np.sum(s > sv_rel * s[0]))
    V = Vh[:rank].conj().T
    V1, V2 = V[:-1], V[1:]
    z = np.conj(np.linalg.eigvals(np.linalg.pinv(V1) @ V2))
    ok = np.abs(np.abs(z) - 1) < z_tol
    zabs = np.abs(z[ok])
    zu = z[ok] / zabs
    omega = -np.angle(zu) / dt
    Z = zu[None, :] ** np.arange(N)[:, None]
    A = np.linalg.lstsq(Z, y, rcond=None)[0]
    o = np.argsort(omega)
    return omega[o], A[o], zabs[o], s, rank, int((~ok).sum())


def fft_power(y, dt, n_pad=8):
    N = len(y)
    w = np.hanning(N)
    Y = np.fft.fft(y * w, n_pad * N)
    om = -2 * np.pi * np.fft.fftfreq(n_pad * N, d=dt)
    o = np.argsort(om)
    return om[o], np.abs(Y[o]) ** 2


def group_lines(omega, amp, tol=1e-7):
    """Merge exactly degenerate lines (+-P, +-k partners, multiplets):
    sum the complex amplitudes of lines within tol in omega."""
    o = np.argsort(omega)
    omega, amp = np.asarray(omega)[o], np.asarray(amp)[o]
    out_o, out_a, members = [], [], []
    for i, (w, a) in enumerate(zip(omega, amp)):
        if out_o and abs(w - out_o[-1]) < tol:
            out_a[-1] += a
            members[-1].append(int(o[i]))
        else:
            out_o.append(float(w)); out_a.append(complex(a)); members.append([int(o[i])])
    return np.array(out_o), np.array(out_a), members


def pencil_recovery(g_o, g_a, T, dt, targets):
    """Pencil on the exact spectral series sum A e^{-i omega t} sampled to T:
    fraction of the exact lines above AMP_REL recovered within 1e-3 / 1e-4
    in omega, and ("ok") within 1e-3 AND a factor W_RATIO in |A|; the same
    for the target (window) lines."""
    t = np.arange(0.0, T + 1e-9, dt)
    C = np.exp(-1j * np.outer(t, g_o)) @ g_a
    om, A, _, _, rank, ndrop = matrix_pencil(C, dt)
    keep = np.abs(A) > AMP_REL * np.max(np.abs(A))
    omk, Ak = om[keep], A[keep]
    sel = np.abs(g_a) > AMP_REL * np.abs(g_a).max()
    ex, exa = g_o[sel], np.abs(g_a[sel])

    def hit(x, ax):
        if not len(omk):
            return np.inf, False
        i = int(np.argmin(np.abs(omk - x)))
        d = float(abs(omk[i] - x))
        return d, bool(d < 1e-3 and 1 / W_RATIO < abs(Ak[i]) / ax < W_RATIO)
    dev = np.array([hit(x, ax)[0] for x, ax in zip(ex, exa)])
    okw = np.array([hit(x, ax)[1] for x, ax in zip(ex, exa)])
    tg = [hit(x, np.abs(g_a[np.argmin(np.abs(g_o - x))])) for x in targets]
    tdev = [float(d) for d, _ in tg]
    return dict(T=float(T), dt=float(dt), n_samples=int(len(t)), rank=int(rank), n_dropped=ndrop,
                n_lines=int(keep.sum()), n_exact=int(len(ex)),
                frac_within_1e3=float(np.mean(dev < 1e-3)), frac_within_1e4=float(np.mean(dev < 1e-4)),
                frac_ok=float(np.mean(okw)),
                max_dev=float(dev.max()), median_dev=float(np.median(dev)),
                target_dev=tdev, targets_within_1e3=int(sum(x < 1e-3 for x in tdev)),
                targets_ok=int(sum(o for _, o in tg)))


# ------------------------------------------------------------------ main
def run(ns, T=200.0, dt=0.1, out=None, log=None, scan_T=(200.0, 400.0, 800.0, 1600.0)):
    t0 = time.time()
    if log is None:
        log = lambda *a: print(f"[{time.time() - t0:7.1f}s]", *a, flush=True)
    out = out or f"data/rt_packet_levels_ns{ns}"

    lat = Z2Lattice(ns)
    nx = lat.nx
    space = GFSpace(lat, q=0)
    H = GFHamiltonian(space, M0, G2, ETA)
    log(f"ns={ns} nx={nx} gf dim={space.dim}")

    # ---- band (all momenta), packet
    band = gf_band(space, M0, G2, ETA, n_per_k=4)
    e0 = band["e0"]
    grid = space.momentum_grid()
    q1 = 2 * np.pi / nx
    x0 = nx // 2
    center = 2 * x0
    mix = gfe.gf_optimize_interpolator(space, band, k0=0.0, sigma_x=SIGMA_X, x0=x0)
    Phi, frac = gfe.gf_packet_target(space, band, 0.0, SIGMA_X, x0, mix["mix"], clean=True)
    Phi_raw, _ = gfe.gf_packet_target(space, band, 0.0, SIGMA_X, x0, mix["mix"], clean=False)
    p_k, p_vac = gfe.band_weights(band, Phi)
    p_k_raw, _ = gfe.band_weights(band, Phi_raw)
    E_pack, sig_pack = H.energy_stats(Phi)
    t2 = space.t2_expect(Phi)
    w_an = np.exp(-2 * SIGMA_X ** 2 * gfe._wrap(band["k"]) ** 2)
    w_an /= w_an.sum()
    log(f"packet: x0={x0} (site center c={center}), sigma_x={SIGMA_X}, band fraction of "
        f"raw interpolator packet {frac:.4f}, mix {mix['mix']}")
    log(f"  |Phi(k)|^2 (clean): " + ", ".join(f"k={k:+.3f}: {p:.4f}" for k, p in zip(band['k'], p_k)))
    log(f"  |Phi(k)|^2 (raw):   " + ", ".join(f"k={k:+.3f}: {p:.4f}" for k, p in zip(band['k'], p_k_raw)))
    log(f"  exp(-2 sigma^2 k^2) normalized: " + ", ".join(f"{p:.4f}" for p in w_an))
    log(f"  sum p_k = {p_k.sum():.6f}, vacuum weight {p_vac:.1e}, <T2> = {t2:.4f}, "
        f"<H> - E_vac = {E_pack - e0:.6f}, sigma_E = {sig_pack:.6f}")

    # band member list (multiplets): S_j, k_j, E_j, c_j
    members = [(i, s) for i, ms in enumerate(band["multiplets"]) for s in ms]
    S = [s for _, s in members]
    k_j = np.array([float(band["k"][i]) for i, _ in members])
    E_j = np.array([e0 + float(band["energy"][i]) for i, _ in members])
    c_j = np.array([np.vdot(s, Phi) for s in S])
    log(f"  {len(S)} band states (k = {np.round(k_j, 3)}); sum |c_j|^2 = {np.sum(np.abs(c_j) ** 2):.8f}")
    i0 = int(np.argmin(np.abs(band["k"])))
    meson0 = band["states"][i0]
    E_M = e0 + float(band["energy"][i0])
    M = E_M - e0

    # ---- ED reference
    ed = np.load(f"data/deep_levels_ns{ns}.npz")
    ed_g, ed_ph, ed_r = ed["gaps"], ed["phases"], ed["refl"]
    assert abs(ed["energies"][0] - e0) < 1e-6

    # ---- operators
    J1 = sum(cur.bond_current(lat, b, ETA) for b in lat.bonds).simplify()
    J0c = pauli_sum(lat.n_qubits, [({lat.site_qubit(center): "Z"}, -0.5)])   # J0(c) - (-1)^c/2
    ops = {"J1": J1, "J0c": J0c}

    # ---- sectors: sparse H, dense spectra, sector maps
    ts = np.arange(0.0, T + 1e-9, dt)
    Nt = len(ts)
    sectors = {}
    for P in grid:
        HP, R, idx = _sector_hamiltonian(space, H, float(P))
        w, U = np.linalg.eigh(HP.toarray())
        sectors[float(P)] = dict(H=HP.tocsc(), idx=idx, w=w, U=U, dim=HP.shape[0])
        log(f"  sector P={P:+.4f}: dim {HP.shape[0]}, E - E_vac in [{w[0] - e0:.4f}, {w[-1] - e0:.4f}]")
    wmax = max(s["w"][-1] for s in sectors.values())
    nyq = np.pi / dt
    log(f"bandwidth: max omega = E_max - E_min(band) = {wmax - E_j.min():.3f}, "
        f"min omega = E_vac - E_max(band) = {e0 - E_j.max():.3f}, Nyquist pi/dt = {nyq:.3f}")
    assert wmax - E_j.min() < nyq and E_j.max() - e0 < nyq, "aliasing"

    # ---- correlators
    cases = {
        "J1_packet": ("J1", Phi, c_j, "packet, J1 = sum_b bond current (P = k_j)"),
        "J1_eigen": ("J1", meson0, None, "eigenstate |k=0>, J1"),
        "J0c_packet": ("J0c", Phi, c_j, f"packet, J0'(c) = -Z_c/2 at c = {center} (all P)"),
        "J0c_eigen": ("J0c", meson0, None, f"eigenstate |k=0>, J0'(c) at c = {center}"),
    }
    npz = dict(t=ts, dt=dt, T=T, E_vac=e0, M=M, q1=q1, band_k=band["k"], band_gap=band["energy"],
               p_k=p_k, p_k_raw=p_k_raw, p_k_analytic=w_an, p_vac=p_vac,
               E_packet=E_pack - e0, sigma_E_packet=sig_pack, k_j=k_j, E_j=E_j, c_j=c_j,
               center=center, x0=x0, sigma_x=SIGMA_X,
               sector_P=np.array(list(sectors)),
               sector_E=np.array([sectors[P]["w"] for P in sectors], dtype=object))
    results = {}
    for name, (opname, state, cvec, label) in cases.items():
        J = ops[opname]
        if cvec is None:
            cvec = np.array([np.vdot(s, state) for s in S])
        chi = space.apply(J, state)                        # J |state>
        norm2 = float(np.vdot(chi, chi).real)
        u_j = [space.apply(J, s) for s in S]               # J |S_j>
        C = np.zeros(Nt, dtype=complex)
        lines_o, lines_a, lines_tag = [], [], []
        leak2, unit_err = 0.0, 0.0
        for P, sec in sectors.items():
            chiP = gf_to_sector(space, chi, P, sec["idx"])
            nP = float(np.vdot(chiP, chiP).real)
            leak2 += nP
            if nP < 1e-24:
                continue
            ujP = np.array([gf_to_sector(space, u, P, sec["idx"]) for u in u_j])  # (nj, dimP)
            psi_t = spla.expm_multiply(-1j * sec["H"], chiP, start=0.0, stop=T,
                                       num=Nt, endpoint=True)                 # (Nt, dimP)
            unit_err = max(unit_err, float(np.max(np.abs(
                np.sum(np.abs(psi_t) ** 2, axis=1) - nP)) / nP))
            ov = psi_t @ ujP.conj().T                                          # <J S_j|psi(t)>
            C += np.sum(np.conj(cvec)[None, :] * np.exp(1j * np.outer(ts, E_j)) * ov, axis=1)
            del psi_t
            # exact line table in this sector: A_{nPj} = c_j* <S_j|J|n,P> <n,P|J|state>
            U, w = sec["U"], sec["w"]
            b_n = U.conj().T @ chiP                                            # <n,P|J|state>
            a_jn = ujP.conj() @ U                                              # <S_j|J|n,P>
            for j in range(len(S)):
                amp = np.conj(cvec[j]) * a_jn[j] * b_n
                for n in np.flatnonzero(np.abs(amp) > 1e-14):
                    lines_o.append(w[n] - E_j[j]); lines_a.append(amp[n])
                    lines_tag.append((int(n), float(P), int(j)))
        leak = abs(leak2 - norm2) / norm2
        lines_o, lines_a = np.array(lines_o), np.array(lines_a)
        amp_of = dict(zip(lines_tag, lines_a))
        C_ex = np.exp(-1j * np.outer(ts, lines_o)) @ lines_a
        ex_err = float(np.max(np.abs(C - C_ex)))
        log(f"{name}: |J state|^2 = {norm2:.6f}, sector completeness {leak:.1e}, unitarity "
            f"max|d norm| {unit_err:.1e}, C(0) = {C[0]:.6f}, {len(lines_o)} exact terms, "
            f"max |C - sum A e^-iwt| = {ex_err:.1e}")
        # merged exact lines (degenerate partners summed)
        g_o, g_a, g_mem = group_lines(lines_o, lines_a)
        gmax = np.max(np.abs(g_a))
        # pencil on the evolved series
        om, A, absz, sv, rank, ndrop = matrix_pencil(C, dt)
        amax = np.max(np.abs(A))
        keep = np.abs(A) > AMP_REL * amax
        om_k, A_k, z_k = om[keep], A[keep], absz[keep]
        fit_err = float(np.max(np.abs(np.exp(-1j * np.outer(ts, om)) @ A - C)))
        n_dens_pencil = int(np.sum(np.abs(A) > DENSITY_REL * amax))
        n_dens_exact = int(np.sum(np.abs(g_a) > DENSITY_REL * gmax))
        n_exact_1e4 = int(np.sum(np.abs(g_a) > AMP_REL * gmax))
        log(f"{name}: pencil rank {rank} (sv[rank-1]/s0 {sv[rank - 1] / sv[0]:.1e}), {ndrop} off-circle "
            f"poles dropped, fit residual {fit_err:.1e}; {keep.sum()} lines above {AMP_REL:g}, "
            f"{n_dens_pencil} above {DENSITY_REL:g} (exact: {n_exact_1e4} / {n_dens_exact}); "
            f"max |1-|z|| kept = {np.max(np.abs(1 - z_k)):.1e}")

        # ---- assignment of every pencil line: nearest exact merged line
        rows = []
        case_a = opname == "J1"
        thr_amp = 1e-3 * AMP_REL * gmax
        for o, a, z in zip(om_k, A_k, z_k):
            je = int(np.argmin(np.abs(g_o - o)))
            d_ex = float(o - g_o[je])
            carrying = []
            for m in g_mem[je]:
                n, P, j = lines_tag[m]
                aa = lines_a[m]
                if abs(aa) < thr_amp:
                    continue
                En = sectors[P]["w"][n]
                carrying.append(dict(n=n, P=P, k_j=float(k_j[j]), j=j, E_n=float(En - e0),
                                     omega_exact=float(En - E_j[j]), residual=float(o - (En - E_j[j])),
                                     exact_amp_abs=float(abs(aa)), exact_amp_phase=float(np.angle(aa)),
                                     in_window=bool(ELASTIC[0] < En - e0 < ELASTIC[1])))
            # energy-only candidates: every (n, P, k_j) with |omega - (E_n(P) - E_j)| < MATCH_TOL
            cands = set()
            for P, sec in sectors.items():
                for j in range(len(S)):
                    if case_a and abs(np.angle(np.exp(1j * (P - k_j[j])))) > 1e-9:
                        continue
                    for n in np.flatnonzero(np.abs(sec["w"] - E_j[j] - o) < MATCH_TOL):
                        cands.add((round(float(sec["w"][n] - e0), 6), round(abs(P), 6),
                                   round(abs(k_j[j]), 6)))
            key_car = {(round(x["E_n"], 6), round(abs(x["P"]), 6), round(abs(x["k_j"]), 6))
                       for x in carrying}
            wratio = float(abs(a) / abs(g_a[je])) if abs(g_a[je]) > 0 else np.inf
            if abs(d_ex) > MATCH_TOL or not carrying:
                status = "unmatched"
            elif not (1 / W_RATIO < wratio < W_RATIO):
                status = "weight-mismatch"
            elif len(key_car) > 1:
                status = "degenerate"      # exactly coincident distinct (E_n, P, k_j)
            else:
                status = "ok"
            ed_hits = []
            for x in carrying:
                selP = np.abs(np.angle(np.exp(1j * (ed_ph - x["P"])))) < 1e-3
                if selP.any():
                    ii = np.flatnonzero(selP)[np.argmin(np.abs(ed_g[selP] - x["E_n"]))]
                    ed_hits.append(dict(ED_gap=float(ed_g[ii]),
                                        ED_refl=None if np.isnan(ed_r[ii]) else float(ed_r[ii]),
                                        diff=float(x["E_n"] - ed_g[ii])))
            rows.append(dict(omega=float(o), weight=float(abs(a)), weight_rel=float(abs(a) / amax),
                             amp_re=float(a.real), amp_im=float(a.imag), abs_z=float(z),
                             exact_omega=float(g_o[je]), exact_weight=float(abs(g_a[je])),
                             d_omega_exact=d_ex, weight_ratio=wratio, status=status,
                             n_energy_candidates=len(cands),
                             energy_ambiguous=bool(len(cands) > len(key_car)),
                             carrying=carrying, ED=ed_hits,
                             in_window=any(x["in_window"] for x in carrying)
                                       and status in ("ok", "degenerate")))
        # ---- elastic-window levels (E_n, P): exposed / missed.  A level is
        # measurable through the merged exact lines (omega = E_n(P) - E_j over
        # the bra momenta k_j) in which its terms dominate (>= half the summed
        # |A| of the merged line) and whose amplitude is above AMP_REL; it is
        # exposed when such a line is reproduced by the pencil (status ok).
        g_members_np = [[(lines_tag[m][0], lines_tag[m][1]) for m in mem] for mem in g_mem]
        line_of_pencil = {}
        for r in rows:
            if r["status"] in ("ok", "degenerate"):
                line_of_pencil[int(np.argmin(np.abs(g_o - r["omega"])))] = r
        exposed, missed, win_targets = [], [], []
        for P, sec in sectors.items():
            w = sec["w"]
            for n in np.flatnonzero((w - e0 > ELASTIC[0]) & (w - e0 < ELASTIC[1])):
                wt = float(sum(abs(lines_a[ii]) for ii, tg in enumerate(lines_tag)
                               if tg[0] == n and abs(tg[1] - P) < 1e-12))
                selP = np.abs(np.angle(np.exp(1j * (ed_ph - P)))) < 1e-3
                ii = np.flatnonzero(selP)[np.argmin(np.abs(ed_g[selP] - (w[n] - e0)))]
                per_line, found = [], []
                for gi, mem in enumerate(g_mem):
                    # own terms: this level and its exactly degenerate partners
                    # (-P twin, k = pi Kramers pair) -- physically one level
                    own = [m for m in mem if abs(sectors[lines_tag[m][1]]["w"][lines_tag[m][0]] - w[n]) < 1e-9]
                    if not own:
                        continue
                    a_own = sum(abs(lines_a[m]) for m in own)
                    a_all = sum(abs(lines_a[m]) for m in mem)
                    if abs(g_a[gi]) < 1e-6 * gmax:
                        continue
                    ks = sorted({float(abs(k_j[lines_tag[m][2]])) for m in own})
                    r = line_of_pencil.get(gi)
                    rec = dict(omega=float(g_o[gi]), k_j=ks, exact_amp=float(abs(g_a[gi])),
                               exact_amp_rel=float(abs(g_a[gi]) / gmax),
                               own_fraction=float(a_own / a_all),
                               above_cut=bool(abs(g_a[gi]) >= AMP_REL * gmax),
                               dominant=bool(a_own >= 0.5 * a_all),
                               pencil_omega=(r["omega"] if r else None),
                               pencil_weight=(r["weight"] if r else None))
                    per_line.append(rec)
                    if rec["above_cut"] and rec["dominant"]:
                        win_targets.append(float(g_o[gi]))
                        if r is not None:
                            found.append(rec)
                usable = [x for x in per_line if x["above_cut"] and x["dominant"]]
                reason = ("" if found else
                          "zero exact weight (symmetry)" if wt < 1e-20 else
                          f"total exact weight {wt / gmax:.1e} of max: below {AMP_REL:g} cut (packet momentum content / coupling)"
                          if not [x for x in per_line if x["above_cut"]] else
                          "only in lines dominated by an exactly degenerate other level"
                          if not usable else
                          "not resolved by the pencil at this T")
                ent = dict(n=int(n), P=float(P), P_over_q=float(P / q1), E_n=float(w[n] - e0),
                           ED_gap=float(ed_g[ii]), ED_refl=None if np.isnan(ed_r[ii]) else float(ed_r[ii]),
                           exact_weight_total=wt, exact_weight_rel=wt / gmax,
                           lines=per_line, n_lines_found=len(found), reason=reason)
                (exposed if found else missed).append(ent)
        win_targets = sorted(set(np.round(win_targets, 9)))
        # ---- resolution
        oo = np.sort(om_k)
        gaps = np.diff(oo)
        oo3 = np.sort(om[np.abs(A) > DENSITY_REL * amax])
        gaps3 = np.diff(oo3)
        go1e3 = np.sort(g_o[np.abs(g_a) > DENSITY_REL * gmax])
        egaps3 = np.diff(go1e3)
        go1e4 = np.sort(g_o[np.abs(g_a) > AMP_REL * gmax])
        egaps4 = np.diff(go1e4)
        # window-line spacing to the nearest OTHER line above 1e-4 (exact)
        win_sep = [float(np.min(np.abs(go1e4[np.abs(go1e4 - x) > 1e-9] - x))) for x in win_targets
                   if np.any(np.abs(go1e4 - x) < 1e-9)]
        # pencil vs T on the exact spectral series (validated above to ex_err)
        wmax_lines = float(np.max(np.abs(g_o[np.abs(g_a) > 1e-14 * gmax])))
        dt_scan = float(min(dt * 2, 0.9 * np.pi / wmax_lines))
        dt_scan = dt if dt_scan < dt else dt_scan
        scan = []
        for Tt in scan_T:
            tsc = time.time()
            scan.append(pencil_recovery(g_o, g_a, Tt, dt_scan, win_targets))
            log(f"{name}: scan T={Tt:.0f} dt={dt_scan:.3f}: rank {scan[-1]['rank']}, lines "
                f"{scan[-1]['n_lines']}/{scan[-1]['n_exact']}, within 1e-3: {scan[-1]['frac_within_1e3']:.2f}, "
                f"1e-4: {scan[-1]['frac_within_1e4']:.2f}, omega+weight ok: {scan[-1]['frac_ok']:.2f}; "
                f"window lines within 1e-3: {scan[-1]['targets_within_1e3']}/{len(win_targets)}, "
                f"omega+weight ok: {scan[-1]['targets_ok']} [{time.time() - tsc:.1f}s]")
        results[name] = dict(label=label, op=opname, norm2=norm2, rank=int(rank), n_dropped=ndrop,
                             fit_residual=fit_err,
                             n_lines_1e4=int(keep.sum()), n_lines_1e3=n_dens_pencil,
                             n_exact_lines_1e4=n_exact_1e4, n_exact_lines_1e3=n_dens_exact,
                             n_exact_terms=int(len(lines_o)), n_exact_distinct=int(len(g_o)),
                             max_abs_amp=float(amax), max_exact_amp=float(gmax),
                             sector_completeness=leak, unitarity_err=unit_err,
                             exact_series_err=ex_err, lines=rows,
                             window_exposed=exposed, window_missed=missed,
                             window_line_omegas=[float(x) for x in win_targets],
                             window_line_min_sep_exact=(min(win_sep) if win_sep else None),
                             min_gap_pencil_1e4=float(gaps.min()) if len(gaps) else None,
                             min_gap_pencil_1e3=float(gaps3.min()) if len(gaps3) else None,
                             min_gap_exact_1e4=float(egaps4.min()) if len(egaps4) else None,
                             min_gap_exact_1e3=float(egaps3.min()) if len(egaps3) else None,
                             T_fourier_1e3=(float(2 * np.pi / egaps3.min()) if len(egaps3) else None),
                             T_fourier_1e4=(float(2 * np.pi / egaps4.min()) if len(egaps4) else None),
                             scan=scan, scan_dt=dt_scan,
                             status_counts={s: sum(r["status"] == s for r in rows)
                                            for s in ("ok", "degenerate", "weight-mismatch", "unmatched")},
                             n_energy_ambiguous=int(sum(r["energy_ambiguous"] for r in rows)))
        npz[f"{name}_t"] = C
        npz[f"{name}_pencil_omega"] = om
        npz[f"{name}_pencil_amp"] = A
        npz[f"{name}_pencil_absz"] = absz
        npz[f"{name}_pencil_sv"] = sv
        npz[f"{name}_exact_omega"] = g_o
        npz[f"{name}_exact_amp"] = g_a
        npz[f"{name}_exact_tags"] = np.array(lines_tag)
        npz[f"{name}_exact_term_omega"] = lines_o
        npz[f"{name}_exact_term_amp"] = lines_a
        fo, fp = fft_power(C, dt)
        npz[f"{name}_fft_omega"] = fo
        npz[f"{name}_fft_power"] = fp
        r_ = results[name]
        log(f"{name}: statuses {r_['status_counts']}, energy-ambiguous {r_['n_energy_ambiguous']}; "
            f"min line gap pencil {r_['min_gap_pencil_1e4']} (1e-4) / {r_['min_gap_pencil_1e3']} (1e-3); "
            f"exact {r_['min_gap_exact_1e4']} / {r_['min_gap_exact_1e3']}; "
            f"T_Fourier(1e-3) {r_['T_fourier_1e3']}; window exposed {len(exposed)}, missed {len(missed)}")
        for r in rows:
            if r["in_window"] or r["status"] != "ok":
                tags = ", ".join(f"(n={x['n']},P={x['P'] / q1:+.0f}q,k={x['k_j'] / q1:+.0f}q,E={x['E_n']:.4f},"
                                 f"|A|={x['exact_amp_abs']:.1e})" for x in r["carrying"][:4])
                log(f"    omega {r['omega']:+9.5f} w {r['weight']:.3e} ({r['weight_rel']:.1e}) d_exact "
                    f"{r['d_omega_exact']:+.1e} [{r['status']}{', E-ambig' if r['energy_ambiguous'] else ''}] {tags}")
        for ent in missed:
            log(f"    MISSED window level E={ent['E_n']:.4f} P={ent['P_over_q']:+.0f}q n={ent['n']} "
                f"refl={ent['ED_refl']}: {ent['reason']}")

    np.savez(out + ".npz", **npz)
    summary = dict(ns=ns, nx=nx, couplings=[M0, G2, ETA], T=T, dt=dt, n_samples=Nt,
                   E_vac=e0, M=M, q1=q1, sigma_x=SIGMA_X, x0=x0, center=center,
                   packet=dict(p_k=dict(zip(map(str, np.round(band["k"], 6)), map(float, p_k))),
                               p_k_raw=dict(zip(map(str, np.round(band["k"], 6)), map(float, p_k_raw))),
                               p_k_analytic=dict(zip(map(str, np.round(band["k"], 6)), map(float, w_an))),
                               p_vac=p_vac, E_minus_Evac=E_pack - e0, sigma_E=sig_pack,
                               band_fraction_raw=frac,
                               mix={f"{k[0]}{k[1]}": [float(v.real), float(v.imag)]
                                    for k, v in mix["mix"].items()},
                               t2=[float(t2.real), float(t2.imag)],
                               c_j=[[float(c.real), float(c.imag)] for c in c_j],
                               k_j=list(map(float, k_j)), E_j_minus_Evac=list(map(float, E_j - e0))),
                   band=dict(k=list(map(float, band["k"])), gap=list(map(float, band["energy"])),
                             degeneracy=list(map(int, band["degeneracy"]))),
                   elastic_window=ELASTIC, sv_rel=SV_REL, amp_rel=AMP_REL, match_tol=MATCH_TOL,
                   z_tol=Z_TOL, nyquist=nyq, max_omega=float(wmax - E_j.min()),
                   correlators=results, wall_s=time.time() - t0)
    with open(out + ".json", "w") as f:
        json.dump(summary, f, indent=1)
    log(f"wrote {out}.npz/.json ({time.time() - t0:.0f}s)")
    return summary


# ---------------------------------------------------------------- figure
def make_figure(path="data/rt_packet_levels.pdf", volumes=(12, 16)):
    """Per volume two rows: (top) |FFT|^2 of C(t) for eigenstate vs packet
    with the pencil lines at M + omega; (bottom) the same pencil lines placed
    at the assigned level energy E_n - E_vac = omega + E_{k_j} - E_vac,
    colored by the bra momentum k_j, labeled (n, P/q) in the elastic window.
    Columns: J1 (translation invariant) and J0'(c) (local)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    avail = [ns for ns in volumes if os.path.exists(f"data/rt_packet_levels_ns{ns}.npz")]
    if not avail:
        return
    fig, axes = plt.subplots(2 * len(avail), 2, figsize=(14, 4.0 * 2 * len(avail)), squeeze=False)
    kcol = ["C3", "C0", "C2", "C1", "C4", "C5"]
    for r, ns in enumerate(avail):
        d = np.load(f"data/rt_packet_levels_ns{ns}.npz", allow_pickle=True)
        with open(f"data/rt_packet_levels_ns{ns}.json") as f:
            js = json.load(f)
        M, q1, T, dt = float(d["M"]), float(d["q1"]), float(d["T"]), float(d["dt"])
        for c, (op, title) in enumerate((("J1", "J1 = sum_b bond current (P = k_j)"),
                                         ("J0c", f"J0'(c) local at c = {int(d['center'])} (all P)"))):
            ax = axes[2 * r, c]
            for name, col, lab in ((f"{op}_eigen", "0.55", "eigenstate |k=0>"),
                                   (f"{op}_packet", "C3", "wavepacket")):
                fo, fp = d[f"{name}_fft_omega"], d[f"{name}_fft_power"]
                ax.plot(M + fo, fp / fp.max(), color=col, lw=0.8, label=f"|FFT|^2 (Hann) {lab}")
            res_e = js["correlators"][f"{op}_eigen"]
            res_p = js["correlators"][f"{op}_packet"]
            ax.vlines([M + ln["omega"] for ln in res_e["lines"]], 1e-8,
                      [ln["weight_rel"] for ln in res_e["lines"]], color="0.3", lw=3, alpha=0.35,
                      label=f"pencil, eigenstate ({res_e['n_lines_1e4']} lines)")
            ax.vlines([M + ln["omega"] for ln in res_p["lines"]], 1e-8,
                      [ln["weight_rel"] for ln in res_p["lines"]], color="C3", lw=1,
                      label=f"pencil, packet ({res_p['n_lines_1e4']} lines)")
            ax.axvspan(*ELASTIC, color="C2", alpha=0.1, lw=0, label="elastic window (k_j = 0 bra)")
            ax.set_yscale("log"); ax.set_ylim(3e-5, 3); ax.set_xlim(-0.6, 9.0)
            ax.set_xlabel("M + omega"); ax.set_ylabel("relative weight / power")
            ax.set_title(f"N_s = {ns}, {title}: C(t) spectra, T = {T:.0f}, dt = {dt}", fontsize=9)
            ax.legend(fontsize=6.5, loc="upper right")
            # assigned levels
            ax = axes[2 * r + 1, c]
            seen = set()
            for ln in res_p["lines"]:
                if ln["status"] not in ("ok", "degenerate"):
                    # pencil line not reproduced by an exact line (this T):
                    # shown at M + omega, no level assignment claimed
                    ax.plot(M + ln["omega"], ln["weight_rel"], "x", color="0.5", ms=3,
                            label=("pencil line without reliable assignment" if "x" not in seen else None))
                    seen.add("x")
                    continue
                x = ln["carrying"][0]
                ik = int(round(abs(x["k_j"]) / q1))
                col = kcol[ik % len(kcol)]
                lab = f"bra k_j = {ik} q" if ik not in seen else None
                seen.add(ik)
                ax.vlines(x["E_n"], 1e-8, ln["weight_rel"], color=col, lw=1.2, label=lab)
                if x["in_window"]:
                    ax.annotate(f"({x['n']},{x['P'] / q1:+.0f})", (x["E_n"], ln["weight_rel"]),
                                fontsize=6, ha="center", va="bottom", xytext=(0, 2),
                                textcoords="offset points", color=col)
            for ln in res_e["lines"]:
                if ln["carrying"]:
                    ax.vlines(ln["carrying"][0]["E_n"], 1e-8, ln["weight_rel"], color="0.3",
                              lw=3, alpha=0.3)
            for ent in res_p["window_missed"]:
                ax.plot(ent["E_n"], 3e-5 * 1.5, "v", color="0.4", ms=4)
            ax.axvspan(*ELASTIC, color="C2", alpha=0.1, lw=0)
            ax.set_yscale("log"); ax.set_ylim(3e-5, 3); ax.set_xlim(-0.6, 9.0)
            ax.set_xlabel("assigned level  E_n(P) - E_vac  (= omega + E_{k_j} - E_vac)")
            ax.set_ylabel("relative pencil weight")
            ax.set_title(f"N_s = {ns}, {op}: packet lines by assigned (n, P/q) [gray: eigenstate; "
                         f"triangles: window levels not exposed]", fontsize=9)
            ax.legend(fontsize=6.5, loc="upper right")
    fig.suptitle("Wavepacket vs eigenstate current correlators: exact real-time spectra and "
                 "matrix-pencil lines", fontsize=11)
    fig.tight_layout()
    fig.savefig(path)
    print("wrote", path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", type=int, default=12)
    ap.add_argument("--T", type=float, default=200.0)
    ap.add_argument("--dt", type=float, default=0.1)
    a = ap.parse_args()
    run(a.ns, a.T, a.dt)
    make_figure()
