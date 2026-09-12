"""Two-meson finite-volume levels from the hadronic tensor's own real-time
correlator (exact, gauge-fixed Q=0 basis), N_s = 20 (N_x = 10).

  C11(t)   = <k=0| J1(t) J1(0) |k=0>,           J1 = sum_b bond_current(b)   (P = 0)
  C00_q(t) = <k=0| J0_q(t) J0_{-q}(0) |k=0>,    J0_q = sum_v e^{-i q x_v} J0(v),
             x_v = v/2, q = 2 pi / N_x                                        (P = q)

Both reduce to an autocorrelation of one ket:
  C(t) = e^{i E_M t} <phi| e^{-iHt} |phi>,   phi = J1|k=0>  resp.  J0_{-q}|k=0>,
so C(t) = sum_n |<n|phi>|^2 e^{-i (E_n - E_M) t}: every frequency omega is an
absolute level gap E_n - E_vac = (E_M - E_vac) + omega.  The ket is projected
on its T2 momentum sector and evolved with the sparse sector Hamiltonian
(gf_engine._sector_hamiltonian + scipy.sparse.linalg.expm_multiply).
Frequencies are extracted with a matrix-pencil (ESPRIT) fit and cross-checked
against a zero-padded FFT and against data/deep_levels_ns20.npz; the exact
spectral weights |<n|phi>|^2 of the lowest sector eigenstates (eigsh) are
computed as well, so "missing" levels can be attributed to a symmetry.

  PYTHONPATH=. python scripts/rt_levels_ns20.py [--T 40] [--dt 0.25] [--tag _T80]
Outputs: data/rt_levels_ns20.npz, .json, .pdf
"""

import argparse
import json
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np
import scipy.linalg as sla
import scipy.sparse.linalg as spla

from htensor.lattice import Z2Lattice
from htensor.currents import bond_current, charge_density
from htensor.gf_engine import (GFSpace, GFHamiltonian, _sector_hamiltonian,
                               _bloch_to_sector)

NS = 20
M0, G2, ETA = 0.7, 1.1, 1.3
M_MESON = 2.7451
ELASTIC = (2 * M_MESON, 6.030)
T0 = time.time()


def log(m):
    print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)


# ------------------------------------------------------------ sector algebra
def sector_project(space, psi, k, idx):
    """Orthogonal projection of a gf-basis vector on the T2 = e^{ik} Bloch
    sector, in sector coordinates (inverse of gf_engine._bloch_to_sector):
    v[c] = sum_{j in orbit(c)} psi[j] e^{+ik m_j} chi_j / sqrt(ell_j)."""
    rep, mrep, chi, ell, _ = space.orbits()
    ci = idx[rep]
    m = ci >= 0
    nk = int(idx.max()) + 1
    v = np.zeros(nk, dtype=complex)
    np.add.at(v, ci[m], psi[m] * np.exp(1j * k * mrep[m]) * chi[m] / np.sqrt(ell[m]))
    return v


def matrix_pencil(y, dt, L=None, rank_tol=1e-11, max_rank=80, max_damp=0.2):
    """ESPRIT / matrix pencil on a uniformly sampled series y_n = y(n dt).
    Hankel Y[i, j] = y[i + j] (N-L rows, L+1 cols), SVD, rank cut at
    s/s0 > rank_tol, rotational invariance of the signal subspace
    (pinv(V[:-1]) V[1:]) gives the poles z_k = e^{-i omega_k dt}; amplitudes
    by Vandermonde least squares.  Returns dict with omega, damping,
    amplitude (complex), weight (= |a|/max|a|), rank, singular values."""
    y = np.asarray(y, dtype=complex)
    N = len(y)
    if L is None:
        L = N // 2
    Y = np.array([y[i:i + L + 1] for i in range(N - L)])
    U, s, Vh = np.linalg.svd(Y, full_matrices=False)
    M = int(np.sum(s > rank_tol * s[0]))
    M = min(M, max_rank)
    V = Vh[:M].T            # rows of Vh span the row space of Y (no conjugation)
    Phi = np.linalg.pinv(V[:-1]) @ V[1:]
    z = np.linalg.eigvals(Phi)
    # poles with |log|z||/dt > max_damp are numerical (the exact signal is
    # a sum of pure phases); drop them before the Vandermonde fit, where
    # a growing pole would otherwise dominate the least squares
    keep = np.abs(np.log(np.abs(z)) / dt) < max_damp
    n_drop = int((~keep).sum())
    z = z[keep]
    n = np.arange(N)
    Z = z[None, :] ** n[:, None]
    a, *_ = np.linalg.lstsq(Z, y, rcond=None)
    omega = -np.angle(z) / dt
    damping = np.log(np.abs(z)) / dt
    resid = np.linalg.norm(Z @ a - y) / np.linalg.norm(y)
    return dict(omega=omega, damping=damping, amp=a, weight=np.abs(a) / np.abs(a).max(),
                rank=M, sv=s, resid=resid, z=z, n_dropped=n_drop)


def fft_peaks(y, dt, pad=1 << 16):
    """Zero-padded FFT of C(t) with a Hann window; returns (omega grid,
    |spectrum|) with omega defined by C(t) ~ e^{-i omega t}."""
    N = len(y)
    w = np.hanning(N)
    F = np.fft.fft(y * w, pad)
    om = -2 * np.pi * np.fft.fftfreq(pad, d=dt)   # e^{-i omega t} <-> +omega
    o = np.argsort(om)
    return om[o], np.abs(F[o])


def local_maxima(x, y, thr):
    i = np.flatnonzero((y[1:-1] > y[:-2]) & (y[1:-1] >= y[2:]) & (y[1:-1] > thr)) + 1
    return x[i], y[i]


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=float, default=40.0)
    ap.add_argument("--dt", type=float, default=0.25)
    ap.add_argument("--neig", type=int, default=36,
                    help="lowest sector eigenstates for the exact-weight cross-check")
    ap.add_argument("--rank-tol", type=float, default=1e-11)
    ap.add_argument("--tag", default="", help="suffix for the output files")
    ap.add_argument("--weight-cut", type=float, default=1e-3)
    args = ap.parse_args()
    T, dt = args.T, args.dt
    nt = int(round(T / dt)) + 1
    ts = np.arange(nt) * dt

    lat = Z2Lattice(NS, pbc=True)
    space = GFSpace(lat, q=0)
    H = GFHamiltonian(space, M0, G2, ETA)
    log(f"GFSpace ns={NS}: dim {space.dim}")
    space.orbits()
    log("orbits done")

    # ---- P = 0 sector: vacuum, meson at rest, low levels
    Hk0, R0, idx0 = _sector_hamiltonian(space, H, 0.0)
    log(f"P=0 sector dim {Hk0.shape[0]}, nnz {Hk0.nnz}")
    w0, v0 = spla.eigsh(Hk0, k=args.neig, which="SA", tol=1e-10, ncv=2 * args.neig + 8)
    o = np.argsort(w0); w0, v0 = w0[o], v0[:, o]
    e_vac, e_mes = float(w0[0]), float(w0[1])
    log(f"E_vac = {e_vac:.6f}, E_meson(k=0) = {e_mes:.6f}, gap = {e_mes - e_vac:.6f}")
    log(f"P=0 gaps: {np.round(w0 - e_vac, 4)}")
    meson = _bloch_to_sector(space, v0[:, 1], 0.0, idx0)
    t2 = space.t2_expect(meson)
    log(f"meson <T2> = {t2:.6f}, |psi| = {np.linalg.norm(meson):.6f}")

    # ---- kets
    J1 = sum(bond_current(lat, b, ETA) for b in lat.bonds).simplify()
    qq = 2 * np.pi / lat.nx
    xv = np.arange(NS) / 2.0
    J0mq = sum(np.exp(+1j * qq * xv[v]) * charge_density(lat, v) for v in range(NS)).simplify()
    phi1 = space.apply(J1, meson)
    phi0 = space.apply(J0mq, meson)
    n1, n0 = np.linalg.norm(phi1), np.linalg.norm(phi0)
    t2_1 = space.t2_expect(phi1) / n1 ** 2
    t2_0 = space.t2_expect(phi0) / n0 ** 2
    log(f"|J1 M| = {n1:.6f}, <T2> = {t2_1:.6f} (phase {np.angle(t2_1):+.4f})")
    log(f"|J0_-q M| = {n0:.6f}, <T2> = {t2_0:.6f} (phase {np.angle(t2_0):+.4f})")
    kq = float(np.round(np.angle(t2_0) / (2 * np.pi / lat.nx)) * 2 * np.pi / lat.nx)
    log(f"J0_-q ket sits in the T2 sector k = {kq:+.4f} (= {kq / (2 * np.pi / lat.nx):+.0f} x 2pi/Nx)")
    # disconnected pieces
    d1 = np.vdot(meson, phi1)
    log(f"<M|J1|M> = {d1:.3e} (C-odd operator between C-odd states: should vanish)")

    Hkq, Rq, idxq = _sector_hamiltonian(space, H, kq)
    log(f"P=q sector dim {Hkq.shape[0]}, nnz {Hkq.nnz}")
    wq, vq = spla.eigsh(Hkq, k=args.neig, which="SA", tol=1e-10, ncv=2 * args.neig + 8)
    o = np.argsort(wq); wq, vq = wq[o], vq[:, o]
    log(f"P=q gaps: {np.round(wq - e_vac, 4)}")

    results = {}
    series = {}
    mp_dropped = {}
    for name, phi, k, Hk, idx, wsec, vsec in [
            ("C11", phi1, 0.0, Hk0, idx0, w0, v0),
            ("C00_q", phi0, kq, Hkq, idxq, wq, vq)]:
        v = sector_project(space, phi, k, idx)
        back = _bloch_to_sector(space, v, k, idx)
        leak = np.linalg.norm(back - phi) / np.linalg.norm(phi)
        log(f"{name}: sector projection leak {leak:.2e}, |v| = {np.linalg.norm(v):.6f}")
        # energy spread of the ket (aliasing check: Nyquist pi/dt)
        hv = Hk @ v
        eb = np.real(np.vdot(v, hv)) / np.vdot(v, v).real
        sig = np.sqrt(max(np.real(np.vdot(hv, hv)) / np.vdot(v, v).real - eb ** 2, 0))
        log(f"{name}: <E>-E_M = {eb - e_mes:.4f}, sigma_E = {sig:.4f}, Nyquist {np.pi / dt:.2f}")
        t1 = time.time()
        U = spla.expm_multiply(-1j * Hk, v, start=0.0, stop=T, num=nt, endpoint=True)
        C = np.exp(1j * e_mes * ts) * (U @ v.conj())      # <v| e^{-iHt} |v>
        log(f"{name}: expm_multiply {nt} points in {time.time() - t1:.1f}s; "
            f"C(0) = {C[0]:.6f}, |C(T)| = {abs(C[-1]):.6f}")
        # unitarity check: norm of last vector
        log(f"{name}: |U(T) v| / |v| - 1 = {np.linalg.norm(U[-1]) / np.linalg.norm(v) - 1:.2e}")
        series[name] = C
        # exact weights on the lowest sector eigenstates
        ov = vsec.conj().T @ v
        wts = np.abs(ov) ** 2
        # matrix pencil
        mp = matrix_pencil(C, dt, rank_tol=args.rank_tol)
        log(f"{name}: pencil rank {mp['rank']}, sv[0..] = {np.array2string(mp['sv'][:mp['rank'] + 3] / mp['sv'][0], precision=2)}, "
            f"fit resid {mp['resid']:.2e}, {mp['n_dropped']} growing/decaying poles dropped")
        keep = mp["weight"] > args.weight_cut
        order = np.argsort(mp["omega"][keep])
        om = mp["omega"][keep][order]; wt = mp["weight"][keep][order]
        amp = mp["amp"][keep][order]; dmp = mp["damping"][keep][order]
        # FFT cross-check
        fo, fa = fft_peaks(C, dt)
        pk_o, pk_a = local_maxima(fo, fa, 1e-3 * fa.max())
        mp_dropped[name] = mp["n_dropped"]
        results[name] = dict(k=k, omega=om, weight=wt, amp=amp, damping=dmp,
                             rank=mp["rank"], sv=mp["sv"], resid=mp["resid"],
                             fft_o=fo, fft_a=fa, fft_pk_o=pk_o, fft_pk_a=pk_a,
                             sec_E=wsec, sec_w=wts, C=C, norm2=float(np.vdot(v, v).real))
        log(f"{name}: extracted (E_abs_gap, weight, damping):")
        for o_, w_, a_, d_ in zip(om, wt, amp, dmp):
            log(f"    E = {e_mes - e_vac + o_:8.4f}  omega = {o_:+8.4f}  w = {w_:.4e}  "
                f"amp = {a_.real:+.4e}{a_.imag:+.1e}i  damp = {d_:+.1e}")
        log(f"{name}: FFT peaks at E = {np.round(e_mes - e_vac + pk_o, 3)}")
        log(f"{name}: exact |<n|phi>|^2 / |phi|^2 on lowest sector levels:")
        for E_, w_ in zip(wsec - e_vac, wts / np.vdot(v, v).real):
            log(f"    E = {E_:8.4f}  w = {w_:.3e}")

    # ---- compare with the ED file
    ed = np.load(f"data/deep_levels_ns{NS}.npz")
    gaps, phases, refl = ed["gaps"], ed["phases"], ed["refl"]
    gap_M = e_mes - e_vac
    out = dict(ns=NS, nx=lat.nx, T=T, dt=dt, E_vac=e_vac, E_meson=e_mes,
               meson_k0_gap=gap_M, elastic_window=list(ELASTIC),
               weight_cut=args.weight_cut, rank_tol=args.rank_tol, correlators=[])
    for name, r in results.items():
        k = r["k"]
        sel = np.abs(np.angle(np.exp(1j * (phases - k)))) < 1e-6
        ed_g, ed_r = gaps[sel], refl[sel]
        win = (ed_g > ELASTIC[0]) & (ed_g < ELASTIC[1])
        levels = []
        matched = set()
        C0 = r["C"][0].real
        for o_, w_, d_, a_ in zip(r["omega"], r["weight"], r["damping"], r["amp"]):
            E = gap_M + o_
            j = int(np.argmin(np.abs(ed_g - E)))
            rec = dict(E_abs_gap=float(E), omega=float(o_), weight=float(w_),
                       damping=float(d_), abs_weight=float(a_.real / C0),
                       matched_ED_gap=float(ed_g[j]),
                       ED_refl=(None if not np.isfinite(ed_r[j]) else float(ed_r[j])),
                       diff=float(E - ed_g[j]),
                       in_window=bool(ELASTIC[0] < E < ELASTIC[1]))
            if abs(E - ed_g[j]) < 0.02:
                matched.add(j)
            levels.append(rec)
        def exact_w(E):
            i = int(np.argmin(np.abs(r["sec_E"] - e_vac - E)))
            return float(r["sec_w"][i] / r["norm2"]) if abs(r["sec_E"][i] - e_vac - E) < 1e-3 else None
        unmatched = [dict(gap=float(ed_g[j]), refl=(None if not np.isfinite(ed_r[j]) else float(ed_r[j])),
                          exact_weight=exact_w(ed_g[j]))
                     for j in np.flatnonzero(win) if j not in matched]
        # exact weights of the sector eigenstates in the window
        sec = [dict(gap=float(E_ - e_vac), weight=float(w_ / r["norm2"]))
               for E_, w_ in zip(r["sec_E"], r["sec_w"])]
        out["correlators"].append(dict(
            name=name, total_momentum=float(k), pencil_rank=int(r["rank"]),
            fit_residual=float(r["resid"]), levels=levels,
            ED_levels_in_window=[dict(gap=float(ed_g[j]), refl=(None if not np.isfinite(ed_r[j]) else float(ed_r[j])),
                                      exact_weight=exact_w(ed_g[j]))
                                 for j in np.flatnonzero(win)],
            n_poles_dropped=int(mp_dropped[name]),
            unmatched_ED_in_window=unmatched,
            fft_peaks_E=[float(gap_M + p) for p in r["fft_pk_o"]],
            exact_sector_weights=sec))
        log(f"{name} (P={k:+.4f}): ED levels in window {np.round(ed_g[win], 4)}; "
            f"unmatched: {[u['gap'] for u in unmatched]}")

    np.savez(f"data/rt_levels_ns{NS}{args.tag}.npz", t=ts, C11=series["C11"], C00_q=series["C00_q"],
             E_vac=e_vac, E_meson=e_mes, q=qq, k_C00=kq, T=T, dt=dt,
             P0_levels=w0, Pq_levels=wq,
             C11_omega=results["C11"]["omega"], C11_weight=results["C11"]["weight"],
             C00_omega=results["C00_q"]["omega"], C00_weight=results["C00_q"]["weight"],
             C11_sector_weights=results["C11"]["sec_w"] / results["C11"]["norm2"],
             C00_sector_weights=results["C00_q"]["sec_w"] / results["C00_q"]["norm2"],
             m0=M0, g2=G2, eta=ETA)
    with open(f"data/rt_levels_ns{NS}{args.tag}.json", "w") as f:
        json.dump(out, f, indent=1)

    # ---- figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for col, (name, r) in enumerate(results.items()):
        k = r["k"]
        ax = axes[0, col]
        C = r["C"] / r["C"][0].real
        ax.plot(ts, C.real, lw=0.8, label="Re")
        ax.plot(ts, C.imag, lw=0.8, label="Im")
        ax.set_xlabel("t"); ax.set_ylabel(f"{name}(t)/{name}(0)")
        ax.set_title(f"{name}, P = {k:+.3f}")
        ax.legend(fontsize=8)
        ax = axes[1, col]
        E = gap_M + r["fft_o"]
        ax.semilogy(E, r["fft_a"] / r["fft_a"].max(), color="0.5", lw=0.8, label="|FFT| (Hann)")
        sel = np.abs(np.angle(np.exp(1j * (phases - k)))) < 1e-6
        for g in gaps[sel]:
            ax.axvline(g, color="C3", lw=0.6, alpha=0.5)
        ax.vlines(gap_M + r["omega"], 1e-5, r["weight"], color="C0", lw=2, label="matrix pencil")
        ax.axvspan(*ELASTIC, color="C2", alpha=0.12, label="elastic window")
        ax.set_xlim(-0.5, 8); ax.set_ylim(1e-5, 2)
        ax.set_xlabel("E - E_vac"); ax.set_ylabel("weight")
        ax.legend(fontsize=8, loc="upper right")
    fig.suptitle(f"ns = {NS}: levels from the real-time correlator (T = {T}, dt = {dt}); red = ED")
    fig.tight_layout()
    fig.savefig(f"data/rt_levels_ns{NS}{args.tag}.pdf")
    log("done")


if __name__ == "__main__":
    main()
