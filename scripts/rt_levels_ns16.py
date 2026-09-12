"""Two-meson finite-volume levels from the hadronic tensor's own real-time
correlator, exactly, at N_s = 16 (N_x = 8), production couplings.

  C11(t)   = <k=0| J1(t) J1(0) |k=0>,   J1 = sum_b bond_current(b)   (P = 0)
  C00_q(t) = <k=0| J0_q(t) J0_{-q}(0) |k=0>,  J0_q = sum_v e^{-i q x_v} J0(v),
             x_v = v/2, q = 2 pi / N_x                               (P = q)

Both are evaluated in the gauge-fixed Q = 0 basis (gf_engine.GFSpace) by
projecting the ket J|k=0> onto its T2 momentum sector and evolving it with
the SPARSE sector Hamiltonian (scipy.sparse.linalg.expm_multiply); the bra
|k=0> is evolved the same way in the P = 0 sector.  Frequencies are then
read off with a matrix-pencil (Hankel-SVD + generalized eigenvalue) fit and
cross-checked against (i) a plain FFT and (ii) the exact spectral weights
|<n|J|k=0>|^2 from a dense diagonalization of the (~3200-dim) sectors,
and compared with data/deep_levels_ns16.npz.

Run:  PYTHONPATH=. python scripts/rt_levels_ns16.py  [--T 40 --dt 0.25]
Writes data/rt_levels_ns16.{npz,json,pdf}.
"""
import argparse
import json
import os
import time

os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np
import scipy.sparse.linalg as spla

from htensor.lattice import Z2Lattice
from htensor.gf_engine import GFSpace, GFHamiltonian, gf_band, _sector_hamiltonian
from htensor import currents as cur

M0, G2, ETA = 0.7, 1.1, 1.3
ELASTIC = (5.490, 6.030)          # 2M < gap < M + M'


# ----------------------------------------------------------- sector maps
# float64 versions of gf_engine._bloch_to_sector and its inverse (the
# library version takes np.sqrt of an int16 array -> float32, 3e-8 noise).
def gf_to_sector(space, psi, k, idx):
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
def matrix_pencil(y, dt, rank=None, sv_rel=1e-7, L=None):
    """y[n] = sum_j A_j z_j^n (uniform samples).  Hankel Y (N-L) x (L+1),
    SVD, keep `rank` (or all singular values > sv_rel * s_max), then
    z = eig(pinv(V1) V2) with V1/V2 the row-shifted right singular vectors;
    amplitudes by least squares.  Returns omega (y ~ sum A e^{-i omega t}),
    A, |z|, singular values, rank."""
    N = len(y)
    L = N // 2 if L is None else L
    Y = np.array([y[i:i + L + 1] for i in range(N - L)])
    U, s, Vh = np.linalg.svd(Y, full_matrices=False)
    if rank is None:
        rank = int(np.sum(s > sv_rel * s[0]))
    V = Vh[:rank].conj().T                    # (L+1) x rank
    V1, V2 = V[:-1], V[1:]
    # the columns of V span the CONJUGATE Vandermonde space (Y = U S V^H),
    # so the shift eigenvalues come out conjugated
    z = np.conj(np.linalg.eigvals(np.linalg.pinv(V1) @ V2))
    omega = -np.angle(z) / dt
    Z = z[None, :] ** np.arange(N)[:, None]
    A = np.linalg.lstsq(Z, y, rcond=None)[0]
    o = np.argsort(omega)
    return omega[o], A[o], np.abs(z[o]), s, rank


def fft_peaks(y, dt, n_pad=16):
    """|FFT| of the series on a zero-padded grid; returns (omega, power)
    sorted by omega with the e^{-i omega t} sign convention."""
    N = len(y)
    w = np.hanning(N)
    Y = np.fft.fft(y * w, n_pad * N)
    om = -2 * np.pi * np.fft.fftfreq(n_pad * N, d=dt)
    o = np.argsort(om)
    return om[o], np.abs(Y[o]) ** 2


def local_maxima(om, p, rel=1e-3):
    idx = [i for i in range(1, len(p) - 1) if p[i] > p[i - 1] and p[i] >= p[i + 1]
           and p[i] > rel * p.max()]
    return om[idx], p[idx]


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", type=int, default=16)
    ap.add_argument("--T", type=float, default=40.0)
    ap.add_argument("--dt", type=float, default=0.25)
    ap.add_argument("--sv-rel", type=float, default=1e-7)
    ap.add_argument("--dense-check", type=int, default=1,
                    help="dense eigh of each sector for exact spectral weights")
    ap.add_argument("--out", default="data/rt_levels_ns16")
    args = ap.parse_args()
    t0 = time.time()
    log = lambda *a: print(f"[{time.time() - t0:7.1f}s]", *a, flush=True)

    lat = Z2Lattice(args.ns)
    space = GFSpace(lat, q=0)
    H = GFHamiltonian(space, M0, G2, ETA)
    log(f"ns={lat.ns} nx={lat.nx} gf dim={space.dim}")

    # -- vacuum + single-meson band (all momenta, for reference energies)
    band = gf_band(space, M0, G2, ETA, n_per_k=3)
    e0 = band["e0"]
    i0 = int(np.argmin(np.abs(band["k"])))
    psi = band["states"][i0]
    psi = psi / np.linalg.norm(psi)
    Ek = float(np.real(np.vdot(psi, H.apply(psi))))
    gap_meson = Ek - e0
    log(f"E_vac={e0:.6f}  E_meson(k=0)={Ek:.6f}  gap={gap_meson:.5f}")
    band_gaps = {float(k): float(e) for k, e in zip(band["k"], band["energy"])}

    # -- currents
    q = 2 * np.pi / lat.nx
    J1 = sum(cur.bond_current(lat, b, ETA) for b in lat.bonds).simplify()
    J0 = {s: sum(np.exp(-1j * s * q * v / 2) * cur.charge_density(lat, v)
                 for v in range(lat.ns)).simplify() for s in (+1, -1)}
    # C-even partner (axial/scalar density) at the same momentum, used only
    # to diagnose the symmetry of levels the vector current cannot reach
    S0 = {s: sum(np.exp(-1j * s * q * v / 2) * cur.axial_charge_density(lat, v)
                 for v in range(lat.ns)).simplify() for s in (+1, -1)}

    grid = space.momentum_grid()
    kq = float(grid[np.argmin(np.abs(grid - q))])          # exact grid value

    corrs = {
        "C11": dict(operator="J1 = sum_b bond_current(b, eta)", k=0.0,
                    ket_op=J1, bra_op=J1, diag_op=None,
                    label="C11(t) = <k=0|J1(t) J1(0)|k=0>"),
        "C00_q": dict(operator="J0_q = sum_v exp(-i q v/2) charge_density(v), q = 2pi/Nx",
                      k=kq, ket_op=J0[-1], bra_op=J0[+1], diag_op=S0[-1],
                      label="C00_q(t) = <k=0|J0_q(t) J0_{-q}(0)|k=0>"),
    }

    ts = np.arange(0.0, args.T + 1e-9, args.dt)
    Nt = len(ts)
    # bra |k=0> evolved in the P = 0 sector (a phase, but done as asked)
    H0, R0, idx0 = _sector_hamiltonian(space, H, 0.0)
    v_bra = gf_to_sector(space, psi, 0.0, idx0)
    assert np.linalg.norm(sector_to_gf(space, v_bra, 0.0, idx0) - psi) < 1e-10
    bra_t = spla.expm_multiply(-1j * H0.tocsc(), v_bra, start=0.0, stop=args.T,
                               num=Nt, endpoint=True)
    phase_err = np.max(np.abs(bra_t - np.exp(-1j * Ek * ts)[:, None] * v_bra[None, :]))
    log(f"bra evolution: max |psi(t) - e^(-iEt) psi| = {phase_err:.2e}")

    out_npz = {"t": ts, "dt": args.dt, "T": args.T, "E_vac": e0, "E_meson_k0": Ek,
               "gap_meson_k0": gap_meson, "q": q,
               "band_k": band["k"], "band_gap": band["energy"]}
    results = []
    ed = np.load(f"data/deep_levels_ns{lat.ns}.npz")
    ed_g, ed_ph, ed_r = ed["gaps"], ed["phases"], ed["refl"]

    for name, c in corrs.items():
        k = c["k"]
        chi_ket = space.apply(c["ket_op"], psi)          # J_{-q} |k=0>
        chi_bra = space.apply(c["bra_op"].adjoint(), psi)   # (J_q)^dag |k=0> = J_{-q}|k=0>
        t2 = space.t2_expect(chi_ket) / np.vdot(chi_ket, chi_ket)
        log(f"{name}: |J psi| = {np.linalg.norm(chi_ket):.6f}, <T2> = {t2:.6f} "
            f"(sector k = {k:+.6f}, e^ik = {np.exp(1j * k):.6f})")
        Hk, R, idx = _sector_hamiltonian(space, H, k)
        v_ket = gf_to_sector(space, chi_ket, k, idx)
        leak = np.linalg.norm(sector_to_gf(space, v_ket, k, idx) - chi_ket) / np.linalg.norm(chi_ket)
        log(f"{name}: sector dim {Hk.shape[0]}, out-of-sector leakage {leak:.1e}")
        # ket evolution with the sparse sector Hamiltonian
        ket_t = spla.expm_multiply(-1j * Hk.tocsc(), v_ket, start=0.0, stop=args.T,
                                   num=Nt, endpoint=True)
        # C(t) = <psi(t)| J_q |phi(t)>, J applied in the gf basis
        C = np.empty(Nt, dtype=complex)
        for n in range(Nt):
            bra_gf = sector_to_gf(space, bra_t[n], 0.0, idx0)
            ket_gf = sector_to_gf(space, ket_t[n], k, idx)
            C[n] = np.vdot(space.apply(c["bra_op"].adjoint(), bra_gf), ket_gf)
        # cross-check: autocorrelation form e^{iE t} <chi| e^{-iHt} |chi>
        C_auto = np.exp(1j * Ek * ts) * (ket_t @ v_ket.conj())
        log(f"{name}: C(0) = {C[0]:.6f}; max |C - autocorr| = {np.max(np.abs(C - C_auto)):.1e}; "
            f"evolved {Nt} steps")
        out_npz[f"{name}_t"] = C

        # matrix pencil
        om, A, absz, sv, rank = matrix_pencil(C, args.dt, sv_rel=args.sv_rel)
        wmax = np.max(np.abs(A))
        keep = np.abs(A) > 1e-3 * wmax
        log(f"{name}: pencil rank {rank} (sv/s0 at cut: {sv[rank - 1] / sv[0]:.1e}, "
            f"next {sv[rank] / sv[0]:.1e}); {keep.sum()} modes above 1e-3")
        modes = [(float(o), float(abs(a)), float(np.real(a)), float(z))
                 for o, a, z in zip(om[keep], A[keep], absz[keep])]
        # rank stability of the lines inside the elastic window
        stab = {}
        for rk in (max(4, rank - 6), rank - 3, rank, rank + 3, rank + 6, rank + 12):
            o_, A_, _, _, _ = matrix_pencil(C, args.dt, rank=rk)
            m_ = (np.abs(A_) > 1e-3 * np.max(np.abs(A_)))
            g_ = gap_meson + o_[m_]
            w_ = (g_ > ELASTIC[0]) & (g_ < ELASTIC[1])
            stab[int(rk)] = [[float(g), float(abs(a))] for g, a in zip(g_[w_], A_[m_][w_])]
        log(f"{name}: window lines vs pencil rank: " +
            "; ".join(f"r={r}: " + ",".join(f"{g:.4f}" for g, _ in v) for r, v in stab.items()))
        # FFT cross-check
        fom, fp = fft_peaks(C, args.dt)
        pk_om, pk_p = local_maxima(fom, fp, rel=1e-3)
        out_npz[f"{name}_pencil_omega"] = om
        out_npz[f"{name}_pencil_amp"] = A
        out_npz[f"{name}_pencil_absz"] = absz
        out_npz[f"{name}_pencil_sv"] = sv
        out_npz[f"{name}_fft_omega"] = fom
        out_npz[f"{name}_fft_power"] = fp

        # exact spectral weights (dense sector eigh) as ground truth
        dense = None
        if args.dense_check:
            w, U = np.linalg.eigh(Hk.toarray())
            a = U.conj().T @ v_ket
            wt = np.abs(a) ** 2
            wt_s = None
            if c["diag_op"] is not None:
                vs = gf_to_sector(space, space.apply(c["diag_op"], psi), k, idx)
                wt_s = np.abs(U.conj().T @ vs) ** 2
            dense = (w - e0, wt, wt_s)
            out_npz[f"{name}_dense_gap"] = w - e0
            out_npz[f"{name}_dense_weight"] = wt
            if wt_s is not None:
                out_npz[f"{name}_dense_weight_Ceven"] = wt_s
            log(f"{name}: dense sector eigh done; total weight {wt.sum():.6f} "
                f"(= |J psi|^2 {np.vdot(v_ket, v_ket).real:.6f}); weight at |omega| > pi/dt: "
                f"{wt[np.abs(w - Ek) > np.pi / args.dt].sum():.1e}")

        # ED comparison in this momentum sector
        sel_ed = np.abs(np.angle(np.exp(1j * (ed_ph - k)))) < 1e-3
        ed_sec = np.flatnonzero(sel_ed)
        extracted = []
        for o, wabs, wre, z in modes:
            gap = gap_meson + o
            j = ed_sec[np.argmin(np.abs(ed_g[ed_sec] - gap))]
            entry = {"omega": o, "E_abs_gap": gap, "weight": wabs, "weight_re": wre,
                     "abs_z": z, "matched_ED_gap": float(ed_g[j]),
                     "diff": float(gap - ed_g[j]),
                     "ED_refl": (None if np.isnan(ed_r[j]) else float(ed_r[j])),
                     "in_elastic_window": bool(ELASTIC[0] < gap < ELASTIC[1])}
            if dense is not None:
                dg, dw, dws = dense
                jd = int(np.argmin(np.abs(dg - gap)))
                entry["dense_gap"] = float(dg[jd])
                entry["dense_weight"] = float(dw[jd])
            extracted.append(entry)
        # ED levels in the window + sector that the correlator did not give
        win = ed_sec[(ed_g[ed_sec] > ELASTIC[0]) & (ed_g[ed_sec] < ELASTIC[1])]
        win_gaps = sorted(set(np.round(ed_g[win], 5)))
        got = [e["E_abs_gap"] for e in extracted if e["in_elastic_window"]]
        unmatched = []
        for g in win_gaps:
            if not any(abs(g - x) < 5e-3 for x in got):
                info = {"ED_gap": float(g)}
                jj = win[np.argmin(np.abs(ed_g[win] - g))]
                info["ED_refl"] = None if np.isnan(ed_r[jj]) else float(ed_r[jj])
                if dense is not None:
                    dg, dw, dws = dense
                    jd = int(np.argmin(np.abs(dg - g)))
                    info["dense_weight_J"] = float(dw[jd])
                    if dws is not None:
                        info["dense_weight_Ceven_density"] = float(dws[jd])
                unmatched.append(info)
        results.append({"name": name, "label": c["label"], "operator": c["operator"],
                        "k": k, "pencil_rank": rank,
                        "sv_ratio_at_cut": float(sv[rank - 1] / sv[0]),
                        "sv_ratio_next": float(sv[rank] / sv[0]),
                        "modes": extracted, "window_lines_vs_rank": stab,
                        "fft_peaks": [{"omega": float(o), "E_abs_gap": float(gap_meson + o),
                                       "power_rel": float(p / pk_p.max())}
                                      for o, p in zip(pk_om, pk_p)],
                        "ED_window_levels": [float(g) for g in win_gaps],
                        "unmatched_ED_levels_in_window": unmatched,
                        "dense_levels_below_6.3": (None if dense is None else
                            [{"gap": float(g), "weight_J": float(wj),
                              "weight_Ceven": (None if dense[2] is None else float(ws))}
                             for g, wj, ws in zip(dense[0], dense[1],
                                                  dense[2] if dense[2] is not None else dense[1])
                             if g < 6.3 and (wj > 1e-12 or (dense[2] is not None and ws > 1e-12))])})
        log(f"{name}: modes (omega, gap, |A|):")
        for e in extracted:
            log(f"    omega={e['omega']:+9.5f}  gap={e['E_abs_gap']:8.5f}  |A|={e['weight']:.4e}"
                f"  |z|={e['abs_z']:.6f}  ED={e['matched_ED_gap']:.5f} diff={e['diff']:+.1e}"
                f"  refl={e['ED_refl']}  {'*' if e['in_elastic_window'] else ''}")
        log(f"{name}: FFT peaks (gap, rel power): " +
            ", ".join(f"{gap_meson + o:.3f}({p / pk_p.max():.2f})" for o, p in zip(pk_om, pk_p)))
        log(f"{name}: unmatched ED levels in window: {unmatched}")

    np.savez(args.out + ".npz", **out_npz)
    summary = {"ns": lat.ns, "nx": lat.nx, "couplings": [M0, G2, ETA], "T": args.T,
               "dt": args.dt, "n_samples": Nt, "E_vac": e0, "E_meson_k0": Ek,
               "gap_meson_k0": gap_meson, "band_gaps": band_gaps,
               "elastic_window": ELASTIC, "bra_phase_err": float(phase_err),
               "correlators": results, "wall_s": time.time() - t0}
    with open(args.out + ".json", "w") as f:
        json.dump(summary, f, indent=1)

    # -- figure: time series + spectral lines vs ED
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = {"C11": "#2a78d6", "C00_q": "#eb6834"}
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for j, (name, c) in enumerate(corrs.items()):
        C = out_npz[f"{name}_t"]
        ax = axes[0, j]
        ax.plot(ts, C.real, color=col[name], lw=1.2, label="Re")
        ax.plot(ts, C.imag, color=col[name], lw=1.2, ls="--", label="Im")
        ax.set_xlabel("t"); ax.set_title(c["label"], fontsize=10)
        ax.legend(frameon=False); ax.grid(alpha=0.2)
        ax = axes[1, j]
        r = results[j]
        k = c["k"]
        sel_ed = np.abs(np.angle(np.exp(1j * (ed_ph - k)))) < 1e-3
        for g in sorted(set(np.round(ed_g[sel_ed], 5))):
            if -0.1 < g < 6.6:
                ax.axvline(g, color="#c3c2b7", lw=0.8, zorder=0)
        for e in r["modes"]:
            ax.vlines(e["E_abs_gap"], 1e-4, e["weight"] / max(x["weight"] for x in r["modes"]),
                      color=col[name], lw=2)
        ax.axvspan(*ELASTIC, color=col[name], alpha=0.08, lw=0)
        ax.set_yscale("log"); ax.set_ylim(5e-4, 2)
        ax.set_xlim(-0.2, 6.6); ax.set_xlabel("E - E_vac  (pencil line = gap_meson + omega)")
        ax.set_ylabel("relative pencil weight")
        ax.set_title(f"{name}: pencil lines (rank {r['pencil_rank']}) vs ED levels (gray)",
                     fontsize=10)
    fig.suptitle(f"ns={lat.ns}: two-meson levels from exact real-time current correlators, "
                 f"T={args.T}, dt={args.dt}", fontsize=11)
    fig.tight_layout()
    fig.savefig(args.out + ".pdf")
    log(f"wrote {args.out}.npz/.json/.pdf")


if __name__ == "__main__":
    main()
