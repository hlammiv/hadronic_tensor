"""Two-meson finite-volume levels from the hadronic tensor's own real-time
correlator (exact, N_s = 12, N_x = 6, PBC, Q = 0).

    C11(t)   = <k=0| J1(t) J1(0) |k=0>,        J1   = sum_b bond_current(b)
    C00_q(t) = <k=0| J0_q(t) J0_{-q}(0) |k=0>, J0_q = sum_v e^{-i q x_v} J0(v),
               q = 2 pi / N_x, x_v = v / 2

with |k=0> the single meson at rest.  Both correlators are autocorrelations
    C(t) = e^{i E_M t} <phi| e^{-iHt} |phi>,  phi = J |k=0>,
so C(t) = sum_n |<n|J|k=0>|^2 e^{-i (E_n - E_M) t}: every frequency omega is
an absolute level gap  E_n - E_vac = M + omega.  phi is evolved in its T2
momentum sector with the sparse sector Hamiltonian (gf_engine) and
scipy.sparse.linalg.expm_multiply; the frequencies are read off with a
matrix-pencil (ESPRIT) analysis of the uniformly sampled series, cross-checked
against an FFT and against the exact spectral decomposition of the sector.

    PYTHONPATH=. python scripts/rt_levels_ns12.py

Outputs: data/rt_levels_ns12.npz (time series + extracted levels),
         data/rt_levels_ns12.json (comparison table),
         data/rt_levels_ns12.pdf (spectra).
"""

import json
import sys
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.linalg import hankel, svd, eigvals, lstsq, pinv

from htensor.lattice import Z2Lattice
from htensor.gf_engine import (GFSpace, GFHamiltonian, gf_band,
                               _sector_hamiltonian, _bloch_to_sector)
from htensor.currents import charge_density, bond_current

NS = 12
M0, G2, ETA = 0.7, 1.1, 1.3
T_MAX = 40.0
DT = 0.1            # Nyquist pi/DT = 31.4 > full bandwidth of H (~25):
                    # no aliasing of high-energy (small-weight) modes
TWO_M = 5.490       # elastic threshold (2 M)
MMP = 6.030         # M M' inelastic threshold
WINDOW = (TWO_M, MMP)
SV_REL = 1e-8       # matrix-pencil singular-value cut (relative)
AMP_REL = 1e-3      # keep modes with weight > AMP_REL * max weight
MATCH_TOL = 0.02    # extracted vs ED level match tolerance

t0 = time.time()


def log(m):
    print(f"[{time.time() - t0:6.1f}s] {m}", flush=True)


# ------------------------------------------------------------ sector maps
def gf_to_sector(space, psi, k, idx):
    """Adjoint of _bloch_to_sector: gf-basis vector -> T2 = e^{ik} Bloch
    sector coordinates (exact when psi lies in that sector)."""
    rep, mrep, chi, ell, _ = space.orbits()
    ci = idx[rep]
    m = ci >= 0
    coef = np.exp(1j * k * mrep[m]) * chi[m] / np.sqrt(ell[m])
    v = np.zeros(idx.max() + 1, dtype=complex)
    np.add.at(v, ci[m], psi[m] * coef)
    return v


def sector_of(space, psi):
    """T2 phase of a (momentum eigen)state, on the grid 2 pi j / nx."""
    t2 = space.t2_expect(psi) / np.vdot(psi, psi)
    if abs(abs(t2) - 1) > 1e-8:
        raise ValueError(f"state is not a T2 eigenstate: |<T2>| = {abs(t2)}")
    k = float(np.angle(t2))
    grid = space.momentum_grid()
    kk = grid[np.argmin(np.abs(np.angle(np.exp(1j * (grid - k)))))]
    return float(kk)


# ------------------------------------------------------- matrix pencil
def matrix_pencil(y, dt, sv_rel=SV_REL, amp_rel=AMP_REL, L=None):
    """y[n] = sum_m a_m z_m^n, z_m = e^{-i omega_m dt}.  Hankel SVD with a
    relative singular-value cut, ESPRIT (shift-invariance of the left
    singular vectors), then a Vandermonde least-squares fit for the
    amplitudes.  Returns (omega, amp, |z|, rank, singular values)."""
    N = len(y)
    L = N // 2 if L is None else L
    Y = hankel(y[:L + 1], y[L:])                 # (L+1) x (N-L)
    U, s, _ = svd(Y, full_matrices=False)
    r = int(np.sum(s > sv_rel * s[0]))
    U1, U2 = U[:-1, :r], U[1:, :r]
    z = eigvals(pinv(U1) @ U2)
    V = z[None, :] ** np.arange(N)[:, None]
    a, *_ = lstsq(V, y)
    omega = -np.angle(z) / dt
    keep = np.abs(a) > amp_rel * np.abs(a).max()
    o = np.argsort(omega[keep])
    return omega[keep][o], a[keep][o], np.abs(z[keep])[o], r, s


def fft_peaks(y, t, npad=16):
    """|FFT| of the windowed series on a fine grid; returns (omega, power)
    and the local maxima above 1e-3 of the largest."""
    N = len(y)
    dt = t[1] - t[0]
    w = np.hanning(N)
    Y = np.fft.fft(y * w, n=npad * N)
    om = -2 * np.pi * np.fft.fftfreq(npad * N, d=dt)   # e^{-i omega t} -> -freq
    o = np.argsort(om)
    om, P = om[o], np.abs(Y[o])
    pk = np.flatnonzero((P[1:-1] > P[:-2]) & (P[1:-1] > P[2:])) + 1
    pk = pk[P[pk] > 1e-3 * P.max()]
    return om, P, om[pk], P[pk]


# ================================================================= main
def main():
    lat = Z2Lattice(NS, pbc=True)
    space = GFSpace(lat, q=0)
    H = GFHamiltonian(space, M0, G2, ETA)
    nx = lat.nx
    log(f"GFSpace ns={NS}: dim {space.dim}")

    # ---- vacuum and meson at rest (P = 0 sector)
    band = gf_band(space, M0, G2, ETA, k=0.0, n_per_k=4, log=log)
    vac, e0 = band["vacuum"], band["e0"]
    meson = band["states"][0]
    M = float(band["energy"][0])
    E_M = e0 + M
    log(f"E_vac = {e0:.6f}, meson k=0 gap M = {M:.6f}, "
        f"<T2> = {space.t2_expect(meson):.4f}")

    # ---- ED reference levels
    ed = np.load(f"data/deep_levels_ns{NS}.npz")
    ed_gaps, ed_ph, ed_refl = ed["gaps"], ed["phases"], ed["refl"]
    assert abs(ed_gaps[1] - M) < 1e-6, (ed_gaps[1], M)
    assert abs(ed["energies"][0] - e0) < 1e-6

    # ---- currents
    J1 = sum(bond_current(lat, b, ETA) for b in lat.bonds).simplify()
    q = 2 * np.pi / nx
    x = np.arange(NS) / 2.0                             # x_v = v/2
    J0q = sum((np.exp(-1j * q * x[v]) * charge_density(lat, v))
              for v in range(NS)).simplify()
    J0mq = sum((np.exp(+1j * q * x[v]) * charge_density(lat, v))
               for v in range(NS)).simplify()

    # ket phi = J |k=0>; the bra of the correlator is <k=0| J^dagger, so
    # C(t) = e^{iE_M t} <phi|e^{-iHt}|phi>:
    #   C11:   phi = J1 |k0>            (J1 hermitian)
    #   C00_q: phi = J0_{-q} |k0>       (J0_q^dagger = J0_{-q})
    kets = {
        "C11": ("sum_b J1_b  (total bond current, P=0)", space.apply(J1, meson)),
        "C00_q": (f"J0_q = sum_v e^(-i q x_v) J0(v), q = 2pi/{nx}",
                  space.apply(J0mq, meson)),
    }

    t = np.arange(0.0, T_MAX + DT / 2, DT)
    N = len(t)
    results, save = {}, {"t": t, "M": M, "e0": e0, "E_M": E_M, "dt": DT,
                         "T": T_MAX, "q": q}
    for name, (opname, phi) in kets.items():
        norm2 = float(np.vdot(phi, phi).real)
        k = sector_of(space, phi)
        elastic = abs(np.vdot(meson, phi)) ** 2
        log(f"{name}: |phi|^2 = {norm2:.6f}, T2 sector k = {k:+.4f} "
            f"(= {k / (2 * np.pi / nx):+.1f} x 2pi/nx), "
            f"elastic |<k0|phi>|^2 = {elastic:.3e}")
        Hk, R, idx = _sector_hamiltonian(space, H, k)
        nk = Hk.shape[0]
        phik = gf_to_sector(space, phi, k, idx)
        back = _bloch_to_sector(space, phik, k, idx)
        leak = np.linalg.norm(back - phi) / np.sqrt(norm2)
        log(f"  sector dim {nk}, nnz {Hk.nnz}, round-trip residual {leak:.2e}")
        assert leak < 1e-6, "phi is not entirely in the momentum sector"  # eigsh tol 1e-9

        # ---- exact real-time evolution (sparse expm_multiply, all times)
        ts = time.time()
        psi_t = spla.expm_multiply(-1j * Hk.tocsc(), phik, start=0.0,
                                   stop=T_MAX, num=N, endpoint=True)
        C = np.exp(1j * E_M * t) * (psi_t @ phik.conj())
        log(f"  expm_multiply {N} samples: {time.time() - ts:.2f}s, "
            f"C(0) = {C[0].real:.6f}, |C(T)| = {abs(C[-1]):.4f}")

        # ---- matrix pencil
        om, amp, zabs, rank, sv = matrix_pencil(C, DT)
        gaps_mp = M + om
        log(f"  matrix pencil: rank {rank}, {len(om)} modes above "
            f"{AMP_REL:g} x max weight; max |1-|z|| = {np.abs(1 - zabs).max():.1e}")
        # ---- FFT cross-check
        om_f, P_f, pk_om, pk_P = fft_peaks(C, t)
        # ---- exact spectral decomposition of the sector (cross-check only)
        w, v = np.linalg.eigh(Hk.toarray())
        wt = np.abs(v.conj().T @ phik) ** 2
        ex_gaps = w - e0
        ex_om = w - E_M

        # ---- ED comparison in this momentum sector
        sel = np.abs(np.angle(np.exp(1j * (ed_ph - k)))) < 1e-3
        sel_gaps, sel_refl = ed_gaps[sel], ed_refl[sel]
        rows, matched = [], set()
        for g_, a_, o_ in zip(gaps_mp, amp, om):
            j = int(np.argmin(np.abs(sel_gaps - g_)))
            d = float(g_ - sel_gaps[j])
            # exact weight of the nearest sector eigenlevel (truth)
            je = int(np.argmin(np.abs(ex_gaps - g_)))
            row = {"omega": float(o_), "E_abs_gap": float(g_),
                   "weight": float(np.abs(a_)), "weight_frac": float(np.abs(a_) / norm2),
                   "amp_phase": float(np.angle(a_)),
                   "matched_ED_gap": float(sel_gaps[j]),
                   "ED_refl": (None if not np.isfinite(sel_refl[j]) else float(sel_refl[j])),
                   "diff": d, "in_window": bool(WINDOW[0] < g_ < WINDOW[1]),
                   "exact_sector_gap": float(ex_gaps[je]),
                   "exact_weight": float(wt[je])}
            rows.append(row)
            if abs(d) < MATCH_TOL:
                matched.add(j)
            flag = "  <-- elastic window" if row["in_window"] else ""
            log(f"    omega {o_:+8.4f}  gap {g_:8.4f}  w {abs(a_):.3e} "
                f"({abs(a_) / norm2:.3e})  ED {sel_gaps[j]:8.4f} "
                f"(refl {sel_refl[j]:+.0f}) diff {d:+.1e}  exact w {wt[je]:.3e}{flag}")
        win = np.flatnonzero((sel_gaps > WINDOW[0]) & (sel_gaps < WINDOW[1]))
        unmatched = []
        for j in win:
            if j in matched:
                continue
            je = int(np.argmin(np.abs(ex_gaps - sel_gaps[j])))
            unmatched.append({"ED_gap": float(sel_gaps[j]),
                              "ED_refl": (None if not np.isfinite(sel_refl[j])
                                          else float(sel_refl[j])),
                              "exact_weight": float(wt[je])})
            log(f"    UNMATCHED ED level gap {sel_gaps[j]:.4f} refl {sel_refl[j]:+.0f}: "
                f"exact spectral weight {wt[je]:.2e}")
        # exact-decomposition levels in window with weight above cut, all sector
        # levels (degenerate pairs of ED at +-k appear once in the sector)
        ex_win = [(float(g), float(ww)) for g, ww in zip(ex_gaps, wt)
                  if WINDOW[0] < g < WINDOW[1]]
        results[name] = {
            "operator": opname, "sector_k": k, "sector_dim": int(nk),
            "norm2": norm2, "elastic_weight": float(elastic),
            "rank": int(rank), "n_modes": int(len(om)),
            "modes": rows, "unmatched_ED_in_window": unmatched,
            "fft_peaks": [{"omega": float(a), "gap": float(M + a), "power": float(p)}
                          for a, p in zip(pk_om, pk_P)],
            "exact_window_levels": ex_win,
            "ED_window_levels": [(float(sel_gaps[j]), None if not np.isfinite(sel_refl[j])
                                  else float(sel_refl[j])) for j in win],
        }
        save[f"{name}_t"] = C
        save[f"{name}_omega"] = om
        save[f"{name}_gap"] = gaps_mp
        save[f"{name}_weight"] = np.abs(amp)
        save[f"{name}_sv"] = sv
        save[f"{name}_fft_omega"] = om_f
        save[f"{name}_fft_power"] = P_f
        save[f"{name}_exact_gaps"] = ex_gaps
        save[f"{name}_exact_weights"] = wt
        save[f"{name}_sector_k"] = k

    # ---- reflection parity of the P=0 ket, directly (PhysicalBasis R)
    try:
        from htensor.gaugefixed import PhysicalBasis
        from htensor import hamiltonian as ham
        basis = PhysicalBasis(lat)
        Hfull = basis.matrix(ham.build_hamiltonian(lat, M0, G2, ETA))
        Rfull, variant = basis.select_reflection(Hfull)
        sel = np.flatnonzero(basis.q == 0)
        # map gf rows -> physical-basis rows through (z, h)
        rows_pb = basis.lut[(space.z << 1) | space.h]
        assert np.all(rows_pb >= 0)
        Rsub = Rfull[rows_pb][:, rows_pb]
        pm, pv = (np.vdot(meson, Rsub @ meson).real,
                  np.vdot(vac, Rsub @ vac).real)
        phi1 = kets["C11"][1]
        pj = np.vdot(phi1, Rsub @ phi1).real / np.vdot(phi1, phi1).real
        log(f"reflection variant {variant}: <R> vac {pv:+.4f}, meson {pm:+.4f}, "
            f"J1|k0> {pj:+.4f}")
        results["C11"]["refl_raw"] = {"vacuum": float(pv), "meson": float(pm),
                                      "J1_ket": float(pj), "variant": str(variant)}
    except Exception as exc:                      # diagnostics only
        log(f"reflection check skipped: {exc!r}")

    np.savez(f"data/rt_levels_ns{NS}.npz", **save)
    with open(f"data/rt_levels_ns{NS}.json", "w") as fh:
        json.dump({"ns": NS, "M": M, "e0": e0, "dt": DT, "T": T_MAX,
                   "window": WINDOW, "results": results}, fh, indent=1)

    # ---- figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(11, 7))
        for col, name in enumerate(kets):
            r = results[name]
            ax = axes[0, col]
            C = save[f"{name}_t"]
            ax.plot(t, C.real, lw=0.8, label="Re")
            ax.plot(t, C.imag, lw=0.8, label="Im")
            ax.set_xlabel("t"); ax.set_ylabel(f"{name}(t)")
            ax.set_title(f"{name}, sector k = {r['sector_k']:+.3f}")
            ax.legend(fontsize=8)
            ax = axes[1, col]
            gf = M + save[f"{name}_fft_omega"]
            ax.plot(gf, save[f"{name}_fft_power"] / save[f"{name}_fft_power"].max(),
                    color="0.6", lw=0.8, label="|FFT| (Hann)")
            ax.vlines(save[f"{name}_gap"], 0, save[f"{name}_weight"] / save[f"{name}_weight"].max(),
                      color="C3", lw=1.5, label="matrix pencil")
            g_ed = [g for g, _ in r["ED_window_levels"]]
            for g in g_ed:
                ax.axvline(g, color="C0", ls=":", lw=0.8)
            ax.axvspan(*WINDOW, color="C2", alpha=0.1, label="elastic window")
            ax.set_xlim(-0.5, 8.0); ax.set_yscale("log"); ax.set_ylim(1e-4, 1.5)
            ax.set_xlabel("E - E_vac"); ax.set_ylabel("weight (normalized)")
            ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(f"data/rt_levels_ns{NS}.pdf")
    except Exception as exc:
        log(f"figure skipped: {exc!r}")
    log("done")
    return results


if __name__ == "__main__":
    main()
