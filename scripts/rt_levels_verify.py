"""Adversarial verification of scripts/rt_levels_ns{12,16,20}.py and
scripts/rt_levels_luscher.py.

    PYTHONPATH=. python scripts/rt_levels_verify.py ns12      # independent dense ED, FFT / NLLS extraction
    PYTHONPATH=. python scripts/rt_levels_verify.py ns20|ns16 # independent sector recomputation (eigsh weights, expm series, FFT/NLLS)
    PYTHONPATH=. python scripts/rt_levels_verify.py stored    # resolution + noise tests on the stored series
    PYTHONPATH=. python scripts/rt_levels_verify.py luscher   # independent Luscher inversion + RMS

Outputs: data/rt_levels_verify_<mode>.json (+ .npz for ns12/ns20).
"""
import json
import sys
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.optimize import least_squares, minimize_scalar, brentq
from scipy.interpolate import CubicSpline
from scipy.linalg import hankel, svd, eigvals, lstsq, pinv

from htensor.lattice import Z2Lattice
from htensor.gf_engine import GFSpace, GFHamiltonian, gf_band, _sector_hamiltonian
from htensor.currents import charge_density, bond_current, axial_charge_density

M0, G2, ETA = 0.7, 1.1, 1.3
WINDOW = (5.490, 6.030)
T0 = time.time()


def log(m):
    print(f"[{time.time() - T0:6.1f}s] {m}", flush=True)


def fl(x):
    return [float(v) for v in np.asarray(x).ravel()]


# ---------------------------------------------------------------- extraction
def fft_lines(C, dt, window="hann", npad=32, rel=1e-3):
    """Periodogram peaks of C(t) = sum a e^{-i w t}; each local maximum is
    refined by maximising |sum_t w(t) C(t) e^{+i w t}| (Fourier magnitude)
    with a bounded scalar search inside the bracket of the discrete peak.
    Returns (omega, |amp| estimate, coarse grid, power)."""
    N = len(C)
    t = np.arange(N) * dt
    w = np.hanning(N) if window == "hann" else np.ones(N)
    Y = np.fft.fft(C * w, n=npad * N)
    om = -2 * np.pi * np.fft.fftfreq(npad * N, d=dt)
    o = np.argsort(om)
    om, P = om[o], np.abs(Y[o])
    pk = np.flatnonzero((P[1:-1] > P[:-2]) & (P[1:-1] > P[2:])) + 1
    pk = pk[P[pk] > rel * P.max()]
    out_w, out_a = [], []
    for i in pk:
        f = lambda x: -abs(np.sum(w * C * np.exp(1j * x * t)))
        r = minimize_scalar(f, bounds=(om[i - 1], om[i + 1]), method="bounded",
                            options={"xatol": 1e-10})
        out_w.append(r.x)
        out_a.append(-r.fun / w.sum())
    return np.array(out_w), np.array(out_a), om, P


def nlls_lines(C, dt, w0, a0):
    """Nonlinear least squares of C(t) = sum_m a_m e^{-i w_m t} (complex a),
    starting from (w0, a0).  No Hankel / pencil / GEVP involved."""
    N = len(C)
    t = np.arange(N) * dt
    K = len(w0)

    def model(p):
        w = p[:K]
        a = p[K:2 * K] + 1j * p[2 * K:]
        return (a[None, :] * np.exp(-1j * np.outer(t, w))).sum(1)

    def res(p):
        d = model(p) - C
        return np.concatenate([d.real, d.imag])

    p0 = np.concatenate([w0, a0.real, a0.imag])
    r = least_squares(res, p0, xtol=1e-14, ftol=1e-14, gtol=1e-14, max_nfev=4000)
    w = r.x[:K]
    a = r.x[K:2 * K] + 1j * r.x[2 * K:]
    o = np.argsort(w)
    return w[o], a[o], float(np.sqrt(np.mean(res(r.x) ** 2)))


def matrix_pencil(y, dt, sv_rel=1e-8, amp_rel=1e-3):
    """Same ESPRIT recipe as the agents' scripts (re-implemented)."""
    N = len(y)
    L = N // 2
    Y = hankel(y[:L + 1], y[L:])
    U, s, _ = svd(Y, full_matrices=False)
    r = int(np.sum(s > sv_rel * s[0]))
    U1, U2 = U[:-1, :r], U[1:, :r]
    z = eigvals(pinv(U1) @ U2)
    V = z[None, :] ** np.arange(N)[:, None]
    a, *_ = lstsq(V, y)
    omega = -np.angle(z) / dt
    keep = np.abs(a) > amp_rel * np.abs(a).max()
    o = np.argsort(omega[keep])
    return omega[keep][o], np.abs(a[keep])[o], np.abs(z[keep])[o], r


def ed_levels(ns, P, refl_rel=None):
    d = np.load(f"data/deep_levels_ns{ns}.npz")
    g, ph, rf = d["gaps"], d["phases"], d["refl"]
    sel = np.abs(np.angle(np.exp(1j * (ph - P)))) < 1e-4
    if refl_rel is not None:
        sel &= (rf * rf[0]) * refl_rel > 0.99
    o = np.argsort(g[sel])
    return g[sel][o], (rf * rf[0])[sel][o]


def nearest(x, arr):
    j = int(np.argmin(np.abs(arr - x)))
    return j, float(x - arr[j])


# ================================================================== ns12
def run_ns12():
    NS = 12
    lat = Z2Lattice(NS, pbc=True)
    space = GFSpace(lat, q=0)
    H = GFHamiltonian(space, M0, G2, ETA)
    D = space.dim
    log(f"ns=12 gf dim {D}: building the FULL Hamiltonian column by column (no sectors)")
    Hm = np.zeros((D, D), dtype=complex)
    e = np.zeros(D, dtype=complex)
    for j in range(D):
        e[:] = 0; e[j] = 1
        Hm[:, j] = H.apply(e)
    assert np.abs(Hm - Hm.conj().T).max() < 1e-12
    w, V = np.linalg.eigh(Hm)
    e0 = float(w[0])
    vac, meson = V[:, 0], V[:, 1]
    Mgap = float(w[1] - e0)
    t2m = space.t2_expect(meson) / np.vdot(meson, meson)
    log(f"E_vac {e0:.6f}, meson gap {Mgap:.6f}, <T2>_meson = {t2m:.4f}")
    ed = np.load("data/deep_levels_ns12.npz")
    log(f"  full-spectrum check vs deep_levels_ns12: max |dE| = "
        f"{np.abs(np.sort(ed['energies']) - w).max():.2e} over {D} levels")

    J1 = sum(bond_current(lat, b, ETA) for b in lat.bonds).simplify()
    q = 2 * np.pi / lat.nx
    x = np.arange(NS) / 2.0
    J0mq = sum(np.exp(+1j * q * x[v]) * charge_density(lat, v) for v in range(NS)).simplify()
    Smq = sum(np.exp(+1j * q * x[v]) * axial_charge_density(lat, v) for v in range(NS)).simplify()
    S0 = sum(axial_charge_density(lat, v) for v in range(NS)).simplify()
    kets = {"C11": space.apply(J1, meson), "C00_q": space.apply(J0mq, meson),
            "S_q (axial, C-even probe)": space.apply(Smq, meson),
            "S_0 (axial, C-even probe)": space.apply(S0, meson)}
    out = {"e0": e0, "M": Mgap}
    theirs = np.load("data/rt_levels_ns12.npz")
    for name, phi in kets.items():
        n2 = float(np.vdot(phi, phi).real)
        t2 = space.t2_expect(phi) / n2
        P = float(np.angle(t2))
        log(f"--- {name}: |phi|^2 = {n2:.5f}, <T2> = {t2:.4f} -> P = {P:+.4f} "
            f"(|<T2>| - 1 = {abs(t2) - 1:.1e})")
        # exact spectral weights, grouped by energy (degenerate pairs summed)
        ov = np.abs(V.conj().T @ phi) ** 2
        gaps = w - e0
        wt = {}
        for g_, o_ in zip(gaps, ov):
            key = round(float(g_), 6)
            wt[key] = wt.get(key, 0.0) + float(o_)
        edg, edr = ed_levels(NS, P)
        rows = []
        for g_ in edg:
            if g_ < 8.0:
                j, dd = nearest(g_, np.array(list(wt)))
                rows.append((float(g_), float(list(wt.values())[j])))
        win = [(g_, ww) for g_, ww in rows if WINDOW[0] < g_ < WINDOW[1]]
        near = [(g_, ww) for g_, ww in rows if WINDOW[1] <= g_ < 6.2]
        log(f"  exact weights |<n|phi>|^2 of ED levels in window: "
            + ", ".join(f"{g_:.4f}:{ww:.2e}" for g_, ww in win))
        log(f"  just above window: " + ", ".join(f"{g_:.4f}:{ww:.2e}" for g_, ww in near))
        out[name] = {"norm2": n2, "P": P, "window_weights": win, "above_window_weights": near,
                     "elastic_weight": float(abs(np.vdot(meson, phi)) ** 2)}
        if name not in ("C11", "C00_q"):
            continue
        # time series from the dense decomposition (exact for all t)
        dt = 0.1
        for T in (40.0, 200.0):
            t = np.arange(0.0, T + dt / 2, dt)
            C = (ov[:, None] * np.exp(-1j * np.outer(w - (e0 + Mgap), t))).sum(0)
            if T == 40.0:
                key = f"{name}_t"
                Ct = theirs[key]
                log(f"  my C(t) vs stored {key} (T=40): max|diff| = "
                    f"{np.abs(C - Ct[:len(C)]).max():.2e}, C(0) = {C[0].real:.5f}")
            for win_ in ("rect", "hann"):
                wf, af, _, _ = fft_lines(C, dt, window=win_)
                sel = (wf + Mgap > 5.0) & (wf + Mgap < 6.3)
                log(f"  FFT ({win_}, T={T:.0f}) lines in 5.0-6.3: "
                    + ", ".join(f"{g_:.4f}({a_:.3f})" for g_, a_ in zip(wf[sel] + Mgap, af[sel])))
                out[name][f"fft_{win_}_T{int(T)}"] = {"gap": fl(wf + Mgap), "amp": fl(af)}
            if T == 200.0:
                # NLLS on the T=40 series seeded by the T=200 FFT lines
                t40 = np.arange(0.0, 40.0 + dt / 2, dt)
                C40 = (ov[:, None] * np.exp(-1j * np.outer(w - (e0 + Mgap), t40))).sum(0)
                wf, af, _, _ = fft_lines(C, dt, window="hann", rel=3e-3)
                wn, an, rms = nlls_lines(C40, dt, wf, af.astype(complex))
                log(f"  NLLS (T=40, {len(wn)} lines seeded by T=200 FFT), fit rms {rms:.1e}:")
                tab = []
                for g_, a_ in zip(wn + Mgap, np.abs(an)):
                    j, dd = nearest(g_, edg)
                    tab.append({"gap": float(g_), "weight": float(a_), "ED": float(edg[j]),
                                "diff": dd})
                    if 5.0 < g_ < 6.6:
                        log(f"     gap {g_:.5f} w {a_:.4f}  ED {edg[j]:.5f} diff {dd:+.1e}")
                out[name]["nlls_T40"] = tab
        # their pencil lines vs my exact weights
        pg = theirs[f"{name}_gap"]; pw = theirs[f"{name}_weight"]
        cmp = []
        for g_, ww in zip(pg, pw):
            j, dd = nearest(g_, edg)
            je, de = nearest(g_, np.array(list(wt)))
            cmp.append({"pencil_gap": float(g_), "pencil_w": float(ww), "ED": float(edg[j]),
                        "diff_ED": dd, "exact_w": float(list(wt.values())[je])})
        out[name]["their_pencil_vs_exact"] = cmp
        log("  their pencil lines vs exact weights: " + ", ".join(
            f"{c['pencil_gap']:.4f}: w {c['pencil_w']:.3f}/{c['exact_w']:.3f} dED {c['diff_ED']:+.0e}"
            for c in cmp if 5.0 < c['pencil_gap'] < 6.3))
    json.dump(out, open("data/rt_levels_verify_ns12.json", "w"), indent=1)
    log("wrote data/rt_levels_verify_ns12.json")


# ================================================================== ns20
def run_sector(NS):
    lat = Z2Lattice(NS, pbc=True)
    space = GFSpace(lat, q=0)
    H = GFHamiltonian(space, M0, G2, ETA)
    log(f"ns={NS} gf dim {space.dim}")
    band = gf_band(space, M0, G2, ETA, k=0.0, n_per_k=2, log=log)
    vac, e0, meson = band["vacuum"], band["e0"], band["states"][0]
    Mgap = float(band["energy"][0])
    log(f"E_vac {e0:.6f}, M = {Mgap:.6f}")
    J1 = sum(bond_current(lat, b, ETA) for b in lat.bonds).simplify()
    phi = space.apply(J1, meson)
    n2 = float(np.vdot(phi, phi).real)
    t2 = space.t2_expect(phi) / n2
    log(f"J1|M>: |phi|^2 {n2:.4f}, <T2> = {t2:.4f}; elastic |<M|phi>|^2 = {abs(np.vdot(meson, phi))**2:.1e}")
    Hk, R, idx = _sector_hamiltonian(space, H, 0.0)
    rep, mrep, chi, ell, _ = space.orbits()
    ci = idx[rep]; m = ci >= 0
    phik = np.zeros(Hk.shape[0], dtype=complex)
    np.add.at(phik, ci[m], phi[m] * chi[m] / np.sqrt(ell[m]))
    log(f"sector dim {Hk.shape[0]}, |phik|^2 = {np.vdot(phik, phik).real:.4f} (should equal |phi|^2)")
    # exact low spectrum of the sector and weights (independent ground truth)
    ts = time.time()
    wk, vk = spla.eigsh(Hk, k=70, which="SA", tol=1e-10, ncv=160)
    o = np.argsort(wk); wk, vk = wk[o], vk[:, o]
    wt = np.abs(vk.conj().T @ phik) ** 2
    log(f"eigsh 70 lowest P=0 levels: {time.time() - ts:.0f}s; sum of weights {wt.sum()/n2:.5f} of |phi|^2")
    edg, edr = ed_levels(NS, 0.0)
    rows = []
    for g_, ww in zip(wk - e0, wt):
        j, dd = nearest(g_, edg)
        rows.append({"gap": float(g_), "weight": float(ww), "ED": float(edg[j]), "dED": dd,
                     "R_rel": float(edr[j])})
    log("  P=0 sector levels in 5.4-6.2 with exact weights (R = vac-normalised ED reflection):")
    for r in rows:
        if 5.4 < r["gap"] < 6.2:
            log(f"     {r['gap']:.5f}  w {r['weight']:.2e}  ED {r['ED']:.5f} (dED {r['dED']:+.0e}, R {r['R_rel']:+.0f})")
    # real-time series, my own FFT / NLLS extraction
    dt, T = 0.25, 40.0
    t = np.arange(0.0, T + dt / 2, dt)
    ts = time.time()
    psi_t = spla.expm_multiply(-1j * Hk.tocsc(), phik, start=0.0, stop=T, num=len(t), endpoint=True)
    C = np.exp(1j * (e0 + Mgap) * t) * (psi_t @ phik.conj())
    log(f"expm_multiply {len(t)} samples: {time.time() - ts:.0f}s")
    theirs = np.load(f"data/rt_levels_ns{NS}.npz")
    C11s = theirs["C11"] if "C11" in theirs else theirs["C11_t"]
    log(f"  my C11(t) vs stored: max|diff| = {np.abs(C - C11s[:len(C)]).max():.2e}")
    # exact model series from eigsh (to see what the 70 levels miss)
    Cex = (wt[:, None] * np.exp(-1j * np.outer(wk - (e0 + Mgap), t))).sum(0)
    log(f"  70-level reconstruction of C11(t): max|diff| = {np.abs(C - Cex).max():.2e}")
    wf, af, _, _ = fft_lines(C, dt, window="hann")
    log("  FFT (Hann, T=40) lines 5.0-6.3: " + ", ".join(
        f"{g_:.4f}({a_:.3f})" for g_, a_ in zip(wf + Mgap, af) if 5.0 < g_ < 6.3))
    # NLLS seeded by the exact eigsh levels with weight > 1e-3 of max (a
    # different extraction; the seed is then perturbed by +-0.02 to check
    # that the fit, not the seed, sets the answer)
    keep = wt > 1e-3 * wt.max()
    w0 = (wk - (e0 + Mgap))[keep]
    rng = np.random.default_rng(1)
    w0p = w0 + rng.uniform(-0.02, 0.02, size=w0.shape)
    wn, an, rms = nlls_lines(C, dt, w0p, wt[keep].astype(complex))
    log(f"  NLLS (T=40, {len(wn)} lines, seeds perturbed by +-0.02) fit rms {rms:.1e}:")
    tab = []
    for g_, a_ in zip(wn + Mgap, np.abs(an)):
        j, dd = nearest(g_, edg)
        tab.append({"gap": float(g_), "weight": float(a_), "ED": float(edg[j]), "dED": dd})
        if 5.4 < g_ < 6.3:
            log(f"     gap {g_:.5f} w {a_:.4f}  ED {edg[j]:.5f} diff {dd:+.1e}")
    # their pencil lines vs ED
    pg = (theirs["C11_omega"] if "C11_omega" in theirs else theirs["C11_pencil_omega"]) + Mgap
    log("  their pencil lines 5.4-6.3 vs ED: " + ", ".join(
        f"{g_:.5f}({nearest(g_, edg)[1]:+.0e})" for g_ in pg if 5.4 < g_ < 6.3))
    np.savez(f"data/rt_levels_verify_ns{NS}.npz", t=t, C11=C, sector_gaps=wk - e0, sector_weights=wt,
             nlls_gap=wn + Mgap, nlls_w=np.abs(an))
    json.dump({"e0": e0, "M": Mgap, "norm2": n2, "sector_levels": rows, "nlls_T40": tab,
               "fft_hann_T40": {"gap": fl(wf + Mgap), "amp": fl(af)}},
              open(f"data/rt_levels_verify_ns{NS}.json", "w"), indent=1)
    log(f"wrote data/rt_levels_verify_ns{NS}.{{json,npz}}")


# ================================================================== stored
def run_stored():
    out = {}
    series = {
        12: ("data/rt_levels_ns12.npz", "C11_t", "C00_q_t", None),
        16: ("data/rt_levels_ns16.npz", "C11_t", "C00_q_t", "gap_meson_k0"),
        20: ("data/rt_levels_ns20.npz", "C11", "C00_q", None),
    }
    rng = np.random.default_rng(0)
    for ns, (fn, k11, k00, kM) in series.items():
        d = np.load(fn)
        t = d["t"]; dt = float(t[1] - t[0])
        Mgap = float(d[kM]) if kM else (float(d["M"]) if "M" in d else float(d["E_meson"]) - float(d["E_vac"]))
        for key, P in ((k11, 0.0), (k00, 2 * np.pi / (ns // 2))):
            C = d[key]
            edg, edr = ed_levels(ns, P)
            edwin = edg[(edg > WINDOW[0]) & (edg < WINDOW[1])]
            log(f"=== ns={ns} {key}: T={t[-1]:.0f}, dt={dt}, N={len(C)}, 2pi/T={2*np.pi/t[-1]:.3f}; "
                f"ED window levels {np.round(edwin, 4)}")
            wf, af, _, _ = fft_lines(C, dt, window="hann")
            g = wf + Mgap
            s = (g > 5.3) & (g < 6.3)
            log("   Hann FFT peaks 5.3-6.3: " + ", ".join(f"{a:.4f}({b:.3g})" for a, b in zip(g[s], af[s])))
            wf, af, _, _ = fft_lines(C, dt, window="rect")
            g = wf + Mgap
            s = (g > 5.3) & (g < 6.3)
            log("   rect FFT peaks 5.3-6.3: " + ", ".join(f"{a:.4f}({b:.3g})" for a, b in zip(g[s], af[s])))
            res = {"ED_window": fl(edwin)}
            for noise in (0.0, 1e-6, 1e-4, 1e-3, 1e-2):
                lines = []
                for trial in range(3 if noise else 1):
                    Cn = C + noise * abs(C[0]) * (rng.standard_normal(len(C)) + 1j * rng.standard_normal(len(C))) / np.sqrt(2)
                    sv_rel = 1e-8 if noise == 0 else max(1e-8, 3 * noise)
                    om, am, zb, r = matrix_pencil(Cn, dt, sv_rel=sv_rel)
                    gg = om + Mgap
                    sel = (gg > WINDOW[0] - 0.02) & (gg < WINDOW[1] + 0.02)
                    lines.append([(float(a), float(b)) for a, b in zip(gg[sel], am[sel] / am.max())])
                # report the deviation of each window ED level from the nearest pencil line
                devs = []
                for e in edwin:
                    dd = [min([abs(a - e) for a, b in ln] + [9.0]) for ln in lines]
                    devs.append(float(np.median(dd)))
                log(f"   pencil noise {noise:.0e}: window lines (trial 0) "
                    + ", ".join(f"{a:.4f}({b:.2g})" for a, b in lines[0])
                    + " | median |dev| from ED window levels: "
                    + ", ".join(f"{e:.4f}:{dv:.1e}" for e, dv in zip(edwin, devs)))
                res[f"noise_{noise:.0e}"] = {"lines_trial0": lines[0], "dev": devs}
            out[f"ns{ns}_{key}"] = res
    json.dump(out, open("data/rt_levels_verify_stored.json", "w"), indent=1)
    log("wrote data/rt_levels_verify_stored.json")


# ================================================================== luscher
def run_luscher():
    k_ed = np.array([0.0, 1.2566, 1.5708, 2.0944, 2.5133, np.pi])
    e_ed = np.array([2.7451, 2.8188, 2.8560, 2.9275, 2.9886, 3.0778])
    E = CubicSpline(k_ed, e_ed, bc_type=((1, 0.0), (1, 0.0)))
    M = e_ed[0]
    E_INEL = 6.0304
    stored = np.load("data/phase_shifts_6vol.npz")["rows"]
    lus = json.load(open("data/rt_levels_luscher.json"))
    pts = lus["points_p0"]
    # my own inversion (same convention as phase_shifts_v2.py, written from the docstring)
    diffs, out = [], []
    for ns in (12, 16, 20):
        L = ns // 2
        d = np.load(f"data/rt_levels_ns{ns}.npz")
        if ns == 12:
            g, wgt = d["C11_gap"], d["C11_weight"]
        elif ns == 16:
            g, wgt = d["C11_pencil_omega"] + float(d["gap_meson_k0"]), np.abs(d["C11_pencil_amp"])
        else:
            g, wgt = d["C11_omega"] + float(d["E_meson"]) - float(d["E_vac"]), d["C11_weight"]
        keep = wgt > 1e-3 * wgt.max()
        g = np.sort(g[keep])
        cands = [x for x in g if 2 * M - 0.06 < x < E_INEL]
        edg, edr = ed_levels(ns, 0.0, refl_rel=+1)
        edw = [x for x in edg if 2 * M - 0.06 < x < E_INEL]
        edm = [x for x in ed_levels(ns, 0.0, refl_rel=-1)[0] if 2 * M - 0.06 < x < E_INEL]
        srows = stored[stored[:, 0] == ns]
        log(f"ns={ns}: RT window cands {np.round(cands, 5)}; ED R=+1 window {np.round(edw, 5)}; "
            f"ED R=-1 window {np.round(edm, 5)}; stored rows n = {srows[:, 3].astype(int)} E2 = {np.round(srows[:, 1], 5)}")
        assert len(cands) == len(edw) == len(srows)
        for n, E2 in enumerate(cands):
            p = brentq(lambda k: 2 * E(k) - E2, 1e-9, np.pi)
            delta = (2 * np.pi * n - p * L) / 2
            delta = (delta + np.pi / 2) % np.pi - np.pi / 2
            sr = srows[np.argmin(np.abs(srows[:, 1] - E2))]
            assert int(sr[3]) == n
            dd = float(delta - sr[4])
            their = [q for q in pts if q["ns"] == ns and q["n"] == n][0]
            diffs.append(dd)
            out.append({"ns": ns, "n": n, "E2_rt": float(E2), "E2_ED": float(sr[1]), "p": float(p),
                        "delta_mine": float(delta), "delta_ED_stored": float(sr[4]), "diff": dd,
                        "their_delta_rt": their["delta_rt"], "their_diff": their["ddelta"]})
            log(f"   n={n} E2={E2:.6f} (ED {sr[1]:.6f}) p={p:.5f} delta={delta:+.6f} "
                f"stored ED {sr[4]:+.6f} diff {dd:+.2e} | theirs {their['delta_rt']:+.6f} ({their['ddelta']:+.2e})")
    diffs = np.array(diffs)
    rms = float(np.sqrt(np.mean(diffs ** 2)))
    log(f"my RMS over {len(diffs)} points = {rms:.3e} (theirs {lus['rms_delta_diff_p0']:.3e}); "
        f"RMS without ns20 n=3: {np.sqrt(np.mean(diffs[:-1]**2)):.2e}")
    # what the T=80 rerun gives for the ns20 n=3 point
    d80 = np.load("data/rt_levels_ns20_T80.npz")
    g80 = np.sort(d80["C11_omega"] + float(d80["E_meson"]) - float(d80["E_vac"]))
    g80 = [x for x in g80 if 5.9 < x < 6.03]
    log(f"T=80 ns20 lines in 5.9-6.03: {np.round(g80, 6)} (ED 6.017185)")
    # sensitivity: d delta / d E2 at each point (how much a level error costs)
    for q in out:
        L = q["ns"] // 2
        dp = 1 / (2 * float(E(q["p"], 1)))
        q["ddelta_dE2"] = float(-L * dp / 2)
    log("d(delta)/d(E2) per point: " + ", ".join(f"ns{q['ns']} n{q['n']}: {q['ddelta_dE2']:+.2f}" for q in out))
    json.dump({"rms": rms, "points": out}, open("data/rt_levels_verify_luscher.json", "w"), indent=1)
    log("wrote data/rt_levels_verify_luscher.json")


if __name__ == "__main__":
    {"ns12": run_ns12, "ns20": lambda: run_sector(20), "ns16": lambda: run_sector(16), "stored": run_stored, "luscher": run_luscher}[sys.argv[1]]()
