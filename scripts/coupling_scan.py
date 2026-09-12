"""Coupling scan for a relativistic-dispersion meson (WS2-A).

Goal: find ONE new coupling point (m0, g2, eta) of the 1+1d Z2 gauge theory
whose lightest meson has a relativistic dispersion, E(k) ~ sqrt(M^2 + k^2).
For E = sqrt(M^2 + k^2), E''(0) = 1/M; for a cosine band
E = M + (W/2)(1 - cos k), E''(0) = W/2, so the figure of merit is

    W * M -> 2        (equivalently  m_curv / M = 1 / (M E''(0)) -> 1).

Everything is exact diagonalization in the gauge-fixed, Gauss-resolved
Q = 0 product basis (htensor/gaugefixed.py: PhysicalBasis, matrix(op,
sub=sel)).  Per coupling point:

  a. vacuum energy and the meson band, identified by CURRENT RESIDUE, not
     by ordering.  Staggered fermions carry the shift symmetry S = T1 C
     (one-staggered-site translation times particle-hole; S^2 = T2), so
     the single-meson band is ONE smooth band E(p) over the fine zone
     p in (-pi, pi] with T2 momentum k = 2p mod 2pi: band-1 is |p| <= pi/2
     (E1(k) = E(k/2)) and band-2 (M') is |p| >= pi/2 (E2(k) = E(pi - k/2),
     hence always "inverted"), and E2(k = pi) = E1(k = pi) EXACTLY (every
     k = pi level is an S-doublet).  The bond current at fine momentum p,
     J1(p) = sum_b e^{i p (b + 1/2)} J1_b (htensor/currents.py
     bond_current), is S-selective, so at each |p| of the fine grid
     p_j = pi j / nx (j = 0..nx) the band level is the LOWEST degenerate
     cluster (T2 momentum 2p) carrying >= 0.5 of the exact sum rule
     ||J1(p)|vac>||^2 + ||J1(-p)|vac>||^2 (guards against intruder /
     scalar bands at large eta).  Momenta come from T2 eigenphases resolved
     cluster by cluster (gaugefixed.deep_spectrum recipe, tolerance 1e-5 --
     eigsh splits degenerate +-k pairs by up to ~1e-5, see
     spectroscopy._t2_phases).
  b. M = E(0), E1(pi) = E(p = pi/2), W = E1(pi) - M, W*M; the periodic
     cosine-series interpolant of the fine band E(|p|) (DCT-I, smooth
     through the zone boundary) gives v_g(k) = dE1/dk = E'(k/2)/2
     (v_g^max, v_g(2 pi / 5)) and the curvature mass 1/E1''(0) =
     4/E''(p = 0).  A clamped cubic spline of E1(|k|)
     (train_cgkA_packets.py band_dispersion) is kept as a cross-check.
  c. M' = E(p = pi) = E2(k = 0); gap M' - E1(pi) = E(pi) - E(pi/2) (the
     top of the fine band above band-1's top; the literal min_k E2 -
     E1(pi) vanishes identically by S); intruder gap min_k [E_int(k) -
     E1(k)] with E_int the lowest cluster at |k| that is neither band-1 nor
     band-2; M*-like levels (P = 0, reflection-even, residue fraction <
     0.1, below 2M, not band-1/2); 2M, M + M'; two-meson threshold
     Theta(k) = min_k1 [E1(k1) + E1(k - k1)] and the margin
     min_k [Theta(k) - E1(k)].
  d. confinement / size proxies: flux-string length L of the k = 0 meson
     (number of links whose electric bit differs from the vacuum's
     dominant pattern; in the gauge-fixed basis the link pattern is fixed
     by the matter configuration and the holonomy bit).  The vacuum
     itself carries fluctuation strings, so the meson size is VACUUM
     SUBTRACTED: size_mean = (<L>_meson - <L>_vac)/2 and size_rms =
     sqrt(<L^2>_meson - <L^2>_vac)/2 in spatial sites (the raw meson RMS
     is kept as size_rms_raw; it grows with volume); vacuum correlation
     length xi from the connected same-sublattice <J0(v) J0(v + 2r)>
     fitted to A cosh((r - nx/2)/xi).
  e. (refine volume only, nx % 5 == 0) plane-wave quasi-PDF <x> of the
     exact band-1 |k = 2 pi / 5> eigenstate, with the convention of
     scripts/quasipdf_matching.py lines 34-54: window x in [-0.5, 1.5],
     taste phase (-1)^z, h(0) = charge density included, normalize the
     shape then take the first moment.

Selection gates (hard): band residue-identified at every fine |p| (both
band-1 and band-2); two-meson margin >= 0.3; meson RMS string size <= 1.5
spatial sites and xi <= 1.5; M' - E1(pi) = E(pi) - E(pi/2) >= 0.1.  Ranking among feasible points: W*M closest to 2 from
below (ties/none: largest) with --rank wm (the brief's proxy), or
|m_curv/M - 1| smallest with --rank mcurv (the direct E''(0) = 1/M test).

Conventions: T2|k> = e^{+ik}|k> (spectroscopy.meson_band); momenta in
inverse spatial sites, k = 2 pi j / nx; energies are gaps above the
vacuum; string lengths in staggered links unless labelled "spatial".

  PYTHONPATH=. .venv/bin/python scripts/coupling_scan.py \\
      --g2 0.8 1.1 1.4 --m0 0.2:0.7:0.1 --eta 1.3:2.6:0.1 --ns 16 --k 60 \\
      --refine-ns 20 --refine-k 80 --top 6 --workers 6 \\
      --out data/coupling_scan.npz --csv data/csv/coupling_scan.csv \\
      --fig data/coupling_scan.pdf
  PYTHONPATH=. .venv/bin/python scripts/coupling_scan.py --validate
"""

import argparse
import csv
import os
import sys
import time
import warnings


def _pre_parse_threads():
    """Pin BLAS threads BEFORE numpy is imported: one thread per worker
    when running a multi-process scan (--threads overrides)."""
    argv = sys.argv
    workers, threads = 1, None
    for i, a in enumerate(argv):
        if a == "--workers" and i + 1 < len(argv):
            workers = int(argv[i + 1])
        if a == "--threads" and i + 1 < len(argv):
            threads = int(argv[i + 1])
    if threads is None:
        threads = 1 if workers > 1 else int(os.environ.get("OMP_NUM_THREADS", "4"))
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = str(threads)


_pre_parse_threads()

import numpy as np                                             # noqa: E402
import scipy.sparse as sp                                      # noqa: E402
import scipy.sparse.linalg as spla                             # noqa: E402
from scipy.interpolate import CubicSpline                      # noqa: E402
from scipy.optimize import curve_fit                           # noqa: E402

from htensor import Z2Lattice                                  # noqa: E402
from htensor import hamiltonian as ham                         # noqa: E402
from htensor.gaugefixed import PhysicalBasis                   # noqa: E402
from htensor.currents import bond_current, charge_density      # noqa: E402
from htensor.quasipdf import wilson_bilinear                   # noqa: E402

K5 = 2 * np.pi / 5                     # packet momentum of the campaign
DEGENERACY_TOL = 1e-5                  # T2 cluster tolerance (see docstring)
RESIDUE_BAND1 = 0.5                    # band-1: >= this fraction of the sum rule
RESIDUE_BAND2 = 0.02                   # "residue-carrying" second level
RESIDUE_MSTAR = 0.1                    # M*-like: below this fraction
GATES = dict(margin=0.3, size=1.5, xi=1.5, gap_M2=0.1)
PRODUCTION = (0.7, 1.1, 1.3)

T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


# ------------------------------------------------------------------ context
class ScanContext:
    """Coupling-independent data for one volume, built once in the parent
    process and inherited by forked workers.

    H(m0, g2, eta) = g2 * Hg + m0 * Hm + eta * Hh on the Q = 0 block
    (hamiltonian.build_hamiltonian is linear in each coupling, seam sign
    included), so a point costs one eigsh plus post-processing."""

    def __init__(self, ns: int):
        lat = Z2Lattice(ns, pbc=True)
        basis = PhysicalBasis(lat)
        sel = np.flatnonzero(basis.q == 0)
        self.lat, self.basis, self.sel, self.ns, self.nx = lat, basis, sel, ns, lat.nx
        self.dim = len(sel)
        self.Hg = basis.matrix(ham.gauge_term(lat, 1.0), sub=sel).real.tocsr()
        self.Hm = basis.matrix(ham.mass_term(lat, 1.0), sub=sel).real.tocsr()
        self.Hh = basis.matrix(ham.hopping_term(lat, 1.0), sub=sel).real.tocsr()
        self.T = basis.translation()[sel][:, sel].tocsr()
        # bond currents (eta = 1) on the Q = 0 block, and their positions
        self.J1 = [basis.matrix(bond_current(lat, b, 1.0), sub=sel).tocsr()
                   for b in range(ns)]
        self.xb = (np.arange(ns) + 0.5) / 2.0            # spatial sites
        self.xstag = np.arange(ns) + 0.5                  # staggered units
        # diagonal staggered charge q_v(j) = ((-1)^v - Z_v)/2, Z = 1 - 2 z
        zsub = basis.z[sel]
        zb = ((zsub[:, None] >> np.arange(ns)) & 1).astype(np.int8)
        stag = ((-1) ** np.arange(ns))[None, :]
        self.q0 = ((stag - (1 - 2 * zb)) / 2.0).T.astype(float)   # (ns, dim)
        self.ebit = basis.ebit[sel]                                # (dim, ns)
        # momentum grid: signed k = 2 pi j / nx in (-pi, pi], and |k|
        nx = lat.nx
        ks = 2 * np.pi * np.arange(nx) / nx
        ks = np.where(ks > np.pi + 1e-9, ks - 2 * np.pi, ks)
        self.ksigned = np.sort(ks)
        self.kabs = np.array(sorted(set(np.round(np.abs(ks), 12))))
        # fine zone: p = pi j / nx, j = 0..2nx-1, wrapped to (-pi, pi]; |p| grid
        ps = np.pi * np.arange(2 * nx) / nx
        ps = np.where(ps > np.pi + 1e-9, ps - 2 * np.pi, ps)
        self.psigned = np.sort(ps)
        self.pabs = np.pi * np.arange(nx + 1) / nx
        # J1(p) = sum_b e^{i p (b + 1/2)} J1_b  (= e^{i k x_b} with k = 2p)
        self.phase_p = np.exp(1j * np.outer(self.xstag, self.psigned))  # (ns, 2nx)
        # reflection: the variant that commutes with a generic H on the block
        self.R, self.R_variant = self._select_reflection()

    def hamiltonian(self, m0, g2, eta):
        return (g2 * self.Hg + m0 * self.Hm + eta * self.Hh).tocsr()

    def _select_reflection(self):
        """Port of PhysicalBasis.select_reflection restricted to the Q = 0
        block (the full-basis H is too large at ns = 20).  R commutes with
        H for all couplings if it does for a generic one."""
        H = self.hamiltonian(*PRODUCTION)
        Td = self.T.conj().T.tocsr()
        for shift in (0, 1, 2):
            for stag in (False, True):
                R = self.basis.reflection(shift, stag)[self.sel][:, self.sel].tocsr()
                if abs(R).sum(axis=1).min() < 0.5:        # leaves the block
                    continue
                if abs(H @ R - R @ H).max() > 1e-9:
                    continue
                assert abs(R @ R - sp.identity(self.dim)).max() < 1e-9
                assert abs(R @ self.T @ R.conj().T - Td).max() < 1e-9
                return R, (shift, stag)
        raise ValueError("no reflection variant commutes with H on Q = 0")


CTX: ScanContext | None = None


# ------------------------------------------------------------- numerics
def resolve_t2(w, v, T, tol=DEGENERACY_TOL):
    """T2 eigenphases resolved inside each degenerate energy cluster
    (gaugefixed.deep_spectrum lines ~272-284).  Returns the rotated
    eigenvectors, phases and |lambda| (|lambda| < 1 flags a level whose
    cluster was truncated by the eigsh cutoff -- excluded downstream), and
    the list of cluster index ranges (i, j)."""
    n = len(w)
    V = np.empty(v.shape, dtype=complex)
    phases, lam = np.empty(n), np.empty(n)
    clusters = []
    i = 0
    while i < n:
        j = i + 1
        while j < n and w[j] - w[i] < tol:
            j += 1
        clusters.append((i, j))
        blk = v[:, i:j]
        ev, U = np.linalg.eig(blk.conj().T @ (T @ blk))
        order = np.argsort(np.angle(ev))
        ev, U = ev[order], U[:, order]
        res = blk @ U
        res = res / np.linalg.norm(res, axis=0)
        V[:, i:j] = res
        phases[i:j], lam[i:j] = np.angle(ev), np.abs(ev)
        i = j
    return V, phases, lam, clusters


def reflection_parities(w, V, phases, R, tol=DEGENERACY_TOL):
    """Reflection parity of the P = 0 levels, relative to the vacuum
    (gaugefixed.deep_spectrum lines ~285-303)."""
    n = len(w)
    par = np.full(n, np.nan)
    i = 0
    while i < n:
        j = i + 1
        while j < n and w[j] - w[i] < tol:
            j += 1
        p0 = [m for m in range(i, j) if abs(phases[m]) < 1e-4]
        if p0:
            q, _ = np.linalg.qr(V[:, p0])
            pe = np.sort(np.real(np.linalg.eigvals(q.conj().T @ (R @ q))))[::-1]
            for m, val in zip(p0, pe):
                par[m] = val
        i = j
    if np.isfinite(par[0]):
        par = par * par[0]
    return par


def trig_coeffs(kabs, E):
    """Cosine-series interpolant through (|k_i|, E_i), i < nk (DCT-I on the
    ring grid): E(k) = sum_{n<nk} a_n cos(n k).  Exact for any band with
    hopping range < nk, periodic by construction."""
    n = np.arange(len(kabs))
    return np.linalg.solve(np.cos(np.outer(kabs, n)), E)


def trig_eval(a, k):
    n = np.arange(len(a))
    return np.cos(np.outer(np.atleast_1d(k), n)) @ a


def trig_vg(a, k):
    n = np.arange(len(a))
    return -(np.sin(np.outer(np.atleast_1d(k), n)) * n) @ a


def trig_curv0(a):
    n = np.arange(len(a))
    return float(-(n**2 * a).sum())


def fit_xi(C, nx):
    """Correlation length from |C(r)|, r = 1..nx//2 spatial sites, with the
    periodic form A cosh((r - nx/2)/xi); fallback: two-point ratio."""
    r = np.arange(1, nx // 2 + 1, dtype=float)
    y = np.abs(C[1:nx // 2 + 1])
    ok = y > 1e-13
    if ok.sum() < 2:
        return np.nan, False
    r, y = r[ok], y[ok]
    two_pt = -1.0 / np.log(y[1] / y[0]) if y[1] < y[0] else np.inf

    def model(rr, lnA, xi):
        return lnA + np.log(np.cosh((rr - nx / 2) / xi))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p, _ = curve_fit(model, r, np.log(y),
                             p0=[np.log(y[0]), max(min(two_pt, 5.0), 0.2)],
                             bounds=([-np.inf, 1e-3], [np.inf, 1e3]), maxfev=2000)
        return float(p[1]), True
    except Exception:
        return float(two_pt), False


def string_size(PL, PL_vac):
    """Flux-string length moments from the length distributions P(L) of the
    k = 0 meson and of the vacuum (L in staggered links).  Sizes in spatial
    sites (L/2), vacuum subtracted: size_mean = excess mean length,
    size_rms = sqrt(<L^2>_meson - <L^2>_vac); size_rms_raw is the
    unsubtracted meson RMS (volume dependent, diagnostic only)."""
    PL, PL_vac = np.asarray(PL, float), np.asarray(PL_vac, float)
    Ls = np.arange(len(PL), dtype=float)
    Lm, Lm_vac = (PL * Ls).sum(), (PL_vac * Ls).sum()
    L2, L2_vac = (PL * Ls**2).sum(), (PL_vac * Ls**2).sum()
    return dict(L_mean=Lm, L_mean_vac=Lm_vac, L_rms=np.sqrt(L2),
                size_rms_raw=np.sqrt(L2) / 2,
                size_rms=np.sqrt(max(L2 - L2_vac, 0.0)) / 2,
                size_mean=(Lm - Lm_vac) / 2, size_excess=(Lm - Lm_vac) / 2)


def apply_gates(row):
    """Hard selection gates (module docstring) -> row['gate_*'], ['feasible']."""
    xi = row.get("xi", np.nan)
    gates = dict(gate_band=bool(row.get("band_ok", False)),
                 gate_margin=bool(row.get("margin", -np.inf) >= GATES["margin"]),
                 gate_size=bool(row.get("size_rms", np.inf) <= GATES["size"]),
                 gate_xi=bool(np.isfinite(xi) and xi <= GATES["xi"]),
                 gate_M2=bool(row.get("gap_M2", -np.inf) >= GATES["gap_M2"]))
    row.update(gates, feasible=all(gates.values()))
    return row


def quasipdf_xmean(ctx: ScanContext, st: np.ndarray, P: float):
    """Plane-wave quasi-PDF first moment <x> of the exact momentum-P
    eigenstate, exactly the convention of scripts/quasipdf_matching.py
    lines 34-54 (exact_quasipdf): h(z) from the seam-free Wilson-line
    bilinear averaged over even origins, h(0) = charge density, taste
    phase (-1)^|z|, x-window [-0.5, 1.5], normalize then first moment."""
    lat, nx, basis, sel = ctx.lat, ctx.nx, ctx.basis, ctx.sel
    xs = np.linspace(-0.5, 1.5, 201)
    zmax = nx // 2
    h = np.empty(zmax + 1, dtype=complex)
    for z in range(zmax + 1):
        if z == 0:
            acc = sum(charge_density(lat, 2 * x0) for x0 in range(nx)) / nx
        else:
            acc = sum(wilson_bilinear(lat, 2 * x0, 2 * z)[0]
                      for x0 in range(nx - z)) / (nx - z)
        O = basis.matrix(acc, sub=sel)
        h[z] = complex(st.conj() @ (O @ st))
    zs = np.arange(-zmax, zmax + 1)
    hz = np.array([h[abs(z)] if z >= 0 else np.conj(h[abs(z)]) for z in zs])
    hz = hz * (-1.0) ** np.abs(zs)
    qt = np.array([np.sum(np.exp(1j * x * P * zs) * hz) for x in xs]).real
    qt = qt / np.trapezoid(qt, xs)
    return float(np.trapezoid(xs * qt, xs))


# -------------------------------------------------------------- one point
def analyze_point(m0, g2, eta, k_levels, ctx=None, want_x=False):
    """All observables of the docstring for one coupling point (dict)."""
    ctx = ctx or CTX
    t0 = time.time()
    ns, nx, kabs, ksigned = ctx.ns, ctx.nx, ctx.kabs, ctx.ksigned
    nk = len(kabs)
    out = dict(m0=m0, g2=g2, eta=eta, ns=ns, nlev=k_levels, eigsh_ok=True)
    H = ctx.hamiltonian(m0, g2, eta)
    try:
        if k_levels >= ctx.dim - 1:
            w, v = np.linalg.eigh(H.toarray())
            w, v = w[:k_levels], v[:, :k_levels]
        else:
            w, v = spla.eigsh(H, k=k_levels, which="SA", tol=1e-10)
        o = np.argsort(w)
        w, v = w[o], v[:, o]
    except Exception as exc:                                   # noqa: BLE001
        out.update(eigsh_ok=False, error=repr(exc))
        return out
    V, phases, lam, clusters = resolve_t2(w, v, ctx.T)
    resolved = lam > 1 - 1e-4
    parity = reflection_parities(w, V, phases, ctx.R)
    gaps = w - w[0]
    vac = V[:, 0]
    out["E0"] = float(w[0])
    out["vac_phase"] = float(phases[0])

    # ---- current residues at the fine momenta: C[n, j] = |<n|J1(p_j)|vac>|^2.
    # Momentum conservation makes exactly one p_j (= the level's fine
    # momentum) contribute per level, so sums over j are convention free.
    Jv = np.column_stack([J @ vac for J in ctx.J1])                # (dim, ns)
    U = Jv @ ctx.phase_p                                           # (dim, 2nx)
    tot = np.linalg.norm(U, axis=0) ** 2
    C = np.abs(V.conj().T @ U) ** 2                                # (nlev, 2nx)
    res = C.sum(axis=1)
    jstar = C.argmax(axis=1)
    frac = np.where(tot[jstar] > 1e-14, res / np.maximum(tot[jstar], 1e-300), 0.0)
    out["all_gaps"], out["all_phases"], out["all_frac"] = gaps, phases, frac
    out["all_parity"], out["all_resolved"] = parity, resolved
    pabs, psigned = ctx.pabs, ctx.psigned
    npf = len(pabs)
    # per-cluster residue at each |p| (sum over +-p and over the cluster)
    cl_E = np.array([gaps[i:j].mean() for i, j in clusters])
    cl_ok = np.array([resolved[i:j].all() for i, j in clusters])
    cl_k = [np.abs(phases[i:j]) for i, j in clusters]
    cl_R = np.zeros((len(clusters), npf))
    ptot = np.zeros(npf)
    for a, pa in enumerate(pabs):
        js = np.flatnonzero(np.abs(np.abs(psigned) - pa) < 1e-9)
        ptot[a] = tot[js].sum()
        for c, (i, j) in enumerate(clusters):
            cl_R[c, a] = C[i:j][:, js].sum()
    cl_frac = cl_R / np.maximum(ptot, 1e-300)[None, :]
    out["ptot"] = ptot

    # ---- band identification per fine |p| (T2 momentum k = 2p mod 2pi)
    Ef = np.full(npf, np.nan); ff = np.full(npf, np.nan)
    cf = np.full(npf, -1, dtype=int)
    okf = np.zeros(npf, bool)
    for a, pa in enumerate(pabs):
        ka = abs(np.angle(np.exp(2j * pa)))
        cand = [c for c in range(1, len(clusters)) if cl_ok[c]
                and np.any(np.abs(cl_k[c] - ka) < 1e-3)]
        if not cand:
            continue
        good = [c for c in cand if cl_frac[c, a] >= RESIDUE_BAND1]
        if good:
            cf[a], okf[a] = good[0], True
        else:                          # fallback for reporting: max residue
            cf[a] = cand[int(np.argmax(cl_frac[cand, a]))]
        Ef[a], ff[a] = cl_E[cf[a]], cl_frac[cf[a], a]
    half = nx // 2
    kabs = ctx.kabs                                # = pabs[:half + 1] * 2
    E1, f1 = Ef[:half + 1], ff[:half + 1]           # band-1: E1(k) = E(k/2)
    E2, f2 = Ef[half:][::-1], ff[half:][::-1]       # band-2: E2(k) = E(pi - k/2)
    # intruder: lowest cluster at T2 momentum |k| that is neither band
    Eint = np.full(half + 1, np.nan)
    for a, ka in enumerate(kabs):
        used = {cf[a], cf[npf - 1 - a]}
        oth = [c for c in range(1, len(clusters)) if cl_ok[c] and c not in used
               and np.any(np.abs(cl_k[c] - ka) < 1e-3)]
        if oth:
            Eint[a] = cl_E[oth[0]]
    out.update(kabs=kabs, pabs=pabs, Efine=Ef, frac_fine=ff, ok_fine=okf,
               E1=E1, frac1=f1, E2=E2, frac2=f2, Eint=Eint,
               band_ok=bool(okf.all()), n_fail=int((~okf).sum()),
               frac_min=float(np.nanmin(ff)) if np.isfinite(ff).any() else np.nan)
    if not np.isfinite(Ef).all():
        out.update(runtime=time.time() - t0)
        return _finish_nan(out)

    # ---- dispersion from the fine band (smooth, periodic in p)
    M, Epi, Mp = float(E1[0]), float(E1[-1]), float(Ef[-1])
    W = Epi - M
    a = trig_coeffs(pabs, Ef)                        # E(p) = sum a_n cos(n p)
    kf = np.linspace(0, np.pi, 721)
    vg = trig_vg(a, kf / 2) / 2                      # dE1/dk = E'(k/2)/2
    curv = trig_curv0(a) / 4                         # E1''(0) = E''(0)/4
    cs = CubicSpline(kabs, E1, bc_type=((1, 0.0), (1, 0.0)))
    out.update(M=M, Epi=Epi, W=W, WM=W * M, trig=a,
               curv0=curv, m_curv=1.0 / curv if curv != 0 else np.inf,
               m_curv_over_M=(1.0 / curv) / M if curv != 0 else np.inf,
               vg_max=float(np.max(np.abs(vg))),
               vg_k5=float(trig_vg(a, K5 / 2)[0] / 2), vg_k5_cs=float(cs(K5, 1)),
               Ek5=float(trig_eval(a, K5 / 2)[0]),
               Wfine=Mp - M)
    # direct relativistic diagnostics: effective light speed at the packet
    # momentum from E^2 = M^2 + c^2 k^2, and the RMS deviation of E1(k)
    # from sqrt(M^2 + k^2) over the grid (in units of M)
    Ek5 = out["Ek5"]
    out["c_k5"] = float(np.sqrt(max(Ek5**2 - M**2, 0.0)) / K5)
    out["rel_rms"] = float(np.sqrt(np.mean(((E1 - np.sqrt(M**2 + kabs**2)) / M) ** 2)))

    # ---- band-2 / M* / thresholds
    out.update(M2min=float(np.nanmin(E2)), M2_k0=Mp, M2_kpi=float(E2[-1]),
               M2_orient="inverted" if E2[-1] < E2[0] else "normal",
               gap_M2=Mp - Epi,                     # E(pi) - E(pi/2)
               gap_M2_k5=float(trig_eval(a, np.pi - K5 / 2)[0] - trig_eval(a, K5 / 2)[0]),
               int_gap=float(np.nanmin(Eint - E1)) if np.isfinite(Eint).any() else np.nan,
               twoM=2 * M, MMp=M + Mp)

    def Efun(kk):                      # E1 at |k| on the ring grid
        return E1[np.argmin(np.abs(kabs[None, :] - np.abs(kk)[:, None]), axis=1)]
    theta = np.empty(len(ksigned))
    for j, kk in enumerate(ksigned):
        k2 = (kk - ksigned + np.pi) % (2 * np.pi) - np.pi
        theta[j] = np.min(Efun(ksigned) + Efun(k2))
    E1s = Efun(ksigned)
    out.update(Theta=theta, margin=float(np.min(theta - E1s)),
               margin_k=float(ksigned[np.argmin(theta - E1s)]))
    excl = set()
    for c in (cf[0], cf[-1]):                      # band-1 and band-2 at k = 0
        excl |= set(range(*clusters[c]))
    mst = [n for n in range(1, len(w))
           if resolved[n] and abs(phases[n]) < 1e-3 and frac[n] < RESIDUE_MSTAR
           and gaps[n] < 2 * M and n not in excl]
    mst_even = [n for n in mst if np.isfinite(parity[n]) and parity[n] > 0]
    out.update(n_Mstar=len(mst), Mstar=float(gaps[mst[0]]) if mst else np.nan,
               Mstar_parity=float(parity[mst[0]]) if mst else np.nan,
               Mstar_even=float(gaps[mst_even[0]]) if mst_even else np.nan)
    i1 = [cf[0]]                                   # k = 0 band-1 cluster

    # ---- flux-string length of the k = 0 meson and of the vacuum
    pvac = np.abs(vac) ** 2
    e_dom = ctx.ebit[int(np.argmax(pvac))]
    L = np.count_nonzero(ctx.ebit != e_dom[None, :], axis=1)
    PL_vac = np.bincount(L, weights=pvac, minlength=ns + 1)
    psi = V[:, clusters[i1[0]][0]]
    PL = np.bincount(L, weights=np.abs(psi) ** 2, minlength=ns + 1)
    out.update(PL=PL, PL_vac=PL_vac, vac_dom_weight=float(pvac.max()))
    out.update(string_size(PL, PL_vac))

    # ---- vacuum correlation length from connected same-sublattice J0 J0
    Qw = ctx.q0 * pvac[None, :]
    G = Qw @ ctx.q0.T                                          # <q_v q_w>
    mean = ctx.q0 @ pvac
    Gc = G - np.outer(mean, mean)
    Cr = np.array([np.mean([Gc[v, (v + 2 * r) % ns] for v in range(ns)])
                   for r in range(nx)])
    xi, xi_fit_ok = fit_xi(Cr, nx)
    out.update(Cr=Cr, xi=xi, xi_fit_ok=xi_fit_ok)

    # ---- quasi-PDF <x> of the exact |k = 2 pi / 5> band-1 state
    out["x_mean"] = np.nan
    if want_x and nx % 5 == 0:
        a5 = int(np.argmin(np.abs(kabs - K5)))
        i, j = clusters[cf[a5]]
        n5 = [n for n in range(i, j) if abs(phases[n] - K5) < 1e-3]
        if n5:
            st = V[:, n5[0]] / np.linalg.norm(V[:, n5[0]])
            out["x_mean"] = quasipdf_xmean(ctx, st, K5)

    apply_gates(out)
    out["runtime"] = time.time() - t0
    return out


def _finish_nan(out):
    for key in ("M", "Epi", "W", "WM", "m_curv", "m_curv_over_M", "vg_max",
                "vg_k5", "vg_k5_cs", "Ek5", "M2min", "M2_k0", "M2_kpi", "gap_M2",
                "gap_M2_k5", "int_gap", "twoM", "MMp", "margin", "Wfine", "c_k5", "rel_rms",
                "margin_k", "Mstar", "L_mean", "L_mean_vac", "L_rms", "size_rms",
                "size_mean", "size_excess", "size_rms_raw", "xi", "x_mean",
                "vac_dom_weight", "curv0", "Mstar_parity", "Mstar_even"):
        out.setdefault(key, np.nan)
    out.setdefault("M2_orient", "n/a")
    out.setdefault("n_Mstar", 0)
    out.setdefault("xi_fit_ok", False)
    for g in ("gate_band", "gate_margin", "gate_size", "gate_xi", "gate_M2"):
        out.setdefault(g, False)
    out.setdefault("feasible", False)
    return out


RANK_MODE = "wm"


def rank_key(row):
    """Ranking among feasible points (infeasible last).
    RANK_MODE = 'wm'    : W*M closest to 2 from below (points above 2 after
                          all points below) -- the brief's cosine-band proxy;
    RANK_MODE = 'mcurv' : |m_curv/M - 1| smallest, i.e. E1''(0) closest to
                          1/M directly (the proxy assumes a pure cosine band
                          in k; the scan shows W*M = 2 at eta ~ 1.9 while
                          m_curv/M = 1 needs eta ~ 2.3)."""
    wm, mc = row.get("WM", np.nan), row.get("m_curv_over_M", np.nan)
    if not row.get("feasible", False) or not np.isfinite(wm):
        return (2, 0, np.inf)
    if RANK_MODE == "mcurv":
        return (0, 0, abs(mc - 1) if np.isfinite(mc) else np.inf)
    return (0 if wm <= 2 else 1, 0, abs(wm - 2))


# --------------------------------------------------------------- driver
def _work(args):
    m0, g2, eta, k, want_x = args
    return analyze_point(m0, g2, eta, k, want_x=want_x)


def run_points(points, ns, k, workers, want_x=False):
    global CTX
    log(f"building ns={ns} context (Q=0 block)")
    CTX = ScanContext(ns)
    log(f"  dim {CTX.dim}, nnz(Hh) {CTX.Hh.nnz}, reflection variant {CTX.R_variant}")
    jobs = [(m0, g2, eta, k, want_x) for (m0, g2, eta) in points]
    rows = []
    if workers > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(workers) as pool:
            for i, row in enumerate(pool.imap_unordered(_work, jobs), 1):
                rows.append(row)
                if i % max(1, len(jobs) // 20) == 0 or i == len(jobs):
                    log(f"  {i}/{len(jobs)} points done "
                        f"(last: m0={row['m0']:.2f} g2={row['g2']:.2f} "
                        f"eta={row['eta']:.2f}, {row.get('runtime', 0):.1f} s)")
    else:
        for i, job in enumerate(jobs, 1):
            row = _work(job)
            rows.append(row)
            log(f"  {i}/{len(jobs)} m0={row['m0']:.2f} g2={row['g2']:.2f} "
                f"eta={row['eta']:.2f}: M={row.get('M', np.nan):.4f} "
                f"W*M={row.get('WM', np.nan):.3f} feasible={row.get('feasible')} "
                f"({row.get('runtime', 0):.1f} s)")
    rows.sort(key=lambda r: (r["g2"], r["m0"], r["eta"]))
    CTX = None
    return rows


SCALARS = ["m0", "g2", "eta", "ns", "nlev", "eigsh_ok", "E0", "M", "Epi", "W",
           "WM", "m_curv", "m_curv_over_M", "curv0", "vg_max", "vg_k5", "vg_k5_cs",
           "Ek5", "Wfine", "c_k5", "rel_rms", "band_ok", "n_fail", "frac_min", "M2min", "M2_k0", "M2_kpi",
           "M2_orient", "gap_M2", "gap_M2_k5", "int_gap", "twoM",
           "MMp", "margin", "margin_k", "n_Mstar", "Mstar", "Mstar_parity",
           "Mstar_even", "L_mean", "L_mean_vac",
           "L_rms", "size_rms_raw", "size_rms", "size_mean", "size_excess", "vac_dom_weight", "xi",
           "xi_fit_ok", "x_mean", "gate_band", "gate_margin", "gate_size",
           "gate_xi", "gate_M2", "feasible", "rank", "runtime"]
ARRAYS = ["kabs", "E1", "frac1", "E2", "frac2", "Eint", "pabs", "Efine",
          "frac_fine", "ok_fine", "Theta", "ptot", "PL", "PL_vac", "Cr", "trig",
          "all_gaps", "all_phases", "all_frac", "all_parity", "all_resolved"]


def assign_ranks(rows):
    order = sorted(range(len(rows)), key=lambda i: rank_key(rows[i]))
    for r in rows:
        r["rank"] = -1
    for pos, i in enumerate(order):
        if rows[i].get("feasible", False):
            rows[i]["rank"] = pos + 1
    return [rows[i] for i in order]


def pack(rows, prefix=""):
    """Rows -> dict of arrays for np.savez (NaN padded)."""
    out = {}
    for key in SCALARS:
        vals = [r.get(key, np.nan) for r in rows]
        if key in ("M2_orient",):
            out[prefix + key] = np.array([str(v) for v in vals])
        elif key in ("band_ok", "eigsh_ok", "xi_fit_ok", "feasible") or key.startswith("gate_"):
            out[prefix + key] = np.array([bool(v) for v in vals])
        else:
            out[prefix + key] = np.array([float(v) if v is not None else np.nan for v in vals])
    for key in ARRAYS:
        arrs = [np.atleast_1d(r[key]) for r in rows if key in r]
        if not arrs:
            continue
        width = max(a.shape[0] for a in arrs)
        mat = np.full((len(rows), width), np.nan)
        for i, r in enumerate(rows):
            if key in r:
                a = np.atleast_1d(r[key]).astype(float)
                mat[i, :a.shape[0]] = a
        out[prefix + key] = mat
    return out


def write_csv(rows, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(SCALARS + [f"E1_{i}" for i in range(len(rows[0]["kabs"]))]
                    + [f"E2_{i}" for i in range(len(rows[0]["kabs"]))])
        for r in rows:
            line = []
            for key in SCALARS:
                v = r.get(key, np.nan)
                if isinstance(v, (bool, np.bool_)):
                    line.append(int(v))
                elif isinstance(v, str):
                    line.append(v)
                else:
                    line.append(f"{float(v):.6g}")
            line += [f"{x:.6f}" for x in r.get("E1", [])]
            line += [f"{x:.6f}" for x in r.get("E2", [])]
            wr.writerow(line)


def print_table(rows, title, n=None):
    cols = [("m0", "{:.2f}"), ("g2", "{:.2f}"), ("eta", "{:.2f}"), ("M", "{:.4f}"),
            ("Epi", "{:.4f}"), ("W", "{:.4f}"), ("WM", "{:.3f}"),
            ("m_curv_over_M", "{:.3f}"), ("c_k5", "{:.3f}"), ("vg_k5", "{:+.3f}"), ("vg_max", "{:.3f}"),
            ("margin", "{:.3f}"), ("size_rms", "{:.3f}"), ("size_excess", "{:.3f}"),
            ("xi", "{:.3f}"), ("gap_M2", "{:+.3f}"), ("int_gap", "{:+.3f}"),
            ("frac_min", "{:.3f}"), ("Mstar", "{:.3f}"), ("x_mean", "{:.4f}")]
    print(f"\n{title}")
    hdr = " ".join(f"{c:>9s}" for c, _ in cols) + "   gates(band,margin,size,xi,M2)"
    print(hdr)
    for r in rows[:n]:
        cells = []
        for c, fmt in cols:
            v = r.get(c, np.nan)
            try:
                cells.append(f"{fmt.format(float(v)):>9s}")
            except (TypeError, ValueError):
                cells.append(f"{'nan':>9s}")
        gates = "".join("Y" if r.get(g, False) else "." for g in
                        ("gate_band", "gate_margin", "gate_size", "gate_xi", "gate_M2"))
        print(" ".join(cells) + f"   {gates}  {'FEASIBLE' if r.get('feasible') else ''}")


# ---------------------------------------------------------------- figure
def make_figure(rows, ranked, path, refine_rows=None):
    """(a) W*M map in (m0, eta) per g2, diverging about W*M = 2, with the
    feasibility mask (white x), the m_curv/M = 1 contour (dashed) and the
    top candidates circled; (b) E(k)/M of the top candidates (refined
    volume when available) vs the cosine and sqrt(M^2 + k^2) forms;
    (c) vacuum-subtracted meson RMS string size and vacuum xi vs W*M;
    (d) W*M/2, m_curv/M and c(2 pi/5) vs eta along the top candidates'
    (m0, g2) slices -- where the two relativistic criteria cross 1."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    try:
        from paper_style import OI
    except Exception:                                          # noqa: BLE001
        OI = ["#0072B2", "#D55E00", "#009E73", "#E69F00", "#56B4E9", "#CC79A7", "k"]
    g2s = sorted(set(r["g2"] for r in rows))
    m0s = sorted(set(r["m0"] for r in rows))
    etas = sorted(set(r["eta"] for r in rows))
    top = [r for r in ranked if r.get("feasible")][:6]
    ncol = max(len(g2s), 3)
    fig = plt.figure(figsize=(3.3 * ncol, 6.0))
    gs = fig.add_gridspec(2, ncol, height_ratios=[1.0, 1.05], hspace=0.45,
                          wspace=0.35, left=0.06, right=0.93, top=0.94, bottom=0.08)
    wm_all = np.array([r.get("WM", np.nan) for r in rows])
    fin = np.isfinite(wm_all)
    vmin, vmax = (np.nanmin(wm_all[fin]), np.nanmax(wm_all[fin])) if fin.any() else (0, 4)
    norm = TwoSlopeNorm(vcenter=2.0, vmin=min(vmin, 1.9), vmax=max(vmax, 2.1))
    pc = None
    for c, g2 in enumerate(g2s):
        ax = fig.add_subplot(gs[0, c])
        grid = np.full((len(etas), len(m0s)), np.nan)
        mcg = np.full_like(grid, np.nan)
        feas = np.zeros_like(grid, dtype=bool)
        for r in rows:
            if r["g2"] != g2:
                continue
            i, j = etas.index(r["eta"]), m0s.index(r["m0"])
            grid[i, j] = r.get("WM", np.nan)
            # curvature is meaningless where the band was not identified
            mcg[i, j] = r.get("m_curv_over_M", np.nan) if r.get("band_ok") else np.nan
            feas[i, j] = r.get("feasible", False)
        dm = (m0s[1] - m0s[0]) if len(m0s) > 1 else 0.1
        de = (etas[1] - etas[0]) if len(etas) > 1 else 0.1
        me = np.array(m0s + [m0s[-1] + dm]) - dm / 2
        ee = np.array(etas + [etas[-1] + de]) - de / 2
        pc = ax.pcolormesh(me, ee, grid, cmap="RdBu_r", norm=norm, shading="flat")
        if np.isfinite(mcg).sum() > 4 and len(m0s) > 1 and len(etas) > 1:
            ax.contour(np.array(m0s), np.array(etas), mcg, levels=[1.0],
                       colors="k", linestyles="--", linewidths=1.0)
            ax.contour(np.array(m0s), np.array(etas), grid, levels=[2.0],
                       colors="k", linestyles=":", linewidths=0.8)
        ii, jj = np.nonzero(~feas & np.isfinite(grid))
        ax.plot(np.array(m0s)[jj], np.array(etas)[ii], "x", color="w", ms=4, mew=0.9)
        for rk, r in enumerate(top):
            if r["g2"] == g2:
                ax.plot(r["m0"], r["eta"], "o", mfc="none", mec=OI[2], ms=9, mew=1.5)
                ax.annotate(str(rk + 1), (r["m0"], r["eta"]), color=OI[2],
                            xytext=(4, 4), textcoords="offset points", fontsize=7)
        if g2 == PRODUCTION[1] and PRODUCTION[0] in m0s and PRODUCTION[2] in etas:
            ax.plot(PRODUCTION[0], PRODUCTION[2], "s", mfc="none", mec="k", ms=8, mew=1.2)
        ax.set_title(f"$g^2 = {g2:.2f}$")
        ax.set_xlabel("$m_0$")
        ax.set_ylabel(r"$\eta$" if c == 0 else "")
        ax.grid(False)
        if c == 0:
            ax.plot([], [], "k--", lw=1.0, label=r"$m_{\rm curv}/M = 1$")
            ax.plot([], [], "k:", lw=0.8, label=r"$W\,M = 2$")
            ax.plot([], [], "x", color="0.5", ms=4, label="fails a gate")
            ax.legend(fontsize=6, loc="lower right", framealpha=0.8)
    cax = fig.add_axes([0.94, 0.58, 0.012, 0.32])
    fig.colorbar(pc, cax=cax, label=r"$W\,M$")

    # (b) E(k) of the top candidates vs cosine and relativistic forms
    ax = fig.add_subplot(gs[1, 0])
    kf = np.linspace(0, np.pi, 200)
    src = refine_rows if refine_rows else top
    for rk, r in enumerate(src[:6]):
        col = OI[rk % len(OI)]
        M, W = r["M"], r["W"]
        ax.plot(r["kabs"], r["E1"] / M, "o", color=col, ms=4,
                label=f"{rk + 1}: ({r['m0']:.1f},{r['g2']:.1f},{r['eta']:.1f}) $N_s$={int(r['ns'])}")
        ax.plot(kf, (M + W / 2 * (1 - np.cos(kf))) / M, "--", color=col, lw=0.7)
        ax.plot(kf, np.sqrt(M**2 + kf**2) / M, ":", color=col, lw=0.7)
    ax.plot([], [], "k--", lw=0.7, label="cosine $(M, W)$")
    ax.plot([], [], "k:", lw=0.7, label=r"$\sqrt{M^2+k^2}$")
    ax.axvline(K5, color="0.6", lw=0.6)
    ax.set_xlabel("$k$")
    ax.set_ylabel("$E(k)/M$")
    ax.legend(fontsize=5.5, loc="upper left")
    # (c) size / xi vs W*M
    ax = fig.add_subplot(gs[1, 1])
    for r in rows:
        if not np.isfinite(r.get("WM", np.nan)):
            continue
        fe = r.get("feasible", False)
        ax.plot(r["WM"], r["size_rms"], "o", color=OI[0] if fe else "0.75", ms=3, alpha=0.8)
        ax.plot(r["WM"], r["xi"], "^", color=OI[1] if fe else "0.85", ms=3, alpha=0.8)
    ax.axhline(GATES["size"], color="k", lw=0.6, ls="--")
    ax.axvline(2.0, color="k", lw=0.6, ls=":")
    ax.plot([], [], "o", color=OI[0], ms=3, label="meson RMS string size (vac. sub.)")
    ax.plot([], [], "^", color=OI[1], ms=3, label=r"vacuum $\xi$")
    ax.plot([], [], "o", color="0.75", ms=3, label="fails a gate")
    ax.set_xlabel(r"$W\,M$")
    ax.set_ylabel("spatial sites")
    ax.set_ylim(0, max(2.5, GATES["size"] * 1.4))
    ax.legend(fontsize=6, loc="upper left")
    # (d) criteria vs eta along the top candidates' (m0, g2) slices
    ax = fig.add_subplot(gs[1, 2])
    seen = []
    for rk, r in enumerate(top):
        key = (r["m0"], r["g2"])
        if key in seen:
            continue
        seen.append(key)
        col = OI[rk % len(OI)]
        sl = sorted([q for q in rows if (q["m0"], q["g2"]) == key
                     and np.isfinite(q.get("WM", np.nan))], key=lambda q: q["eta"])
        e = [q["eta"] for q in sl]
        ax.plot(e, [q["WM"] / 2 for q in sl], ":", color=col, lw=0.9)
        ax.plot(e, [q["m_curv_over_M"] for q in sl], "-", color=col, lw=1.0,
                label=f"$(m_0, g^2) = ({key[0]:.1f}, {key[1]:.1f})$")
        ax.plot(e, [q["c_k5"] for q in sl], "--", color=col, lw=0.9)
        ax.plot([q["eta"] for q in sl if not q["feasible"]],
                [q["m_curv_over_M"] for q in sl if not q["feasible"]], "x", color=col, ms=4)
    ax.axhline(1.0, color="k", lw=0.6)
    ax.plot([], [], "k-", lw=1.0, label=r"$m_{\rm curv}/M$")
    ax.plot([], [], "k--", lw=0.9, label=r"$c(2\pi/5)$")
    ax.plot([], [], "k:", lw=0.9, label=r"$W\,M/2$")
    ax.set_xlabel(r"$\eta$")
    ax.set_ylabel("criterion (relativistic = 1)")
    ax.set_ylim(0.4, 2.2)
    ax.legend(fontsize=5.5, loc="upper right")
    for c in range(3, ncol):
        fig.add_subplot(gs[1, c]).axis("off")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path)
    log(f"figure -> {path}")


# ----------------------------------------------------------------- args
def parse_range(tokens):
    vals = []
    for t in tokens:
        if ":" in t:
            a, b, s = map(float, t.split(":"))
            n = int(round((b - a) / s)) + 1
            vals += list(np.round(a + s * np.arange(n), 6))
        else:
            vals.append(float(t))
    return sorted(set(vals))


def validate():
    """Cross-check against stored deep levels (production couplings):
    gaps + |T2 phases| at ns=12 (data/deep_levels_ns12.npz) and the band-1
    E(k) at ns=20 quoted in the WS2-A brief."""
    global CTX
    for ns in (12,):
        ref = np.load(f"data/deep_levels_ns{ns}.npz")
        CTX = ScanContext(ns)
        k = min(40, len(ref["gaps"]))
        row = analyze_point(*PRODUCTION, k_levels=k)
        dg = np.abs(row["all_gaps"][:k] - ref["gaps"][:k]).max()
        # the last cluster can straddle the eigsh cutoff (flagged unresolved)
        kk = k - 4
        dp = np.abs(np.sort(np.abs(row["all_phases"][:kk]))
                    - np.sort(np.abs(ref["phases"][:kk]))).max()
        log(f"ns={ns}: max |gap diff| {dg:.2e}, max |phase| diff {dp:.2e} "
            f"(first {kk} levels; unresolved flags: "
            f"{np.flatnonzero(~row['all_resolved'])})")
        log(f"  E1(k) = {np.round(row['E1'], 4)} at |k| = {np.round(row['kabs'], 3)}")
        log(f"  frac1 = {np.round(row['frac1'], 3)}  band_ok={row['band_ok']}")
        log(f"  E2(k) = {np.round(row['E2'], 4)} frac2 = {np.round(row['frac2'], 3)}")
        log(f"  fine band E(p) = {np.round(row['Efine'], 4)} at p = {np.round(row['pabs'], 3)}")
        log(f"  Eint(k) = {np.round(row['Eint'], 4)}  int_gap={row['int_gap']:.3f} "
            f"gap_M2={row['gap_M2']:.3f}")
        if "refl" in ref:
            m0 = np.abs(ref["phases"][:k]) < 1e-3
            mine = row["all_parity"][:k][m0]
            log(f"  P=0 parities mine {np.round(mine, 2)} vs stored "
                f"{np.round(ref['refl'][:k][m0], 2)}")
        log(f"  M={row['M']:.4f} W*M={row['WM']:.3f} m_curv/M={row['m_curv_over_M']:.3f} "
            f"margin={row['margin']:.3f} size_rms={row['size_rms']:.3f} xi={row['xi']:.3f} "
            f"Mstar={row['Mstar']} n_Mstar={row['n_Mstar']} R variant={CTX.R_variant}")
        assert dg < 1e-6 and dp < 1e-3
    CTX = None
    log("validation OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--g2", nargs="+", default=["0.8", "1.1", "1.4"])
    ap.add_argument("--m0", nargs="+", default=["0.2:0.7:0.1"])
    ap.add_argument("--eta", nargs="+", default=["1.3:2.6:0.1"])
    ap.add_argument("--ns", type=int, default=16)
    ap.add_argument("--k", type=int, default=60, help="eigsh levels at --ns")
    ap.add_argument("--refine-ns", type=int, default=0)
    ap.add_argument("--refine-k", type=int, default=80)
    ap.add_argument("--top", type=int, default=6)
    ap.add_argument("--extra", nargs="*", default=[],
                    help="extra m0,g2,eta triples appended to the refine list")
    ap.add_argument("--refine-only", action="store_true",
                    help="skip the scan; rank the stored --out and refine")
    ap.add_argument("--fig-only", action="store_true",
                    help="redraw --fig from the stored --out (scan + refine)")
    ap.add_argument("--out", default="data/coupling_scan.npz")
    ap.add_argument("--csv", default="data/csv/coupling_scan.csv")
    ap.add_argument("--fig", default="data/coupling_scan.pdf")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--rank", choices=["wm", "mcurv"], default="wm",
                    help="ranking: W*M -> 2 (brief) or m_curv/M -> 1 (direct)")
    ap.add_argument("--validate", action="store_true")
    args = ap.parse_args()
    global RANK_MODE
    RANK_MODE = args.rank
    if args.validate:
        validate()
        return

    if args.fig_only:
        z = np.load(args.out, allow_pickle=True)
        rows = unpack(z)
        for r in rows:
            if "PL" in r and np.isfinite(r.get("M", np.nan)):
                r.update(string_size(r["PL"], r["PL_vac"]))
                apply_gates(r)
        ranked = assign_ranks(rows)
        ref = unpack(z, prefix="ref_") if "ref_m0" in z else None
        if ref:
            ref = assign_ranks(ref)
            print_table(ref, "refined rows (stored)")
        print_table(ranked, f"top {args.top} by --rank {args.rank}", args.top)
        make_figure(rows, ranked, args.fig, ref)
        return
    if args.refine_only:
        z = np.load(args.out, allow_pickle=True)
        rows = unpack(z)
        for r in rows:                     # re-derive sizes and gates
            if "PL" in r and "PL_vac" in r and np.isfinite(r.get("M", np.nan)):
                r.update(string_size(r["PL"], r["PL_vac"]))
                apply_gates(r)
        log(f"loaded {len(rows)} scan rows from {args.out} (gates re-derived)")
    else:
        g2s, m0s, etas = parse_range(args.g2), parse_range(args.m0), parse_range(args.eta)
        points = [(m0, g2, eta) for g2 in g2s for m0 in m0s for eta in etas]
        log(f"scan: {len(points)} points at ns={args.ns}, k={args.k}, "
            f"workers={args.workers}, threads={os.environ['OMP_NUM_THREADS']}")
        rows = run_points(points, args.ns, args.k, args.workers)
    ranked = assign_ranks(rows)
    n_feas = sum(r.get("feasible", False) for r in rows)
    log(f"{n_feas}/{len(rows)} points pass all gates")
    for g in ("gate_band", "gate_margin", "gate_size", "gate_xi", "gate_M2"):
        log(f"  {g}: {sum(1 for r in rows if not r.get(g, False))} failures")
    other = "mcurv" if args.rank == "wm" else "wm"
    print_table(ranked, f"top {args.top} by --rank {args.rank} (scan, ns={rows[0]['ns']})",
                args.top)
    RANK_MODE = other
    print_table(sorted(rows, key=rank_key),
                f"for comparison: top {args.top} by --rank {other}", args.top)
    RANK_MODE = args.rank
    print_table([r for r in ranked if r["m0"] == PRODUCTION[0] and r["g2"] == PRODUCTION[1]
                 and r["eta"] == PRODUCTION[2]], "production reference (0.7, 1.1, 1.3)")
    write_csv(ranked, args.csv)
    save = pack(ranked)
    save["gates"] = np.array([f"{k}={v}" for k, v in GATES.items()])

    refine_rows = None
    if args.refine_ns:
        cand = [(r["m0"], r["g2"], r["eta"]) for r in ranked if r.get("feasible")][:args.top]
        for t in args.extra:
            m0, g2, eta = map(float, t.split(","))
            if (m0, g2, eta) not in cand:
                cand.append((m0, g2, eta))
        log(f"refining {len(cand)} points at ns={args.refine_ns}, k={args.refine_k}")
        refine_rows = run_points(cand, args.refine_ns, args.refine_k,
                                 min(args.workers, len(cand)), want_x=True)
        refine_rows = assign_ranks(refine_rows)
        print_table(refine_rows, f"refined (ns={args.refine_ns})")
        save.update(pack(refine_rows, prefix="ref_"))
        write_csv(refine_rows, args.csv.replace(".csv", "_refine.csv"))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez(args.out, **save)
    log(f"saved {args.out} and {args.csv}")
    make_figure(rows, ranked, args.fig, refine_rows)


def unpack(z, prefix=""):
    """Inverse of pack for the scalar + array columns (refine-only mode)."""
    n = len(z[prefix + "m0"])
    rows = []
    for i in range(n):
        r = {}
        for key in SCALARS:
            if prefix + key in z:
                v = z[prefix + key][i]
                r[key] = v.item() if hasattr(v, "item") else v
        for key in ARRAYS:
            if prefix + key in z:
                a = z[prefix + key][i]
                r[key] = a[np.isfinite(a)] if key not in ("all_parity",) else a
        rows.append(r)
    return rows


if __name__ == "__main__":
    main()
