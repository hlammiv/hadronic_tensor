"""Moving-frame (total momentum P != 0) two-meson Luscher analysis at CGK-A
(m0, g2, eta) = (0.1, 0.4, 1.0): densify the elastic MM phase shift delta(E)
near the C-even resonance by adding P = 2*pi*m/Nx (m = 1, 2) finite-volume
levels to the known P = 0 set.

Everything is post-processing of the stored deep spectra
data/deep_levels_cgkinA_ns{8..20}.npz (gaps, T2 phases, P=0 reflection
parities) -- no new diagonalization.

Method (1+1d Bethe-Yang, the analog of Rummukainen-Gottlieb):
  a level at total momentum P = 2*pi*m/L (L = ns/2 physical sites) solves
     E2_lab = E(k1) + E(k2),   k1 + k2 = P,
     k1 L + 2 delta = 2*pi*n1  (and identically for constituent 2),
  with E(k) the one-meson lattice dispersion (band-1 spline, ns=20).
  E(k1) + E(P-k1) is monotonic for k1 in (P/2, P/2+pi) (the stationary
  points are at relative momentum 0 and pi), so k1 is the unique root there
  and delta = fold[(2*pi*n1 - k1 L)/2] into (-pi/2, pi/2].  At P = 0 this
  is EXACTLY the stored pipeline (scripts/factorization.py): p = k1,
  delta = (2*pi*n - p L)/2.  n1 is fixed by counting interacting levels up
  from the free tower of the sector: the j-th interacting MM level takes
  the j-th free pair {a, m-a}, i.e. n1 = ceil(m/2) + j.

  The collapse variable is the interaction-frame (CM) energy
     E_cm = 2 E(k_rel),  k_rel = k1 - P/2,
  which reduces to E2_lab at P = 0 and maps each sector bottom to 2M.  The
  lab energy is NOT the right variable at P != 0 (the resonance would
  appear boosted); the collapse of the moving-frame points onto the P = 0
  curve in E_cm is checked (and reported) rather than assumed.

Level identification per sector (ns = 12..20, m = 0, 1, 2):
  - m = 0: reflection-even (R = +1) levels in (2M, MM'-threshold), exactly
    the factorization.py / phase_shifts_v2.py selection (R = -1 removes the
    M'' one-body state at 1.941 and the odd MM'-threshold level).
  - m != 0: no R label exists (R maps P -> -P).  One-body contaminants are
    removed by proximity (< TOL3) to the empirical band-3 (M'') dispersion
    E3(P), fitted to volume-consistent anchor levels across ns = 14..20
    (E3 runs 1.941 at P=0 -> 2.0215 at P=pi, straight through the elastic
    window).  If TWO sector levels fall within TOL3 of E3(P) the pair is an
    avoided crossing: the closer is dropped as one-body and the other is
    dropped as ambiguous, but it still consumes a slot in the free-tower
    counting.  The vector band E(P) < 1.44 and scalar band E'(P) < 1.50
    stay below the window for every sector used.
  - MM' (vector+scalar) contamination: per-sector inelastic lab threshold
    thr(P) = min_k [E1(k) + E'(P-k)] from the empirical band-2 spline;
    levels above thr(P) - THR_PAD are dropped (the P=0 near-threshold MM'
    level sits within 0.01 of thr, so the pad matters).

Identification is CONFIRMED EXACTLY by scripts/moving_frame_residue_check.py
(ns <= 16 ED): momentum-projected interpolator residues obey an exact
selection rule -- every sector level is either MM-coupled (res ~ 1e-2) or
decoupled at machine precision (res < 1e-25), even inside near-degenerate
doublets.  The band-3 tower carries an exact conserved label at all P (the
C/shift quantum number, reducing to R = -1 at P = 0) and never mixes with
MM.  One point dropped by the proximity heuristic is reinstated on that
evidence (ns=14 m=1, E=1.9595: its 1.9586 partner is the decoupled band-3
state); the ns=20 m=2 doublet 1.9724/1.9762 stays dropped because no
residue computation is affordable at ns=20 to tell the pair apart.

SANITY GATE (stage 1): the P = 0 arm of this pipeline must reproduce the
stored data/factorization_cgkA.npz level set and the known delta(E) curve
(1.29 at E=1.745, min 1.06 near 1.93, through pi/2 near 2.11) before any
moving-frame point is used.

RESULT (stage 3, run 2026-07-21): the moving-frame points are mutually
consistent at FIXED P across volumes (<= 0.03 rad at P = 2.094 between
L = 6 and L = 9) but do NOT collapse onto the P = 0 delta(E) under ANY
single energy variable: deviations grow smoothly with P, from +0.04..+0.13
rad at P ~ 0.6-0.8 to +1.3 rad at P = 2.09, and for P >= 1.4 the sector
phase (mod pi) leaves the range spanned by delta(E) entirely, so no
reparametrization of the energy axis can restore a collapse.  The two-body
phase of these composite mesons is genuinely frame-dependent, phi(P, E) --
the 1+1d Rummukainen-Gottlieb analog assumed here requires a boost
invariance this lattice theory does not have at P ~ 1.  Fits:
  A: P = 0 only (the 15-point baseline);
  B: P = 0 plus m = 1 (P <= 1.05), one nuisance frame term  g0 P^2;
  C: all sectors,  (g0 + g1 (E - 1.7)) P^2.
delta(E) = a + b (E-1.7) + atan2(Gamma/2, E_R - E) (+ frame term), fitted
on mod-pi residuals; 1-sigma errors from the residual-scaled covariance.

  PYTHONPATH=. .venv/bin/python scripts/moving_frame_luscher.py
"""

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq, least_squares

M0, G2, ETA = 0.1, 0.4, 1.0            # CGK-A
VOLS = (12, 14, 16, 18, 20)
MSEC = (0, 1, 2)                       # P = 2*pi*m/L sectors
TOL3 = 0.025                           # band-3 (M'') proximity cut
THR_PAD = 0.02                         # stay this far below the MM' threshold
EWIN = (1.72, 2.30)                    # elastic fit window in E_cm
PTOL = 1e-4                            # T2-phase sector match

# ---------------------------------------------------------------- dispersion
# band-1 spline exactly as scripts/factorization.py (ns=20, min gap per
# |phase| with 0.1 < gap < 1.5, clamped ends)
d20 = np.load("data/deep_levels_cgkinA_ns20.npz")
_g, _ph = d20["gaps"], d20["phases"]
_ks, _es = [], []
for kk in np.unique(np.round(np.abs(_ph), 6)):
    m = np.isclose(np.abs(_ph), kk) & (_g > 0.1) & (_g < 1.5)
    if m.any():
        _ks.append(kk); _es.append(_g[m].min())
E1 = CubicSpline(np.array(_ks), np.array(_es), bc_type=((1, 0.0), (1, 0.0)))
M = float(E1(0))


def efold(k):
    """E(k) for any real k (even, 2*pi-periodic)."""
    k = np.abs((np.asarray(k, float) + np.pi) % (2 * np.pi) - np.pi)
    return E1(k)


# ------------------------------------------------- empirical band-2 (scalar)
# second level of each sector at the two largest volumes -> spline in |K|
_b2 = {}
for ns in (18, 20):
    d = np.load(f"data/deep_levels_cgkinA_ns{ns}.npz")
    g, ph = d["gaps"], d["phases"]
    L = ns // 2
    for mm in range(0, L // 2 + 1):
        P = 2 * np.pi * mm / L
        sel = np.abs(np.angle(np.exp(1j * (ph - P)))) < PTOL
        lv = np.sort(g[sel & (g > 0.05)])
        cand = lv[(lv > 1.39) & (lv < 1.51)]
        if len(cand):
            _b2[round(P, 6)] = float(cand[0])   # ns=20 overwrites ns=18
_k2 = np.array(sorted(_b2))
E2b = CubicSpline(_k2, np.array([_b2[k] for k in _k2]),
                  bc_type=((1, 0.0), (1, 0.0)))


def e2fold(k):
    k = np.abs((np.asarray(k, float) + np.pi) % (2 * np.pi) - np.pi)
    return E2b(k)


_KGRID = np.linspace(-np.pi, np.pi, 4001)


def mmprime_thr(P):
    """Lab-frame MM' (vector+scalar) threshold in the P sector."""
    return float(np.min(efold(_KGRID) + e2fold(P - _KGRID)))


# ------------------------------------------- empirical band-3 (M'') curve
# anchors: P=0 (R=-1 level, ns=20) and P=pi (lower of the one-body doublet,
# ns=16/20); then iterate: harmonic fit -> adopt the unique sector level
# within 0.02 of the prediction (ns >= 14; skip sectors where two levels
# match -- avoided crossings) -> refit.
def _sector_levels(ns, mm):
    d = np.load(f"data/deep_levels_cgkinA_ns{ns}.npz")
    g, ph = d["gaps"], d["phases"]
    L = ns // 2
    P = 2 * np.pi * mm / L
    sel = np.abs(np.angle(np.exp(1j * (ph - P)))) < PTOL
    return P, np.sort(g[sel & (g > 0.05)])


def _b3_fit(nodes):
    A = np.array([[1.0, 1 - np.cos(p), 1 - np.cos(2 * p)] for p, _ in nodes])
    y = np.array([e for _, e in nodes])
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    return lambda P, c=c: c[0] + c[1] * (1 - np.cos(P)) \
        + c[2] * (1 - np.cos(2 * P)), c


_d20 = np.load("data/deep_levels_cgkinA_ns20.npz")
_rf = _d20["refl"] * _d20["refl"][0]
_p0 = np.abs(np.angle(np.exp(1j * _d20["phases"]))) < PTOL
_m3 = _p0 & (_rf < -0.99) & (_d20["gaps"] > 1.90) & (_d20["gaps"] < 1.96)
_anch0 = float(_d20["gaps"][_m3].min())          # 1.9411
_anchpi = float(_sector_levels(20, 5)[1][2])     # lower of the pi doublet
nodes = [(0.0, _anch0), (np.pi, _anchpi)]
E3c, _ = _b3_fit(nodes + [(np.pi / 2, 0.5 * (_anch0 + _anchpi))])
for _ in range(2):
    nodes = [(0.0, _anch0), (np.pi, _anchpi)]
    for ns in (14, 16, 18, 20):
        L = ns // 2
        for mm in range(1, L // 2 + (0 if ns in (16, 20) else 1)):
            P, lv = _sector_levels(ns, mm)
            near = lv[np.abs(lv - E3c(P)) < 0.02]
            if len(near) == 1:
                nodes.append((P, float(near[0])))
    E3c, _c3 = _b3_fit(nodes)
print(f"dispersion: M = {M:.4f}, E(pi) = {float(E1(np.pi)):.4f}; "
      f"band-2(0) = {float(E2b(0)):.4f}; "
      f"band-3: {_c3[0]:.4f} + {_c3[1]:.4f}(1-cosP) + {_c3[2]:.4f}(1-cos2P) "
      f"({len(nodes)} nodes)")
THR0 = mmprime_thr(0.0)
print(f"P=0 MM' threshold (empirical bands) = {THR0:.4f} "
      f"(stored pipeline used 2.2651)")

# ------------------------------------------------------------- delta machinery


def solve_k1(E2, P):
    """Unique root of E(k1) + E(P-k1) = E2 on the monotonic branch
    k1 in (P/2, P/2 + pi); None if E2 is below the branch bottom."""
    lo, hi = P / 2 + 1e-9, P / 2 + np.pi - 1e-9
    f = lambda k: float(efold(k) + efold(P - k)) - E2
    if f(lo) > 0 or f(hi) < 0:
        return None
    return brentq(f, lo, hi, xtol=1e-12)


def fold(d):
    return (d + np.pi / 2) % np.pi - np.pi / 2


def extract(E2, P, L, a):
    """(delta, k1, E_cm) for a level E2 in sector P with tower integer a."""
    k1 = solve_k1(E2, P)
    if k1 is None:
        return None
    dl = fold((2 * np.pi * a - k1 * L) / 2)
    Ecm = float(2 * efold(k1 - P / 2))
    return dl, k1, Ecm


# ============================== stage 1: P = 0 sanity gate ==================
print("\n=== stage 1: P = 0 sanity gate ===")
fact = np.load("data/factorization_cgkA.npz")
known = {}
for e, n in zip(fact["E"], fact["ns"]):
    known.setdefault(int(n), []).append(float(e))
known = {n: sorted(v) for n, v in known.items()}

gate_ok = True
gate_pts = []          # (ns, E2, a, delta, Ecm)
for ns in VOLS:
    d = np.load(f"data/deep_levels_cgkinA_ns{ns}.npz")
    g, ph = d["gaps"], d["phases"]
    rf = d["refl"] * d["refl"][0]
    L = ns // 2
    p0 = np.abs(np.angle(np.exp(1j * ph))) < PTOL
    cands = np.sort(g[p0 & (rf > 0.99) & (g > 2 * M + 1e-3)
                       & (g < THR0 - 2e-3)])
    ref = known.get(ns, [])
    match = (len(cands) == len(ref)
             and np.allclose(cands, ref, atol=5e-4))
    if not match:
        gate_ok = False
    print(f"ns={ns}: R=+1 P=0 window levels {np.round(cands, 4)} "
          f"vs factorization {np.round(ref, 4)}  "
          f"{'MATCH' if match else '*** MISMATCH ***'}")
    for j, E2 in enumerate(cands):
        dl, k1, Ecm = extract(E2, 0.0, L, j)
        assert abs(Ecm - E2) < 1e-9        # E_cm == E_lab at P = 0
        gate_pts.append((ns, E2, j, dl, Ecm))

# ns=22 stored levels (from the factorization cache; no new diagonalization)
for j, E2 in enumerate(known.get(22, [])):
    dl, k1, Ecm = extract(E2, 0.0, 11, j)
    gate_pts.append((22, E2, j, dl, Ecm))

gate_pts.sort(key=lambda r: r[1])
gE = np.array([r[1] for r in gate_pts])
gD = np.unwrap(np.array([r[3] for r in gate_pts]), period=np.pi)
print("\nP=0 delta(E) (unwrapped):")
for (ns, E2, j, dl, _), du in zip(gate_pts, gD):
    print(f"  ns={ns:2d} n={j}  E={E2:.4f}  delta={du:+.4f}")
gspl = CubicSpline(gE, gD)
chk = [("delta(1.745) ~ 1.29", float(gspl(1.745)), 1.29),
       ("min ~ 1.06 near 1.93", float(np.min(gspl(np.linspace(1.8, 2.0, 201)))), 1.06),
       ("E(delta=pi/2) ~ 2.11",
        float(brentq(lambda e: gspl(e) - np.pi / 2, 2.0, 2.2)), 2.11)]
for lab, got, want in chk:
    ok = abs(got - want) < 0.05
    gate_ok &= ok
    print(f"  {lab}: got {got:.3f}  ({'ok' if ok else '*** FAIL ***'})")
if not gate_ok:
    raise SystemExit("SANITY GATE FAILED -- stopping before moving frames.")
print("sanity gate PASSED")

# ============================== stage 2: moving frames ======================
print("\n=== stage 2: moving-frame sectors (m = 1, 2) ===")
# residue-check reinstatements: (ns, m, round(E,4)) -> reason
REINSTATE = {(14, 1, 1.9595):
             "C-even by ED residue 6.8e-2; 1.9586 partner exactly decoupled"}
rows = []      # dict per point
for ns in VOLS:
    L = ns // 2
    for mm in (1, 2):
        P, lv = _sector_levels(ns, mm)
        thr = mmprime_thr(P)
        # free MM tower {a, m-a}, a = ceil(m/2) + j on the branch
        a0 = (mm + 1) // 2
        free = [float(efold(2 * np.pi * a / L)
                      + efold(2 * np.pi * (mm - a) / L))
                for a in range(a0, a0 + 6)]
        win = lv[(lv > free[0] - 0.15) & (lv < 2.45)]
        e3 = float(E3c(P))
        near3 = np.abs(win - e3) < TOL3
        tags = []
        if near3.sum() >= 2:
            # avoided crossing: closest is one-body, the rest ambiguous
            i1b = int(np.argmin(np.abs(win - e3)))
            for i in range(len(win)):
                tags.append("band3" if i == i1b else
                            ("ambig" if near3[i] else "MM"))
        else:
            tags = ["band3" if near3[i] else "MM" for i in range(len(win))]
        print(f"\nns={ns} m={mm} P={P:.4f}  band3(P)={e3:.4f} "
              f"thr_MM'={thr:.4f}")
        print(f"  free tower: {np.round(free[:4], 4)}")
        j = 0
        for Elab, tag in zip(win, tags):
            if tag == "ambig" and (ns, mm, round(float(Elab), 4)) in REINSTATE:
                print(f"  {Elab:.4f}: reinstated -- "
                      f"{REINSTATE[(ns, mm, round(float(Elab), 4))]}")
                tag = "MM"
            if tag == "band3":
                print(f"  {Elab:.4f}: one-body (band-3), excluded")
                continue
            a = a0 + j
            j += 1                      # MM and ambig both consume a slot
            if tag == "ambig":
                print(f"  {Elab:.4f}: slot a={a} DROPPED "
                      f"(within {TOL3} of band-3, no ns>16 residue check "
                      f"affordable -- cannot tell the C-sectors apart)")
                continue
            out = extract(Elab, P, L, a)
            if out is None:
                print(f"  {Elab:.4f}: slot a={a} below branch bottom "
                      f"2E(P/2)={2 * float(efold(P / 2)):.4f} -- no real k1")
                continue
            dl, k1, Ecm = out
            keep, why = True, ""
            if Elab > thr - THR_PAD:
                keep, why = False, f"above MM' thr-{THR_PAD}"
            elif not (EWIN[0] <= Ecm <= EWIN[1]):
                keep, why = False, f"E_cm outside {EWIN}"
            elif abs(Elab - free[j - 1]) > 0.35:
                keep, why = False, "free-tower assignment unsafe (>0.35)"
            rows.append(dict(ns=ns, m=mm, P=P, L=L, Elab=Elab, a=a, k1=k1,
                             Ecm=Ecm, delta=dl, keep=keep))
            print(f"  {Elab:.4f}: a={a} (free {free[j - 1]:.4f})  "
                  f"k1={k1:.4f}  E_cm={Ecm:.4f}  delta={dl:+.4f}"
                  f"{'' if keep else '  [dropped: ' + why + ']'}")

kept = [r for r in rows if r["keep"]]
print(f"\nmoving-frame points kept: {len(kept)} "
      f"(of {len(rows)} extracted)")

# ============================== stage 3: collapse + fit =====================
print("\n=== stage 3: collapse test ===")
# collapse of the new points onto the P=0 spline, in E_cm vs E_lab
dev_cm, dev_lab = [], []
for r in kept:
    if gE[0] <= r["Ecm"] <= gE[-1]:
        r["dev"] = float(fold(r["delta"] - fold(float(gspl(r["Ecm"])))))
        dev_cm.append(r["dev"])
    else:
        r["dev"] = np.nan
    if gE[0] <= r["Elab"] <= gE[-1]:
        dev_lab.append(fold(r["delta"] - fold(float(gspl(r["Elab"])))))
print(f"collapse vs P=0 spline:  E_cm : rms {np.sqrt(np.mean(np.array(dev_cm)**2)):.4f} rad "
      f"(max |dev| {np.max(np.abs(dev_cm)):.4f}, {len(dev_cm)} pts)")
print(f"                         E_lab: rms {np.sqrt(np.mean(np.array(dev_lab)**2)):.4f} rad "
      f"(max |dev| {np.max(np.abs(dev_lab)):.4f}, {len(dev_lab)} pts)")
print("deviation grows smoothly with P (frame effect), NOT with 1/L:")
for r in sorted(kept, key=lambda r: (r["P"], r["Ecm"])):
    print(f"  P={r['P']:.4f} (ns={r['ns']:2d} m={r['m']})  E_cm={r['Ecm']:.4f}"
          f"  delta={r['delta']:+.4f}  dev={r['dev']:+.4f}")

# same-P, different-L consistency (diagnostic sectors incl. ns=8,10 and m=3,
# outside the fit set; band-3 levels skipped by the same curve criterion)
print("\nsame-P cross-volume consistency (quantization self-consistent,")
print("phase frame-dependent -- these agree with each other, not with P=0):")
DIAG = {1.2566: [(10, 1, 2.0588, 1)], 1.5708: [(8, 1, 2.1094, 1)],
        2.0944: [(18, 3, 1.9782, 2), (18, 3, 2.1137, 3)]}
for P, extra in DIAG.items():
    pts = [(r["ns"], r["m"], r["Elab"], r["a"]) for r in kept
           if abs(r["P"] - P) < 1e-6] + extra
    for ns, mm, Elab, a in sorted(pts, key=lambda t: t[2]):
        dl, k1, Ecm = extract(Elab, P, ns // 2, a)
        print(f"  P={P:.4f}  ns={ns:2d} m={mm} L={ns//2:2d}  "
              f"E_cm={Ecm:.4f}  delta={dl:+.4f}")

print("\n=== fits:  delta = a + b(E-1.7) + atan2(Gamma/2, E_R-E)"
      " [+ (g0 + g1(E-1.7)) P^2] ===")


def bw_fit(E, dlt, P, label, nfr=0):
    """Mod-pi residual fit; nfr = number of frame parameters (0, 1, 2)."""
    E, dlt, P = map(np.asarray, (E, dlt, P))

    def resid(p):
        a, b, er, gam = p[:4]
        g0 = p[4] if nfr > 0 else 0.0
        g1 = p[5] if nfr > 1 else 0.0
        mod = a + b * (E - 1.7) + np.arctan2(gam / 2, er - E) \
            + (g0 + g1 * (E - 1.7)) * P ** 2
        return fold(mod - dlt)

    p0 = [1.2, -0.6, 2.13, 0.28, 0.3, 0.0][:4 + nfr]
    lo = [-10, -10, 1.9, 0.01, -5, -5][:4 + nfr]
    hi = [10, 10, 2.4, 1.0, 5, 5][:4 + nfr]
    fit = least_squares(resid, p0, bounds=(lo, hi))
    dof = len(E) - len(p0)
    s2 = float(fit.fun @ fit.fun) / dof
    cov = s2 * np.linalg.inv(fit.jac.T @ fit.jac)
    err = np.sqrt(np.diag(cov))
    a, b, er, gam = fit.x[:4]
    fr = "".join(f"  g{i}={fit.x[4+i]:+.3f}({err[4+i]*1e3:.0f})"
                 for i in range(nfr))
    print(f"{label}: N={len(E)}  E_R={er:.4f}({err[2]*1e4:.0f})  "
          f"Gamma={gam:.4f}({err[3]*1e4:.0f})  a={a:.3f}({err[0]*1e3:.0f}) "
          f"b={b:.3f}({err[1]*1e3:.0f}){fr}  "
          f"rms={np.sqrt(s2 * dof / len(E)):.4f}")
    return fit.x, err, np.sqrt(s2 * dof / len(E))


gD_raw = np.array([r[3] for r in gate_pts])
fitA = bw_fit(gE, gD_raw, np.zeros(len(gE)), "A (P=0, baseline)      ")
m1 = [r for r in kept if r["m"] == 1]
fitB = bw_fit(np.concatenate([gE, [r["Ecm"] for r in m1]]),
              np.concatenate([gD_raw, [r["delta"] for r in m1]]),
              np.concatenate([np.zeros(len(gE)), [r["P"] for r in m1]]),
              "B (P=0 + m=1, g0)      ", nfr=1)
allE = np.concatenate([gE, [r["Ecm"] for r in kept]])
allD = np.concatenate([gD_raw, [r["delta"] for r in kept]])
allP = np.concatenate([np.zeros(len(gE)), [r["P"] for r in kept]])
allNs = np.concatenate([np.array([r[0] for r in gate_pts], dtype=float),
                        [r["ns"] for r in kept]])
fitC = bw_fit(allE, allD, allP, "C (all sectors, g0+g1) ", nfr=2)
# naive universal-delta fit on everything, for the record
bw_fit(allE, allD, np.zeros(len(allE)), "naive (no frame term)  ")

# per-point table
print("\nper-point table (E = E_cm; delta folded to (-pi/2, pi/2]):")
meta = [(0.0, int(r[0]), r[1], r[2]) for r in gate_pts] \
    + [(r["P"], r["ns"], r["Elab"], r["a"]) for r in kept]
print(f"{'E_cm':>7} {'delta':>8} {'P':>7} {'L':>3} {'ns':>3} "
      f"{'E_lab':>8} {'n1':>3}")
for i in np.argsort(allE):
    Pm, nsm, elab, aa = meta[i]
    print(f"{allE[i]:7.4f} {allD[i]:+8.4f} {Pm:7.4f} {nsm // 2:3d} {nsm:3d} "
          f"{elab:8.4f} {aa:3d}")

np.savez("data/moving_frame_delta.npz",
         E=allE, delta=allD, P=allP, ns=allNs,
         E_lab=np.array([m[2] for m in meta]),
         n1=np.array([m[3] for m in meta]),
         fitA=fitA[0], fitA_err=fitA[1],
         fitB=fitB[0], fitB_err=fitB[1],
         fitC=fitC[0], fitC_err=fitC[1],
         notes="E=E_cm=2E(krel); delta folded to (-pi/2,pi/2]; fit params "
               "(a,b,E_R,Gamma[,g0,g1]); frame term (g0+g1(E-1.7))P^2 -- "
               "the moving-frame phase is frame-dependent, see script "
               "docstring; P=0 points are the trustworthy delta(E)")
print("\nsaved data/moving_frame_delta.npz")

print("""
VERDICT: the moving-frame levels are clean, exactly-identified MM states,
but their Bethe-Yang phase is frame-dependent: it matches the P=0 delta(E)
only in the P -> 0 limit (deviation ~ +0.3 P^2 rad, i.e. +0.04..+0.13 at
m=1, up to +1.3 rad at P=2.09, volume-stable at fixed P).  Moving frames
therefore do NOT provide model-independent extra delta(E) points here; the
P=0 result E_R = 2.128(10), Gamma = 0.277(12) stands, and the frame-
corrected fits B/C above quantify (not remove) the model dependence of any
moving-frame-augmented result.""")
