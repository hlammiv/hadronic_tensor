"""Elastic phase shifts from the two-meson levels extracted out of the
REAL-TIME hadronic-tensor correlators (scripts/rt_levels_ns{12,16,20}.py,
matrix-pencil poles of C11(t) = <M|J1(t)J1|M> at P = 0 and of
C00_q(t) = <M|J0_q(t)J0_{-q}|M> at P = 2pi/N_x), inverted with the SAME
finite-volume quantization condition the paper uses in Sec. VI.1
(eq:luscher, scripts/phase_shifts_v2.py):

    p N_x + 2 delta(p) = 2 pi n,     E2 = 2 E(p),

E(p) the clamped-spline single-meson dispersion of phase_shifts_v2.py, n
counted up from the free tower (n = 0 allowed), branch (-pi/2, pi/2].
The result is compared point by point with the ED-based curve stored in
data/phase_shifts_6vol.npz (the rows behind figures/phase_shift_collapse.pdf).

Moving-frame (P = 2pi/N_x) levels are converted with the 1+1d Bethe-Yang
condition of scripts/moving_frame_luscher.py (solve_k1 / fold / extract
copied verbatim, production dispersion substituted):
    E2 = E(k1) + E(P - k1),  k1 N_x + 2 delta = 2 pi n1,  E_cm = 2 E(k1 - P/2).
Those points are a supplement only: there is no stored production-coupling
moving-frame curve to compare with, and at CGK-A the moving-frame phase was
found to be frame-dependent (moving_frame_luscher.py docstring), so they are
compared with the ED levels of the same sector and with the P = 0 curve, and
NOT folded into the RMS.

Outputs: data/rt_levels_luscher.{npz,json,pdf}, data/rt_levels_report.md.
    PYTHONPATH=. python scripts/rt_levels_luscher.py
"""

import json
import os
import sys

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_style import OI, MARKERS  # noqa: E402

VOLS = (12, 16, 20)
WEIGHT_CUT = 1e-3          # keep pencil modes with weight > 1e-3 x max
                           # (the cut used in the rt_levels_ns* reports)

# ------------------------------------------------- dispersion (phase_shifts_v2)
k_ed = np.array([0.0, 1.2566, 1.5708, 2.0944, 2.5133, np.pi])
e_ed = np.array([2.7451, 2.8188, 2.8560, 2.9275, 2.9886, 3.0778])
E = CubicSpline(k_ed, e_ed, bc_type=((1, 0.0), (1, 0.0)))
M = e_ed[0]
E_INEL = 6.0304            # P=0 MM' threshold E1(0)+E2(0) (phase_shifts_v2)


def efold(k):
    """E(k) for any real k (even, 2pi-periodic) -- moving_frame_luscher.py."""
    k = np.abs((np.asarray(k, float) + np.pi) % (2 * np.pi) - np.pi)
    return E(k)


# ------------------------------------------- band-2 (M') dispersion from ED
# second band of deep_levels_ns20 per momentum (3.1 < gap < 3.4), plus the
# stored band-2 minimum MPRIME at k = pi (the band is inverted).
_d20 = np.load("data/deep_levels_ns20.npz")
_ph, _g = _d20["phases"], _d20["gaps"]
_b2k, _b2e = [], []
for mm in range(0, 5):
    P = 2 * np.pi * mm / 10
    sel = np.abs(np.angle(np.exp(1j * (_ph - P)))) < 1e-4
    lv = np.sort(_g[sel & (_g > 3.1) & (_g < 3.4)])
    _b2k.append(P); _b2e.append(float(lv[0]))
_b2k.append(np.pi); _b2e.append(float(np.load("data/phase_shifts_6vol.npz")["MPRIME"]))
E2b = CubicSpline(np.array(_b2k), np.array(_b2e), bc_type=((1, 0.0), (1, 0.0)))


def e2fold(k):
    k = np.abs((np.asarray(k, float) + np.pi) % (2 * np.pi) - np.pi)
    return E2b(k)


_KGRID = np.linspace(-np.pi, np.pi, 4001)


def mmprime_thr(P):
    """Lab-frame MM' threshold in the sector of total momentum P."""
    return float(np.min(efold(_KGRID) + e2fold(P - _KGRID)))


# ------------------------------------------------- delta machinery (copied)
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


def delta_p0(E2, L, n):
    """P = 0 inversion exactly as phase_shifts_v2.py."""
    p = brentq(lambda q: 2 * E(q) - E2, 1e-9, np.pi)
    delta = (2 * np.pi * n - p * L) / 2
    return fold(delta), p


# ------------------------------------------- real-time levels (pencil poles)
def rt_levels(ns):
    """(gaps, weights) of the P=0 (C11) and P=q (C00_q) pencil poles,
    weights relative to the largest, cut at WEIGHT_CUT."""
    d = np.load(f"data/rt_levels_ns{ns}.npz")
    if ns == 12:
        p0 = (d["C11_gap"], d["C11_weight"])
        pq = (d["C00_q_gap"], d["C00_q_weight"])
    elif ns == 16:
        gm = float(d["gap_meson_k0"])
        p0 = (d["C11_pencil_omega"] + gm, np.abs(d["C11_pencil_amp"]))
        pq = (d["C00_q_pencil_omega"] + gm, np.abs(d["C00_q_pencil_amp"]))
    elif ns == 20:
        gm = float(d["E_meson"]) - float(d["E_vac"])
        p0 = (d["C11_omega"] + gm, d["C11_weight"])
        pq = (d["C00_omega"] + gm, d["C00_weight"])
    out = {}
    for key, (g, w) in (("P0", p0), ("Pq", pq)):
        g, w = np.asarray(g, float), np.asarray(w, float)
        keep = w > WEIGHT_CUT * w.max()
        o = np.argsort(g[keep])
        out[key] = (g[keep][o], (w[keep] / w.max())[o])
    return out


def ed_sector(ns, P, refl_plus=False):
    d = np.load(f"data/deep_levels_ns{ns}.npz")
    g, ph = d["gaps"], d["phases"]
    sel = np.abs(np.angle(np.exp(1j * (ph - P)))) < 1e-4
    if refl_plus:
        rf = d["refl"] * d["refl"][0]
        sel &= rf > 0.99
    return np.sort(g[sel])


# ------------------------------------------------- stored ED phase shifts
stored = np.load("data/phase_shifts_6vol.npz")
rows_ed = stored["rows"]              # ns, E2, p, n, delta
# ERE fit exactly as phase_shift_figure.py (all 16 ED points)
X = np.vstack([rows_ed[:, 2], rows_ed[:, 2] ** 3]).T
coef, *_ = np.linalg.lstsq(X, np.pi / 2 - rows_ed[:, 4], rcond=None)
ere = lambda p: np.pi / 2 - coef[0] * p - coef[1] * p ** 3
resid_ere = (np.pi / 2 - X @ coef) - rows_ed[:, 4]
cov = np.mean(resid_ere ** 2) * np.linalg.inv(X.T @ X)
print(f"dispersion: M = {M:.4f}, window 2M = {2*M:.4f} .. {E_INEL:.4f}; "
      f"ERE from stored ED rows: a = {coef[0]:.4f}, r = {-coef[1]:.4f}")

# ================================================= P = 0 (C11) conversion
print("\n=== P = 0 levels from C11(t) -> delta(p) ===")
pts = []          # dicts per point
for ns in VOLS:
    L = ns // 2
    g, w = rt_levels(ns)["P0"]
    cands = [(gg, ww) for gg, ww in zip(g, w) if 2 * M - 0.06 < gg < E_INEL]
    ed_win = [e for e in ed_sector(ns, 0.0, refl_plus=True)
              if 2 * M - 0.06 < e < E_INEL]
    ed_rows = rows_ed[rows_ed[:, 0] == ns]
    print(f"ns={ns} (N_x={L}): RT window levels {np.round([c[0] for c in cands], 5)}"
          f"  ED (R=+1) {np.round(ed_win, 5)}")
    for j, (E2, wt) in enumerate(cands):
        if E2 <= 2 * M + 1e-9:
            print(f"   E2={E2:.5f} below threshold, skipped"); continue
        dl, p = delta_p0(E2, L, j)
        # matching stored ED row: same ns, nearest E2
        i = int(np.argmin(np.abs(ed_rows[:, 1] - E2)))
        ed = ed_rows[i]
        assert int(ed[3]) == j, f"n mismatch ns={ns} E2={E2}"
        dl_ed, p_ed = delta_p0(float(ed[1]), L, j)   # re-inverted here
        assert abs(dl_ed - ed[4]) < 1e-6 and abs(p_ed - ed[2]) < 1e-6
        pts.append(dict(ns=ns, P=0.0, n=j, E2_rt=float(E2), E2_ed=float(ed[1]),
                        dE=float(E2 - ed[1]), p_rt=float(p), p_ed=float(ed[2]),
                        delta_rt=float(dl), delta_ed=float(ed[4]),
                        ddelta=float(dl - ed[4]), weight_rel=float(wt),
                        dev_ere=float(dl - ere(p))))
        print(f"   n={j} E2_rt={E2:.6f} E2_ed={ed[1]:.6f} dE={E2-ed[1]:+.1e}"
              f"  p={p:.5f}  delta_rt={dl:+.5f} delta_ed={ed[4]:+.5f}"
              f"  diff={dl-ed[4]:+.2e}  (rel.weight {wt:.3g})")
    missed = [e for e in ed_win
              if not any(abs(e - c[0]) < 2e-3 for c in cands)]
    if missed:
        print(f"   *** ED R=+1 window levels not seen in C11: {missed}")

dd = np.array([q["ddelta"] for q in pts])
rms = float(np.sqrt(np.mean(dd ** 2)))
print(f"\nP=0: {len(pts)} points, RMS(delta_rt - delta_ed) = {rms:.2e} rad, "
      f"max |diff| = {np.abs(dd).max():.2e} rad")

# ================================================= P = 2pi/N_x (C00_q)
print("\n=== P = 2pi/N_x levels from C00_q(t) -> Bethe-Yang delta (supplement) ===")
mf = []
for ns in VOLS:
    L = ns // 2
    P = 2 * np.pi / L
    thr = mmprime_thr(P)
    g, w = rt_levels(ns)["Pq"]
    ed_all = ed_sector(ns, P)
    ed_win = [e for e in ed_all if 2 * M < e < E_INEL]
    cands = [(gg, ww) for gg, ww in zip(g, w) if 2 * M < gg < thr - 1e-3]
    print(f"ns={ns} P={P:.4f}: MM' threshold {thr:.4f}; RT window levels "
          f"{np.round([c[0] for c in cands], 5)}; ED sector levels in "
          f"(2M, {E_INEL}) {np.round(ed_win, 5)}")
    for j, (E2, wt) in enumerate(cands):
        a = 1 + j                    # n1 = ceil(m/2) + j, m = 1
        r = extract(E2, P, L, a)
        if r is None:
            print(f"   E2={E2:.5f} below the branch bottom, skipped"); continue
        dl, k1, Ecm = r
        e_ed = float(ed_all[np.argmin(np.abs(ed_all - E2))])
        dl_ed, _, Ecm_ed = extract(e_ed, P, L, a)
        p_rel = k1 - P / 2
        mf.append(dict(ns=ns, P=float(P), n1=a, E2_rt=float(E2), E2_ed=e_ed,
                       dE=float(E2 - e_ed), k1=float(k1), p_rel=float(p_rel),
                       Ecm=float(Ecm), delta_rt=float(dl), delta_ed=float(dl_ed),
                       ddelta=float(dl - dl_ed), weight_rel=float(wt),
                       dev_ere=float(dl - ere(p_rel))))
        print(f"   n1={a} E2_rt={E2:.6f} E2_ed={e_ed:.6f} dE={E2-e_ed:+.1e}"
              f"  k1={k1:.4f} p_rel={p_rel:.4f} E_cm={Ecm:.4f}"
              f"  delta={dl:+.4f} (ED-level {dl_ed:+.4f}, diff {dl-dl_ed:+.1e});"
              f"  vs P=0 ERE curve {dl-ere(p_rel):+.3f}")
    absent = [e for e in ed_win if not any(abs(e - c[0]) < 2e-3 for c in cands)]
    print(f"   ED sector levels in the window absent from C00_q (zero weight, "
          f"C-odd MM'): {np.round(absent, 5)}")

# ================================================= figure
fig, (ax, axr) = plt.subplots(2, 1, figsize=(3.3, 3.9), constrained_layout=True,
                              gridspec_kw=dict(height_ratios=[2.6, 1.0]),
                              sharex=True)
cmap = {ns: (OI[i], MARKERS[i]) for i, ns in enumerate(range(8, 21, 2))}
# ED curve: ERE fit + band + all 16 ED points (open markers)
pf = np.linspace(0, rows_ed[:, 2].max() * 1.03, 200)
J = np.vstack([pf, pf ** 3])
band = np.sqrt(np.einsum("ip,ij,jp->p", J, cov, J))
ax.fill_between(pf, ere(pf) - band, ere(pf) + band, color="0.35", alpha=0.18,
                lw=0, zorder=0)
ax.plot(pf, ere(pf), "-", color="0.35", lw=1.1, zorder=1, label="ED ERE fit")
for ns in sorted(set(rows_ed[:, 0].astype(int))):
    m = rows_ed[:, 0] == ns
    c, mk = cmap[ns]
    ax.plot(rows_ed[m, 2], rows_ed[m, 4], mk, mfc="none", mec=c, ms=6.5,
            mew=0.9, ls="none",
            label=f"ED $N_x={ns//2}$" if ns in VOLS else None, zorder=2)
# real-time points (filled, same marker/colour as their volume)
for ns in VOLS:
    c, mk = cmap[ns]
    sel = [q for q in pts if q["ns"] == ns]
    ax.plot([q["p_rt"] for q in sel], [q["delta_rt"] for q in sel], mk,
            color=c, ms=3.6, ls="none", label=f"$C_{{11}}$ $N_x={ns//2}$",
            zorder=3)
    selm = [q for q in mf if q["ns"] == ns]
    ax.plot([q["p_rel"] for q in selm], [q["delta_rt"] for q in selm], "x",
            color=c, ms=5, mew=1.0, ls="none", zorder=3,
            label=(r"$C_{00}$, $P=2\pi/N_x$" if ns == VOLS[0] else None))
ax.axhline(0, color="0.7", lw=0.6)
ax.axhline(np.pi / 2, color="0.7", lw=0.6, ls=":")
ax.text(0.05, np.pi / 2 + 0.05, r"$\pi/2$", fontsize=9, color="0.4")
ax.set_ylim(-0.75, 1.80)
ax.set_ylabel(r"elastic phase shift $\delta(p)$")
ax.legend(fontsize=5.6, ncol=2, loc="lower left", framealpha=0.9,
          columnspacing=0.6, handletextpad=0.3, labelspacing=0.35,
          borderpad=0.3, handlelength=1.2)
# residual panel: |delta_rt - delta_ed| per point (log)
for ns in VOLS:
    c, mk = cmap[ns]
    sel = [q for q in pts if q["ns"] == ns]
    axr.plot([q["p_rt"] for q in sel],
             [max(abs(q["ddelta"]), 1e-12) for q in sel], mk, color=c,
             ms=3.6, ls="none")
    selm = [q for q in mf if q["ns"] == ns]
    axr.plot([q["p_rel"] for q in selm],
             [max(abs(q["ddelta"]), 1e-12) for q in selm], "x", color=c,
             ms=5, mew=1.0, ls="none")
axr.set_yscale("log")
axr.set_ylim(1e-12, 1e-1)
axr.set_yticks([1e-11, 1e-8, 1e-5, 1e-2])
axr.axhline(rms, color="0.35", lw=0.8, ls="--")
axr.text(0.05, rms * 1.8, f"RMS $= {rms:.1e}$", fontsize=6.5, color="0.35")
axr.set_ylabel(r"$|\delta_{\rm rt}-\delta_{\rm ED}|$", fontsize=8)
axr.set_xlabel("relative momentum $p$")
fig.savefig("data/rt_levels_luscher.pdf", dpi=200)
print("wrote data/rt_levels_luscher.pdf")

# ================================================= outputs
np.savez("data/rt_levels_luscher.npz",
         p0=np.array([[q["ns"], q["E2_rt"], q["p_rt"], q["n"], q["delta_rt"],
                       q["E2_ed"], q["delta_ed"], q["ddelta"]] for q in pts]),
         p0_columns="ns,E2_rt,p,n,delta_rt,E2_ed,delta_ed,ddelta",
         mf=np.array([[q["ns"], q["P"], q["E2_rt"], q["k1"], q["p_rel"],
                       q["Ecm"], q["n1"], q["delta_rt"], q["E2_ed"],
                       q["delta_ed"], q["ddelta"], q["dev_ere"]] for q in mf]),
         mf_columns="ns,P,E2_rt,k1,p_rel,Ecm,n1,delta_rt,E2_ed,delta_ed,ddelta,dev_ere",
         rms_p0=rms, ere_coef=coef, M=M, E_INEL=E_INEL)
json.dump(dict(M=M, E_INEL=E_INEL, ere_a=float(coef[0]), ere_r=float(-coef[1]),
               rms_delta_diff_p0=rms, points_p0=pts, points_moving_frame=mf),
          open("data/rt_levels_luscher.json", "w"), indent=1)
print("wrote data/rt_levels_luscher.npz / .json")
