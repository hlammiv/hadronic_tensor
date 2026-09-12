"""Pool the forward-campaign kingston bits into the tier-3 C(t,x) slices
(t=0.5, 1.0) and apply the wing-anchored b-sector correction everywhere
(t=0.5, 1.0, 2.0).

Estimator: tier-3 conventions replicated exactly (J0(v) = id_b(v) I -
Z_v/2, sx = <X_anc (x) J0> - id_b <X_anc>, per-site kappa_v =
sx_m/sx_ideal0, per-site beta 0.5-anchor).

b-sector residual: pooled statistics exposed a parity-structured bias of
the beta model (even-site wings +0.07-0.08 at t=1.0, identical in both
sessions and readout schemes -> a state-dependent damping asymmetry
between mirror and physics, not readout).  The wings are spectator
vacuum with known truth, so the residual is measured there per parity
and subtracted (wing-anchored correction, blind to the packet); its
wing scatter enters the error in quadrature.  Same philosophy as the
dM/dm0 wing floor.

Merge: calibrated Re slices inverse-variance combined across sessions
(chi-square inflated).  Im stays tier-3-only (W00 assembly and fig 14
use Re).  Tier-3 Re error approximated as cerr/sqrt(2).

  PYTHONPATH=. .venv/bin/python scripts/losch_t3_pool.py
"""
import json
import sys
import numpy as np

sys.path.insert(0, "scripts")
from htensor import Z2Lattice
from htensor.measure import split_current
from htensor import currents as cur

NS, NSYS, C = 50, 100, 24
KAP_GUARD = 0.02
WING = np.abs(np.arange(NS) - C) >= 6

lat = Z2Lattice(NS, pbc=True)
id_b = np.array([split_current(cur.charge_density(lat, v))[0]
                 for v in range(NS)]).real
CZ = -0.5
meta = json.load(open("data/hw/job_tier3a_d9d3emkinv1c73aoguc0.json"))
cm = meta["circ_meta"][0]
id_a, c_a = float(cm["id_c"]), float(cm["ins_coeff"])

idl = np.load("data/hw_t3_ideal50_k1.26.npz")
it0 = int(np.argmin(np.abs(idl["times"])))
sx_i0 = np.array([idl[f"XB_{v}"][it0] for v in range(NS)]) \
    - id_b * idl["X"][it0]
b_i0 = np.array([idl[f"B_{v}"][it0] for v in range(NS)])
A0 = b_i0[C]


def b_ideal(t):
    row = int(np.argmin(np.abs(idl["times"] - t)))
    return np.array([idl[f"B_{v}"][row] for v in range(NS)])


def wing_anchor(b_cal, bit):
    """Per-parity wing-residual subtraction + its scatter as systematic."""
    out, syst = b_cal.copy(), np.zeros(NS)
    for par in (0, 1):
        s = WING & (np.arange(NS) % 2 == par)
        allp = np.arange(NS) % 2 == par
        delta = np.mean(b_cal[s] - bit[s])
        out[allp] -= delta
        syst[allp] = np.std(b_cal[s] - bit[s]) / np.sqrt(s.sum())
    return out, syst


d1 = np.load("data/hw/losch_bits_d9lsndnbupns73e7uobg.npz")
d2 = np.load("data/hw/losch_bits_d9not7csfqic73ara4m0.npz")
d3p = np.load("data/hw/losch_bits_d9npki460llc73cacsig.npz")  # pass 3 CZ
d5p = np.load("data/hw/losch_bits_d9nq7as60llc73cadh2g.npz")  # pass 5 rzz
d7p = np.load("data/hw/losch_bits_d9nqeq8qs0bc73e3od9g.npz")  # pass 7 rzz
d8p = np.load("data/hw/losch_bits_d9nqtfeij12s73fu7fbg.npz")  # pass 8 rzz
d9p = np.load("data/hw/losch_bits_d9nra4ssfqic73arckc0.npz")  # pass 9 rzz
d10p = np.load("data/hw/losch_bits_d9nrg0mij12s73fu81fg.npz")  # pass 10


def pooled_obs(key, files=(d1, d2)):
    b = np.concatenate([d[key][:, ::-1] for d in files
                        if key in d.files])
    z = 1.0 - 2.0 * b[:, 0:NSYS:2].astype(np.float64)
    xa = 1.0 - 2.0 * b[:, NSYS].astype(np.float64)
    N = len(b)
    Bshot = id_b[None, :] + CZ * z
    XBshot = xa[:, None] * Bshot
    sx = XBshot.mean(0) - id_b * xa.mean()
    sxe = (XBshot - id_b[None, :] * xa[:, None]).std(0) / np.sqrt(N)
    return dict(sx=sx, sxe=sxe, B=Bshot.mean(0),
                Be=Bshot.std(0) / np.sqrt(N), N=N)


def rebuild_C(anc, b_cal):
    return anc + id_a * (b_cal - id_b) + id_b * (A0 - id_a) + id_a * id_b


TIER3 = {0.5: "data/hw/job_tier3a_d9d3emkinv1c73aoguc0.npz",
         1.0: "data/hw/job_tier3b_d9d45acinv1c73aohsj0.npz",
         1.5: None,                                  # forward-campaign only
         2.0: "data/hw/job_tier3d_d9d518sjeosc73fh2prg.npz",
         # rzz (pass 5) single-session slices; 2.0 exists in BOTH forms:
         # tier3d (CZ, 47/50 masked) -> BFIX kept as cross-check, and the
         # fractional 2.0/3.0 slices written by the rzz loop below
         }
OUT = {0.5: "data/hw/job_tier3a_POOLED.npz",
       1.0: "data/hw/job_tier3b_POOLED.npz",
       1.5: "data/hw/job_tier3e_T15.npz",
       2.0: "data/hw/job_tier3d_BFIX.npz"}
RZZ_OUT = {0.5: "data/hw/job_tier3j_T05RZZ.npz",
           1.0: "data/hw/job_tier3i_T10RZZ.npz",
           1.5: "data/hw/job_tier3h_T15RZZ.npz",
           2.0: "data/hw/job_tier3f_T20RZZ.npz",
           3.0: "data/hw/job_tier3g_T30RZZ.npz"}
RZZ_FILES = {0.5: (d10p,), 1.0: (d9p,), 1.5: (d7p,), 2.0: (d5p, d8p),
             3.0: (d5p,)}  # t2.0: passes
# 5+8 pooled (kappa_c drift 0.9%, slice chi2/dof 1.65 across sessions)
NEWJOBS = ["d9lsndnbupns73e7uobg", "d9not7csfqic73ara4m0"]
T15JOBS = ["d9npki460llc73cacsig"]
RZZJOBS = ["d9nq7as60llc73cadh2g"]
tru = np.load("data/w_meson_ns50_k1.26_v3.npz")


def new_session_slice(t, files):
    """Single-session calibrated Re slice from the forward-campaign bits."""
    bit = b_ideal(t)
    p = pooled_obs(f"t{t:.1f}", files)
    m = pooled_obs(f"m{t:.1f}", files)
    kap = np.where(np.abs(sx_i0) > KAP_GUARD, m["sx"] / sx_i0, np.nan)
    kap = np.where(np.isnan(kap), np.nanmedian(kap), kap)
    bet = np.where(np.abs(b_i0 - 0.5) > KAP_GUARD,
                   (m["B"] - 0.5) / (b_i0 - 0.5), 1.0)
    b_cal = 0.5 + (p["B"] - 0.5) / bet
    b_calf, sN = wing_anchor(b_cal, bit)
    kerr = np.where(np.abs(sx_i0) > KAP_GUARD, m["sxe"] / np.abs(sx_i0),
                    np.median(m["sxe"]) / KAP_GUARD) / np.abs(kap)
    anc_new = c_a * p["sx"] / kap
    cerr = np.sqrt((c_a / kap) ** 2 * p["sxe"] ** 2
                   + (c_a / kap) ** 2 * p["sx"] ** 2 * kerr ** 2
                   + (id_a * p["Be"] / np.abs(bet)) ** 2
                   + (id_a * sN) ** 2)
    C_cal = rebuild_C(anc_new, b_calf)
    C_raw = rebuild_C(c_a * p["sx"], p["B"])
    return dict(C_cal=C_cal, C_raw=C_raw, C_err=cerr, kap=kap, bet=bet,
                b_cal=b_calf, N=p["N"])

for t, path in TIER3.items():
    if path is None:                       # t=1.5: forward-campaign only
        s = new_session_slice(t, (d3p,))
        ti = int(np.argmin(np.abs(tru["times"] - t)))
        Ctru = tru["corr_wp"][ti, :NS].real
        okw = WING & (s["kap"] > 0.05)
        print(f"t={t}: single-session slice, kappa_c {s['kap'][C]:.3f}, "
              f"masked {(np.abs(s['kap']) <= 0.05).sum()}/50, "
              f"{s['N']} shots"
              + (f", wing offset {np.mean(s['C_cal'][okw]-Ctru[okw]):+.4f}"
                 if okw.sum() else ""))
        np.savez(OUT[t], times=np.array([t]),
                 probes=np.arange(NS),
                 C=(s["C_raw"] + 0j)[None, :],
                 C_cal=(s["C_cal"] + 0j)[None, :],
                 C_err=s["C_err"][None, :],
                 kappa_v=s["kap"][None, :], beta_v=s["bet"][None, :],
                 b_cal=s["b_cal"][None, :], id_a=id_a, c_a=c_a,
                 id_b=id_b, tier=3, backend="ibm_kingston",
                 nshot=s["N"], merged_jobs=np.array(T15JOBS))
        print(f"   -> {OUT[t]}")
        continue
    bit = b_ideal(t)
    d3 = np.load(path)
    C3 = d3["C_cal"][0]
    e3 = d3["C_err"][0]
    e3r = e3 / np.sqrt(2)
    kap3, bet3 = d3["kappa_v"][0], d3["beta_v"][0]
    b3 = d3["b_cal"][0]
    anc3 = C3 - (id_a * (b3 - id_b) + id_b * (A0 - id_a) + id_a * id_b)
    b3f, s3 = wing_anchor(b3, bit)
    C3f = rebuild_C(anc3.real, b3f) + 1j * anc3.imag
    e3f = np.sqrt(e3 ** 2 + (id_a * s3) ** 2)
    e3fr = np.sqrt(e3r ** 2 + (id_a * s3) ** 2)

    if t in (0.5, 1.0):
        p = pooled_obs(f"t{t:.1f}")
        m = pooled_obs(f"m{t:.1f}")
        kap = np.where(np.abs(sx_i0) > KAP_GUARD, m["sx"] / sx_i0, np.nan)
        kap = np.where(np.isnan(kap), np.nanmedian(kap), kap)
        bet = np.where(np.abs(b_i0 - 0.5) > KAP_GUARD,
                       (m["B"] - 0.5) / (b_i0 - 0.5), 1.0)
        b_cal = 0.5 + (p["B"] - 0.5) / bet
        b_calf, sN = wing_anchor(b_cal, bit)
        kerr = np.where(np.abs(sx_i0) > KAP_GUARD,
                        m["sxe"] / np.abs(sx_i0),
                        np.median(m["sxe"]) / KAP_GUARD) / np.abs(kap)
        anc_new = c_a * p["sx"] / kap
        cerr = np.sqrt((c_a / kap) ** 2 * p["sxe"] ** 2
                       + (c_a / kap) ** 2 * p["sx"] ** 2 * kerr ** 2
                       + (id_a * p["Be"] / np.abs(bet)) ** 2
                       + (id_a * sN) ** 2)
        C_new = rebuild_C(anc_new, b_calf)

        both = (kap > 0.05) & (kap3 > 0.05)
        dev = (C_new[both] - C3f[both].real) / np.sqrt(cerr[both] ** 2
                                                       + e3fr[both] ** 2)
        chi2 = float(np.mean(dev ** 2))
        if chi2 > 4.0:
            raise SystemExit(f"VALIDATION FAILED t={t}: chi2 {chi2:.2f}")

        w_new, w_t3 = 1.0 / cerr ** 2, 1.0 / e3fr ** 2
        wsum = w_new + w_t3
        Re_m = (w_new * C_new + w_t3 * C3f.real) / wsum
        e_m = 1.0 / np.sqrt(wsum)
        chi = (w_new * (C_new - Re_m) ** 2 + w_t3 * (C3f.real - Re_m) ** 2)
        e_m = e_m * np.sqrt(np.maximum(chi, 1.0))
        C_out = Re_m + 1j * C3f.imag
        e_out = np.sqrt(e_m ** 2 + e3fr ** 2)
        wb = 1.0 / np.maximum(p["Be"] ** 2 + sN ** 2, 1e-12)
        wb3 = 1.0 / np.maximum((e3 / (2 * id_a)) ** 2 + s3 ** 2, 1e-12)
        b_out = (wb * b_calf + wb3 * b3f) / (wb + wb3)
        kap_out = 0.5 * (kap + kap3)
        bet_out = 0.5 * (bet + bet3)
        nshot = int(d3["nshot"]) + 2 * 70_000
        jobs = list(d3["merged_jobs"]) + NEWJOBS
    else:
        chi2 = np.nan
        C_out, e_out, b_out = C3f, e3f, b3f
        kap_out, bet_out = kap3, bet3
        nshot, jobs = int(d3["nshot"]), list(d3["merged_jobs"])

    ti = int(np.argmin(np.abs(tru["times"] - t)))
    Ctru = tru["corr_wp"][ti, :NS].real
    okw = WING & (kap_out > 0.05)
    print(f"t={t}: chi2/dof {chi2:.2f}  final wing offset vs truth "
          f"{np.mean(C_out.real[okw] - Ctru[okw]):+.4f}  "
          f"(was {np.mean(C3.real[okw] - Ctru[okw]):+.4f})")

    out = {k: d3[k] for k in d3.files}
    out["C_cal"] = C_out[None, :]
    out["C_err"] = e_out[None, :]
    out["b_cal"] = b_out[None, :]
    out["kappa_v"] = kap_out[None, :]
    out["beta_v"] = bet_out[None, :]
    out["nshot"] = nshot
    out["merged_jobs"] = np.array(jobs)
    np.savez(OUT[t], **out)
    print(f"   -> {OUT[t]}")

# ---- fractional (rzz, untwirled) single-session slices ----
for t, outp in RZZ_OUT.items():
    s = new_session_slice(t, RZZ_FILES[t])
    ti = int(np.argmin(np.abs(tru["times"] - t)))
    Ctru = tru["corr_wp"][ti, :NS].real
    ok = s["kap"] > 0.05
    okw = WING & ok
    print(f"t={t} rzz: kappa_c {s['kap'][C]:.3f}, masked "
          f"{int((~ok).sum())}/50, wing offset vs truth "
          f"{np.mean(s['C_cal'][okw]-Ctru[okw]):+.4f}, "
          f"median err {np.median(s['C_err'][ok]):.4f}")
    if t == 2.0:
        dref = np.load(OUT[2.0])
        both = ok & (dref["kappa_v"][0] > 0.05)
        if both.sum():
            dev = (s["C_cal"][both] - dref["C_cal"][0].real[both]) \
                / np.sqrt(s["C_err"][both]**2 + dref["C_err"][0][both]**2/2)
            print(f"   vs CZ tier3d on {both.sum()} common sites: "
                  f"chi2/dof {np.mean(dev**2):.2f}")
    np.savez(outp, times=np.array([t]), probes=np.arange(NS),
             C=(s["C_raw"] + 0j)[None, :],
             C_cal=(s["C_cal"] + 0j)[None, :],
             C_err=s["C_err"][None, :],
             kappa_v=s["kap"][None, :], beta_v=s["bet"][None, :],
             b_cal=s["b_cal"][None, :], id_a=id_a, c_a=c_a,
             id_b=id_b, tier=3, backend="ibm_kingston",
             nshot=s["N"],
             merged_jobs=np.array(
                 {0.5: ["d9nrg0mij12s73fu81fg"],
                  1.0: ["d9nra4ssfqic73arckc0"],
                  1.5: ["d9nqeq8qs0bc73e3od9g"]}.get(t, RZZJOBS)))
    print(f"   -> {outp}")

# ---- final t=0.5: merge the fractional slice with the CZ pooled one ----
a = np.load("data/hw/job_tier3a_POOLED.npz")
j = np.load("data/hw/job_tier3j_T05RZZ.npz")
okm = (a["kappa_v"][0] > 0.05) & (j["kappa_v"][0] > 0.05)
dev = (j["C_cal"][0].real[okm] - a["C_cal"][0].real[okm]) / \
    np.sqrt(j["C_err"][0][okm]**2 + a["C_err"][0][okm]**2 / 2)
chi2 = float(np.mean(dev**2))
print(f"t=0.5 rzz-vs-CZ: chi2/dof {chi2:.2f} over {okm.sum()} sites")
assert chi2 < 4.0, "t=0.5 merge gate failed"
eA = a["C_err"][0] / np.sqrt(2)
wA, wJ = 1.0 / eA**2, 1.0 / j["C_err"][0]**2
Re_m = (wA * a["C_cal"][0].real + wJ * j["C_cal"][0].real) / (wA + wJ)
e_m = 1.0 / np.sqrt(wA + wJ)
chi = wA * (a["C_cal"][0].real - Re_m)**2 + wJ * (j["C_cal"][0].real - Re_m)**2
e_m = e_m * np.sqrt(np.maximum(chi, 1.0))
out = {k: a[k] for k in a.files}
out["C_cal"] = (Re_m + 1j * a["C_cal"][0].imag)[None, :]
out["C_err"] = np.sqrt(e_m**2 + eA**2)[None, :]
out["kappa_v"] = (0.5 * (a["kappa_v"][0] + j["kappa_v"][0]))[None, :]
out["nshot"] = int(a["nshot"]) + int(j["nshot"])
out["merged_jobs"] = np.array(list(a["merged_jobs"])
                              + ["d9nrg0mij12s73fu81fg"])
np.savez("data/hw/job_tier3k_T05MERGED.npz", **out)
print(f"t=0.5 MERGED: median err {np.median(out['C_err'][0]):.4f} "
      f"(CZ was {np.median(a['C_err'][0]):.4f}), nshot {out['nshot']}")
print("   -> data/hw/job_tier3k_T05MERGED.npz")
