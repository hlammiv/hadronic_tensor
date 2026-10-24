"""Apply the prep-state model's time-dependent prep factors to the slices.

The mirror calibration divides by kappa = N(t) f(0): the circuit damping
times the prepared state's EQUAL-TIME factor on the probe (note sec. 14).
The right divisor for the dynamical signal is N(t) f(t), with f(t) the
prepared state's factor on the probe at time t, which
scripts/prep_factor_model.py predicts by propagating a noise-damaged
preparation through the ideal dynamics.  The residual bias of a calibrated
ancilla term is therefore f(t) / f_used(0), where f_used(0) is whatever the
pipeline divided by: the probe's own factor (guard-passing probes), the
median over the guard-passing probes (median fill), or the insertion
site's (prep-blind fill, `htq_hw.analyze.prep_blind_fill`).  This script
multiplies the ancilla term of every slice by the inverse of that bias,
summed correctly over the Pauli terms and insertion families a component
combines (weights: the Ns = 50 ideal term signals times the pipeline's
coefficients), on the probes the model covers: |offset from the insertion|
<= 4 (the trained block, where the Ns = 10 model ring is faithful to
Ns = 50; the two ideals are compared per probe and time and the correction
skipped where they differ).  All factors are ratios of CONNECTED ancilla
signals, XB_v - id_b <A>, the quantity the mirror calibrates.  Everything
else is copied.

A term is left uncorrected (ratio 1) where the model cannot say: ideal
connected signal below --sigmin, Monte Carlo error above --relmax of the
signal, or a bias outside [1/--maxcorr, --maxcorr].

Errors: the model's Monte Carlo error on f(t), plus half the difference
between the two noise models when both are given, as a relative error on
the corrected ancilla term, added in quadrature to C_err.

  PYTHONPATH=. python scripts/prepfactor_slices.py --card prod --model data/work/prep_factor/run_prod_gate.npz \
      --alt data/work/prep_factor/run_prod_post.npz --dst data/hw/runs/phoenix/2026-10-06/analysis/2026-10-23_prepfactor/slices
"""
import argparse
import glob
import json
import os
import re
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from htq_hw import analyze as hqa  # noqa: E402

CARDS = {"prod": "prod_k1.26_s0.75_ns50", "relA": "relA_k1.26_s0.75_ns50"}
SRC = "data/hw/runs/phoenix/2026-10-06/analysis/2026-10-22_prepblind_kappa/slices"
CENTER = 24
NS = 50
REACH = 4                     # block half-width: offsets the model covers
IDEAL_TOL = 0.15              # | |ideal_10| - |ideal_50| | / max_v |ideal_50| above which the ring is not faithful
GUARD = hqa.KAP_GUARD
# component -> (families, probe rows in the model's S array: 0 = J0, 1 = T1, 2 = T2)
LAYOUT = {"00": (("j0",), (0,)), "10": (("j0",), (1, 2)),
          "01": (("j1p1", "j1p2"), (0,)), "11": (("j1p1", "j1p2"), (1, 2))}
# pipeline coefficients: c_a per family (x eta for j1, htq_hw.analyze.FAMILY_COEFF) and
# +-eta/4 per J1 Pauli term (<X J1_b> = eta/4 (xT1/kappa1 - xT2/kappa2))
ROW_SIGN = {0: 1.0, 1: 1.0, 2: -1.0}


def tnum(f):
    m = re.search(r"_t([0-9.]+)(_dt[0-9.]+)?\.npz$", f)
    return float(m.group(1)), m.group(2)


def idb(n):
    """id_b of the J0 probe operator J0_v = id_b + c_b Z_v on sites 0..n-1 (c_b = -id_b)."""
    return 0.5 * (-1.0) ** np.arange(n)


class Model:
    """A prep_factor_model run; S(fam, noisy) -> (nt, 3, ns10) CONNECTED ancilla signals."""

    def __init__(self, path):
        d = np.load(path, allow_pickle=True)
        self.d, self.times, self.c10 = d, d["times"], int(d["center"])
        self.ns10 = int(d["ns"])
        self.p = float(d["ps"][0])
        self.fams = [k[len("ideal_S_"):] for k in d.files if k.startswith("ideal_S_") and not k.endswith("_err")]
        self._cache = {}

    def tindex(self, t):
        i = int(np.argmin(np.abs(self.times - t)))
        return i if abs(self.times[i] - t) < 1e-9 else None

    def insertion_onept(self, fam, noisy):
        """<A> in the prepared state for the family's Pauli insertion A (the
        Hadamard test's <X_anc>): from the t = 0 one-points at the insertion."""
        pre = f"p{self.p:g}" if noisy else "ideal"
        c = self.c10
        if fam == "j0":            # <Z_c> from <J0_c> = id_b + c_b <Z_c>, id_b = +1/2 at the even centre
            return 1.0 - self.d[f"{pre}_b"][0, c] / 0.5
        if fam == "j1p1":
            return float(self.d[f"{pre}_one1"][0, c])
        if fam == "j1p2":
            return float(self.d[f"{pre}_one2"][0, c])
        raise KeyError(fam)

    def S(self, fam, noisy: bool):
        key = (fam, noisy)
        if key not in self._cache:
            pre = f"p{self.p:g}" if noisy else "ideal"
            S = np.array(self.d[f"{pre}_S_{fam}"], dtype=float)
            S[:, 0, :] -= idb(self.ns10)[None, :] * self.insertion_onept(fam, noisy)
            self._cache[key] = S
        return self._cache[key]

    def Serr(self, fam):
        return self.d[f"p{self.p:g}_S_{fam}_err"]


def factor_table(model: Model, comp: str, t: float, kap_rows: list, ideal50: dict, cuts: dict):
    """-> (factor (50,), relerr (50,), covered (50,) bool, skipped: list of reasons) for one slice.

    kap_rows: the slice's kappa per probe row (one array for J0 slices, two
    for J1 slices: kappa1_v, kappa2_v).  ideal50: {fam: (rows, 50)
    connected ideal term signals at t from the Ns = 50 grid}, the weights
    and the faithfulness reference."""
    fams, rows = LAYOUT[comp]
    it, i0 = model.tindex(t), 0
    factor, relerr, covered = np.ones(NS), np.zeros(NS), np.zeros(NS, bool)
    skipped = []
    if it is None or any(fam not in model.fams for fam in fams) or any(ideal50.get(fam) is None for fam in fams):
        return factor, relerr, covered, ["no model time or family"]
    num, den, var = np.zeros(NS), np.zeros(NS), np.zeros(NS)
    for fam in fams:
        Sn, Si, Se = model.S(fam, True), model.S(fam, False), model.Serr(fam)
        g50 = ideal50[fam]
        cfam = hqa.FAMILY_COEFF[fam] * (2.3 if fam.startswith("j1") else 1.0)   # eta cancels in the ratio anyway
        fc0 = Sn[i0, 0, model.c10] / Si[i0, 0, model.c10]
        for ri, row in enumerate(rows):
            ok0 = np.abs(Si[i0, row]) > GUARD
            f0 = np.where(ok0, Sn[i0, row] / np.where(ok0, Si[i0, row], 1.0), np.nan)
            fmed = np.nanmedian(f0) if np.isfinite(f0).any() else fc0
            kap = kap_rows[min(ri, len(kap_rows) - 1)]
            kmed = np.median(kap)
            scale = max(np.abs(g50[ri]).max(), 1e-9)
            for off in range(-REACH, REACH + 1):
                v, u = CENTER + off, model.c10 + off
                if not (0 <= u < model.ns10):
                    continue
                w = cfam * ROW_SIGN[row] * g50[ri, v]
                num[v] += w
                si_t, sn_t = Si[it, row, u], Sn[it, row, u]
                r, reason = 1.0, None
                # what the pipeline divided by, in units of N(t)
                prep_blind = (row in (1, 2) and comp in ("10", "11")
                              and off in ((-1, 0, 1) if comp == "11" else (-1, 0)))
                if prep_blind:
                    f_used = fc0
                elif abs(kap[v] - kmed) < 1e-9 and not ok0[u]:
                    f_used = fmed
                elif ok0[u]:
                    f_used = f0[u]
                else:
                    f_used = fmed
                if abs(abs(si_t) - abs(g50[ri, v])) > IDEAL_TOL * scale:
                    reason = "ring not faithful"
                elif abs(si_t) < cuts["sigmin"]:
                    reason = "ideal below sigmin"
                elif Se[it, row, u] > cuts["relmax"] * abs(si_t):
                    reason = "MC error"
                else:
                    r = (sn_t / si_t) / f_used
                    if not (1.0 / cuts["maxcorr"] <= r <= cuts["maxcorr"]):
                        reason, r = f"bias {r:.2f} outside range", 1.0
                if reason:
                    skipped.append(f"{fam} row{row} v{v}: {reason}")
                else:
                    covered[v] = True
                    var[v] += (w * Se[it, row, u] / abs(si_t) / f_used) ** 2
                den[v] += w * r
    sel = covered & (np.abs(den) > 1e-12) & (np.abs(num) > 1e-12)
    factor[sel] = num[sel] / den[sel]
    relerr[sel] = np.sqrt(var[sel]) / np.abs(den[sel])
    covered &= sel
    return factor, relerr, covered, skipped


def ideal_rows(grids, fam, comp, t):
    """(rows, 50) connected ideal term signals: XB_v - id_b X for J0 probes, XT1/XT2 for J1 terms."""
    g = grids[fam]
    it = int(np.argmin(np.abs(g["times"] - t)))
    if abs(g["times"][it] - t) > 1e-9:
        return None
    rows = LAYOUT[comp][1]
    out = []
    for r in rows:
        if r == 0:
            out.append(np.array([g[f"XB_{v}"][it] for v in range(NS)]) - idb(NS) * float(g["X"][it]))
        else:
            out.append(np.array([g[f"XT{r}_{v}"][it] for v in range(NS)]))
    return np.array(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--card", choices=sorted(CARDS), required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--alt", default=None, help="second noise model, for the model-dependence error")
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--sigmin", type=float, default=0.05)
    ap.add_argument("--relmax", type=float, default=0.35)
    ap.add_argument("--maxcorr", type=float, default=3.0)
    args = ap.parse_args()
    cuts = dict(sigmin=args.sigmin, relmax=args.relmax, maxcorr=args.maxcorr)
    sub = CARDS[args.card]
    model = Model(args.model)
    alt = Model(args.alt) if args.alt else None
    grids = {fam: np.load(f"htq_hw/cards/{sub}/ideal_{fam}.npz") for fam in ("j0", "j1p1", "j1p2")}
    out = f"{args.dst}/{sub}"
    os.makedirs(out, exist_ok=True)
    log = {"settings": dict(model=args.model, alt=args.alt, src=args.src, **cuts), "slices": {}}
    for f in sorted(glob.glob(f"{args.src}/{sub}/slice_*.npz")):
        shutil.copy2(f, out)
    for f in sorted(glob.glob(f"{args.src}/{sub}/slice_*.npz")):
        base = os.path.basename(f)
        comp = base.split("_")[1]
        t, sfx = tnum(f)
        if comp not in LAYOUT or sfx or t <= 0:
            continue
        d = dict(np.load(f, allow_pickle=True))
        fams, rows = LAYOUT[comp]
        if rows == (0,):
            kap_rows = [d["kappa_v"][0]]
            A0 = float(grids[fams[0]]["insert_1pt"])
            id_a, id_b, b = float(d["id_a"]), d["id_b"], d["b_cal"][0].real
            base_terms = id_a * (b - id_b) + id_b * (A0 - id_a) + id_a * id_b
        else:
            kap_rows = [d["kappa1_v"][0], d["kappa2_v"][0]]
            base_terms = float(d["id_a"]) * d["one_pt"][0]
        anc = d["C_cal"][0].real - base_terms
        ideal50 = {fam: ideal_rows(grids, fam, comp, t) for fam in fams}
        fac, rel, cov, skipped = factor_table(model, comp, t, kap_rows, ideal50, cuts)
        if alt is not None:
            fac2, _, cov2, _ = factor_table(alt, comp, t, kap_rows, ideal50, cuts)
            both = cov & cov2
            rel = np.sqrt(rel ** 2 + np.where(both, (0.5 * (fac - fac2) / np.where(fac != 0, fac, 1.0)) ** 2, 0.0))
        if not cov.any():
            log["slices"][base] = {"factor": {}, "skipped": skipped}
            continue
        anc_new = anc * fac
        C_new = d["C_cal"][0].real + (anc_new - anc)
        err = d["C_err"][0].copy()
        err[cov] = np.hypot(err[cov], np.abs(anc_new[cov]) * rel[cov])
        d["C_cal"] = (C_new + 0j)[None, :]
        d["C_err"] = err[None, :]
        d["prep_factor"] = np.array(f"prep-state model {os.path.basename(args.model)} (p={model.p:g}): ancilla term "
                                    f"multiplied by f_used(0)/f(t) on probes {np.where(cov)[0].tolist()}"
                                    + (f"; model-dependence error from {os.path.basename(args.alt)}" if alt else ""))
        np.savez(f"{out}/{base}", **d)
        log["slices"][base] = {"factor": {int(v): round(float(fac[v]), 3) for v in np.where(cov)[0]},
                               "relerr": {int(v): round(float(rel[v]), 3) for v in np.where(cov)[0]},
                               "anc_old": {int(v): round(float(anc[v]), 4) for v in np.where(cov)[0]},
                               "skipped": skipped}
    json.dump(log, open(f"{args.dst}/prepfactor_log_{args.card}.json", "w"), indent=1)
    n = sum(1 for s in log["slices"].values() if s["factor"])
    print(args.card, n, "slices corrected under", out)
    for k in sorted(log["slices"]):
        s = log["slices"][k]
        print(f"  {k:22s}", s["factor"], f"({len(s['skipped'])} terms skipped)" if s["skipped"] else "")


if __name__ == "__main__":
    main()
