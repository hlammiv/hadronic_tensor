"""Hardware equal-time prep factors, per probe, from the mirror kappas.

kappa_v(t) = N(t) f_v(0) for a guard-passing probe and kappa_c(t) = N(t) f_c(0)
for the Z probe on the insertion site, so kappa_v / kappa_c measures the
prepared state's equal-time factor on probe v relative to the insertion
site's, free of the circuit damping.  Averaged over the early slices
(t <= T_EARLY), where the mirror's own late-time shortfall has not set in;
the J1 terms are weight-3 probes and get the weight law exp(+0.05 (3-1) t)
back.  Written as json for scripts/prep_factor_model.py --target, keyed by
card, probe kind and offset from the insertion.

  PYTHONPATH=. python scripts/prep_factor_targets.py [--slices DIR] [--out FILE]
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from htq_hw import analyze as hqa  # noqa: E402

CARDS = {"prod": "prod_k1.26_s0.75_ns50", "relA": "relA_k1.26_s0.75_ns50"}
CENTER = 24
T_EARLY = 2.0
ERR_FLOOR = 0.03
OFFSETS = range(-4, 5)


def tnum(f):
    m = re.search(r"_t([0-9.]+)(_dt[0-9.]+)?\.npz$", f)
    return float(m.group(1)), m.group(2)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--slices", default="data/hw/runs/phoenix/2026-10-06/analysis/2026-10-22_prepblind_kappa/slices")
    ap.add_argument("--out", default="data/work/prep_factor/targets.json")
    args = ap.parse_args()
    out = {}
    for card, sub in CARDS.items():
        g = np.load(f"htq_hw/cards/{sub}/ideal_j0.npz")
        ideal = {"J0": np.array([g[f"XB_{v}"][0] for v in range(50)]),
                 "T1": np.array([g[f"XT1_{b}"][0] for b in range(50)]),
                 "T2": np.array([g[f"XT2_{b}"][0] for b in range(50)])}
        rows = {"J0": [], "T1": [], "T2": []}
        for f in sorted(glob.glob(f"{args.slices}/{sub}/slice_00_t*.npz")):
            t, sfx = tnum(f)
            if sfx or t == 0 or t > T_EARLY:
                continue
            k00 = np.load(f)["kappa_v"][0]
            kc = k00[CENTER]
            rows["J0"].append(k00 / kc)
            f10 = f.replace("slice_00_", "slice_10_")
            if os.path.exists(f10):
                d = np.load(f10, allow_pickle=True)
                w = np.exp(hqa.WEIGHT_RATE * 2 * t)
                rows["T1"].append(d["kappa1_v"][0] / kc * w)
                rows["T2"].append(d["kappa2_v"][0] / kc * w)
        out[card] = {}
        for kind, lst in rows.items():
            R = np.array(lst)
            tab = {}
            for off in OFFSETS:
                v = CENTER + off
                if abs(ideal[kind][v]) <= hqa.KAP_GUARD or (kind == "J0" and off == 0):
                    continue      # median-filled on hardware, or the normalisation itself
                vals = R[:, v]
                tab[str(off)] = [float(vals.mean()), float(max(vals.std(ddof=1) / np.sqrt(len(vals)), ERR_FLOOR))]
            out[card][kind] = tab
            print(card, kind, {k: f"{m:.3f}+-{e:.3f}" for k, (m, e) in tab.items()})
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
