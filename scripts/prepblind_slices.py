"""Apply `htq_hw.analyze.prep_blind_fill` to slices that were reduced before
it existed (campaign note of 2026-10-22, sec. 14).

The per-shot bits of the phoenix run are not local, so `analyze` cannot be
re-run; this rewrites the slice files instead.  On the J1 probes that share a
qubit with the insertion (j0: bonds 23, 24; j0d: 24, 25; j1p*: 23, 24, 25) the
median-filled kappa is replaced by the 00 component's kappa at the insertion
site at the same t, which is the one prep-blind anchor the mirror has (it is
the merged j0 Z-mirror kappa, what `prep_blind_fill` borrows across jobs), and
the calibrated signal is rebuilt from the stored raw one: C_cal = raw / kappa +
id_a <J1>.  The error on those bonds is the shot part (the median of
C_err x kappa over the other unmasked bonds, divided by the new kappa) plus a
10% model term.  Every other slice is copied unchanged, so the output directory
is a complete slice set for `scripts/hw_w_tensor.py --slices`.

The probe-weight law exp(-WEIGHT_RATE (3 - 1) t) of `prep_blind_fill` is
applied, so the adjusted bonds match what a fresh reduction would give.

  PYTHONPATH=. python scripts/prepblind_slices.py
"""
import glob
import json
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from htq_hw import analyze as hqa  # noqa: E402
SRC = "data/hw/runs/phoenix/2026-10-06/analysis"
DST = f"{SRC}/2026-10-22_prepblind_kappa/slices"
CARDS = {"prod": "prod_k1.26_s0.75_ns50", "relA": "relA_k1.26_s0.75_ns50"}
CENTER = 24
# component -> (J1 bonds whose kappa is replaced).  j0 insertion at c: bonds c-1, c.
# j0d insertion at c+1: bonds c, c+1.  J1 insertion on bond c: bonds c-1, c, c+1.
BONDS = {"10": (CENTER - 1, CENTER), "10d": (CENTER, CENTER + 1), "11": (CENTER - 1, CENTER, CENTER + 1)}
DELTA_MODEL = hqa.PREP_BLIND_MODEL_ERR      # relative model error on the transferred kappa
log = {}
for card, sub in CARDS.items():
    out = f"{DST}/{sub}"; os.makedirs(out, exist_ok=True)
    for f in sorted(glob.glob(f"{SRC}/{sub}/slice_*.npz")):
        shutil.copy2(f, out)
    for comp, bonds in BONDS.items():
        for f in sorted(glob.glob(f"{SRC}/{sub}/slice_{comp}_t*.npz")):
            base = os.path.basename(f); tag = base[len(f"slice_{comp}_"):-4]      # 't0.5' or 't0.5_dt0.25'
            d = dict(np.load(f, allow_pickle=True))
            t = float(d["t"])
            if t <= 0:
                continue
            f00 = f"{SRC}/{sub}/slice_00_{tag}.npz"
            if not os.path.exists(f00):
                print("  no 00 anchor for", base); continue
            k_anchor = float(np.load(f00)["kappa_v"][0][CENTER]) * float(np.exp(-hqa.WEIGHT_RATE * 2 * t))
            C, Cc, one, id_a = d["C"][0], d["C_cal"][0], d["one_pt"][0], float(d["id_a"])
            k1, k2, err = d["kappa1_v"][0].copy(), d["kappa2_v"][0].copy(), d["C_err"][0].copy()
            anc_raw = C - id_a * one
            other = np.ones(len(k1), bool); other[list(bonds)] = False; other &= d["mask"][0]
            pool = err[other] * np.minimum(k1, k2)[other]
            pool = pool[np.isfinite(pool) & (pool > 0)]
            if len(pool) == 0:
                pool = (err * np.minimum(k1, k2)); pool = pool[np.isfinite(pool) & (pool > 0)]
            shot = float(np.median(pool))     # ~ A*sigma, the per-bond shot error before 1/kappa
            Cc_new = Cc.copy()
            for b in bonds:
                anc_cal = anc_raw[b] / k_anchor
                Cc_new[b] = anc_cal + id_a * one[b]
                err[b] = np.hypot(shot / k_anchor, abs(anc_cal) * DELTA_MODEL)
                k1[b] = k2[b] = k_anchor
            d["C_cal"] = Cc_new[None, :]; d["C_err"] = err[None, :]
            d["kappa1_v"] = k1[None, :]; d["kappa2_v"] = k2[None, :]; d["kappa_v"] = np.minimum(k1, k2)[None, :]
            d["kappa_fill"] = np.array(f"prep-blind (post-processed): bonds {list(bonds)} take the 00 kappa at site {CENTER} "
                                       f"times exp(-{hqa.WEIGHT_RATE} * 2 * t) = {k_anchor:.3f} at the same t; "
                                       f"model error {DELTA_MODEL:.2f}; shot {shot:.4f}")
            np.savez(f"{out}/{base}", **d)
            log[f"{card}/{base}"] = dict(k_anchor=k_anchor, old=[float(np.load(f)["C_cal"][0][b].real) for b in bonds],
                                         new=[float(Cc_new[b].real) for b in bonds], old_kappa=[float(np.load(f)["kappa1_v"][0][b]) for b in bonds])
json.dump(log, open(f"{DST}/../prepblind_log.json", "w"), indent=1)
for k in ("prod/slice_10_t0.5.npz", "prod/slice_10_t1.0.npz", "relA/slice_10_t0.5.npz", "relA/slice_11_t0.5.npz", "relA/slice_11_t1.0.npz"):
    print(k, {a: np.round(v, 3) for a, v in log[k].items()})
print(len(log), "slices rewritten under", DST)
