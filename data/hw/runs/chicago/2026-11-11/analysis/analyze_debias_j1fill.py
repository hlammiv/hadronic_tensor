"""htq_hw analyze, de-biased ancilla (analyze_debias.py), plus a TEST rule for
the far J1 probes.  Not adopted; 2026-11-11, for Henry's question whether
W10/W11 on the twirled Chicago run are recoverable in software.

Standard rule: a J1 probe on bond b takes kappa = sx_mirror / sx_ideal(0)
where |sx_ideal(0)| > 0.02, the prep-blind j0 Z-mirror kappa at the insertion
on the two insertion bonds, and the MEDIAN of the guard-passing bonds
elsewhere.  On the production card only bonds 22, 25, 27 pass the guard, so
48 bonds inherit the median of three weak odd-site bonds.

Test rule: every bond that fails the guard and is not an insertion bond
takes the same-job j0 Z-mirror kappa at its own two sites (mean of sites b
and b + 1, as the W00 reduction uses them), times the probe-weight law
exp(-WEIGHT_RATE (3 - 1) t), with the anchor's error plus the prep-blind
model error 0.10 in quadrature.  This is the prep-blind rule extended from
the insertion bonds to all anchorless bonds.  Guard-passing bonds keep
their own measured kappa.

usage: same arguments as analyze_debias.py
"""
import os
import sys

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
_debias = os.path.join(_here, "..", "..", "2026-11-06", "analysis", "2026-11-07_debias",
                       "analyze_debias.py")
_src = open(_debias).read().split("from htq_hw.__main__ import main")[0]
exec(compile(_src, _debias, "exec"))          # installs the de-biased estimate_probes on A

_orig_fill = A.prep_blind_fill
_far_report = []


def prep_blind_fill(kap, kerr, job, t, family, dt, anc, ideal_j0, center, log=None, anchor_jobs=None):
    kap, kerr, bonds, anchor_id = _orig_fill(kap, kerr, job, t, family, dt, anc, ideal_j0,
                                             center, log, anchor_jobs)
    if not bonds:
        return kap, kerr, bonds, anchor_id
    ns = len(kap)
    kanc = "X" if anc == "Y" else anc
    anchor = next((j for j in [job] + [a for a in (anchor_jobs or []) if a is not job]
                   if j.mirror_name(t, "Z", kanc, "j0", dt) is not None), None)
    k0, e0 = A.kappa_for_basis(anchor, t, "j0", dt, anc, ideal_j0.sx_ideal0("J0"))
    guard = (np.abs(ideal_j0.sx_ideal0("J1", 1)) > A.KAP_GUARD) | \
            (np.abs(ideal_j0.sx_ideal0("J1", 2)) > A.KAP_GUARD)
    far = [b for b in range(ns) if not guard[b] and b not in bonds]
    w = np.exp(-A.WEIGHT_RATE * 2 * t)
    kap, kerr = np.array(kap, float), np.array(kerr, float)
    for b in far:
        s, s2 = b, (b + 1) % ns
        kap[b] = 0.5 * (k0[s] + k0[s2]) * w
        kerr[b] = np.hypot(0.5 * np.hypot(e0[s], e0[s2]), A.PREP_BLIND_MODEL_ERR)
    _far_report.append((family, t, float(np.median(kap[far]))))
    return kap, kerr, sorted(set(bonds) | set(far)), anchor_id


A.prep_blind_fill = prep_blind_fill
from htq_hw.__main__ import main   # noqa: E402
rc = main(sys.argv[1:])
if _far_report:
    m = np.array([r[2] for r in _far_report])
    print(f"far-bond J1 fill from the j0 Z mirror at the bond's sites: {len(m)} calls, "
          f"median kappa {np.median(m):.3f}, range {m.min():.3f} .. {m.max():.3f}")
sys.exit(rc)
