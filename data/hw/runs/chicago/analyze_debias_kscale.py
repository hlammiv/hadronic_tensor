"""htq_hw analyze, de-biased ancilla (analyze_debias.py), with every mirror
kappa multiplied by the measured physics/mirror loss ratio r(t): the
mirror-understatement systematic adopted 2026-11-19.

    HTQ_KAPPA_SCALE=mirror_ratio_<run>.npz   (from mirror_systematic/measure_mirror_ratio.py)

r(t) is the card's far-site ratio (<card tag>_r on <card tag>_times),
linearly interpolated, r(0) = 1; outside the measured times the nearest
value is held.  The scale is applied where kappa is formed
(analyze.kappa_from_mirror), so it reaches the J0 and J1 kappas and the
prep-blind fill alike, and the calibrated signal becomes signal / (kappa r).
The slice still records the UNSCALED kappa_v, so the map's kappa floor
masks the same sites as the nominal reduction and the shift measures the
scale alone (analyze's own 0.05 guard does see the scaled value).  The
systematic is the change in each asymmetry against the nominal reduction,
central value unchanged (choice of 2026-11-19).
"""
import os
import sys

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
_debias = os.path.join(_here, "2026-11-06", "analysis", "2026-11-07_debias", "analyze_debias.py")
_src = open(_debias).read().split("from htq_hw.__main__ import main")[0]
exec(compile(_src, _debias, "exec"))

_z = np.load(os.environ["HTQ_KAPPA_SCALE"], allow_pickle=True)
_tag = _card.split("_")[0]
_T = np.concatenate([[0.0], np.asarray(_z[f"{_tag}_times"], float)])
_R = np.concatenate([[1.0], np.asarray(_z[f"{_tag}_r"], float)])
_now = {"t": None}


def r_of(t):
    return float(np.interp(float(t), _T, _R))


_kfm = A.kappa_from_mirror


def kappa_from_mirror(sx_m, sxe_m, sx_i0):
    kap, kerr = _kfm(sx_m, sxe_m, sx_i0)
    t = _now["t"]
    if t is None or t <= 0:
        return kap, kerr
    return kap * r_of(t), kerr                 # kerr is relative, unchanged


_cs = A.component_slice


def component_slice(comp, t, *a, **k):
    _now["t"] = float(t)
    try:
        out = _cs(comp, t, *a, **k)
    finally:
        _now["t"] = None
    if out is not None and float(t) > 0 and "kappa_v" in out:
        out["kappa_v"] = out["kappa_v"] / r_of(t)        # the mask sees the nominal kappa
        out["kappa_scale"] = np.array(r_of(t))
    return out


A.kappa_from_mirror = kappa_from_mirror
A.component_slice = component_slice
print(f"kappa scale from {os.environ['HTQ_KAPPA_SCALE']} ({_tag}): r(2) {r_of(2):.2f}, r(4) {r_of(4):.2f}, "
      f"r(6) {r_of(6):.2f}", file=sys.stderr)
from htq_hw.__main__ import main  # noqa: E402
sys.exit(main(sys.argv[1:]))
