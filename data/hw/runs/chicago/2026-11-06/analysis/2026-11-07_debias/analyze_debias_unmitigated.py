"""analyze_debias.py with every mitigation switched off, for the talk's
"no mitigation" stage under the de-bias rule: ancilla X bias removed per pub
(analyze_debias.py), no mirror division (kappa = beta = 1, the mirror still
required to exist), no wing anchor.  Gauss post-selection is off by passing
--postselect 0.  Same recipe as campaign/prelim_2026-11-02/unmitigated/
analyze_unmitigated.py, plus the bias removal."""
import os
import runpy
import sys

import numpy as np

import htq_hw.analyze as A

_k = A.kappa_for_basis


def kappa_one(job, t, family, dt, anc, sx_i0):
    k, e = _k(job, t, family, dt, anc, sx_i0)
    return (None, None) if k is None else (np.ones_like(k), np.zeros_like(e))


A.kappa_for_basis = kappa_one
A.beta_from_mirror = lambda B_m, b_i0: np.ones_like(np.asarray(B_m, float))
A.wing_anchor = lambda b_cal, b_ideal, center: (b_cal, np.zeros_like(np.real(b_cal)))
runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "analyze_debias.py"),
               run_name="__main__")
