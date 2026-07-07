"""Exact post-hoc repair of the id_b <A(t)> assembly bug (2026-07-07).

hadamard_correlator_aer captured <A> on the first time point of each call;
production scripts call it once per t, so every stored correlator row i
carries id_b[j] * <A(t_i)> where the correct term is id_b[j] * <A(0)>.
The repair is exact from stored data: the insertion J0(center) is itself
probe column CENTER, so

    corr[i, j] -= id_b[j] * (one_pt[i, CENTER] - one_pt[0, CENTER]).

Only J0 probe columns (id_b = (-1)^v / 2) are affected; J1 columns have
id_b = 0.  Vacuum-block grids are stationary (<A(t)> = <A(0)>) and thus
untouched by construction, but are corrected anyway for uniformity.

Applies in place, adds flag idfix=1; skips files already flagged.

  PYTHONPATH=. .venv/bin/python scripts/fix_id_term.py <file.npz> [...]
"""

import sys

import numpy as np

CENTER = 24
NS = 50
id_b = 0.5 * (-1.0) ** np.arange(NS)


def fix_corr(corr, one_pt):
    """corr, one_pt: (nt, nprobe) with J0 probes in the first NS columns."""
    da = one_pt[:, CENTER] - one_pt[0, CENTER]
    fixed = corr.copy()
    fixed[:, :NS] = corr[:, :NS] - da[:, None] * id_b[None, :]
    return fixed


for path in sys.argv[1:]:
    d = dict(np.load(path, allow_pickle=True))
    if d.get("idfix") is not None:
        print(f"{path}: already fixed, skipping")
        continue
    if "corr" in d:                                   # sysscan / ladder style
        d["corr"] = fix_corr(d["corr"], d["one_pt"])
    elif "corr_wp" in d:                              # production style
        d["corr_wp"] = fix_corr(d["corr_wp"], d["one_pt_wp"])
        d["corr_vac"] = fix_corr(d["corr_vac"], d["one_pt_vac"])
    else:
        print(f"{path}: no recognized corr keys, skipping")
        continue
    d["idfix"] = np.int64(1)
    np.savez(path, **d)
    print(f"{path}: repaired")
