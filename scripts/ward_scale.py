"""Use the lattice Ward identity to measure a J1-probe calibration scale.

The momentum-space Ward form is, per insertion index nu,

    q0 W^{0 nu} - qhat(q1) W^{1 nu} = 0,

so each nu pairs a J0-probe component with a J1-probe one: nu = 0 pairs
(00, 10) and nu = 1 pairs (01, 11).  If the J1 probe carries a
multiplicative calibration error, W^{1 nu} -> s^-1 W^{1 nu}_true for one
common s, then the least-squares scale that best satisfies the identity,

    s = <q0 W^{0 nu}, qhat W^{1 nu}> / || qhat W^{1 nu} ||^2,

measures it, with no reference input at all.  Two independent determinations
(nu = 0 and nu = 1) is the consistency check.

A J1-INSERTION error cannot be measured this way: it multiplies both terms
of the nu = 1 equation equally and so divides out.  The identity sees the
probe side only.

The identity is not exact for the windowed one-sided transform (t = 0
boundary and window-derivative terms, see tensor.ward_residual_q), so the
MPS reference pushed through the same pipeline is the control: s_hw must be
read against s_ref, not against 1.

  PYTHONPATH=. python scripts/ward_scale.py <reduction.npz> [...]

with the reduction npz written by scripts/hw_w_tensor.py.
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from htensor import asymmetry as asym, tensor  # noqa: E402

PAIRS = {"0": ("00", "10"), "1": ("01", "11")}


def report(path, wedge_only):
    d = np.load(path, allow_pickle=True)
    q0, q1 = d["q0"], d["q1"]
    reg = asym.region_mask(q0) if wedge_only else np.ones(len(q0), bool)
    print(f"\n=== {os.path.basename(path)}"
          f"   region: {'wedge q0 in [%g, %g]' % asym.WEDGE if wedge_only else 'full q0 grid'}")
    print(f"{'nu':>4}{'map':>6}{'s':>9}{'resid s=1':>11}{'resid at s':>12}"
          f"{'1/A(J1 probe)':>15}")
    for nu, (c0, c1) in PAIRS.items():
        if f"W_hw_{c0}" not in d.files or f"W_hw_{c1}" not in d.files:
            continue
        for tag in ("hw", "ref"):
            maps = {c: d[f"W_{tag}_{c}"] for c in (c0, c1)}
            r = tensor.ward_scale(maps, q0, q1, reg)[nu]
            s, r1, rs = r["s"], r["resid_1"], r["resid_s"]
            inv_a = ""
            if tag == "hw" and f"W_ref_{c1}" in d.files:
                try:
                    a1 = asym.amplitude(d[f"W_hw_{c1}"], d[f"W_ref_{c1}"], q0)
                    inv_a = f"{1.0 / a1:15.3f}" if a1 else ""
                except ValueError:
                    inv_a = ""
            print(f"{nu:>4}{tag:>6}{s:9.3f}{r1:11.3f}{rs:12.3f}{inv_a}")
    # stability of s across q0: a single scale should not drift
    print("  s over q0 thirds (hardware), as a stability check:")
    for nu, (c0, c1) in PAIRS.items():
        if f"W_hw_{c0}" not in d.files or f"W_hw_{c1}" not in d.files:
            continue
        idx = np.where(reg)[0]
        thirds = np.array_split(idx, 3)
        vals = []
        maps = {c: d[f"W_hw_{c}"] for c in (c0, c1)}
        for part in thirds:
            m = np.zeros(len(q0), bool)
            m[part] = True
            vals.append(tensor.ward_scale(maps, q0, q1, m)[nu]["s"])
        lo = f"[{q0[idx[0]]:+.2f}, {q0[idx[-1]]:+.2f}]"
        print(f"    nu={nu} over {lo}: " + ", ".join(f"{v:.3f}" for v in vals))


p = argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("npz", nargs="+")
p.add_argument("--full-grid", action="store_true",
               help="fit over the whole q0 grid instead of the asymmetry wedge")
args = p.parse_args()
for path in args.npz:
    report(path, not args.full_grid)
