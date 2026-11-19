"""Mirror-understatement systematic: asymmetry (wedge [0, 2.5], t_max 6)
nominal and with kappa x r(t); the shift is the systematic.

usage: shift_table.py LABEL NOMINAL_DIR SCALED_DIR [LABEL NOMINAL_DIR SCALED_DIR ...]
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from jackknife_table import parse  # noqa: E402

args = sys.argv[1:]
print(f"{'run':10s} {'particle':12s} {'W':3s} {'nominal':>8s} {'shot':>6s} {'window':>7s} {'scaled':>8s} {'syst':>6s}")
for i in range(0, len(args), 3):
    lab, nom, sca = args[i:i + 3]
    a, b = parse(f"{nom}/asymmetry.txt"), parse(f"{sca}/asymmetry.txt")
    for key in sorted(a):
        if key not in b:
            continue
        p, q = a[key], b[key]
        print(f"{lab:10s} {key[0]:12s} {key[1]:3s} {p[0]:8.2f} {p[1]:6.2f} {p[2]:7.2f} {q[0]:8.2f} {abs(q[0] - p[0]):6.2f}")
