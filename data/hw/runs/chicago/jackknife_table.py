"""Jackknife table of the boost asymmetry (wedge [0, 2.5], t_max 6 maps):
pooled A with its shot and window errors, the G leave-one-group-out values
and err^2 = (G-1)/G sum_g (A_-g - mean)^2.

usage: jackknife_table.py POOLED_DIR JK_DIR G
"""
import re
import sys

import numpy as np

ROW = re.compile(r"^\s*\[\+0\.00, 2\.50\]\s+\d+\s+(\S+)\s+(\S+)\s+(\S+)")


def parse(path):
    out, card, comp = {}, None, None
    for line in open(path):
        if line.startswith("=== "):
            card = ("slow (prod)" if "/prod_t6.npz" in line else
                    "fast (relA)" if "/relA_t6.npz" in line else None)
        m = re.match(r"\s+(\d\d): max", line)
        if m:
            comp = m.group(1)
        m = ROW.match(line)
        if m and card and comp:
            try:
                out[(card, comp)] = tuple(float(x) for x in m.groups())
            except ValueError:
                pass
    return out


def main():
    pooled_dir, jk_dir, G = sys.argv[1], sys.argv[2], int(sys.argv[3])
    pooled = parse(f"{pooled_dir}/asymmetry.txt")
    drops = [parse(f"{jk_dir}/g{g}/asymmetry.txt") for g in range(G)]
    print(f"jackknife over {G} groups of twirl instances; wedge [0, 2.5], t_max 6")
    print(f"{'particle':12s} {'W':3s} {'A':>7s} {'shot':>6s} {'window':>7s} {'jk err':>7s} {'A/err':>6s}   leave-one-out")
    for key in sorted(pooled):
        p = pooled[key]
        th = [d.get(key) for d in drops]
        if any(t is None for t in th):
            print(f"{key[0]:12s} {key[1]:3s} {p[0]:7.2f} (jackknife incomplete)")
            continue
        th = np.array([t[0] for t in th])
        jk = np.sqrt((G - 1) / G * np.sum((th - th.mean()) ** 2))
        flag = "  UNSTABLE: pooled outside the leave-one-out range" if not th.min() <= p[0] <= th.max() else ""
        print(f"{key[0]:12s} {key[1]:3s} {p[0]:7.2f} {p[1]:6.2f} {p[2]:7.2f} {jk:7.2f} {p[0] / np.hypot(jk, p[2]):6.1f}   "
              + " ".join(f"{t:.2f}" for t in th) + flag)


if __name__ == "__main__":
    main()
