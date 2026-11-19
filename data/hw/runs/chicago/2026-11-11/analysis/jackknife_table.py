"""Delete-one-instance jackknife of the boost asymmetry A (wedge [0, 2.5])
of the 11 Nov twirled maps: pooled value, its quoted shot (+window) error,
the jackknife error over the K = 4 twirl instances, and the single-instance
scatter error std/sqrt(K) for comparison.  2026-11-14."""
import re, sys
import numpy as np
K = 4
ROW = re.compile(r"^\s*\[\+0\.00, 2\.50\]\s+\d+\s+(\S+)\s+(\S+)\s+(\S+)")


def parse(path):
    out, card, comp = {}, None, None
    for line in open(path):
        if line.startswith("=== "):
            card = ("slow (prod)" if "/prod_t6.npz" in line else
                    "fast (relA)" if "/relA_t6.npz" in line else None)   # t_max 6 maps only
        m = re.match(r"\s+(\d\d): max", line)
        if m:
            comp = m.group(1)
        m = ROW.match(line)
        if m and card and comp:
            try:
                out[(card, comp)] = tuple(float(x) for x in m.groups())
            except ValueError:
                out[(card, comp)] = None
    return out


pooled = parse("debias_w1/asymmetry.txt")
drops = [parse(f"jackknife/d{i}/asymmetry.txt") for i in range(K)]
singles = [parse(f"instances/i{i}/asymmetry.txt") for i in range(K)]
print(f"{'particle':12s} {'W':3s} {'pooled A':>9s} {'shot':>6s} {'window':>6s} | {'leave-one-out A':>27s} {'jk err':>7s} | {'single-inst err':>15s} | {'A / jk err':>10s}")
for key in sorted(pooled):
    p = pooled[key]
    th = [d.get(key) for d in drops]
    si = [s.get(key) for s in singles]
    if p is None or any(t is None for t in th):
        print(f"{key[0]:12s} {key[1]:3s} (missing)"); continue
    th = np.array([t[0] for t in th])
    jk = np.sqrt((K - 1) / K * np.sum((th - th.mean()) ** 2))
    se = (np.std([s[0] for s in si if s], ddof=1) / np.sqrt(K)) if all(si) else float("nan")
    print(f"{key[0]:12s} {key[1]:3s} {p[0]:9.2f} {p[1]:6.2f} {p[2]:6.2f} | {' '.join(f'{t:6.2f}' for t in th):>27s} {jk:7.2f} | {se:15.2f} | {p[0] / np.hypot(jk, p[2]):10.1f}")
