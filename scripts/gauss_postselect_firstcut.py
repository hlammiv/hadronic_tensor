"""First-cut Gauss post-selection diagnostics on raw per-shot bits (note sec. 17a).

    PYTHONPATH=. python scripts/gauss_postselect_firstcut.py acceptance [--raw DIR]
    PYTHONPATH=. python scripts/gauss_postselect_firstcut.py undamp [--times 0 0.5 1 2 4 6 8]
    PYTHONPATH=. python scripts/gauss_postselect_firstcut.py vacuum

acceptance: kept fraction of every j0 / plain Z-setting pub of the two packet
  cards at windows 0..3, with the measured <G_centre>.
undamp:     matched-filter amplitude of the connected ancilla signal against the
  card's ideal grid (sites 1..6 from the insertion) for the j0 physics pub and
  its mirror (against the t = 0 row) at windows 0, 1, 2.  The ratio
  physics / mirror is the calibrated amplitude the slices would carry.
vacuum:     the same calibrated amplitude on the two vacuum cards, where the
  ideal is known at every time (closed test; ideal = 1).

Uses htq_hw.analyze.gauss_keep_mask, the mask `analyze --postselect W` applies.
"""

import argparse
import glob
import re
import sys

import numpy as np

sys.path.insert(0, ".")
from htq_hw import analyze as A  # noqa: E402
from htq_hw.model import Lattice  # noqa: E402

NS, CENTER = 50, 24
RAW = "data/hw/runs/phoenix/2026-10-06/raw"
CARDS = {"prod-bridge": "prod_k1.26_s0.75_ns50", "relA-core": "relA_k1.26_s0.75_ns50"}


def load_pubs(raw, pattern):
    """{(preset, card, family, kind, t): bits} for pub names matching ``pattern``
    (groups: preset, card, family, kind t|m, time)."""
    out = {}
    for f in sorted(glob.glob(f"{raw}/htq_bits_*.npz")):
        z = np.load(f, allow_pickle=False)
        for n in map(str, z["pub_names"]):
            m = re.match(pattern, n)
            if m:
                pre, card, fam, kind, t = m.groups()
                out[(pre, card, fam, kind, float(t))] = z[n]
    return out


def ideal_sx(card):
    """t -> connected ancilla signal XB_v - id_b <X> of the card's j0 grid."""
    g = np.load(f"htq_hw/cards/{card}/ideal_j0.npz")
    times = [round(float(x), 3) for x in g["times"]]
    idb = np.array([(-1) ** v / 2 for v in range(NS)])

    def at(t):
        if round(t, 3) not in times:
            return None
        i = times.index(round(t, 3))
        return np.array([g[f"XB_{v}"][i] for v in range(NS)]) - idb * g["X"][i]
    return at


def amplitude(bits, lat, window, ideal, reg):
    keep = A.gauss_keep_mask(bits, lat, CENTER, window)
    r = A.estimate_probes(bits[keep], lat, "Z")
    den = (ideal[reg] ** 2).sum()
    return (float((ideal[reg] * r["sx"][reg]).sum() / den),
            float(np.sqrt(((ideal[reg] * r["sxe"][reg]) ** 2).sum()) / den), int(keep.sum()))


def cmd_acceptance(a):
    lat = Lattice(NS)
    pubs = load_pubs(a.raw, r"(prod-bridge|relA-core)\.(\S+?):(j0|plain)_([tm])([\d.]+)_Z$")
    for key in sorted({k[:4] for k in pubs}):
        print(f"== {key[0]} {key[2]} {'mirror' if key[3] == 'm' else 'physics'}   "
              "(shots, kept at w=0,1,2,3, <G_centre>)")
        for k in sorted(k for k in pubs if k[:4] == key):
            b = pubs[k]
            acc = [A.gauss_keep_mask(b, lat, CENTER, w).mean() for w in (0, 1, 2, 3)]
            g = A.estimate_probes(b, lat, "Z")["gauss"][CENTER]
            print(f"  t={k[4]:4.1f} N={len(b):6d}  " + " ".join(f"{x:.3f}" for x in acc) + f"   G_c={g:+.3f}")


def cmd_undamp(a):
    lat = Lattice(NS)
    x = np.arange(NS) - CENTER
    x = np.minimum(abs(x), NS - abs(x))
    reg = (x >= 1) & (x <= 6)
    pubs = load_pubs(a.raw, r"(prod-bridge|relA-core)\.(\S+?):(j0)_([tm])([\d.]+)_Z$")
    pubs = {k: v for k, v in pubs.items() if k[4] in a.times}
    for pre, card in CARDS.items():
        at = ideal_sx(card)
        print(f"== {pre}: amplitude vs ideal, sites 1..6 from the centre  [amp +- err, kept]")
        print("   kind  t       w=0               w=1               w=2")
        for k in sorted(k for k in pubs if k[0] == pre):
            kind, t = k[3], k[4]
            ide = at(0.0 if kind == "m" else t)
            if ide is None:
                continue
            cells = [amplitude(pubs[k], lat, w, ide, reg) for w in (0, 1, 2)]
            print(f"   {kind}  {t:3.1f}  " + "  ".join(f"{v:6.3f}+-{e:.3f} {n:6d}" for v, e, n in cells))


def cmd_vacuum(a):
    lat = Lattice(NS)
    x = np.arange(NS) - CENTER
    x = np.minimum(abs(x), NS - abs(x))
    reg = (x >= 1) & (x <= 6)
    pubs = load_pubs(a.raw, r"(vac-w00)\.(\S+?):(j0)_([tm])([\d.]+)_Z$")
    for card in ("prod_vac_ns50", "relA_vac_ns50"):
        at = ideal_sx(card)
        print(f"== {card}: calibrated amplitude physics/mirror, ideal = 1   (kept physics shots at w=2)")
        print("    t     w=0      w=1      w=2")
        for t in sorted({k[4] for k in pubs if k[1] == card and k[3] == "t"}):
            if at(t) is None or ("vac-w00", card, "j0", "m", t) not in pubs:
                continue
            row = []
            for w in (0, 1, 2):
                p = amplitude(pubs[("vac-w00", card, "j0", "t", t)], lat, w, at(t), reg)
                m = amplitude(pubs[("vac-w00", card, "j0", "m", t)], lat, w, at(0.0), reg)
                row.append(f"{p[0] / m[0]:6.3f}" + (f" ({p[2]})" if w == 2 else "   "))
            print(f"   {t:3.1f}  " + "  ".join(row))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("what", choices=("acceptance", "undamp", "vacuum"))
    p.add_argument("--raw", default=RAW)
    p.add_argument("--times", type=float, nargs="+", default=(0.0, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0))
    a = p.parse_args()
    {"acceptance": cmd_acceptance, "undamp": cmd_undamp, "vacuum": cmd_vacuum}[a.what](a)
