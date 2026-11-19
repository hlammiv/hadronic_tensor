"""Measure how much the mirror understates the physics circuit's loss
(adopted as a systematic 2026-11-19).

In the empty-lattice region far from the packet (|n - 24| >= 8) the
density barely moves, so the decay of <Z_n> there is noise.  For every t
of a card:

    physics survival  s_p(n, t) = [Z_hw(n, t) / Z_hw(n, 0)] / [Z_id(n, t) / Z_id(n, 0)]
    mirror survival   s_m(n, t) =  Z_mir(n, t) / Z_hw(n, 0)        (ideal mirror = identity)
    ratio             r(n, t)   =  s_p / s_m

r(t) = median over the far sites is the number the systematic uses
(choice of 2026-11-19: far-site Z, one ratio per t per run).  Recorded as
well, for checking later: r per site at every site, the far-site medians
by parity, and the centre-site cross-check from the ancilla correlator,
c_24 hardware / ideal at t over mirror c_24 / own t = 0 (it uses the ideal
answer, so it is a check, not an input).

usage: measure_mirror_ratio.py OUT.npz BITS_GLOB [CARD ...]
"""
import glob
import sys

import numpy as np

import htq_hw.analyze as A
from htq_hw.campaign import pub_name, split_prefix
from htq_hw.model import Lattice

NS, CENTER, FAR = 50, 24, 8
lat = Lattice(NS)
ZS = [lat.site_qubit(v) for v in range(NS)]
far = np.array([n for n in range(NS) if min((n - CENTER) % NS, (CENTER - n) % NS) >= FAR])


def collect(pattern, card):
    pubs = {}
    for p in sorted(glob.glob(pattern)):
        z = np.load(p, allow_pickle=True)
        for k in z.files:
            if z[k].dtype != np.uint8 or z[k].ndim != 2:
                continue
            pre, bare = split_prefix(k)
            if A.card_of(pre) == card:
                pubs.setdefault(bare, []).append((p, k))
    return pubs


def stats(pubs, name):
    """Pooled over every file carrying the pub: <X_anc>, <Z_n>, c_n = <X Z_n> - <X><Z_n>."""
    arrs = [np.load(p, allow_pickle=True)[k] for p, k in pubs[name]]
    s = A.signs_from_bits(np.concatenate(arrs, axis=0))
    x, zz = s[:, lat.ancilla], s[:, ZS]
    X, Z = x.mean(), zz.mean(0)
    return X, Z, (x[:, None] * zz).mean(0) - X * Z, len(s)


def main():
    out_path, pattern, cards = sys.argv[1], sys.argv[2], sys.argv[3:] or ["prod_k1.26_s0.75_ns50",
                                                                          "relA_k1.26_s0.75_ns50"]
    rec = {"far_sites": far, "center": CENTER, "bits_glob": pattern, "cards": np.array(cards)}
    for card in cards:
        pubs = collect(pattern, card)
        g = A.IdealGrid(f"htq_hw/cards/{card}/ideal_j0.npz", expect_ns=NS)
        idb = g.id_b()
        sur = A.load_wing_surrogate(f"htq_hw/refs/wing_surrogate_{card.split('_')[0]}.npz")

        def zid(t):
            """Ideal <Z_n>: the card's grid where it has the row; beyond it (relA stops at
            t = 3) the far sites take B(0) plus the vacuum surrogate's per-parity shift
            (A.wing_target's rule, 1e-3 of the grid at t <= 3) and the rest are NaN."""
            if g.has_row(t):
                return 2 * (idb - g.vec("B", t))
            out = np.full(NS, np.nan)
            i = int(np.argmin(np.abs(sur["times"] - t)))
            if abs(sur["times"][i] - t) > 1e-9:
                raise KeyError(f"no surrogate row at t={t}")
            b0 = g.vec("B", 0.0)        # surrogate = B(t) - B(0) per parity, as in A.wing_target
            for n in far:
                out[n] = 2 * (idb[n] - b0[n] - (sur["even"][i] if n % 2 == 0 else sur["odd"][i]))
            return out

        def cid(t):
            if not g.has_row(t):
                return np.full(NS, np.nan)
            X = float(g.scalar("X", t))
            return 2 * (idb * X - g.vec("XB", t)) - X * zid(t)
        ref = pub_name("j0", 0.0, False, "Z")
        if ref not in pubs:
            print(f"{card}: no t = 0 reference, skipped")
            continue
        _, Z0, c0, _ = stats(pubs, ref)
        times = sorted({float(A.parse_pub_name(n)["t"]) for n in pubs
                        if n.startswith("j0_t") and n.endswith("_Z") and float(A.parse_pub_name(n)["t"]) > 0
                        and pub_name("j0", float(A.parse_pub_name(n)["t"]), True, "Z") in pubs})
        rs, rev, rod, rsite, rc, nsh = [], [], [], [], [], []
        for t in times:
            _, Zp, cp, n_p = stats(pubs, pub_name("j0", t, False, "Z"))
            _, Zm, cm, n_m = stats(pubs, pub_name("j0", t, True, "Z"))
            sp = (Zp / Z0) / (zid(t) / zid(0.0))
            sm = Zm / Z0
            r = sp / sm
            rsite.append(r)
            rs.append(np.nanmedian(r[far]))
            rev.append(np.nanmedian(r[far[far % 2 == 0]]))
            rod.append(np.nanmedian(r[far[far % 2 == 1]]))
            rc.append((cp[CENTER] / cid(t)[CENTER]) / (cm[CENTER] / c0[CENTER]))
            nsh.append((n_p, n_m))
        tag = card.split("_")[0]
        rec.update({f"{tag}_times": np.array(times), f"{tag}_r": np.array(rs), f"{tag}_r_even": np.array(rev),
                    f"{tag}_r_odd": np.array(rod), f"{tag}_r_site": np.array(rsite),
                    f"{tag}_r_centre_corr": np.array(rc), f"{tag}_shots": np.array(nsh)})
        print(f"{card}: far-site ratio physics / mirror survival (median over {len(far)} sites)")
        print("     t     r   even    odd | centre-correlator check")
        for i, t in enumerate(times):
            print(f"  {t:5.2f} {rs[i]:5.2f}  {rev[i]:5.2f}  {rod[i]:5.2f} | {rc[i]:5.2f}")
    np.savez(out_path, **rec)
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
