"""Per-site mirror kappa for the ibm_chicago mirror diagnostic (8 Nov 2026).

Compares the mirror-diag variants (control at mirror angle 1e-8, eps05 at
mirror angle 0.05; the twirled variant was refused by the Runtime, error
1519) with each other and with the 6 November production run, site by site,
at t = 2, 4, 6 on the production card.  The question is whether the odd
sites around the insertion (21, 23, 25, 27), whose mirror kappa drifted
through zero in production, recover when the mirror block is a real
evolution instead of RZZ(1e-8) layers.

kappa_n = sx_mirror_n / sx_ideal_n, with sx the connected ancilla signal
(estimate_probes) and the ideal taken at the mirror's own time: t = 0 for
the 1e-8 mirror, t = eps for the raised one (linearly interpolated between
the t = 0 and t = 0.25 rows of the card's grid; the shift is printed).  No
guard, no median fill: the raw ratio per site is what the diagnostic is
about.  The physics circuits' own damping sx_phys / sx_ideal(t) is printed
beside it; in production the two agreed at the even sites and disagreed at
the odd ones.  The ancilla X bias (mean <X_anc> - ideal, additive, depth
independent) is removed per pub unless --raw-ancilla is given; both the
physics and the mirror bias are printed.

usage:
  python mirror_diag_compare.py --card prod_k1.26_s0.75_ns50 \
      --variant control "data/hw/runs/chicago/2026-11-08/raw/mirror_diag_control/htq_bits_*.npz" \
      --variant eps05   "data/hw/runs/chicago/2026-11-08/raw/mirror_diag_eps05/htq_bits_*.npz" \
      --production      "data/hw/runs/chicago/2026-11-06/raw/main_rzz/htq_bits_*.npz"
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())
import htq_hw.analyze as A                      # noqa: E402
from htq_hw.campaign import pub_name, split_prefix   # noqa: E402
from htq_hw.model import Lattice                # noqa: E402

TIMES = (2.0, 4.0, 6.0)
CENTER = 24
GROUPS = {"centre": [24], "even 22,26": [22, 26], "odd 21,23,25,27": [21, 23, 25, 27],
          "next 20,28": [20, 28], "far |n-24|>=8": [n for n in range(50) if abs(n - 24) >= 8]}


def signal(bits, lat, x_ideal, debias):
    """Connected J0 ancilla signal per site, its error, the raw <X_anc> and the bias removed."""
    s = A.signs_from_bits(bits)
    xa_raw = s[:, lat.ancilla].astype(float)
    bias = float(xa_raw.mean()) - x_ideal
    xa = xa_raw - bias if debias else xa_raw
    z = s[:, [lat.site_qubit(v) for v in range(lat.ns)]]
    idb = np.array([(-1) ** v / 2 for v in range(lat.ns)])
    conn = xa[:, None] * (idb[None, :] - z / 2) - idb[None, :] * xa[:, None]
    N = len(s)
    return conn.mean(0), conn.std(0) / np.sqrt(N), float(xa_raw.mean()), bias, N


def find_pubs(paths, card, names):
    """{bare name: (path, key)} for the wanted bare names of this card, first file wins."""
    found = {}
    for p in paths:
        z = np.load(p, allow_pickle=True)
        for k in z.files:
            pre, bare = split_prefix(k)
            if A.card_of(pre) == card and bare in names and bare not in found:
                found[bare] = (p, k)
    return found


def load(found, name):
    p, k = found[name]
    return np.load(p, allow_pickle=True)[k]


def mirror_eps_of(paths):
    """mirror_eps from the job records next to the raw files (jobs/<sub>/htq_job_<id>.json), else 1e-8."""
    for p in paths:
        j = p.replace("/raw/", "/jobs/").replace("htq_bits_", "htq_job_")
        j = os.path.splitext(j)[0] + ".json"
        if os.path.exists(j):
            opt = json.load(open(j)).get("options", {}) or {}
            if "mirror_eps" in opt:
                return float(opt["mirror_eps"]), opt
            return 1e-8, opt
    return 1e-8, {}


def ideal_at(grid, t):
    if grid.has_row(t):
        return grid.s_ideal("X", t)
    if t < 0.25:                                  # the raised mirror angle: interpolate 0 .. 0.25
        s0, s1 = grid.s_ideal("X", 0.0), grid.s_ideal("X", 0.25)
        return s0 + (t / 0.25) * (s1 - s0)
    raise KeyError(t)


def analyse(label, paths, card, grid, lat, debias, eps_override=None):
    eps, opt = mirror_eps_of(paths)
    if eps_override is not None:
        eps = eps_override
    x_ideal = float(grid.scalar("X", 0.0))
    names = {pub_name("j0", 0.0, False, "Z")} | {pub_name("j0", t, m, "Z") for t in TIMES for m in (False, True)}
    found = find_pubs(paths, card, names)
    sx_i0 = grid.s_ideal("X", 0.0)
    sx_im = ideal_at(grid, eps)                   # the mirror's own target
    print(f"\n=== {label}: {len(paths)} files, {len(found)}/{len(names)} pubs, mirror_eps {eps:g}, "
          f"options {opt or '(none recorded)'}, ancilla {'de-biased' if debias else 'raw'}")
    if eps > 1e-4:
        d = sx_im - sx_i0
        odd = GROUPS["odd 21,23,25,27"]
        print(f"    ideal mirror target moved from t=0 to t={eps:g}: max |shift| {np.abs(d).max():.4f}, "
              f"at the odd sites {np.abs(d[odd]).max():.4f} against |sx_i0| {np.abs(sx_i0[odd]).min():.3f}..{np.abs(sx_i0[odd]).max():.3f}")
    out = {}
    n0 = pub_name("j0", 0.0, False, "Z")
    if n0 in found:
        sx0, sxe0, xa0, b0, N0 = signal(load(found, n0), lat, x_ideal, debias)
        k0 = sx0 / sx_i0
        print(f"    t=0 reference: N {N0}, <X_anc> {xa0:+.4f} (ideal {x_ideal:+.4f}, bias {b0:+.4f}); "
              f"sx_hw/sx_ideal centre {k0[CENTER]:.3f}, odd {np.round(k0[GROUPS['odd 21,23,25,27']], 3)}")
        out[0.0] = dict(kap=k0, phys=k0)
    for t in TIMES:
        pn, mn = pub_name("j0", t, False, "Z"), pub_name("j0", t, True, "Z")
        if pn not in found or mn not in found:
            print(f"    t={t}: missing {'physics' if pn not in found else ''} {'mirror' if mn not in found else ''}")
            continue
        sxp, sxpe, xap, bp, Np = signal(load(found, pn), lat, x_ideal, debias)
        sxm, sxme, xam, bm, Nm = signal(load(found, mn), lat, x_ideal, debias)
        kap, kerr = sxm / sx_im, sxme / np.abs(sx_im)
        kap0 = sxm / sx_i0
        sx_it = grid.s_ideal("X", t)
        ok = np.abs(sx_it) > 0.02
        phys = np.where(ok, sxp / np.where(ok, sx_it, 1.0), np.nan)
        out[t] = dict(kap=kap, kerr=kerr, kap0=kap0, phys=phys, xap=xap, xam=xam, bp=bp, bm=bm)
        print(f"  t={t:.0f}: N phys {Np} mirror {Nm}; <X_anc> phys {xap:+.4f} mirror {xam:+.4f} "
              f"(bias {bp:+.4f} / {bm:+.4f})")
        print(f"    {'site':>5} {'sx_ideal0':>9} {'kappa':>7} {'+-':>6} {'phys/ideal(t)':>13}")
        for n in range(18, 31):
            print(f"    {n:>5} {sx_i0[n]:>+9.4f} {kap[n]:>7.3f} {kerr[n]:>6.3f} "
                  f"{phys[n]:>13.3f}" + ("   <- odd" if n % 2 else ""))
        for g, sites in GROUPS.items():
            km = np.nanmedian(kap[sites]); pm = np.nanmedian(phys[sites])
            print(f"    {g:<18} kappa median {km:+.3f}   physics damping median {pm:+.3f}   "
                  f"kappa against t=0 ideal {np.nanmedian(kap0[sites]):+.3f}")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--card", default="prod_k1.26_s0.75_ns50")
    ap.add_argument("--ns", type=int, default=50)
    ap.add_argument("--variant", nargs=2, action="append", metavar=("LABEL", "GLOB"), default=[])
    ap.add_argument("--production", metavar="GLOB", help="6 Nov raw files for the same card")
    ap.add_argument("--raw-ancilla", action="store_true", help="do not remove the ancilla X bias")
    ap.add_argument("--eps", type=float, action="append", metavar="LABEL=EPS", default=[],
                    help="override the mirror angle of a variant when the job record lacks it")
    args = ap.parse_args(argv)
    grid = A.IdealGrid(f"htq_hw/cards/{args.card}/ideal_j0.npz", expect_ns=args.ns)
    lat = Lattice(args.ns)
    debias = not args.raw_ancilla
    overrides = dict((s.split("=")[0], float(s.split("=")[1])) for s in args.eps)
    results = {}
    if args.production:
        paths = sorted(glob.glob(args.production))
        results["production 6 Nov"] = analyse("production 6 Nov", paths, args.card, grid, lat, debias, 1e-8)
    for label, pattern in args.variant:
        paths = sorted(glob.glob(pattern))
        if not paths:
            print(f"\n=== {label}: no files match {pattern}")
            continue
        results[label] = analyse(label, paths, args.card, grid, lat, debias, overrides.get(label))
    # side by side
    labels = [l for l in results if results[l]]
    if len(labels) > 1:
        print("\n=== side by side: median mirror kappa per group (kappa against the mirror's own ideal)")
        print(f"    {'t':>3} {'group':<18}" + "".join(f"{l:>20}" for l in labels))
        for t in TIMES:
            for g, sites in GROUPS.items():
                row = []
                for l in labels:
                    r = results[l].get(t)
                    row.append(f"{np.nanmedian(r['kap'][sites]):>+20.3f}" if r else f"{'-':>20}")
                print(f"    {t:>3.0f} {g:<18}" + "".join(row))
        print("\n=== side by side: site 25 (odd, next to the insertion) and site 24 (centre)")
        for t in TIMES:
            row = "  ".join(f"{l}: 25 {results[l][t]['kap'][25]:+.3f}+-{results[l][t]['kerr'][25]:.3f}, "
                            f"24 {results[l][t]['kap'][24]:+.3f}+-{results[l][t]['kerr'][24]:.3f}"
                            for l in labels if t in results[l])
            print(f"    t={t:.0f}: {row}")


if __name__ == "__main__":
    main()
