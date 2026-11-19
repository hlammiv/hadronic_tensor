"""Is the mirror kappa a property of the logical site or of the physical qubit?

The 8 Nov mirror-diag layout runs through the same region of ibm_chicago as
the 6 Nov production layout, in the opposite direction: 90 of the 101 wires
share physical qubits, and the centre site 24 of 8 Nov (q108) was the failing
odd site 25 of 6 Nov.  This script pairs the per-site mirror kappas of the two
runs by physical qubit and by logical site and reports which pairing
correlates.  It also checks whether each mirror pub returns the matter Z
profile to the t=0 reference (the mirror's job), independent of the ancilla.

usage: python kappa_by_physical_qubit.py [--card prod_k1.26_s0.75_ns50]
"""
import argparse, glob, json, os, sys
import numpy as np
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import htq_hw.analyze as A
from htq_hw.campaign import pub_name
from htq_hw.model import Lattice
import mirror_diag_compare as M

RUN6 = 'data/hw/runs/chicago/2026-11-06'
RUN8 = 'data/hw/runs/chicago/2026-11-08'


def kappas(paths, card, grid, lat, eps):
    names = {pub_name("j0", 0.0, False, "Z")} | {pub_name("j0", t, m, "Z") for t in M.TIMES for m in (False, True)}
    found = M.find_pubs(paths, card, names)
    x_ideal = float(grid.scalar("X", 0.0))
    sx_im = M.ideal_at(grid, eps)
    out, prof = {}, {}
    raw0 = M.load(found, pub_name("j0", 0.0, False, "Z"))
    zsites = [lat.site_qubit(v) for v in range(lat.ns)]
    prof[0.0] = A.signs_from_bits(raw0)[:, zsites].mean(0)
    out[0.0] = M.signal(raw0, lat, x_ideal, True)[0] / grid.s_ideal("X", 0.0)
    for t in M.TIMES:
        rm = M.load(found, pub_name("j0", t, True, "Z"))
        rp = M.load(found, pub_name("j0", t, False, "Z"))
        out[t] = M.signal(rm, lat, x_ideal, True)[0] / sx_im
        prof[(t, 'm')] = A.signs_from_bits(rm)[:, zsites].mean(0)
        prof[(t, 'p')] = A.signs_from_bits(rp)[:, zsites].mean(0)
    return out, prof


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--card', default='prod_k1.26_s0.75_ns50')
    args = ap.parse_args()
    grid = A.IdealGrid(f"htq_hw/cards/{args.card}/ideal_j0.npz", expect_ns=50)
    lat = Lattice(50)
    L6 = json.load(open(f'{RUN6}/acceptance_21-49-59_889395Z.json'))['embedding']['layout']
    L8 = json.load(open(f'{RUN8}/acceptance_mirror_diag_control.json'))['embedding']['layout']
    k6, p6 = kappas(sorted(glob.glob(f'{RUN6}/raw/main_rzz/htq_bits_*.npz')), args.card, grid, lat, 1e-8)
    runs = {'control': (f'{RUN8}/raw/mirror_diag_control', 1e-8), 'eps05': (f'{RUN8}/raw/mirror_diag_eps05', 0.05)}
    k8 = {lab: kappas(sorted(glob.glob(f'{d}/htq_bits_*.npz')), args.card, grid, lat, eps) for lab, (d, eps) in runs.items()}
    q6 = {L6[lat.site_qubit(v)]: v for v in range(50)}      # physical qubit -> 6 Nov logical site
    q8 = {L8[lat.site_qubit(v)]: v for v in range(50)}
    shared = sorted(set(q6) & set(q8))
    print(f"matter qubits shared by the two layouts: {len(shared)} of 50; ancilla 6 Nov q{L6[lat.ancilla]}, 8 Nov q{L8[lat.ancilla]}")
    sx0 = grid.s_ideal("X", 0.0)
    for lab, (k8l, _) in k8.items():
        print(f"\n=== 6 Nov production vs 8 Nov {lab}: mirror kappa paired by physical qubit")
        print(f"    {'phys q':>6} {'site6':>5} {'site8':>5} | " + " ".join(f"{'k6 t'+str(int(t)):>7} {'k8 t'+str(int(t)):>7} |" for t in M.TIMES))
        for q in shared:
            s6, s8 = q6[q], q8[q]
            if abs(sx0[s6]) < 0.03 or abs(sx0[s8]) < 0.03:
                continue                      # kappa undefined where the t=0 signal is tiny
            row = " ".join(f"{k6[t][s6]:>+7.3f} {k8l[t][s8]:>+7.3f} |" for t in M.TIMES)
            print(f"    {q:>6} {s6:>5} {s8:>5} | {row}" + ("   <- q108" if q == 108 else ""))
        print("    correlation of kappa across the two runs (sites with |sx_ideal0| > 0.03 in both roles):")
        for t in M.TIMES:
            a = np.array([k6[t][q6[q]] for q in shared if abs(sx0[q6[q]]) > 0.03 and abs(sx0[q8[q]]) > 0.03])
            b = np.array([k8l[t][q8[q]] for q in shared if abs(sx0[q6[q]]) > 0.03 and abs(sx0[q8[q]]) > 0.03])
            ok = np.isfinite(a) & np.isfinite(b)
            r_phys = np.corrcoef(a[ok], b[ok])[0, 1]
            sites = [v for v in range(50) if abs(sx0[v]) > 0.03]
            c = np.array([k6[t][v] for v in sites]); d = np.array([k8l[t][v] for v in sites])
            r_log = np.corrcoef(c, d)[0, 1]
            print(f"      t={t:.0f}: by physical qubit r = {r_phys:+.2f} (n={ok.sum()})   by logical site r = {r_log:+.2f} (n={len(sites)})")
    # does the mirror return the Z profile to the reference?
    print("\n=== matter <Z_n> profile: RMS over sites of (mirror - ref) and (physics - ref), ref = t=0 pub of the same run")
    for lab, prof in [('production 6 Nov', p6)] + [(l, k8[l][1]) for l in k8]:
        ref = prof[0.0]
        row = "  ".join(f"t={t:.0f}: mirror {np.sqrt(np.mean((prof[(t,'m')]-ref)**2)):.3f} physics {np.sqrt(np.mean((prof[(t,'p')]-ref)**2)):.3f}" for t in M.TIMES)
        print(f"    {lab:<18} {row}")
        print(f"    {'':<18} centre site 24 <Z>: ref {ref[24]:+.3f}  " + "  ".join(f"t={t:.0f} mirror {prof[(t,'m')][24]:+.3f} physics {prof[(t,'p')][24]:+.3f}" for t in M.TIMES))
        print(f"    {'':<18} site 25      <Z>: ref {ref[25]:+.3f}  " + "  ".join(f"t={t:.0f} mirror {prof[(t,'m')][25]:+.3f} physics {prof[(t,'p')][25]:+.3f}" for t in M.TIMES))


if __name__ == '__main__':
    main()
