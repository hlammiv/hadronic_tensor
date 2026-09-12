"""Noisy dress rehearsal, step 2: average Pauli-trajectory calibration grids
(scripts/hw_cal_grids.py --noise --mirror) per insertion family and predict
the mirror-calibration damping kappa for every probe type -- the input to
the shots plan (htq_hw campaign.shots_plan) and the README's kappa-vs-depth
table.

For each family (j0, j1p1, j1p2) and time t:
  J0 probes : sx_v = XB_v - id_b(v) X              kappa_v = <sx_v>_mirror,noisy / sx_v^ideal(0)
  J1 probes : sx^{(k)}_b = XT{k}_b                 kappa^{(k)}_b likewise (traceless term, no identity part)
  calibrated physics = sx_noisy(t) / kappa_v(t) vs the ideal sx(t): the recovery error per site.
Guard: |sx_ideal(0)| > GUARD else kappa is median-filled (ibm_hardware.analyze_t3 convention).

  PYTHONPATH=. .venv/bin/python scripts/hw_cal_assemble.py --ideal "data/hw_cal_prod_ns50_{family}_k1.26_s0.75.npz" \
      --noisy "data/hw_cal_prod_ns50_{family}_k1.26_s0.75_noise*.npz" --families j0 j1p1 j1p2
"""

import argparse
import glob

import numpy as np

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--ideal", required=True, help="template with {family}")
p.add_argument("--noisy", required=True, help="glob template with {family}")
p.add_argument("--families", nargs="+", default=["j0", "j1p1", "j1p2"])
p.add_argument("--guard", type=float, default=0.02)
p.add_argument("--kap-min", type=float, default=0.05, help="sites with kappa below this are masked")
p.add_argument("--out", default="data/hw_cal_kappa.npz")
args = p.parse_args()

out = {}
for fam in args.families:
    idl = np.load(args.ideal.format(family=fam), allow_pickle=True)
    files = sorted(glob.glob(args.noisy.format(family=fam)))
    if not files:
        print(f"{fam}: no noisy files"); continue
    ds = [np.load(f, allow_pickle=True) for f in files]
    # the noisy runs may use a coarser time grid than the ideal reference:
    # work on the intersection, indexing each side by its own row
    t_i, t_n = np.asarray(idl["times"], float), np.asarray(ds[0]["times"], float)
    times = np.array([t for t in t_n if np.min(np.abs(t_i - t)) < 1e-9])
    irow = np.array([int(np.argmin(np.abs(t_i - t))) for t in times])
    nrow = np.array([int(np.argmin(np.abs(t_n - t))) for t in times])
    ns = len(idl["probes"])
    id_b = np.array([(-1) ** v / 2 for v in range(ns)])
    has_j0 = f"XB_0" in idl.files
    has_j1 = f"XT1_0" in idl.files
    has_mirror = any(k.startswith("m_") for k in ds[0].files)

    def sx_j0(d, prefix=""):
        XB = np.array([d[f"{prefix}XB_{v}"] for v in range(ns)]).T
        return XB - id_b[None, :] * d[f"{prefix}X"][:, None]

    def sx_j1(d, k, prefix=""):
        return np.array([d[f"{prefix}XT{k}_{b}"] for b in range(ns)]).T

    print(f"== family {fam}: {len(files)} trajectories, times {times.tolist()}, mirror={has_mirror}")
    for name, fn in ([("J0", lambda d, pr="": sx_j0(d, pr))] if has_j0 else []) + \
                    ([(f"J1T{k}", (lambda kk: lambda d, pr="": sx_j1(d, kk, pr))(k)) for k in (1, 2)] if has_j1 else []):
        ideal = fn(idl)[irow]
        phys = np.mean([fn(d) for d in ds], axis=0)[nrow]
        mirr = np.mean([fn(d, "m_") for d in ds], axis=0)[nrow] if has_mirror else None
        vis = np.abs(ideal[0]) > args.guard
        rows = []
        for i, t in enumerate(times):
            if mirr is not None and t > 0:
                kap = np.where(vis, mirr[i] / np.where(vis, ideal[0], 1.0), np.nan)
                kap = np.where(np.isnan(kap), np.nanmedian(kap), kap)
            else:
                kap = np.where(vis, phys[i] / np.where(vis, ideal[i], 1.0), np.nan)  # t=0: direct ratio
                kap = np.where(np.isnan(kap), np.nanmedian(kap), kap)
            ok = vis & (kap > args.kap_min)
            cal = np.where(ok, phys[i] / np.where(ok, kap, 1.0), np.nan)
            err = np.abs(cal - ideal[i])[ok]
            rows.append((t, float(np.nanmedian(kap[vis])), float(np.min(kap[vis])), int(vis.sum()),
                         int((vis & ~ok).sum()), float(np.sqrt(np.mean(err ** 2))) if err.size else np.nan))
            out[f"{fam}_{name}_kappa_t{t:.1f}"] = kap
            out[f"{fam}_{name}_mask_t{t:.1f}"] = ok
        print(f"  probes {name}: visible sites {int(vis.sum())}/{ns} (|sx_ideal(0)|>{args.guard})")
        print("     t    kappa_med  kappa_min  n_vis  masked  rms(cal-ideal)")
        for t, km, kmin, nv, nm, rms in rows:
            print(f"   {t:4.1f}   {km:8.3f}  {kmin:8.3f}  {nv:5d}  {nm:6d}   {rms:.4f}")
np.savez(args.out, **out)
print("wrote", args.out)
