"""Read a prep-state model run (scripts/prep_factor_model.py --stage run) and
print what the maps need from it:

  * the equal-time factors f_v(0) against the hardware targets (the fit)
  * f_v(t) / f_v(0) for the J0 probes and the J1 terms on the block sites,
    i.e. the residual bias of the mirror calibration per probe per time
  * the insertion site alone, against the measured calibrated/ideal there
    (note sec. 15a: 0.93 0.86 0.76 0.73 0.75 0.78 0.75 0.81 at prod,
    t = 0.5 .. 4.0; relA 0.76 0.45 0.54 0.50 0.41)
  * the imaginary branch at the insertion site, f_im(t) = Im S_noisy / Im S_ideal,
    against the kingston archive (note sec. 15f: 0.24, 0.03 at t = 0.5;
    0.16 at t = 1.0; 0.02 at t = 2.0 on the prod card)

  PYTHONPATH=. python scripts/prep_factor_report.py data/work/prep_factor/run_prod_gate.npz [--target targets.json]
"""
import argparse
import json

import numpy as np

GUARD = 0.02
MEASURED_SITE = {"prod": [0.93, 0.86, 0.76, 0.73, 0.75, 0.78, 0.75, 0.81],
                 "relA": [0.76, 0.45, 0.54, 0.50, 0.41]}
KINGSTON_IM = {0.5: (0.24, 0.03), 1.0: (0.16,), 2.0: (0.02,)}


def ratio(num, den, guard=GUARD):
    ok = np.abs(den) > guard
    return np.where(ok, num / np.where(ok, den, 1.0), np.nan)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npz")
    ap.add_argument("--target", default="data/work/prep_factor/targets.json")
    args = ap.parse_args()
    d = np.load(args.npz, allow_pickle=True)
    card, model, p = str(d["card"]), str(d["model"]), float(d["ps"][0])
    times, c, ns = d["times"], int(d["center"]), int(d["ns"])
    fams = [k[len("ideal_S_"):] for k in d.files if k.startswith("ideal_S_") and not k.endswith("_err")]
    np.set_printoptions(precision=2, suppress=True, linewidth=170)
    print(f"{card} / {model} p={p:g}: {int(d['ntraj'])} damaged trajectories, clean weight {float(d['p_clean'][0]):.3f}, "
          f"families {fams}, t = 0 .. {times[-1]:g}")
    Si, Sn, Se = d["ideal_S_j0"], d[f"p{p:g}_S_j0"], d[f"p{p:g}_S_j0_err"]
    f = ratio(Sn, Si)                      # (nt, 3, ns)
    fe = ratio(Se, np.abs(Si))
    offs = np.arange(ns) - c
    # ---- the fit
    try:
        tg = json.load(open(args.target))[card]
    except (OSError, KeyError):
        tg = None
    f0 = f[0] / f[0, 0, c]
    print("\nequal-time factors f_v(0)/f_c(0) (model) against the hardware targets (kappa_v/kappa_c):")
    chi, n = 0.0, 0
    for kind, row in (("J0", 0), ("T1", 1), ("T2", 2)):
        line = f"  {kind}: "
        for off in range(-4, 5):
            v = c + off
            if not np.isfinite(f0[row, v]):
                continue
            hw = tg[kind].get(str(off)) if tg else None
            line += f" [{off:+d}] {f0[row, v]:.2f}" + (f" (hw {hw[0]:.2f})" if hw else "")
            if hw:
                chi += ((f0[row, v] - hw[0]) / hw[1]) ** 2
                n += 1
        print(line)
    print(f"  chi2/n = {chi:.1f}/{n};  f_c(0) = {f[0, 0, c]:.3f}")
    # ---- time dependence
    print("\nf_v(t)/f_v(0), J0 probes (rows: t; columns: offset -4 .. +4; blank where the ideal is below the guard):")
    hdr = "   t   " + "".join(f"{o:+6d}" for o in range(-4, 5))
    print(hdr)
    for i, t in enumerate(times):
        row = f"  {t:4.1f} "
        for off in range(-4, 5):
            v = c + off
            r = f[i, 0, v] / f[0, 0, v] if np.isfinite(f[i, 0, v]) and np.isfinite(f[0, 0, v]) else np.nan
            row += f"{r:6.2f}" if np.isfinite(r) else "      "
        print(row)
    print("\nf_b(t) for the J1 terms under the j0 insertion, divided by the insertion site's f_c(0) (the prep-blind fill's"
          " assumption) on the adjacent bonds and by f_b(0) elsewhere; T1 | T2:")
    for i, t in enumerate(times):
        if t == 0:
            continue
        row = f"  {t:4.1f} "
        for trow in (1, 2):
            for off in range(-4, 4):
                b = c + off
                den = f[0, 0, c] if off in (-1, 0) else f[0, trow, b]
                r = f[i, trow, b] / den if np.isfinite(f[i, trow, b]) and np.isfinite(den) else np.nan
                row += f"{r:6.2f}" if np.isfinite(r) else "      "
            row += "  | "
        print(row)
    # ---- the insertion site
    print("\ninsertion site: model f_c(t)/f_c(0) against the measured calibrated/ideal (15a)")
    meas = MEASURED_SITE.get(card, [])
    for i, t in enumerate(times):
        if t == 0:
            continue
        r = f[i, 0, c] / f[0, 0, c]
        m = meas[i - 1] if i - 1 < len(meas) else None
        print(f"  t={t:3.1f}  model {r:5.2f} +- {fe[i, 0, c]:.2f}" + (f"   measured {m:.2f}" if m else ""))
    # ---- the imaginary branch
    if "ideal_Simag_j0" in d.files:
        Ii, In, Ie = d["ideal_Simag_j0"], d[f"p{p:g}_Simag_j0"], d[f"p{p:g}_Simag_j0_err"]
        print("\nimaginary branch at the insertion site: f_im(t) = Im S_noisy / Im S_ideal (kingston measured in brackets)")
        for i, t in enumerate(times):
            if t == 0 or abs(Ii[i, 0, c]) < 0.005:
                continue
            k = KINGSTON_IM.get(float(t))
            print(f"  t={t:3.1f}  ideal Im {Ii[i, 0, c]:+.3f}  model f_im {In[i, 0, c] / Ii[i, 0, c]:5.2f} +- {Ie[i, 0, c] / abs(Ii[i, 0, c]):.2f}"
                  + (f"   [kingston {', '.join(f'{x:.2f}' for x in k)}]" if k and card == "prod" else ""))
        print("  neighbours (offset -1, +1) f_im at t = 0.5, 1.0:",
              [round(float(In[i, 0, c + o] / Ii[i, 0, c + o]), 2) for i in (1, 2) if i < len(times)
               for o in (-1, 1) if abs(Ii[i, 0, c + o]) > 0.005])


if __name__ == "__main__":
    main()
