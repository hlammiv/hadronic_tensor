"""Where the J1-probe excess enters: the hardware against the ideal, stage by
stage, per time and per bond.

For every component on a card the calibrated connected ancilla signal (the
quantity the mirror kappa and the probe calibration produce) is compared to
the ideal grid's at the same t, as a matched-filter amplitude over probes;
for the J1 probes the raw (uncalibrated) signal is compared as well, which
is the forward damping the mirror is supposed to measure.  The ideal grids
match the MPS reference to a few percent at t > 0, so a departure from 1 is
the hardware's.

What this found on phoenix (campaign note, sections 13 and 14): the J0
probes calibrate to 0.9 to 1.0 at early t; the J1 probes came out 1.4 to 2.5
times the ideal at t <= 1 on every card, at dt = 0.5 and at the dt = 0.25
control alike, with the excess on the bonds that contain the insertion qubit.
The cause was the kappa fill on those bonds: their equal-time ideal is zero
by anticommutation, the median fill carried the prepared state's reduced XY
correlations (0.39), and the first-step current there is prep-blind (sourced
by Z^2 = 1) and damped only by the circuit (0.78, the J0 kappa at the
insertion site).  `htq_hw.analyze.prep_blind_fill` is the fix; --run points
this script at a slice set reduced with it.  The 01 / 11 rows compare the
two-family hardware signal with the two-family ideal (an earlier version
used one family and reported 2.7 / 2.3 at t = 0.5; the right numbers are
1.3 / 1.2).

  PYTHONPATH=. python scripts/j1_probe_stages.py prod relA [--tmax 6]
      [--run data/hw/runs/phoenix/2026-10-06/analysis/2026-10-22_prepblind_kappa/slices]
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from htq_hw import analyze as hqa  # noqa: E402

RUN = "data/hw/runs/phoenix/2026-10-06/analysis"
CARDS = {"prod": "prod_k1.26_s0.75_ns50", "relA": "relA_k1.26_s0.75_ns50"}
# comp -> (family, probe, insertion label)
# comp -> insertion label; families and probe kind come from hqa.COMPONENT_LAYOUT
INSERTION = {"00": "site 24", "10": "site 24", "00d": "site 25", "10d": "site 25",
             "01": "bond 24", "11": "bond 24"}


def mf(h, i):
    h, i = np.asarray(h, float), np.asarray(i, float)
    return float((h * i).sum() / (i * i).sum()), float(np.corrcoef(h, i)[0, 1])


def signals(d, ideals, comp, t, eta):
    """-> (raw, cal, ideal) connected ancilla signals on the probes, or raw
    None for J0 probes (their raw rebuild needs the wing anchor).

    `ideals` maps family -> IdealGrid.  The ideal is summed over the
    component's insertion families exactly as `component_slice` sums the
    hardware parts: the J1 insertion is two families (j1p1 + j1p2), and
    comparing their summed hardware signal with one family's ideal is off by
    about 2 (which is what the first version of this script did)."""
    fams, probe = hqa.COMPONENT_LAYOUT[comp]
    first = ideals[fams[0]]
    C, Cc = d["C"][0].real, d["C_cal"][0].real
    id_a, A0 = first.id_a, first.A0           # the first family's, as component_slice uses

    def c_a(fam):
        return hqa.FAMILY_COEFF[fam] * (eta if fam.startswith("j1") else 1.0)
    if probe == "J1":
        one = d["one_pt"][0]
        idl = sum(c_a(f) * eta / 4 * (ideals[f].s_ideal("X", t, "J1", 1) - ideals[f].s_ideal("X", t, "J1", 2))
                  for f in fams)
        return C - id_a * one, Cc - id_a * one, idl
    b, idb = d["b_cal"][0].real, first.id_b()
    cal = Cc - id_a * (b - idb) - idb * (A0 - id_a) - id_a * idb     # rebuild_C inverted
    return None, cal, sum(c_a(f) * ideals[f].s_ideal("X", t, "J0") for f in fams)


def card_report(card, tmax, tag, run):
    sub = CARDS[card]
    D = f"{run}/{sub}/"
    ideals = {}
    for fam in ("j0", "j0d", "j1p1", "j1p2"):
        path = f"htq_hw/cards/{sub}/ideal_{fam}.npz"
        if os.path.exists(path):
            ideals[fam] = hqa.IdealGrid(path)
    eta = float(ideals["j0"].z["eta"])
    print(f"\n=== {card}  eta {eta}{'  ' + tag if tag else ''}   slices {D}\n"
          f"    (amplitude = <hw, ideal>/<ideal, ideal> over probes, r = correlation)")
    for comp, ins in INSERTION.items():
        fams, probe = hqa.COMPONENT_LAYOUT[comp]
        if any(f not in ideals for f in fams) or not os.path.exists(D + f"slice_{comp}_t0.5{tag}.npz"):
            continue
        ideal = ideals[fams[0]]
        times = [t for t in np.arange(0.5, tmax + 1e-9, 0.5)
                 if ideal.has_row(t) and os.path.exists(D + f"slice_{comp}_t{t:.1f}{tag}.npz")]
        print(f" {comp}: {probe} probes, insertion {ins}, families {'+'.join(fams)}")
        print(f"   {'t':>4} | {'raw/ideal':>9} {'r':>5} | {'kappa':>6} | {'cal/ideal':>9} {'r':>5} | "
              f"per bond (|ideal| > 25% of max)" if probe == "J1" else
              f"   {'t':>4} | {'kappa':>6} | {'cal/ideal':>9} {'r':>5}")
        num, den = np.zeros(50), np.zeros(50)
        for t in times:
            d = np.load(D + f"slice_{comp}_t{t:.1f}{tag}.npz", allow_pickle=True)
            raw, cal, idl = signals(d, ideals, comp, t, eta)
            kap = d["kappa_v"][0]
            ok = np.isfinite(cal) & (kap > hqa.KAP_MASK)
            a_cal, r_cal = mf(cal[ok], idl[ok])
            weight = ok & (np.abs(idl) > 0.10 * np.abs(idl).max())
            num[weight] += (cal * idl)[weight]
            den[weight] += (idl * idl)[weight]
            if probe == "J1":
                a_raw, r_raw = mf(raw[ok], idl[ok])
                big = np.where(ok & (np.abs(idl) > 0.25 * np.abs(idl).max()))[0]
                print(f"   {t:4.1f} | {a_raw:9.3f} {r_raw:5.2f} | {np.median(kap[ok]):6.3f} | "
                      f"{a_cal:9.3f} {r_cal:5.2f} | "
                      + "  ".join(f"b{b}:{cal[b] / idl[b]:.2f}" for b in big))
            else:
                print(f"   {t:4.1f} | {np.median(kap[ok]):6.3f} | {a_cal:9.3f} {r_cal:5.2f}")
        prof = np.where(den > 0, num / np.maximum(den, 1e-30), np.nan)
        lab = "b" if probe == "J1" else "s"
        print("   over all t, per probe: " + "  ".join(
            f"{lab}{i}:{prof[i]:.2f}" for i in range(19, 30) if np.isfinite(prof[i])))


p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("cards", nargs="*", default=list(CARDS), choices=list(CARDS))
p.add_argument("--tmax", type=float, default=6.0)
p.add_argument("--dt-control", action="store_true",
               help="read the dt = 0.25 control slices instead (t = 0.5 and 1.0)")
p.add_argument("--run", default=RUN,
               help="directory holding <card>/slice_*.npz (default: the campaign's; point it at a "
                    "post-processed set such as .../2026-10-22_prepblind_kappa/slices)")
args = p.parse_args()
for card in args.cards:
    card_report(card, args.tmax, "_dt0.25" if args.dt_control else "", args.run)
