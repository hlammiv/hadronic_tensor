"""Where the J1-probe excess enters: the hardware against the ideal, stage by
stage, per time and per bond.

For every component on a card the calibrated connected ancilla signal (the
quantity the mirror kappa and the probe calibration produce) is compared to
the ideal grid's at the same t, as a matched-filter amplitude over probes;
for the J1 probes the raw (uncalibrated) signal is compared as well, which
is the forward damping the mirror is supposed to measure.  The ideal grids
match the MPS reference to a few percent at t > 0, so a departure from 1 is
the hardware's.

What this found on phoenix (campaign note, section 13): the J0 probes
calibrate to 0.9 to 1.0 at early t; the J1 probes come out 1.4 to 2.5 times
the ideal at t <= 1 on every card, at dt = 0.5 and at the dt = 0.25 control
alike (so it is not Trotter error), with a raw forward damping of 0.73 at
t = 0.5 where the mirror says kappa = 0.39 and the t = 0 pub reads 0.42.
At t = 0.5 the excess sits on the bonds that contain the insertion qubit
and moves with the insertion (the j0d family); by t >= 1 it has spread over
the active bonds.  The eps-angle mirror cannot see it, because it carries
no dynamics to propagate anything from the insertion.

  PYTHONPATH=. python scripts/j1_probe_stages.py prod relA [--tmax 6]
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
LAYOUT = {"00": ("j0", "J0", "site 24"), "10": ("j0", "J1", "site 24"),
          "00d": ("j0d", "J0", "site 25"), "10d": ("j0d", "J1", "site 25"),
          "01": ("j1p1", "J0", "bond 24"), "11": ("j1p1", "J1", "bond 24")}


def mf(h, i):
    h, i = np.asarray(h, float), np.asarray(i, float)
    return float((h * i).sum() / (i * i).sum()), float(np.corrcoef(h, i)[0, 1])


def signals(d, ideal, comp, t, eta):
    """-> (raw, cal, ideal) connected ancilla signals on the probes, or raw
    None for J0 probes (their raw rebuild needs the wing anchor)."""
    fam, probe, _ = LAYOUT[comp]
    c_a = hqa.FAMILY_COEFF[fam] * (eta if fam.startswith("j1") else 1.0)
    C, Cc = d["C"][0].real, d["C_cal"][0].real
    id_a, A0 = ideal.id_a, ideal.A0           # the family's own, as component_slice uses
    if probe == "J1":
        one = d["one_pt"][0]
        idl = c_a * eta / 4 * (ideal.s_ideal("X", t, "J1", 1) - ideal.s_ideal("X", t, "J1", 2))
        return C - id_a * one, Cc - id_a * one, idl
    b, idb = d["b_cal"][0].real, ideal.id_b()
    cal = Cc - id_a * (b - idb) - idb * (A0 - id_a) - id_a * idb     # rebuild_C inverted
    return None, cal, c_a * ideal.s_ideal("X", t, "J0")


def card_report(card, tmax, tag):
    sub = CARDS[card]
    D = f"{RUN}/{sub}/"
    ideals = {}
    for fam in ("j0", "j0d", "j1p1"):
        path = f"htq_hw/cards/{sub}/ideal_{fam}.npz"
        if os.path.exists(path):
            ideals[fam] = hqa.IdealGrid(path)
    eta = float(ideals["j0"].z["eta"])
    print(f"\n=== {card}  eta {eta}{'  ' + tag if tag else ''}   "
          f"(amplitude = <hw, ideal>/<ideal, ideal> over probes, r = correlation)")
    for comp, (fam, probe, ins) in LAYOUT.items():
        if fam not in ideals or not os.path.exists(D + f"slice_{comp}_t0.5{tag}.npz"):
            continue
        ideal = ideals[fam]
        times = [t for t in np.arange(0.5, tmax + 1e-9, 0.5)
                 if ideal.has_row(t) and os.path.exists(D + f"slice_{comp}_t{t:.1f}{tag}.npz")]
        print(f" {comp}: {probe} probes, insertion {ins}, family {fam}")
        print(f"   {'t':>4} | {'raw/ideal':>9} {'r':>5} | {'kappa':>6} | {'cal/ideal':>9} {'r':>5} | "
              f"per bond (|ideal| > 25% of max)" if probe == "J1" else
              f"   {'t':>4} | {'kappa':>6} | {'cal/ideal':>9} {'r':>5}")
        num, den = np.zeros(50), np.zeros(50)
        for t in times:
            d = np.load(D + f"slice_{comp}_t{t:.1f}{tag}.npz", allow_pickle=True)
            raw, cal, idl = signals(d, ideal, comp, t, eta)
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
args = p.parse_args()
for card in args.cards:
    card_report(card, args.tmax, "_dt0.25" if args.dt_control else "")
