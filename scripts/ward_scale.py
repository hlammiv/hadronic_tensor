"""Use the lattice Ward identity, exactly, to test the J1-probe calibration.

Per insertion index nu the continuity equation pairs a J0-probe component
with a J1-probe one: nu = 0 pairs (00, 10) and nu = 1 pairs (01, 11).
tensor.ward_pair pushes both sides through the production one-sided
windowed transform in a form that is exact for it (the naive
q0 W^{0 nu} = qhat W^{1 nu} drops the t = 0, t_max and window-derivative
terms and is off by up to 44% on the reference), so a == b when the two
grids obey continuity and a = s b when the J1 PROBE carries a common
multiplicative calibration error s.  A J1-INSERTION error divides out.

Three things decide how to read the numbers:

  * The control.  The unmasked MPS reference on both sides must return 1
    (it does, to 1%); that is what makes the hardware's departure a
    statement about the hardware.
  * The mask.  A rejected probe on the ridge breaks the identity by itself
    (prod loses site 27, the peak of G^{00}, at t >= 2 and the masked
    reference returns 0.62), so masked points are filled from the
    reference before the test, on hardware and reference alike.
  * Regression dilution.  s_lo = <a,b>/<b,b> is biased low by noise on b,
    s_hi = <a,a>/<a,b> is biased high by noise on a.  The hardware J0 side
    is too noisy in this combination for a reference-free fit to bracket
    anything (prod: [0.36, 2.2]), so the quotable number is the mixed
    fit, reference 00 against hardware 10, where s_hi is unbiased: it is
    the matched-filter amplitude of the hardware J1 map against the
    template that continuity builds from the J0 reference.

  PYTHONPATH=. python scripts/ward_scale.py [--derive] <reduction.npz> [...]

--derive turns the identity around: tensor.ward_derived_W1 builds the J1
map continuity implies from the J0 grid alone, and the report gives its
matched-filter amplitude against the reference J1 map next to the measured
hardware J1 map's.  At relA the derived W10 lands on the reference at
A = 0.98 (odd part, |q1| >= 1.1) where the measured one gives 1.92.

with the reduction npz written by scripts/hw_w_tensor.py, next to its
_command.json (for the reference files).
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from htensor import asymmetry as asym, tensor  # noqa: E402

PAIRS = {"0": ("00", "10"), "1": ("01", "11")}
DT, DX = 0.5, 0.5


def load(path):
    d = np.load(path, allow_pickle=True)
    cmd = json.load(open(os.path.splitext(path)[0] + "_command.json"))
    j1 = cmd["source_j1"] if "01" in cmd["components"] else None
    return d, tensor.load_mps_components(cmd["source_j0"], j1)


def grids(d, g, comp):
    """-> times, x, G_ref (unmasked, Re), G_hw with masked points filled."""
    times = d[f"times_{comp}"]
    ti = [int(np.argmin(np.abs(g.times - t))) for t in times]
    if not np.allclose(g.times[ti], times, rtol=0, atol=1e-9):
        raise ValueError(f"{comp}: the reference has no slice at one of {times}")
    ref = np.ascontiguousarray(g.g[comp][ti].real)
    hw = np.where(d[f"mask_{comp}"], d[f"G_hw_{comp}"], ref)
    return times, d[f"x_{comp}"], ref, hw


def report(path, wedge_only, forms, derive=False):
    d, g = load(path)
    q0, q1, sig_x = d["q0"], d["q1"], float(d["sig_x"])
    reg = asym.region_mask(q0) if wedge_only else np.ones(len(q0), bool)
    print(f"\n=== {os.path.basename(path)}   region: "
          f"{'wedge q0 in [%g, %g]' % asym.WEDGE if wedge_only else 'full q0 grid'}")
    for nu, (c0, c1) in PAIRS.items():
        if f"G_hw_{c0}" not in d.files or f"G_hw_{c1}" not in d.files:
            continue
        times, x0, r0, h0 = grids(d, g, c0)
        _, x1, r1, h1 = grids(d, g, c1)
        sig_t = float(d[f"sig_t_{c0}"])
        filled = {c: int((~d[f"mask_{c}"]).sum()) for c in (c0, c1)}
        inv_a = ""
        try:
            a1 = asym.amplitude(d[f"W_hw_{c1}"], d[f"W_ref_{c1}"], q0)
            inv_a = f"   1/A({c1}) = {1 / a1:.3f}" if a1 else ""
        except ValueError:
            pass
        print(f" nu = {nu}  ({c0} J0 probe, {c1} J1 probe)  t_max {times[-1]:g}  sigma_t {sig_t:g}"
              f"   masked points filled from the reference: {filled}{inv_a}")
        print(f"   {'J0 side / J1 side':<20}" + "".join(
            f"   {f:<12}{'s_lo':>7}{'s_hi':>7}{'resid 1':>9}{'resid lo':>9}" for f in forms))
        for lab, A, B in (("ref / ref  (control)", r0, r1), ("ref / hw", r0, h1),
                          ("hw  / ref", h0, r1), ("hw  / hw", h0, h1)):
            row = f"   {lab:<20}"
            for form in forms:
                a, b = tensor.ward_pair(times, x0, x1, A, B, q0, q1, sig_t, sig_x, DT, DX, form)
                r = tensor.ward_scale(a, b, reg)
                row += (f"   {'':<12}{r['s_lo']:7.3f}{r['s_hi']:7.3f}"
                        f"{r['resid_1']:9.3f}{r['resid_lo']:9.3f}")
            print(row)
        # stability of the quotable fit across q0
        a, b = tensor.ward_pair(times, x0, x1, r0, h1, q0, q1, sig_t, sig_x, DT, DX, forms[0])
        idx = np.where(reg)[0]
        vals = []
        for part in np.array_split(idx, 3):
            m = np.zeros(len(q0), bool)
            m[part] = True
            r = tensor.ward_scale(a, b, m)
            vals.append(f"[{r['s_lo']:.2f}, {r['s_hi']:.2f}]")
        print(f"   ref / hw, {forms[0]} form, [s_lo, s_hi] over q0 thirds of "
              f"[{q0[idx[0]]:+.2f}, {q0[idx[-1]]:+.2f}]: " + "  ".join(vals))
        if derive:
            derived(d, c1, times, x0, x1, r0, h0, q0, q1, sig_t, sig_x, reg)


def derived(d, c1, times, x0, x1, r0, h0, q0, q1, sig_t, sig_x, reg):
    """The J1 map continuity implies from the J0 grid, against the reference
    J1 map, with the measured hardware J1 map for comparison."""
    W1_ref, W1_hw = d[f"W_ref_{c1}"], d[f"W_hw_{c1}"]
    odd = lambda m: 0.5 * (m - m[:, ::-1])

    def amp(H, R, sel):
        Hs, Rs = H[reg][:, sel], R[reg][:, sel]
        return float((Hs * Rs).sum() / (Rs * Rs).sum()), float(np.corrcoef(Hs.ravel(), Rs.ravel())[0, 1])

    print(f"   W{c1} derived from the J0 grid through continuity, matched filter against the reference W{c1}:")
    print(f"   {'J0 grid':<22}{'|q1| >=':>8} | {'A full':>7}{'r':>6} | {'A odd':>7}{'r':>6}")
    for lab, G0 in (("reference (control)", r0), ("hardware (filled)", h0)):
        W1_der = tensor.ward_derived_W1(times, x0, x1, G0, q0, q1, sig_t, sig_x, DT, DX)
        for cut in (0.5, 1.1, 1.6):
            sel = np.abs(q1) >= cut
            af, rf = amp(W1_der, W1_ref, sel)
            ao, ro = amp(odd(W1_der), odd(W1_ref), sel)
            print(f"   {lab:<22}{cut:8.1f} | {af:7.3f}{rf:6.2f} | {ao:7.3f}{ro:6.2f}")
    for cut in (0.5, 1.1, 1.6):
        sel = np.abs(q1) >= cut
        ao, ro = amp(odd(W1_hw), odd(W1_ref), sel)
        print(f"   {'measured hardware':<22}{cut:8.1f} | {'':>7}{'':>6} | {ao:7.3f}{ro:6.2f}")


p = argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("npz", nargs="+")
p.add_argument("--full-grid", action="store_true",
               help="fit over the whole q0 grid instead of the asymmetry wedge")
p.add_argument("--form", default="both", choices=("integral", "differential", "both"),
               help="which exact form(s) to report")
p.add_argument("--derive", action="store_true",
               help="also report the J1 map continuity derives from the J0 grid")
args = p.parse_args()
for path in args.npz:
    report(path, not args.full_grid,
           ("integral", "differential") if args.form == "both" else (args.form,), args.derive)
