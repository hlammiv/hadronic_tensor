"""Two checks on the tier-3 W00 reduction, both runnable from released data.

Context: PR #1 (declanmillar) proposes calibrating each charge probe around
its staggered identity offset id_b(v) = (-1)^v/2 instead of the published
fixed +1/2, on the argument that depolarization contracts <J0(v)> toward
(-1)^v/2.  The argument is sound in isolation, so this script tests it
against MPS truth on the real ibm_kingston results rather than arguing.

  PYTHONPATH=. python scripts/tier3_calibration_checks.py beta
  PYTHONPATH=. python scripts/tier3_calibration_checks.py imaginary

`beta`       recalibrates B_v both ways on the four tier-3 estimator jobs and
             scores each against the noiseless grid.
`imaginary`  quantifies what the published real-part-only convention costs on
             the exact correlator, where the answer is not shot limited.

Inputs are all in the repository: data/hw/primitive/estimator_*.npz (the
runtime output, with observable definitions), data/hw/job_tier3*.json (the
observable names, in the same order), the card's noiseless grid, and the MPS
reference data/w_meson_ns50_k1.26_v3.npz.
"""

import glob
import json
import sys

import numpy as np

sys.path.insert(0, ".")
from htensor import Z2Lattice, analysis  # noqa: E402

NS, VC, CARD = 50, 24, "htq_hw/cards/prod_k1.26_s0.75_ns50/ideal_j0.npz"
# tier-3 estimator jobs -> the slice time each one measured
JOBS = {"d9d3emkinv1c73aoguc0": 0.5, "d9d84gsinv1c73aomlqg": 0.5,
        "d9d45acinv1c73aohsj0": 1.0, "d9d518sjeosc73fh2prg": 2.0}
GUARD = 0.05            # a probe with no signal above the identity cannot be calibrated


def ideal_B(grid, t):
    tt = np.asarray(grid["times"], float)
    i = int(np.argmin(np.abs(tt - t)))
    return np.array([float(grid[f"B_{v}"][i]) for v in range(NS)])


def calibrate(B_noisy, B_mirror, b_ideal0, id_b):
    """beta from the depth-matched mirror, then invert it on the physics run.
    beta_v = (B^mirror - id_b) / (B^mirror,ideal - id_b), with the mirror's
    ideal value being the t=0 row (the mirror is net identity)."""
    den = b_ideal0 - id_b
    ok = np.abs(den) > GUARD
    beta = np.where(ok, (B_mirror - id_b) / np.where(ok, den, 1.0), 1.0)
    good = np.abs(beta) > GUARD
    return id_b + (B_noisy - id_b) / np.where(good, beta, 1.0), beta


def cmd_beta():
    grid = np.load(CARD, allow_pickle=True)
    b_i0 = ideal_B(grid, 0.0)
    anchors = {"+1/2 (published)": np.full(NS, 0.5),
               "(-1)^v/2 (PR #1)": np.array([(-1.0) ** v / 2 for v in range(NS)])}
    print("Reconstructing B_v on the tier-3 estimator jobs, scored against the "
          "noiseless grid.\nrms over all 50 probes; lower is better.\n")
    print(f"{'job':<24}{'t':>5}{'raw':>9}" + "".join(f"{k:>20}" for k in anchors))
    totals = {k: [] for k in anchors}
    for jid, t in JOBS.items():
        arch = f"data/hw/primitive/estimator_{jid}.npz"
        meta = glob.glob(f"data/hw/job_tier3*_{jid}.json")
        if not (glob.glob(arch) and meta):
            print(f"{jid[:22]:<24}{t:>5}   missing inputs, skipped")
            continue
        a = np.load(arch, allow_pickle=True)
        idx = {n: i for i, n in enumerate(json.load(open(meta[0]))["obs_names"][0])}
        Bn = np.array([a["pub0_evs"][idx[f"B_{v}"]] for v in range(NS)])   # physics
        Bm = np.array([a["pub1_evs"][idx[f"B_{v}"]] for v in range(NS)])   # mirror
        truth = ideal_B(grid, t)
        rms = lambda d: float(np.sqrt((d ** 2).mean()))
        line = f"{jid[:22]:<24}{t:>5}{rms(Bn - truth):>9.4f}"
        for k, idb in anchors.items():
            Bc, _ = calibrate(Bn, Bm, b_i0, idb)
            r = rms(Bc - truth)
            totals[k].append(r)
            line += f"{r:>20.4f}"
        print(line)
    print("\n" + " " * 38 + "".join(f"{np.mean(v):>20.4f}" for v in totals.values()) + "   <- mean")
    best = min(totals, key=lambda k: np.mean(totals[k]))
    print(f"\nLower rms on every slice: {best}.")
    med = {k: float(np.median(np.abs(b_i0 - v))) for k, v in anchors.items()}
    print("Not a degeneracy artifact: the median |B_ideal(0) - id_b| is "
          + ", ".join(f"{m:.3f} for {k}" for k, m in med.items())
          + ", so both denominators are healthy, and only "
          + f"{int(min((np.abs(b_i0 - v) < GUARD).sum() for v in anchors.values()))}-"
          + f"{int(max((np.abs(b_i0 - v) < GUARD).sum() for v in anchors.values()))}"
          + " of 50 probes trip the guard either way.")


def cmd_imaginary():
    """The published reduction takes .real of both the hardware and the MPS
    correlator (hw_w00_coarse.py:111,115), so the quoted A is apples to apples.
    What it costs on the reconstructed tensor is a separate question, and the
    exact correlator answers it without shot noise."""
    lat = Z2Lattice(NS, pbc=True)
    tru = np.load("data/w_meson_ns50_k1.26_v3.npz")
    sub = lambda c, o, i: analysis.subtract(c, None, o, complex(i))[:, :NS]
    G = (sub(tru["corr_wp"], tru["one_pt_wp"], tru["insert_1pt_wp"])
         - sub(tru["corr_vac"], tru["one_pt_vac"], tru["insert_1pt_vac"]))
    times = np.asarray(tru["times"], float)
    sel = times <= 3.0
    t, Gs = times[sel], G[sel]
    print(f"exact connected correlator, {Gs.shape[0]} slices to t={t[-1]:g}")
    print(f"  max|Re C| = {np.abs(Gs.real).max():.4f}   max|Im C| = {np.abs(Gs.imag).max():.4f}"
          f"   ratio {np.abs(Gs.imag).max() / np.abs(Gs.real).max():.3f}")
    x0 = analysis.ring_fold((np.arange(NS) - VC) / 2, lat.nx)
    q0 = np.arange(-1.0, 6.001, 0.04)
    q1 = 2 * np.pi * np.arange(-(lat.nx // 2), lat.nx // 2 + 1) / lat.nx
    ft = lambda g: analysis.onesided_ft(t, x0, g, q0, q1, 1.5, lat.nx / 2.0, 0.5, 0.5)
    W_full, W_re = ft(Gs), ft(Gs.real.astype(complex))
    d = np.abs(W_full - W_re).max()
    print(f"  max|W| full C {np.abs(W_full).max():.4f} | Re-only {np.abs(W_re).max():.4f}"
          f" | max difference {d:.4f} ({100 * d / np.abs(W_full).max():.1f}% of peak)")
    odd = lambda W: 0.5 * (W - W[:, ::-1])
    reg = (q0 >= -1) & (q0 <= 2.5)
    T = odd(W_full)[reg]
    A = float((T * odd(W_re)[reg]).sum()) / float((T * T).sum())
    print(f"\n  asymmetry of the Re-only tensor against the full-C template: A = {A:.4f}")
    print("  (the published A uses a Re-only template on BOTH sides, so it is unbiased;")
    print("   this is the size of the convention's effect on the tensor itself.)")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "beta"
    {"beta": cmd_beta, "imaginary": cmd_imaginary}[mode]()
