"""Does the mirror kappa measured on weight-1 (J^0) probes calibrate the
weight-3 (J^1) probes in the same circuit?

This decides whether the W^{01}/W^{11} pubs need their own mirror family in
the phoenix manifest, or can share the J^0 mirror.  A single Pauli
trajectory is a deterministic Pauli frame (kappa = +-1); only the ensemble
average damps, so this runs at N_s = 16 where hundreds of trajectories are
affordable.  The ratio is what transfers to N_s = 50: it is set by the
operator weight and the local gate structure, not by the volume, and it is
trajectory-correlated between probe types measured in the SAME circuit, so
it converges much faster than either kappa separately.

  PYTHONPATH=. .venv/bin/python scripts/kappa_ratio.py [glob-dir] [ns]
"""

import glob
import sys

import numpy as np

DIR = sys.argv[1] if len(sys.argv) > 1 else "data/work/kappa16"
NS = int(sys.argv[2]) if len(sys.argv) > 2 else 16
GUARD = 0.02
NBOOT = 2000
ID_B = np.array([(-1) ** v / 2 for v in range(NS)])


def sx_j0(d, pr=""):
    XB = np.array([d[f"{pr}XB_{v}"] for v in range(NS)]).T
    return XB - ID_B[None, :] * d[f"{pr}X"][:, None]


def sx_t(d, k, pr=""):
    return np.array([d[f"{pr}XT{k}_{b}"] for b in range(NS)]).T


PROBES = {"J0": lambda d, pr="": sx_j0(d, pr),
          "T1": lambda d, pr="": sx_t(d, 1, pr),
          "T2": lambda d, pr="": sx_t(d, 2, pr)}


def load(fam):
    idl = np.load(f"{DIR}/k16_{fam}_ideal.npz", allow_pickle=True)
    ds = []
    for f in sorted(glob.glob(f"{DIR}/k16_{fam}_noise*.npz")):
        try:
            d = np.load(f, allow_pickle=True)
            _ = d["m_X"]          # completeness check (mirror rows present)
            ds.append(d)
        except Exception:
            pass
    return idl, ds


rng = np.random.default_rng(0)
for fam in ("j0", "j1p1", "j1p2"):
    try:
        idl, ds = load(fam)
    except FileNotFoundError:
        continue
    if not ds:
        continue
    times = np.asarray(idl["times"], float)
    # per-trajectory, per-time site-mean of sx_mirror / sx_ideal(t=0)
    per = {}
    for name, fn in PROBES.items():
        ideal = fn(idl)
        vis = np.abs(ideal[0]) > GUARD
        if vis.sum() == 0:
            continue
        per[name] = np.array([(fn(d, "m_")[:, vis] / ideal[0][vis]).mean(axis=1) for d in ds])
    n = len(ds)
    print(f"\n== family {fam}: {n} trajectories, {NS} sites, guard {GUARD}")
    print(f"{'t':>5} " + " ".join(f"{k:>16}" for k in per) + "   " +
          " ".join(f"{k+'/J0':>14}" for k in per if k != "J0"))
    for i, t in enumerate(times):
        cells, ratios = [], []
        boot = rng.integers(0, n, size=(NBOOT, n))
        for name in per:
            v = per[name][:, i]
            cells.append(f"{v.mean():>9.4f}+-{v[boot].mean(axis=1).std():<5.3f}")
        for name in per:
            if name == "J0":
                continue
            num, den = per[name][:, i], per["J0"][:, i]
            r = num.mean() / den.mean()
            rb = num[boot].mean(axis=1) / den[boot].mean(axis=1)   # correlated: same trajectories
            ratios.append(f"{r:>7.3f}+-{rb.std():<5.3f}")
        print(f"{t:>5.1f} " + " ".join(cells) + "   " + " ".join(ratios))
