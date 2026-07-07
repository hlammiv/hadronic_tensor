"""MPS/Trotter error-budget table (task 14) from the sysscan campaign.

Each config reran the rest-packet J0 correlator grid (t = 0..6) with one
knob moved off the production anchor (cap=None, trunc=1e-8, dt=0.5).
Reported per config: rms and max deviation of the CONNECTED correlator
from the anchor, normalized to the anchor's max amplitude -- the numbers
for the paper's uncertainty table.

  PYTHONPATH=. .venv/bin/python scripts/error_budget.py
"""

import numpy as np

CONFIGS = [
    ("chiscanA_tr1e6",     "truncation 1e-6 (10x looser)"),
    ("chiscanB_tr1e10",    "truncation 1e-10 (100x tighter)"),
    ("chiscanB_cap64",     "bond cap 64"),
    ("chiscanA_cap256",    "bond cap 256"),
    ("chiscanA_dt025_rest", "Trotter dt 0.25 (half step)"),
]


def connected(path):
    d = np.load(path)
    corr, one, ins = d["corr"], d["one_pt"], complex(d["insert_1pt"])
    return corr - one * ins, d["times"]


G0, t0 = connected("data/sysscan_chiscanB_anchor.npz")
scale = np.abs(G0).max()
print(f"anchor: |G|_max = {scale:.4f} on t = {t0[0]:.1f}..{t0[-1]:.1f} "
      f"({G0.shape[1]} probes)")
print(f"\n{'config':<24s} {'rms/|G|max':>10s} {'max/|G|max':>10s}")
rows = []
for tag, label in CONFIGS:
    G, t = connected(f"data/sysscan_{tag}.npz")
    assert np.allclose(t, t0)
    dev = G - G0
    r = float(np.sqrt(np.mean(np.abs(dev) ** 2)) / scale)
    m = float(np.abs(dev).max() / scale)
    rows.append((tag, r, m))
    print(f"{label:<24s} {r:10.5f} {m:10.5f}")

np.savez("data/error_budget.npz",
         rows=np.array([(r, m) for _, r, m in rows]),
         tags=[t for t, _, _ in rows], scale=scale)
print("\nsaved data/error_budget.npz")
