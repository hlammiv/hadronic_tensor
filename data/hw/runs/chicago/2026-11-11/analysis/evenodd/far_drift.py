"""Vacuum-region sites (|n-24| >= 8): <Z_n> should be static in both the
mirror and the physics circuit.  Survival Z(t)/Z(0) by site parity (even
sites sit in |0>, odd in |1>), per run, mirror and physics."""
import glob, os, sys, pickle
import numpy as np
sys.path.insert(0, os.getcwd())
import htq_hw.analyze as A
from htq_hw.campaign import pub_name, split_prefix
from htq_hw.model import Lattice
np.set_printoptions(linewidth=250, precision=2, suppress=True)
CARD = "prod_k1.26_s0.75_ns50"
lat = Lattice(50); zs = [lat.site_qubit(v) for v in range(50)]
RUNS = {"phoenix": ("data/hw/runs/phoenix/2026-10-06/raw/htq_bits_*.npz", "prod-bridge"),
        "chi 6Nov": ("data/hw/runs/chicago/2026-11-06/raw/main_rzz/htq_bits_*.npz", "prod-bridge"),
        "chi11 twl": ("data/hw/runs/chicago/2026-11-11/raw/main_rzz_twirl/htq_bits_*.npz", "prod-bridge")}
far = np.array([n for n in range(50) if abs(n - 24) >= 8]); ev, od = far[far % 2 == 0], far[far % 2 == 1]
times = [0.5 * k for k in range(1, 13)]
out = {}
for lab, (pat, preset) in RUNS.items():
    pubs = {}
    for p in sorted(glob.glob(pat)):
        z = np.load(p, allow_pickle=True)
        for k in z.files:
            if z[k].dtype != np.uint8 or z[k].ndim != 2: continue
            pre, bare = split_prefix(k)
            if A.card_of(pre) == CARD and pre.startswith(preset) and bare not in pubs: pubs[bare] = (p, k)
    Zof = lambda n: A.signs_from_bits(np.load(pubs[n][0], allow_pickle=True)[pubs[n][1]])[:, zs].mean(0)
    Z0 = Zof(pub_name("j0", 0.0, False, "Z")); out[lab] = {"Z0": Z0}
    print(f"\n=== {lab}: t=0 |Z| far even {np.median(Z0[ev]):+.3f} odd {np.median(Z0[od]):+.3f}")
    print("   t   | mirror Z/Z0 even  odd | physics Z/Z0 even  odd | plain Z/Z0 even odd")
    for t in times:
        row = [f"  {t:4.1f} |"]
        for kind in ("m", "p", "plain"):
            n = pub_name("j0", t, kind == "m", "Z") if kind != "plain" else pub_name("plain", t, False, "Z")
            if n in pubs:
                Z = Zof(n); out[lab][(kind, t)] = Z
                row.append(f"   {np.median((Z/Z0)[ev]):5.2f} {np.median((Z/Z0)[od]):5.2f}   |")
            else: row.append("      -     -   |")
        print("".join(row))
pickle.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "far_drift.pkl"), "wb"))
