"""Gauss-law syndromes G_n = (-1)^n z_n x_{l-} x_{l+} at t = 0, per site, on
every dataset: the prepared state should give +1 everywhere."""
import glob, os, sys
import numpy as np
sys.path.insert(0, os.getcwd())
import htq_hw.analyze as A
from htq_hw.campaign import pub_name, split_prefix
from htq_hw.model import Lattice
np.set_printoptions(linewidth=250, precision=2, suppress=True)
CARD = sys.argv[1]; lat = Lattice(50)
RUNS = {"phoenix": "data/hw/runs/phoenix/2026-10-06/raw/htq_bits_*.npz",
        "chi 6Nov": "data/hw/runs/chicago/2026-11-06/raw/main_rzz/htq_bits_*.npz",
        "chi 8Nov": "data/hw/runs/chicago/2026-11-08/raw/mirror_diag_control/htq_bits_*.npz",
        "chi10 ctl": "data/hw/runs/chicago/2026-11-10/raw/mirror_diag_control2/htq_bits_*.npz",
        "chi10 twl": "data/hw/runs/chicago/2026-11-10/raw/mirror_diag_twirl/htq_bits_*.npz",
        "chi11 twl": "data/hw/runs/chicago/2026-11-11/raw/main_rzz_twirl/htq_bits_*.npz"}
want = pub_name("j0", 0.0, False, "Z"); sites = list(range(14, 35))
print(CARD, "sites", np.array(sites))
for lab, pat in RUNS.items():
    for p in sorted(glob.glob(pat)):
        z = np.load(p, allow_pickle=True); done = False
        for k in z.files:
            if z[k].dtype != np.uint8 or z[k].ndim != 2: continue
            pre, bare = split_prefix(k)
            if A.card_of(pre) == CARD and bare == want:
                s = A.signs_from_bits(z[k])
                zz = s[:, [lat.site_qubit(v) for v in range(50)]]; xl = s[:, [lat.link_qubit(n) for n in range(50)]]
                G = np.array([(-1) ** n * zz[:, n] * xl[:, (n - 1) % 50] * xl[:, n] for n in range(50)]).mean(1)
                XL = xl.mean(0)
                far = [n for n in range(50) if abs(n - 24) >= 8]
                print(f"{lab:10s} G_n   {G[sites]}  far median {np.median(G[far]):.2f}")
                print(f"{'':10s} <X_l> {XL[sites]}  (link n joins sites n, n+1)")
                done = True; break
        if done: break
