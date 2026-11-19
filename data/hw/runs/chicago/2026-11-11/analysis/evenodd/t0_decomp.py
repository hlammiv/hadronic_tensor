"""t = 0 reference pub, per site: one-point <Z_n>, ancilla <X>, and the truly
connected ancilla-site correlator c_n = <X Z_n> - <X><Z_n>, hardware vs ideal."""
import glob, json, os, sys
import numpy as np
sys.path.insert(0, os.getcwd())
import htq_hw.analyze as A
from htq_hw.campaign import pub_name, split_prefix
from htq_hw.model import Lattice
np.set_printoptions(linewidth=250, precision=3, suppress=True)
CARD = sys.argv[1] if len(sys.argv) > 1 else "prod_k1.26_s0.75_ns50"
lat = Lattice(50); zs = [lat.site_qubit(v) for v in range(50)]
g = A.IdealGrid(f"htq_hw/cards/{CARD}/ideal_j0.npz", expect_ns=50)
idb = g.id_b(); Xi = float(g.scalar("X", 0.0))
Bi = g.vec("B", 0.0); zi = 2 * (idb - Bi)                 # ideal <Z_n> (ancilla-averaged)
XBi = g.vec("XB", 0.0); XZi = 2 * (idb * Xi - XBi)         # ideal <X Z_n>
ci = XZi - Xi * zi
RUNS = {"phoenix": "data/hw/runs/phoenix/2026-10-06/raw/htq_bits_*.npz",
        "chi 6Nov": "data/hw/runs/chicago/2026-11-06/raw/main_rzz/htq_bits_*.npz",
        "chi 8Nov": "data/hw/runs/chicago/2026-11-08/raw/mirror_diag_control/htq_bits_*.npz",
        "chi10 ctl": "data/hw/runs/chicago/2026-11-10/raw/mirror_diag_control2/htq_bits_*.npz",
        "chi10 twl": "data/hw/runs/chicago/2026-11-10/raw/mirror_diag_twirl/htq_bits_*.npz",
        "chi11 twl": "data/hw/runs/chicago/2026-11-11/raw/main_rzz_twirl/htq_bits_*.npz"}
want = pub_name("j0", 0.0, False, "Z")
sites = list(range(16, 33))
print("card", CARD, " ideal <X_anc> =", round(Xi, 4))
print("site        ", np.array(sites))
print("ideal <Z_n> ", zi[sites])
print("ideal c_n   ", ci[sites])
print("ideal XZ    ", XZi[sites])
res = {}
for lab, pat in RUNS.items():
    arrs = []
    for p in sorted(glob.glob(pat)):
        z = np.load(p, allow_pickle=True)
        for k in z.files:
            if z[k].dtype != np.uint8 or z[k].ndim != 2: continue
            pre, bare = split_prefix(k)
            if A.card_of(pre) == CARD and bare == want:
                arrs.append((k, z[k]))
    if not arrs:
        print(lab, "no t=0 pub"); continue
    for k, b in arrs:
        s = A.signs_from_bits(b); x = s[:, lat.ancilla]; zz = s[:, zs]
        N = len(s); X = x.mean(); Z = zz.mean(0); XZ = (x[:, None] * zz).mean(0); c = XZ - X * Z
        ce = (x[:, None] * zz).std(0) / np.sqrt(N)
        tag = f"{lab} [{k.split('.')[0]}]"
        print(f"\n{tag}: N {N}, <X_anc> {X:+.4f}")
        print("  <Z_n> hw     ", Z[sites])
        print("  Z hw/ideal   ", (Z / zi)[sites])
        print("  c_n hw       ", c[sites])
        print("  c hw/ideal   ", np.where(np.abs(ci) > 0.01, c / ci, np.nan)[sites], " (err ~", round(float(np.median(ce)), 4), "abs)")
        res[tag] = dict(Z=Z, c=c, X=X)
        far = np.array([n for n in range(50) if abs(n - 24) >= 8])
        ev, od = far[far % 2 == 0], far[far % 2 == 1]
        print(f"  far sites Z hw/ideal: even median {np.median((Z/zi)[ev]):.3f}, odd median {np.median((Z/zi)[od]):.3f}; "
              f"ideal Z even {np.median(zi[ev]):+.3f} odd {np.median(zi[od]):+.3f}")
