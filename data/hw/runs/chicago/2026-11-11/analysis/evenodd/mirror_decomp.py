"""Mirror and physics pubs at t = 2, 4, 6: per site <Z_n> and the connected
ancilla-site correlator c_n = <X Z_n> - <X><Z_n>, relative to the SAME run's
t = 0 reference (so prep fidelity divides out) for the mirror, and to the
ideal at t for the physics."""
import glob, os, sys
import numpy as np
sys.path.insert(0, os.getcwd())
import htq_hw.analyze as A
from htq_hw.campaign import pub_name, split_prefix
from htq_hw.model import Lattice
np.set_printoptions(linewidth=250, precision=2, suppress=True)
CARD = "prod_k1.26_s0.75_ns50"
lat = Lattice(50); zs = [lat.site_qubit(v) for v in range(50)]
g = A.IdealGrid(f"htq_hw/cards/{CARD}/ideal_j0.npz", expect_ns=50)
idb = g.id_b()
def ideal(t):
    X = float(g.scalar("X", t)); z = 2 * (idb - g.vec("B", t)); XZ = 2 * (idb * X - g.vec("XB", t))
    return X, z, XZ - X * z
RUNS = {"phoenix": ("data/hw/runs/phoenix/2026-10-06/raw/htq_bits_*.npz", "prod-bridge"),
        "chi 6Nov": ("data/hw/runs/chicago/2026-11-06/raw/main_rzz/htq_bits_*.npz", "prod-bridge"),
        "chi10 ctl": ("data/hw/runs/chicago/2026-11-10/raw/mirror_diag_control2/htq_bits_*.npz", "mirror-diag"),
        "chi10 twl": ("data/hw/runs/chicago/2026-11-10/raw/mirror_diag_twirl/htq_bits_*.npz", "mirror-diag"),
        "chi11 twl": ("data/hw/runs/chicago/2026-11-11/raw/main_rzz_twirl/htq_bits_*.npz", "prod-bridge")}
sites = list(range(19, 30))
def stats(b):
    s = A.signs_from_bits(b); x = s[:, lat.ancilla]; zz = s[:, zs]
    X = x.mean(); Z = zz.mean(0); c = (x[:, None] * zz).mean(0) - X * Z
    return X, Z, c
Xi0, zi0, ci0 = ideal(0.0)
print("sites            ", np.array(sites))
print("ideal c_n(0)     ", ci0[sites]); print("ideal Z_n(0)     ", zi0[sites])
for lab, (pat, preset) in RUNS.items():
    pubs = {}
    for p in sorted(glob.glob(pat)):
        z = np.load(p, allow_pickle=True)
        for k in z.files:
            if z[k].dtype != np.uint8 or z[k].ndim != 2: continue
            pre, bare = split_prefix(k)
            if A.card_of(pre) == CARD and pre.startswith(preset) and bare not in pubs:
                pubs[bare] = (p, k)
    ld = lambda n: np.load(pubs[n][0], allow_pickle=True)[pubs[n][1]]
    X0, Z0, c0 = stats(ld(pub_name("j0", 0.0, False, "Z")))
    print(f"\n=== {lab}: t=0 c/ideal {(c0/ci0)[sites]}")
    for t in (2.0, 4.0, 6.0):
        mn, pn = pub_name("j0", t, True, "Z"), pub_name("j0", t, False, "Z")
        if mn not in pubs: continue
        Xm, Zm, cm = stats(ld(mn)); Xp, Zp, cp = stats(ld(pn))
        Xi, zi, ci = ideal(t)
        print(f"  t={t:.0f} mirror <X> {Xm:+.3f} (t=0 {X0:+.3f}); physics <X> {Xp:+.3f} (ideal {Xi:+.3f})")
        print(f"     mirror c_n / ideal c_n(0)  {(cm/ci0)[sites]}")
        print(f"     mirror c_n / own t=0 c_n   {(cm/c0)[sites]}")
        print(f"     mirror Z_n / own t=0 Z_n   {(Zm/Z0)[sites]}")
        print(f"     mirror Z_n - t=0 Z_n       {(Zm-Z0)[sites]}")
        print(f"     physics c_n hw             {cp[sites]}")
        print(f"     physics c_n ideal          {ci[sites]}")
