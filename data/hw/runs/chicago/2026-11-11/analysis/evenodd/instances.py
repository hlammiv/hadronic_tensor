"""Twirled runs: split each merged pub into its 4 instances and look at the
t = 0 prepared state (one-points and connected correlators near the
insertion) and at the physics correlator, instance by instance."""
import glob, os, sys
import numpy as np
sys.path.insert(0, os.getcwd())
import htq_hw.analyze as A
from htq_hw.campaign import pub_name, split_prefix
from htq_hw.model import Lattice
np.set_printoptions(linewidth=250, precision=2, suppress=True)
CARD = "prod_k1.26_s0.75_ns50"; lat = Lattice(50); zs = [lat.site_qubit(v) for v in range(50)]
g = A.IdealGrid(f"htq_hw/cards/{CARD}/ideal_j0.npz", expect_ns=50); idb = g.id_b()
def ideal(t):
    X = float(g.scalar("X", t)); z = 2 * (idb - g.vec("B", t)); return X, z, 2 * (idb * X - g.vec("XB", t)) - X * z
sites = list(range(20, 29))
def stats(b):
    s = A.signs_from_bits(b); x = s[:, lat.ancilla]; zz = s[:, zs]
    X = x.mean(); Z = zz.mean(0); return X, Z, (x[:, None] * zz).mean(0) - X * Z
pubs = {}
for p in sorted(glob.glob("data/hw/runs/chicago/2026-11-11/raw/main_rzz_twirl/htq_bits_*.npz")):
    z = np.load(p, allow_pickle=True)
    for k in z.files:
        if z[k].dtype == np.uint8 and z[k].ndim == 2 and A.card_of(split_prefix(k)[0]) == CARD: pubs[k] = p
def show(key, t):
    b = np.load(pubs[key], allow_pickle=True)[key]; n = len(b) // 4
    Xi, zi, ci = ideal(t)
    print(f"\n{key}  (N {len(b)})")
    print(f"   ideal       <X> {Xi:+.3f}  Z {zi[sites]}  c {ci[sites]}")
    Zs, cs = [], []
    for i in range(4):
        X, Z, c = stats(b[i * n:(i + 1) * n]); Zs.append(Z); cs.append(c)
        print(f"   instance {i}  <X> {X:+.3f}  Z {Z[sites]}  c {c[sites]}")
    Zs, cs = np.array(Zs), np.array(cs)
    se = 1 / np.sqrt(n)
    print(f"   spread over instances: Z {Zs.std(0)[sites]}  c {cs.std(0)[sites]}   (shot noise per instance ~{se:.3f})")
P = "prod_k1.26_s0.75_ns50:"
show("prod-bridge." + P + pub_name("j0", 0.0, False, "Z"), 0.0)
show("prod-quarter." + P + pub_name("j0", 0.0, False, "Z"), 0.0)
for t in (2.0, 4.0):
    show("prod-bridge." + P + pub_name("j0", t, True, "Z"), 0.0)
    show("prod-bridge." + P + pub_name("j0", t, False, "Z"), t)
