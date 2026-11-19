"""Per-instance mirror kappa for the 10 Nov pre-twirled Chicago run.

The merged raw file concatenates the K = 4 twirl instances in instance order
(htq_hw.campaign.fetch), 15,000 shots each, so instance i is rows
[15000 i, 15000 (i+1)).  Kappa per instance tells whether the improvement
over the back-to-back control is an averaging effect (instances scatter
around the control value) or a per-circuit effect (every instance beats the
control).  Same estimator as mirror_diag_compare.py (de-biased ancilla).
"""
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "2026-11-08", "analysis"))
import htq_hw.analyze as A                      # noqa: E402
from htq_hw.campaign import pub_name            # noqa: E402
from htq_hw.model import Lattice                # noqa: E402
import mirror_diag_compare as M                 # noqa: E402

CARD = "prod_k1.26_s0.75_ns50"
RUN = "data/hw/runs/chicago/2026-11-10/raw"
grid = A.IdealGrid(f"htq_hw/cards/{CARD}/ideal_j0.npz", expect_ns=50)
lat = Lattice(50)
x_ideal = float(grid.scalar("X", 0.0))
sx_i0 = grid.s_ideal("X", 0.0)
names = {pub_name("j0", t, m, "Z") for t in M.TIMES for m in (False, True)}

ctrl = M.find_pubs(sorted(glob.glob(f"{RUN}/mirror_diag_control2/htq_bits_*.npz")), CARD, names)
twl = M.find_pubs(sorted(glob.glob(f"{RUN}/mirror_diag_twirl/htq_bits_*.npz")), CARD, names)
z = np.load(twl[pub_name("j0", 2.0, True, "Z")][0], allow_pickle=True)
K = int(z["twirl_instances"]); print("instances", K, list(z["twirl_jobs"]))

groups = dict(M.GROUPS)
rows = []
for t in M.TIMES:
    mn, pn = pub_name("j0", t, True, "Z"), pub_name("j0", t, False, "Z")
    bm, bp = M.load(twl, mn), M.load(twl, pn)
    cm, cp = M.load(ctrl, mn), M.load(ctrl, pn)
    n = len(bm) // K
    sx_it = grid.s_ideal("X", t)
    ok = np.abs(sx_it) > 0.02
    def kap_phys(bits_m, bits_p):
        sxm = M.signal(bits_m, lat, x_ideal, True)[0]
        sxp = M.signal(bits_p, lat, x_ideal, True)[0]
        return sxm / sx_i0, np.where(ok, sxp / np.where(ok, sx_it, 1.0), np.nan)
    kc, pc = kap_phys(cm, cp)
    kall, pall = kap_phys(bm, bp)
    inst = [kap_phys(bm[i * n:(i + 1) * n], bp[i * n:(i + 1) * n]) for i in range(K)]
    # sanity: the mean over instances of the per-instance signal equals the pooled one
    print(f"\nt={t:.0f}: centre kappa  control2 {kc[24]:+.3f}  twirl pooled {kall[24]:+.3f}  "
          f"instances {[f'{k[24]:+.3f}' for k, _ in inst]}  (+-{0.5 / np.sqrt(n) / abs(sx_i0[24]):.3f} each)")
    print(f"      centre physics/ideal(t)  control2 {pc[24]:+.3f}  twirl pooled {pall[24]:+.3f}  "
          f"instances {[f'{p[24]:+.3f}' for _, p in inst]}")
    for g, sites in groups.items():
        ki = [np.nanmedian(k[sites]) for k, _ in inst]
        print(f"      {g:<18} kappa median  control2 {np.nanmedian(kc[sites]):+.3f}  pooled {np.nanmedian(kall[sites]):+.3f}  "
              f"instances {['%+.3f' % v for v in ki]}  spread {np.std(ki):.3f}")
    print(f"      site 25  control2 {kc[25]:+.3f}  pooled {kall[25]:+.3f}  instances {[f'{k[25]:+.3f}' for k, _ in inst]}")
    print(f"      site 27  control2 {kc[27]:+.3f}  pooled {kall[27]:+.3f}  instances {[f'{k[27]:+.3f}' for k, _ in inst]}")
    # how many of the 50 sites pass the production kappa floor 0.10
    print(f"      sites with kappa > 0.10: control2 {(kc > 0.10).sum()}/50, twirl pooled {(kall > 0.10).sum()}/50, "
          f"> 0.30: {(kc > 0.30).sum()} / {(kall > 0.30).sum()}; median over all sites {np.median(kc):+.3f} / {np.median(kall):+.3f}")
    # physics damping over the sites with a usable ideal signal
    print(f"      physics/ideal(t) median over usable sites: control2 {np.nanmedian(pc):+.3f}  twirl {np.nanmedian(pall):+.3f}")
