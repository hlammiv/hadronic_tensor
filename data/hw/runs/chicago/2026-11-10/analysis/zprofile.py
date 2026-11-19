"""Mirror return of the matter Z profile, 10 Nov control2 vs twirl (same
layout, 2.5 min apart), with the 8 Nov control and 6 Nov production for
scale.  RMS over the 50 sites of <Z_n>(mirror t) - <Z_n>(t=0 ref), beside
<Z_n>(physics t) - <Z_n>(t=0 ref).  Independent of the ancilla."""
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
lat = Lattice(50)
zs = [lat.site_qubit(v) for v in range(50)]
names = {pub_name("j0", 0.0, False, "Z")} | {pub_name("j0", t, m, "Z") for t in M.TIMES for m in (False, True)}
RUNS = {"production 6 Nov": "data/hw/runs/chicago/2026-11-06/raw/main_rzz/htq_bits_*.npz",
        "control 8 Nov": "data/hw/runs/chicago/2026-11-08/raw/mirror_diag_control/htq_bits_*.npz",
        "control2 10 Nov": "data/hw/runs/chicago/2026-11-10/raw/mirror_diag_control2/htq_bits_*.npz",
        "twirl 10 Nov": "data/hw/runs/chicago/2026-11-10/raw/mirror_diag_twirl/htq_bits_*.npz"}
print(f"{'run':<18}{'t':>3}{'RMS mirror-ref':>16}{'RMS physics-ref':>17}{'centre |dZ| mirror':>20}{'odd |dZ| mirror':>17}")
for label, pat in RUNS.items():
    found = M.find_pubs(sorted(glob.glob(pat)), CARD, names)
    ref = A.signs_from_bits(M.load(found, pub_name("j0", 0.0, False, "Z")))[:, zs].mean(0)
    for t in M.TIMES:
        zm = A.signs_from_bits(M.load(found, pub_name("j0", t, True, "Z")))[:, zs].mean(0)
        zp = A.signs_from_bits(M.load(found, pub_name("j0", t, False, "Z")))[:, zs].mean(0)
        dm, dp = zm - ref, zp - ref
        odd = [21, 23, 25, 27]
        print(f"{label:<18}{t:>3.0f}{np.sqrt(np.mean(dm**2)):>16.3f}{np.sqrt(np.mean(dp**2)):>17.3f}"
              f"{abs(dm[24]):>20.3f}{np.sqrt(np.mean(dm[odd]**2)):>17.3f}")
