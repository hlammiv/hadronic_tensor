"""Stage B(ii): CGK-style two-packet collision at CGK-A couplings, ns=30,
with the Stage-A certified wavepacket blocks.  Records the site-resolved
charge-density profile <J0(n,t)> (and, via the production machinery, the
C(t,x) correlator as a by-product) for the delay extraction.

  PYTHONPATH=. python scripts/stageB_collision_cgkA.py <j>   # j = 2 or 3
j=3: K=+-1.2566 (on-resonance; predicted Wigner delay ~ +4.2)
j=2: K=+-0.8378 (off-resonance null; predicted ~ +0.2)

  PYTHONPATH=. python scripts/stageB_collision_cgkA.py <j> [ns] [tmax]
"""
import sys
import time

import numpy as np

from htensor import Z2Lattice, stateprep, wavepacket, backends
from htensor import currents as cur

M0, G2, ETA = 0.1, 0.4, 1.0            # CGK-A
J = int(sys.argv[1])
NS = int(sys.argv[2]) if len(sys.argv) > 2 else 30
TMAX = float(sys.argv[3]) if len(sys.argv) > 3 else 44.0
TIMES = np.arange(0.0, TMAX + 1e-9, 1.0)
DT_TARGET = 0.25
TRUNC = 1e-8
KTAG = {3: ("1.26", "-1.26"), 2: ("0.84", "-0.84")}[J]
C1, C2 = 8, 22                          # even staggered sites, 14 apart

t0 = time.time()
lat = Z2Lattice(NS, pbc=True)
zv = np.load("data/cgkA_vacuum_thetas.npz")
prep = stateprep.vacuum_ansatz(lat, zv["thetas"])
for ktag, center in ((KTAG[0], C1), (KTAG[1], C2)):
    z = np.load(f"data/wpCGKA_params_k{ktag}_L3.npz")
    params = wavepacket.params_from_vector(z["vec"], list(z["offsets"]),
                                           int(z["L"]))
    prep.compose(wavepacket.block_circuit(lat, center, params), inplace=True)
    print(f"[{time.time()-t0:6.0f}s] packet K={ktag} at site {center}",
          flush=True)

vc = (C1 + C2) // 2 + 1                 # expected collision site (even)
probes = [cur.charge_density(lat, v) for v in range(NS)]
insert = cur.charge_density(lat, vc)

# prep the two-packet state ONCE with a bond cap (the internal uncapped
# path stalls on the composed circuit), then reuse across slices --
# the synthesis-ladder pattern
mps, perm = backends.prepare_state_mps(lat, prep, NS, cap=512, trunc=1e-10)
print(f"[{time.time()-t0:6.0f}s] two-packet MPS prepared (cap=512)",
      flush=True)

rows = []
for t in TIMES:
    d = backends.hadamard_correlator_aer(
        lat, None, insert, probes, M0, G2, ETA, [t], dt_target=DT_TARGET,
        method="matrix_product_state", mps_trunc=TRUNC,
        initial_mps=mps, initial_perm=perm,
        stationary_1pt=False)
    rows.append(d)
    dens = d.probe_expect[0].real
    print(f"[{time.time()-t0:6.0f}s] t={t:4.0f} done  "
          f"max|dJ0| = {np.abs(dens - dens.mean()).max():.4f}", flush=True)

corr = np.concatenate([d.correlator for d in rows], axis=0)
dens = np.concatenate([d.probe_expect for d in rows], axis=0)
np.savez(f"data/stageB_collision_j{J}_ns{NS}.npz", times=TIMES,
         density=dens, corr=corr, insert_1pt=rows[0].insert_expect,
         ns=NS, c1=C1, c2=C2, j=J, m0=M0, g2=G2, eta=ETA,
         dt_target=DT_TARGET, trunc=TRUNC)
print(f"[{time.time()-t0:6.0f}s] saved data/stageB_collision_j{J}_ns{NS}.npz",
      flush=True)
