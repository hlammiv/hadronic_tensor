"""Vacuum-polarization W^{00}(q0,q1) at a published collision study's couplings.
Usage: PYTHONPATH=. python scripts/rt_wvac_couplings.py <tag> <NS>
  tag in {dhk, cgkel, cgkinA, cgkinB}: couplings read from data/deep_levels_<tag>_ns8.npz
Same protocol as scripts/run_w00_vacuum_ns50.py (t = 0..6 step 0.5, dt 0.1, MPS trunc 1e-8).
Output: data/rt_wvac_<tag>_ns<NS>.npz
"""
import sys, time
import numpy as np
from htensor import Z2Lattice, stateprep, backends
from htensor import currents as cur
tag, NS = sys.argv[1], int(sys.argv[2])
TMAX = float(sys.argv[3]) if len(sys.argv) > 3 else 6.0
CHI = int(sys.argv[5]) if len(sys.argv) > 5 else None   # optional MPS bond cap
lv = np.load(f"data/deep_levels_{tag}_ns8.npz")
M0, G2, ETA = float(lv["m0"]), float(lv["g2"]), float(lv["eta"])
TIMES = np.arange(0.0, TMAX + 0.01, 0.5); DT_TARGET = 0.1; TRUNC = 1e-8
out_path = f"data/rt_wvac_{tag}_ns{NS}.npz" if TMAX == 6.0 else f"data/rt_wvac_{tag}_ns{NS}_T{int(TMAX)}.npz"
if CHI: out_path = out_path.replace(".npz", f"_chi{CHI}.npz")
t0 = time.time()
print(f"tag={tag} NS={NS} couplings (m0,g2,eta)=({M0},{G2},{ETA})", flush=True)
NL = int(sys.argv[4]) if len(sys.argv) > 4 else 2
cands = {ref: stateprep.optimize_vacuum(Z2Lattice(6, pbc=True), M0, G2, ETA, n_layers=NL, restarts=2, link_ref=ref) for ref in ("+", "-")}
REF = max(cands, key=lambda r: cands[r]["fidelity"]); thetas = cands[REF]["thetas"]
print(f"vacuum: link_ref={REF} n_layers={NL} fidelity(ns=6)={cands[REF]['fidelity']:.6f} (other ref {cands[min(cands, key=lambda r: cands[r]['fidelity'])]['fidelity']:.2e})", flush=True)
print(f"[{time.time()-t0:6.0f}s] vacuum angles from ns=6 optimization", flush=True)
lat = Z2Lattice(NS, pbc=True)
prep = stateprep.vacuum_ansatz(lat, thetas, link_ref=REF)
vc = NS // 2
probes = [cur.charge_density(lat, v) for v in range(NS)]
insert = cur.charge_density(lat, vc)
rows = []
for t in TIMES:
    d = backends.hadamard_correlator_aer(lat, prep, insert, probes, M0, G2, ETA, [t],
                                         dt_target=DT_TARGET, method="matrix_product_state",
                                         mps_trunc=TRUNC, mps_max_bond=CHI, max_threads=8, stationary_1pt=False if t == 0.0 else True)
    rows.append(d)
    print(f"[{time.time()-t0:6.0f}s] t={t:4.1f} done  C(t,x=0) = {d.correlator[0, vc]:.6f}", flush=True)
corr = np.concatenate([d.correlator for d in rows], axis=0)
probe_1pt = np.concatenate([d.probe_expect for d in rows], axis=0)
np.savez(out_path, times=TIMES, corr=corr, probe_1pt=probe_1pt, insert_1pt=rows[0].insert_expect,
         ns=NS, vc=vc, m0=M0, g2=G2, eta=ETA, dt_target=DT_TARGET, trunc=TRUNC, thetas=thetas, tag=tag, link_ref=REF, n_layers=NL, chi=CHI if CHI else 0)
print(f"[{time.time()-t0:6.0f}s] saved {out_path}", flush=True)
