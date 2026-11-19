import glob, json, os, sys
import numpy as np
sys.path.insert(0, os.getcwd())
import htq_hw.analyze as A
from htq_hw.campaign import pub_name, split_prefix
from htq_hw.model import Lattice
np.set_printoptions(linewidth=250, precision=2, suppress=True)
CARD = "prod_k1.26_s0.75_ns50"; lat = Lattice(50); zs = [lat.site_qubit(v) for v in range(50)]
def props(path):
    d = json.load(open(path)); T1, T2 = {}, {}
    for q, items in enumerate(d["qubits"]):
        for it in items:
            if it["name"] == "T1": T1[q] = it["value"] * (1e6 if it.get("unit") == "s" else 1)
            if it["name"] == "T2": T2[q] = it["value"] * (1e6 if it.get("unit") == "s" else 1)
    return T1, T2
RUNS = {"chi 8Nov ctl (untwirled)": ("data/hw/runs/chicago/2026-11-08", "mirror_diag_control"),
        "chi10 ctl (untwirled)": ("data/hw/runs/chicago/2026-11-10", "mirror_diag_control2"),
        "chi10 twl": ("data/hw/runs/chicago/2026-11-10", "mirror_diag_twirl")}
for lab, (run, sub) in RUNS.items():
    cal = sorted(glob.glob(run + "/calibration/backend_properties_*.json"))[0]
    T1, T2 = props(cal)
    jr = json.load(open(sorted(glob.glob(f"{run}/jobs/{sub}/htq_job_*.json"))[0]))
    layout = jr.get("initial_layout") or jr["embedding"]["layout"]
    pubs = {}
    for p in sorted(glob.glob(f"{run}/raw/{sub}/htq_bits_*.npz")):
        z = np.load(p, allow_pickle=True)
        for k in z.files:
            if z[k].dtype == np.uint8 and z[k].ndim == 2: pubs[split_prefix(k)[1]] = (p, k)
    Zof = lambda n: A.signs_from_bits(np.load(pubs[n][0], allow_pickle=True)[pubs[n][1]])[:, zs].mean(0)
    Z0 = Zof(pub_name("j0", 0.0, False, "Z"))
    far = np.array([n for n in range(50) if abs(n - 24) >= 6])
    t1 = np.array([T1[layout[zs[n]]] for n in range(50)])
    print(f"\n=== {lab}: device T1 median {np.median(list(T1.values())):.0f} us, matter-qubit T1 median {np.median(t1):.0f} us (min {t1.min():.0f})")
    for t in (2.0, 4.0, 6.0):
        for kind, nm in (("mirror", pub_name("j0", t, True, "Z")), ("physics", pub_name("j0", t, False, "Z"))):
            r = Zof(nm) / Z0
            od, ev = far[far % 2 == 1], far[far % 2 == 0]
            loss_od, loss_ev = 1 - r[od], 1 - r[ev]
            c_od = np.corrcoef(loss_od, 1 / t1[od])[0, 1]; c_ev = np.corrcoef(loss_ev, 1 / t1[ev])[0, 1]
            # implied decay time if the odd-minus-even excess loss is amplitude damping: Z -> -(1 - 2p)
            p = (np.median(loss_od) - np.median(loss_ev)) / 2
            print(f"  t={t:.0f} {kind:7s}: Z/Z0 even {np.median(r[ev]):.2f} odd {np.median(r[od]):.2f}; "
                  f"corr(loss, 1/T1) odd {c_od:+.2f} even {c_ev:+.2f}; excess odd decay prob {p:.2f}")
