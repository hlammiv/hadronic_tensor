"""htq_hw analyze with the ancilla's X-readout bias removed per pub.

On ibm_chicago (6 Nov 2026) the ancilla reads <X> = +0.06 at t = 0 in every
X-basis pub, physics and mirror alike, against the ideal -0.045; its Y
readout and the mirror phase (1.8 deg) are clean, and the offset does not
grow with depth.  An additive bias b on the ancilla enters every probe as
b <O_n>: the connected J0 estimator picks up -b <z_n>/2 (staggered, 0.015
per site here) and the J1 bond estimator b <T_b>.  Both are of the order
of the weak-site mirror signals, so kappa goes negative at the sites with
|sx0| ~ 0.03 and the median fill for the wings follows it below the mask.

Here b = mean(x_anc) - X_ideal(t = 0) is estimated per pub from its own
shots (the ideal is state-independent and constant in t) and subtracted
from the ancilla sign before the products are formed; the centre site,
whose density is 0.5, is unaffected.  Phoenix has the same bias at a third
the size, so the same rule is applied to both devices.
"""
import sys
import numpy as np
import htq_hw.analyze as A

_card = sys.argv[sys.argv.index("--card") + 1]
_grid = A.IdealGrid(f"htq_hw/cards/{_card}/ideal_j0.npz", expect_ns=50)
X_IDEAL = float(_grid.scalar("X", 0.0))
_report = {}


def estimate_probes(bits, lat, setting, eta=A.ETA):
    s = A.signs_from_bits(bits)
    anc = lat.ancilla
    xa_raw = s[:, anc].astype(float)
    bias = float(xa_raw.mean()) - X_IDEAL
    xa = xa_raw - bias
    N = len(s)
    out = {"xa": float(xa_raw.mean()), "xa_err": float(xa_raw.std() / np.sqrt(N)), "N": N,
           "setting": setting, "xa_bias": bias}
    _report.setdefault(setting, []).append(bias)
    if setting == "Z":
        z = s[:, [lat.site_qubit(v) for v in range(lat.ns)]]
        idb = np.array([(-1) ** v / 2 for v in range(lat.ns)])
        J0 = idb[None, :] - z / 2
        xJ0 = xa[:, None] * J0
        conn = xJ0 - idb[None, :] * xa[:, None]
        x_links = s[:, [lat.link_qubit(n) for n in range(lat.ns)]]
        G = np.array([(-1) ** n * z[:, n] * x_links[:, (n - 1) % lat.ns] * x_links[:, n]
                      for n in range(lat.ns)]).T
        out.update(J0=J0.mean(0), J0_err=J0.std(0) / np.sqrt(N),
                   xJ0=xJ0.mean(0), xJ0_err=xJ0.std(0) / np.sqrt(N), id_b=idb,
                   sx=conn.mean(0), sxe=conn.std(0) / np.sqrt(N),
                   B=J0.mean(0), Be=J0.std(0) / np.sqrt(N),
                   gauss=G.mean(0), gauss_err=G.std(0) / np.sqrt(N))
        return out
    T = np.empty((N, lat.ns))
    term = np.empty(lat.ns, int)
    for b in range(lat.ns):
        a, bb, l = lat.bond_qubits(b)
        sign = lat.seam_sign if lat.is_seam(b) else 1
        T[:, b] = sign * s[:, a] * s[:, l] * s[:, bb]
        term[b] = A.j1_term_in_setting(b, setting)
    xT = xa[:, None] * T
    out.update(T=T.mean(0), T_err=T.std(0) / np.sqrt(N),
               xT=xT.mean(0), xT_err=xT.std(0) / np.sqrt(N), term=term,
               sx=xT.mean(0), sxe=xT.std(0) / np.sqrt(N))
    return out


A.estimate_probes = estimate_probes
from htq_hw.__main__ import main
rc = main(sys.argv[1:])
for setting, b in _report.items():
    b = np.array(b)
    print(f"ancilla X bias removed ({_card}, X_ideal {X_IDEAL:+.4f}, setting {setting}): "
          f"{len(b)} pubs, mean {b.mean():+.4f}, min {b.min():+.4f}, max {b.max():+.4f}")
sys.exit(rc)
