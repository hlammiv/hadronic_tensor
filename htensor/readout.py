"""Readout-basis settings and bitstring estimators for the four-component
campaign (reference implementation; the hardware package htq_hw re-implements
these without htensor and is cross-checked against this module).

Settings (logical qubit -> measurement basis):
  'Z'   : matter Z, links X.  J^0 probes; Gauss syndromes
          G_n = (-1)^n Z_n X_{l-} X_{l+} come for free.
  'XYA' : even matter sites Y, odd matter sites X, links Z.
  'XYB' : even matter sites X, odd matter sites Y, links Z.
The bulk bond current on bond b (sites a = b, b' = b+1, link l) is
(eta/4)(Y_a X_b' - X_a Y_b') Z_l.  With N_s even the alternation closes on
the ring, so XYA reads the Y_a X_b' term on even bonds and the X_a Y_b' term
on odd bonds; XYB the complement.  Two settings therefore cover both terms
of all N_s bond currents (the seam bond additionally needs the JW string,
which on Gauss-law states equals seam_sign -- see trotter.seam_sign -- and
is applied as a sign).  The ancilla is read in X (Re) or Y (Im).
"""

import numpy as np
from qiskit import QuantumCircuit

from .lattice import Z2Lattice
from .trotter import seam_sign

SETTINGS = ("Z", "XYA", "XYB")


def basis_map(lat: Z2Lattice, setting: str, anc: int | None = None,
              anc_basis: str = "X") -> dict:
    """logical qubit -> 'X' | 'Y' | 'Z'."""
    m = {}
    for n in range(lat.ns):
        s, l = lat.site_qubit(n), lat.link_qubit(n)
        if setting == "Z":
            m[s], m[l] = "Z", "X"
        elif setting == "XYA":
            m[s], m[l] = ("Y" if n % 2 == 0 else "X"), "Z"
        elif setting == "XYB":
            m[s], m[l] = ("X" if n % 2 == 0 else "Y"), "Z"
        else:
            raise ValueError(setting)
    if anc is not None:
        m[anc] = anc_basis
    return m


def apply_readout_rotations(qc: QuantumCircuit, bmap: dict):
    """Rotate each qubit so a Z measurement reads the requested basis."""
    for q, p in sorted(bmap.items()):
        if p == "X":
            qc.h(q)
        elif p == "Y":
            qc.sdg(q)
            qc.h(q)


def j1_term_in_setting(lat: Z2Lattice, bond: int, setting: str) -> int:
    """Which J^1 Pauli term the setting reads on `bond`: 1 = Y_a X_b (coeff
    +eta/4), 2 = X_a Y_b (coeff -eta/4).  None for the 'Z' setting."""
    if setting == "Z":
        return None
    a_even = bond % 2 == 0
    if setting == "XYA":
        return 1 if a_even else 2
    return 2 if a_even else 1


def signs_from_bits(bits: np.ndarray) -> np.ndarray:
    """0/1 bit array (shots, qubits), column q = logical qubit q -> +-1."""
    return 1.0 - 2.0 * np.asarray(bits, dtype=np.float64)


def estimate_probes(bits: np.ndarray, lat: Z2Lattice, setting: str, anc: int,
                    eta: float = 1.0) -> dict:
    """Per-shot estimators from one readout setting.

    Returns dict with, for 'Z': 'xJ0' (ns,) = <X_anc J^0(v)> and 'J0' (ns,)
    = <J^0(v)>; for XY settings: 'xT' (ns,) = <X_anc T_b> and 'T' (ns,) =
    <T_b> where T_b is the Pauli term read on bond b (see j1_term_in_setting),
    with the seam sign applied, plus 'term' (ns,) = 1|2 and 'xa' = <X_anc>.
    Errors ('*_err') are standard errors of the shot mean."""
    s = signs_from_bits(bits)
    xa = s[:, anc]
    N = len(s)
    out = {"xa": float(xa.mean()), "N": N}
    if setting == "Z":
        z = s[:, [lat.site_qubit(v) for v in range(lat.ns)]]
        idb = np.array([(-1) ** v / 2 for v in range(lat.ns)])
        J0 = idb[None, :] - z / 2
        xJ0 = xa[:, None] * J0
        out.update(J0=J0.mean(0), J0_err=J0.std(0) / np.sqrt(N),
                   xJ0=xJ0.mean(0), xJ0_err=xJ0.std(0) / np.sqrt(N), id_b=idb)
        return out
    T = np.empty((N, lat.ns))
    term = np.empty(lat.ns, int)
    for b in range(lat.ns):
        a, l, bb = lat.site_qubit(b), lat.link_qubit(b), lat.site_qubit(b + 1)
        sign = seam_sign(lat) if lat.is_seam(b) else 1
        T[:, b] = sign * s[:, a] * s[:, l] * s[:, bb]
        term[b] = j1_term_in_setting(lat, b, setting)
    xT = xa[:, None] * T
    out.update(T=T.mean(0), T_err=T.std(0) / np.sqrt(N),
               xT=xT.mean(0), xT_err=xT.std(0) / np.sqrt(N), term=term)
    return out


def combine_j1(resA: dict, resB: dict, eta: float) -> tuple[np.ndarray, np.ndarray]:
    """<X_anc J^1_b> = (eta/4)(<X T^{(1)}_b> - <X T^{(2)}_b>) from the two
    XY settings (independent shots, variances add).  -> (value, err)."""
    ns = len(resA["term"])
    val, err = np.empty(ns), np.empty(ns)
    for b in range(ns):
        r1, r2 = (resA, resB) if resA["term"][b] == 1 else (resB, resA)
        val[b] = eta / 4 * (r1["xT"][b] - r2["xT"][b])
        err[b] = eta / 4 * np.hypot(r1["xT_err"][b], r2["xT_err"][b])
    return val, err
