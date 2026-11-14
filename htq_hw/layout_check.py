"""Gauss-law check of a layout from its t = 0 reference pub.

The prepared state obeys Gauss' law, so every syndrome
G_n = (-1)^n Z_n X_{l-} X_{l+} (l- = link n-1, l+ = link n) is +1 on an
ideal device.  On hardware the far sites read 0.6-0.75 (readout and
preparation error); a site whose matter qubit, link qubits or couplers are
bad reads far less.  On ibm_chicago 2026-11-11 sites 27/28 read 0.11-0.14
and 22/23 read 0.17-0.43 against a far median of 0.65, and those sites are
where the J1 maps and the boost asymmetry failed; on ibm_phoenix 2026-10-06
the worst packet site (26/27, the known bad link) read 0.36-0.39 against
0.57-0.61.  So the check costs nothing but the t = 0 reference job, which
the production run needs anyway: submit that job first (``--only-jobs 0``),
fetch it, run ``gauss-check``, and submit the rest only when it passes on
the same layout.

PASS when every packet site's G_n >= ``min_ratio`` x the far-site median
(default 0.5: Phoenix passes at 0.62, 11 November fails at 0.17).  The ratio,
not an absolute floor, so a device whose readout is uniformly worse is not
refused for that.
"""
import json

import numpy as np

from . import CENTER
from .model import Lattice

MIN_RATIO = 0.5
FAR = 8            # far sites: |n - centre| >= FAR, cyclically


def gauss_profile(bits, lat: Lattice) -> tuple[np.ndarray, np.ndarray]:
    """(shots, n_wires) 0/1 Z-basis readout of a t = 0 pub with the links read
    in X (the production Z-readout reference) -> (G_n, shot error), n = 0..ns-1."""
    s = 1.0 - 2.0 * np.asarray(bits, dtype=np.float64)
    z = s[:, [lat.site_qubit(n) for n in range(lat.ns)]]
    x = s[:, [lat.link_qubit(n) for n in range(lat.ns)]]
    g = np.stack([(-1) ** n * z[:, n] * x[:, (n - 1) % lat.ns] * x[:, n] for n in range(lat.ns)], axis=1)
    return g.mean(0), g.std(0, ddof=1) / np.sqrt(len(g))


def far_sites(ns: int, center: int = CENTER, far: int = FAR) -> list[int]:
    return [n for n in range(ns) if min((n - center) % ns, (center - n) % ns) >= far]


def check(g: np.ndarray, err: np.ndarray, sites, center: int = CENTER,
          min_ratio: float = MIN_RATIO) -> dict:
    """-> {'status', 'far_median', 'sites': [{site, G, err, ratio, ok}], 'failing'}."""
    ns = len(g)
    ref = float(np.median(g[far_sites(ns, center)]))
    rows, failing = [], []
    for n in sites:
        ratio = float(g[n] / ref) if ref > 0 else float("nan")
        ok = bool(np.isfinite(ratio) and ratio >= min_ratio)
        rows.append({"site": int(n), "G": float(g[n]), "err": float(err[n]), "ratio": ratio, "ok": ok})
        if not ok:
            failing.append(int(n))
    status = "PASS" if ref > 0 and not failing else "FAIL"
    return {"status": status, "far_median": ref, "min_ratio": float(min_ratio), "sites": rows,
            "failing": failing}


def find_reference(z, card: str | None = None):
    """The j0 t = 0 Z-readout pub in a bits file (``htq_bits_*.npz``):
    -> (pub name, array).  With several cards present ``card`` picks one."""
    from .analyze import card_of
    from .campaign import pub_name, split_prefix
    want = pub_name("j0", 0.0, False, "Z")
    hits = []
    for name in [str(x) for x in z["pub_names"]]:
        pre, bare = split_prefix(name)
        if bare == want and (card is None or card_of(pre) == card):
            hits.append(name)
    if not hits:
        raise ValueError(f"no {want} pub" + (f" of card {card}" if card else "") + " in this bits file")
    if len(hits) > 1:
        raise ValueError(f"{len(hits)} {want} pubs ({hits}); pass --card to pick one")
    return hits[0], z[hits[0]]


def format_report(name: str, res: dict, layout=None, lat: Lattice | None = None) -> str:
    lines = [f"{name}: far-site median G = {res['far_median']:.3f}; packet sites need "
             f">= {res['min_ratio']:.2f} x that"]
    for r in res["sites"]:
        q = ""
        if layout is not None and lat is not None:
            n = r["site"]
            q = (f"  site q{layout[lat.site_qubit(n)]}, links q{layout[lat.link_qubit((n - 1) % lat.ns)]}"
                 f" q{layout[lat.link_qubit(n)]}")
        lines.append(f"  site {r['site']:2d}: G = {r['G']:+.3f} +- {r['err']:.3f}  ratio {r['ratio']:.2f}"
                     f"  {'ok' if r['ok'] else 'FAIL'}{q}")
    lines.append(f"{res['status']}" + (f": sites {res['failing']} are below the ratio; the layout "
                                       f"has a bad qubit or coupler there" if res["failing"] else ""))
    return "\n".join(lines)


def write_record(path: str, payload: dict) -> None:
    with open(path, "w") as f:
        json.dump(payload, f, indent=1)
