"""Packet excitation energies dE = <H>_packet - <H>_vacuum at Ns = 50.

dE fixes where the hadronic tensor's q0 support ends.  The ordered-product
transform W(q0) = sum_t e^{i q0 t} <psi| J(t) J(0) |psi> has weight only at
q0 = E_n - E_psi, and E_n >= E_0, so

    W(q0, q1) = 0   for   q0 < -dE,      dE = E_psi - E_0,

with a tail of order the packet's own energy spread above that edge.  The
published q0 window starts at -1, so none of it is kinematically forbidden
and the map cannot be null-tested inside it (see scripts/w_map_symmetry.py).

Both energies are already certified by stored_prep and cached in
data/work/prep/, which is a working cache and not in the repository, so this
reduces them to a committed artifact.  To rebuild a missing cache, run the
prep that wants it, e.g.

    PYTHONPATH=. python scripts/qpdf_card_refs.py

  PYTHONPATH=. python scripts/packet_excitation.py
  -> data/packet_excitation_ns50.npz
"""

import argparse
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_outputs import in_place  # noqa: E402

CACHE = "data/work/prep"
# (tag, packet label) pairs to tabulate; the vacuum is "<tag>_ns50_vac".
CARDS = [("prod", "k1.26_s0.75"), ("prod", "k1.26_s1.00"), ("prod", "k1.26_s1.50"),
         ("prod", "k0.00_s0.75"),
         ("relA", "k1.26_s0.75"), ("relA", "k1.26_s1.00"), ("relA", "k1.26_s1.50")]
ANC = 48


def cached_energy(tag, label, anc=ANC, ns=50):
    path = os.path.join(CACHE, f"{tag}_ns{ns}_{label}_anc{anc}.pkl")
    if not os.path.exists(path):
        raise SystemExit(f"missing prep cache {path}; see the module docstring")
    with open(path, "rb") as f:
        ck = pickle.load(f)
    if ck["ns"] != ns or ck["anc_site"] != anc:
        raise SystemExit(f"{path}: ns/anc_site mismatch ({ck['ns']}, {ck['anc_site']})")
    return float(ck["H"])


p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--out", default=None, help="output npz (default: in place under data/)")
args = p.parse_args()

vac = {tag: cached_energy(tag, "vac") for tag in dict.fromkeys(t for t, _ in CARDS)}
for tag, e in vac.items():
    print(f"{tag} vacuum  <H> = {e:+.6f}")

rows = []
for tag, label in CARDS:
    e = cached_energy(tag, label)
    rows.append((tag, label, e, vac[tag], e - vac[tag]))
    print(f"{tag} {label:12s} <H> = {e:+.6f}   dE = {e - vac[tag]:.6f}")

out = args.out or in_place("data/packet_excitation_ns50")
np.savez(out if out.endswith(".npz") else out + ".npz",
         tag=np.array([r[0] for r in rows]), label=np.array([r[1] for r in rows]),
         h_packet=np.array([r[2] for r in rows]), h_vacuum=np.array([r[3] for r in rows]),
         de=np.array([r[4] for r in rows]), ns=50, anc_site=ANC,
         source="data/work/prep (stored_prep certified <H>)")
print(f"wrote {out}")
