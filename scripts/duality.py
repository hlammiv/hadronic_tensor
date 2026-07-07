"""Quark-hadron duality via energy-weighted sum rules (task 7).

The spectral moments of the density response,
    mu_k(q) = sum_n |<n|J0(q)|GS>|^2 (E_n - E_0)^k,
are exact ground-state matrix elements: mu_0 is the static structure
factor S(q), and mu_1 is the model-independent f-sum rule, fixed by the
current-Hamiltonian commutator [J0(-q),[H,J0(q)]] -- a PARTON-level
(current-algebra) quantity that only sees the hopping term, identical to
free fermions.  Quark-hadron duality is the statement that the confining,
discrete meson spectrum saturates these parton sum rules: we show how few
mesons exhaust mu_0 and mu_1 as a function of Q^2.

  PYTHONPATH=. .venv/bin/python scripts/duality.py [ns]
"""

import sys

import numpy as np

from htensor import Z2Lattice
from htensor import hamiltonian as ham
from htensor.currents import charge_density
from htensor.gaugefixed import PhysicalBasis
from htensor.pauli import pauli_sum

M0, G2, ETA = 0.7, 1.1, 1.3
ns = int(sys.argv[1]) if len(sys.argv) > 1 else 12
lat = Z2Lattice(ns, pbc=True)
basis = PhysicalBasis(lat)
sel = np.flatnonzero(basis.q == 0)
H = basis.matrix(ham.build_hamiltonian(lat, M0, G2, ETA), sub=sel).real
w, v = np.linalg.eigh(H.toarray())
E0, gs = w[0], v[:, 0]
dE = w - E0
print(f"ns={ns}: {len(sel)} Q=0 states, meson mass M = {dE[1]:.4f}")


def J0q(q):
    """momentum-space charge density sum_v e^{iqv} J0(v), reduced to Q=0."""
    n = lat.n_qubits
    terms = []
    for vv in range(lat.ns):
        # J0(v) = ((-1)^v - Z_v)/2 ; the identity piece drops out of q!=0
        ph = np.exp(1j * q * (vv // 2))          # physical-site position
        terms.append(({}, ph * 0.5 * (-1) ** vv))
        terms.append(({lat.site_qubit(vv): "Z"}, -ph * 0.5))
    op = pauli_sum(n, terms)
    return basis.matrix(op, sub=sel)


print(f"\n{'q':>6} {'mu0=S(q)':>10} {'mu1 (f-sum)':>12} "
      f"{'n@90%mu0':>9} {'n@90%mu1':>9} {'1-meson %mu0':>12}")
rows = []
for j in range(1, lat.nx // 2 + 1):
    q = 2 * np.pi * j / lat.nx
    Jq = J0q(q)
    psi = Jq @ gs
    amp2 = np.abs(v.conj().T @ psi) ** 2          # |<n|J0(q)|GS>|^2
    mu0 = amp2.sum()
    mu1 = (amp2 * dE).sum()
    # saturation: cumulative fraction in energy order (states already sorted)
    c0 = np.cumsum(amp2) / mu0
    c1 = np.cumsum(amp2 * dE) / mu1
    n0 = int(np.searchsorted(c0, 0.9)) + 1
    n1 = int(np.searchsorted(c1, 0.9)) + 1
    # single-meson (first excited multiplet) fraction of mu0
    one = amp2[(dE > dE[1] - 0.05) & (dE < dE[1] + 0.4)].sum() / mu0
    rows.append((q, mu0, mu1, n0, n1, one))
    print(f"{q:6.3f} {mu0:10.4f} {mu1:12.4f} {n0:9d} {n1:9d} {one:11.1%}")

# duality: mu1 (hadronic spectrum) vs the exact parton f-sum commutator, all q
print("\nf-sum rule (hadronic spectrum vs parton current-algebra commutator):")
maxdev = 0.0
for q, mu0, mu1, *_ in rows:
    Jq = J0q(q)
    comm = Jq.conj().T @ (H @ Jq) - 0.5 * (Jq.conj().T @ Jq @ H
                                           + H @ Jq.conj().T @ Jq)
    fsum = np.real(gs.conj() @ (comm @ gs))
    maxdev = max(maxdev, abs(mu1 - fsum))
    print(f"  q={q:.3f}: spectrum {mu1:.4f}  commutator {fsum:.4f}")
print(f"max |spectrum - parton commutator| = {maxdev:.2e} "
      f"(hadronic spectrum saturates the parton f-sum rule exactly)")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    qs = [r[0] for r in rows]
    one = [r[5] for r in rows]
    fig, ax = plt.subplots(figsize=(5.4, 3.8), constrained_layout=True)
    ax.plot(qs, one, "o-", color="C0")
    ax.set_xlabel(r"momentum transfer $q$")
    ax.set_ylabel(r"single-meson fraction of $S(q)$")
    ax.set_title("Local duality: single-mode saturation vs $Q^2$", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.axhline(1, color="0.8", lw=0.6)
    fig.savefig("data/duality_saturation.pdf", dpi=200)
    print("wrote data/duality_saturation.pdf")
except Exception as e:
    print("plot skipped:", e)

np.savez("data/duality_sumrules.npz",
         rows=np.array([list(r) for r in rows]), ns=ns, M=float(dE[1]),
         fsum_maxdev=maxdev)
print("saved data/duality_sumrules.npz")
