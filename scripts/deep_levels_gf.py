"""Gauge-fixed deep spectra: validate against the full-space runs, then
compute volumes ARPACK cannot reach.

  validate : (a) exact isometry cross-check at ns=6 and ns=8 -- embed every
             reduced basis state into the full 2^(2ns) space and compare
             H, T2, and J0 matrix elements against htensor.exact /
             spectroscopy.translate;
             (b) gaps + T2 phases vs stored deep_levels_ns{8,10,12}.npz.
  run <ns> [k] : compute the deep spectrum (k=None -> all levels, dense)
             and save data/deep_levels_ns{ns}.npz.

  PYTHONPATH=. .venv/bin/python scripts/deep_levels_gf.py validate
  PYTHONPATH=. .venv/bin/python scripts/deep_levels_gf.py run 14
"""

import sys
import time

import numpy as np

from htensor import Z2Lattice
from htensor import hamiltonian as ham
from htensor.currents import charge_density
from htensor.exact import to_sparse
from htensor.gaugefixed import PhysicalBasis, deep_spectrum
from htensor.spectroscopy import translate

M0, G2, ETA = 0.7, 1.1, 1.3
t0 = time.time()


def log(m):
    print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)


def embed(basis, lat):
    """Isometry V (2^nq x dim): reduced basis -> full-space product states."""
    nq = lat.n_qubits
    V = np.zeros((1 << nq, basis.dim))
    plus = np.array([1.0, 1.0]) / np.sqrt(2)
    minus = np.array([1.0, -1.0]) / np.sqrt(2)
    for j in range(basis.dim):
        amp = np.array([1.0])
        # qiskit ordering: qubit q is bit q of the full-space index; build
        # by iterating qubits high-to-low so kron order matches indexing
        for q in range(nq - 1, -1, -1):
            if q % 2 == 0:                     # site qubit
                zb = (basis.z[j] >> (q // 2)) & 1
                vec = np.array([1.0 - zb, float(zb)])
            else:                              # link qubit
                vec = plus if basis.xbit[j, q // 2] == 0 else minus
            amp = np.kron(amp, vec)
        V[:, j] = amp
    return V


if sys.argv[1] == "validate":
    for ns in (6, 8):
        lat = Z2Lattice(ns, pbc=True)
        basis = PhysicalBasis(lat)
        V = embed(basis, lat)
        for name, op in [("H", ham.build_hamiltonian(lat, M0, G2, ETA)),
                         ("J0(1)", charge_density(lat, 1)),
                         ("J0(2)", charge_density(lat, 2))]:
            full = to_sparse(op)
            d = np.abs(V.T @ (full @ V) - basis.matrix(op).toarray()).max()
            log(f"ns={ns} {name}: |V^T O V - O_red|_max = {d:.2e}")
            assert d < 1e-12, name
        Tfull = np.column_stack([translate(V[:, j], lat)
                                 for j in range(basis.dim)])
        d = np.abs(V.T @ Tfull - basis.translation().toarray()).max()
        log(f"ns={ns} T2: |V^T T V - T_red|_max = {d:.2e}")
        assert d < 1e-12, "T2 convention mismatch"
    for ns in (8, 10, 12):
        lat = Z2Lattice(ns, pbc=True)
        ref = np.load(f"data/deep_levels_ns{ns}.npz")
        k = len(ref["gaps"])
        gaps, phases, _ = deep_spectrum(lat, M0, G2, ETA, k=k)
        dg = np.abs(gaps[:k] - ref["gaps"]).max()
        # phases compare as sorted multisets within each degenerate cluster
        # (the stored resolution order inside +-k pairs is arbitrary)
        dp, i = 0.0, 0
        while i < k:
            j = i + 1
            while j < k and ref["gaps"][j] - ref["gaps"][i] < 1e-6:
                j += 1
            d = np.angle(np.exp(1j * (np.sort(phases[i:j])
                                      - np.sort(ref["phases"][i:j]))))
            dp = max(dp, np.abs(d).max())
            i = j
        b = PhysicalBasis(lat)
        log(f"ns={ns}: max |gap diff| = {dg:.2e}, max cluster |phase diff| = "
            f"{dp:.2e} ({k} levels, Q=0 dim {int((b.q == 0).sum())})")
    log("validation complete")

elif sys.argv[1] == "run":
    # run <ns> [k] [m0 g2 eta tag]   (tag -> deep_levels_<tag>_ns{ns}.npz)
    ns = int(sys.argv[2])
    k = int(sys.argv[3]) if len(sys.argv) > 3 else None
    tag = ""
    if len(sys.argv) > 4:
        M0, G2, ETA = map(float, sys.argv[4:7])
        tag = sys.argv[7] + "_"
    lat = Z2Lattice(ns, pbc=True)
    basis = PhysicalBasis(lat)
    log(f"ns={ns} ({M0}, {G2}, {ETA}): physical dim {basis.dim}, "
        f"Q=0 block {int((basis.q == 0).sum())}")
    gaps, phases, energies = deep_spectrum(lat, M0, G2, ETA, k=k)
    np.savez(f"data/deep_levels_{tag}ns{ns}.npz", gaps=gaps, phases=phases,
             energies=energies, m0=M0, g2=G2, eta=ETA)
    log(f"saved data/deep_levels_{tag}ns{ns}.npz ({len(gaps)} levels)")
    for g, p in list(zip(gaps, phases))[:30]:
        log(f"  gap {g:8.4f}   T2 phase {p:+8.4f}")
