"""ED cross-check for scripts/moving_frame_luscher.py: momentum-projected
interpolator residues confirming the one-body (band-3 / M'') vs two-meson
identification in the P != 0 sectors at CGK-A.

O_P = sum_b e^{i P b/2} hop_b couples the vacuum to momentum -+P states;
within a degenerate +-P cluster the summed residue is basis-independent.
ns = 12, 14 dense; ns = 16 sparse (Q=0 dims 1848, 6864, 25740).

FINDING (2026-07-21): the residues obey an EXACT selection rule -- every
sector level is either MM-coupled (res ~ 1e-2) or decoupled at machine
precision (res < 1e-25), even inside near-degenerate doublets (ns=14 m=1:
1.9586 decoupled / 1.9595 coupled, split 9e-4).  The band-3 (M'') tower and
the 2.33-2.44 odd-channel states carry an exact conserved label at ALL P
(the C / shift quantum number; at P=0 it reduces to the R = -1 tag), so
band-3 NEVER mixes with the MM tower: the moving-frame level identification
is exact, and the observed P-dependence of the extracted phase cannot be
blamed on one-body contamination.

  PYTHONPATH=. .venv/bin/python scripts/moving_frame_residue_check.py"""
import numpy as np
import scipy.sparse.linalg as spla
from htensor import Z2Lattice
from htensor import hamiltonian as ham
from htensor.hamiltonian import hop_term
from htensor.gaugefixed import PhysicalBasis

for ns in (12, 14, 16):
    lat = Z2Lattice(ns, pbc=True)
    basis = PhysicalBasis(lat)
    sel = np.flatnonzero(basis.q == 0)
    H = basis.matrix(ham.build_hamiltonian(lat, 0.1, 0.4, 1.0), sub=sel).real
    if ns <= 14:
        w, v = np.linalg.eigh(H.toarray())
        w, v = w[:120], v[:, :120]
    else:
        w, v = spla.eigsh(H, k=120, which="SA")
        o = np.argsort(w); w, v = w[o], v[:, o]
    T = basis.translation()[sel][:, sel]
    L = ns // 2
    # T2 phases per level (cluster-resolved)
    phases = np.empty(len(w)); vres = v.astype(complex).copy()
    i = 0
    while i < len(w):
        j = i + 1
        while j < len(w) and w[j] - w[i] < 1e-6:
            j += 1
        blk = vres[:, i:j]
        tb = blk.conj().T @ (T @ blk)
        ev, U = np.linalg.eig(tb)
        order = np.argsort(np.angle(ev))
        phases[i:j] = np.angle(ev[order])
        vres[:, i:j] = blk @ U[:, order]
        i = j
    dE = w - w[0]
    print(f"\n===== ns={ns} (L={L}) =====")
    for mm in (1, 2):
        P = 2 * np.pi * mm / L
        Op = sum(np.exp(1j * P * b / 2) * basis.matrix(
            hop_term(lat, b, 1.0), sub=sel) for b in range(ns))
        r_p = np.abs(vres.conj().T @ (Op @ vres[:, 0])) ** 2
        Om = sum(np.exp(-1j * P * b / 2) * basis.matrix(
            hop_term(lat, b, 1.0), sub=sel) for b in range(ns))
        r_m = np.abs(vres.conj().T @ (Om @ vres[:, 0])) ** 2
        res = r_p + r_m
        insec = np.abs(np.angle(np.exp(1j * (phases - P)))) < 1e-3
        print(f"-- m={mm} P={P:.4f}:")
        for i in np.flatnonzero(insec):
            if 1.3 < dE[i] < 2.45:
                print(f"   dE={dE[i]:.4f}  res={res[i]:.3e}")
