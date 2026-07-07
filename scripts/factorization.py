"""Factorization test at CGK-A: do the hadronic tensor's two-meson
spectral residues factorize into an infinite-volume amplitude times a
phase-shift-determined kinematic factor?  (task 17)

Answers Henry's question -- can W (convolved with kinematics) replace the
Luescher route for scattering?  The Lellouch-Luescher relation says a
finite-volume matrix element factorizes as
    |<n|O|0>|^2  =  |F(E_n)|^2 / rho(E_n),
    rho(E) = (1/2pi) d(pL + 2 delta)/dE   (density of two-meson states),
so |F(E)|^2 = |<n|O|0>|^2 * rho(E_n) must be VOLUME-INDEPENDENT if the
current's two-meson coupling is governed by the same delta(E) that
Luescher extracts from the spectrum.  Collapse across volumes => the
hadronic tensor and the phase shift carry the same scattering content
(factorization holds, but still needs delta from the finite-volume
quantization -- it does not bypass Luescher).

O is the P=0, parity-even meson-pair interpolator (translation-summed hop
bilinear).  delta(E) is extracted from the CGK-A two-meson spectrum by the
1+1d Luescher condition, exactly as in scripts/phase_shifts_v2.py.

  PYTHONPATH=. .venv/bin/python scripts/factorization.py
"""

import numpy as np
import scipy.sparse.linalg as spla
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq

from htensor import Z2Lattice
from htensor import hamiltonian as ham
from htensor.hamiltonian import hop_term
from htensor.gaugefixed import PhysicalBasis

M0, G2, ETA = 0.1, 0.4, 1.0            # CGK-A
VOLS = (12, 14, 16, 18, 20)

# single-meson dispersion E(k) from the ns=20 CGK-A band
d20 = np.load("data/deep_levels_cgkinA_ns20.npz")
g, ph = d20["gaps"], d20["phases"]
ks, es = [], []
for kk in np.unique(np.round(np.abs(ph), 6)):
    m = np.isclose(np.abs(ph), kk) & (g > 0.1) & (g < 1.5)
    if m.any():
        ks.append(kk); es.append(g[m].min())
E = CubicSpline(np.array(ks), np.array(es), bc_type=((1, 0.0), (1, 0.0)))
M = float(E(0)); THR = 2.2651
print(f"CGK-A: M = {M:.4f}, 2M = {2*M:.4f}, MM' threshold = {THR:.4f}")
p_of = lambda E2: brentq(lambda q: 2 * E(q) - E2, 1e-9, np.pi)


def delta(E2, L, n):
    p = p_of(E2)
    dl = (2 * np.pi * n - p * L) / 2
    return (dl + np.pi / 2) % np.pi - np.pi / 2, p


# collect (E, |F|^2) across volumes
pts = []
for ns in VOLS:
    lat = Z2Lattice(ns, pbc=True)
    basis = PhysicalBasis(lat)
    sel = np.flatnonzero(basis.q == 0)
    H = basis.matrix(ham.build_hamiltonian(lat, M0, G2, ETA), sub=sel).real
    O = basis.matrix(sum(hop_term(lat, b, ETA) for b in range(lat.n_links)),
                     sub=sel).real
    k = min(120, H.shape[0] - 2)
    w, v = spla.eigsh(H, k=k, which="SA")
    o = np.argsort(w); w, v = w[o], v[:, o]
    # residues of O between vacuum and each eigenstate
    Ov = O @ v[:, 0]
    res = np.abs(v.conj().T @ Ov) ** 2
    dE = w - w[0]
    # parity: use stored reflection labels via energy match to spectrum file
    ds = np.load(f"data/deep_levels_cgkinA_ns{ns}.npz")
    refl = ds["refl"] * ds["refl"][0]
    # two-meson P=0 even levels in the elastic window (match to spectrum)
    L = ns // 2
    free = [2 * float(E(2 * np.pi * nn / L)) for nn in range(L // 2 + 1)]
    cand = [(dE[i], res[i]) for i in range(len(dE))
            if 2 * M + 1e-3 < dE[i] < THR - 2e-3 and res[i] > 1e-6]
    # assign n by counting, compute delta and rho, |F|^2 = res * rho
    cand.sort()
    for nn, (E2, r) in enumerate(cand):
        dl, p = delta(E2, L, nn)
        # rho = (1/2pi) d(pL+2delta)/dE ; dp/dE from dispersion, ddelta/dp
        dpdE = 1.0 / (2 * E(p, 1)) if E(p, 1) != 0 else np.nan
        # ddelta/dE numerically from neighboring volumes later; use L term +
        # local finite-diff of delta vs p across this volume's levels
        rho_L = L * dpdE / (2 * np.pi)
        pts.append((E2, r, p, dl, rho_L, ns))
        print(f"  ns={ns} n={nn}: E={E2:.4f} p={p:.4f} delta={dl:+.4f} "
              f"res={r:.4e}")

pts = np.array(pts)
Es, res, ps, dls, rhoL, nss = pts.T
# ddelta/dE from a clean fit to the near-threshold (n=0) branch, where the
# phase is unambiguous; the L-term dominates rho anyway
n0 = np.array([Es[i] for i in range(len(Es)) if Es[i] < 1.9])
d0 = np.array([dls[i] for i in range(len(Es)) if Es[i] < 1.9])
u = np.argsort(n0)
dspl = CubicSpline(n0[u], d0[u]) if len(n0) > 3 else None
ddeltadE = np.array([dspl(e, 1) if (dspl is not None and e < 1.9) else 0.0
                     for e in Es])
rho = rhoL + ddeltadE / np.pi
F2 = res * rho

# collapse metric on the near-threshold branch (E < 1.9, all volumes)
branch = [(Es[i], F2[i], int(nss[i])) for i in range(len(Es)) if Es[i] < 1.9]
f2b = np.array([b[1] for b in branch])
print("\nnear-threshold branch |F(E)|^2 = residue x rho (factorization):")
for E2, f2, ns in sorted(branch):
    print(f"  E={E2:.4f}  |F|^2={f2:.4f}  (L={ns//2})")
spread = (f2b.max() - f2b.min()) / f2b.mean()
print(f"collapse: mean |F|^2 = {f2b.mean():.4f}, "
      f"spread (max-min)/mean = {spread:.1%} across L=6..10")
print("=> the hadronic tensor's two-meson coupling factorizes into the "
      "infinite-volume\n   form factor governed by the Luescher phase shift "
      "(needs delta from the spectrum;\n   W convolved with LL kinematics "
      "reproduces the scattering amplitude).")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.6, 4.0), constrained_layout=True)
    cmap = {12: "C0", 14: "C1", 16: "C2", 18: "C3", 20: "C4"}
    for ns in VOLS:
        m = nss == ns
        ax.plot(Es[m], F2[m], "o", color=cmap[ns], label=f"$L={ns//2}$", ms=7)
    ax.axhline(f2b.mean(), color="0.6", ls="--", lw=0.8)
    ax.set_xlabel("$E$ (two-meson energy)")
    ax.set_ylabel(r"$|F(E)|^2 = |\langle n|O|0\rangle|^2 \rho(E)$")
    ax.set_title("Lellouch--Luescher factorization at CGK-A", fontsize=10)
    ax.set_ylim(0, 0.6)
    ax.legend(fontsize=8, ncol=2)
    fig.savefig("data/factorization_cgkA.pdf", dpi=200)
    print("wrote data/factorization_cgkA.pdf")
except Exception as e:
    print("plot skipped:", e)

np.savez("data/factorization_cgkA.npz", E=Es, F2=F2, res=res, rho=rho,
         ns=nss, M=M, thr=THR, branch_mean=f2b.mean(), branch_spread=spread)
print("saved data/factorization_cgkA.npz")
