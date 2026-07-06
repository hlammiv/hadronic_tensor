"""Gauge-fixed (physical-sector) exact spectroscopy.

In the product basis {sites in the Z basis} x {links in the X basis} every
Gauss operator G_n = (-1)^n Z_n X_{n-1,n} X_{n,n+1} is diagonal, so the
G_n = +1 sector is spanned by product states whose link pattern is fixed by
the matter configuration and a single holonomy bit:

    e_n = s_n e_{n-1},   e_n = 1 - 2 x_n,   s_n = (-1)^n (1 - 2 z_n),

with ring closure prod_n s_n = 1 selecting the admissible matter parities.
Any gauge-invariant Pauli string maps this basis to itself up to a sign, so
reduced matrices are built mechanically from the SAME SparsePauliOp objects
used everywhere else -- no rederivation of seam/JW/staggering conventions.
Dimensions: 2^ns physical states, a few thousand after the Q = 0 cut
(ns = 14: 6864), versus 2^(2 ns) for the penalized full-space solver.

Validation: scripts/deep_levels_gf.py reproduces the stored full-space
deep_levels_ns{8,10,12}.npz gaps and T2 phases.
"""

import numpy as np
import scipy.sparse as sp

from .lattice import Z2Lattice
from . import hamiltonian as ham
from .currents import charge_density


def _terms(op):
    """SparsePauliOp -> [({qubit: 'X'|'Y'|'Z'}, complex coeff), ...]."""
    op = op.simplify()
    n = op.num_qubits
    out = []
    for label, c in zip(op.paulis.to_labels(), op.coeffs):
        ops = {n - 1 - i: ch for i, ch in enumerate(label) if ch != "I"}
        out.append((ops, complex(c)))
    return out


class PhysicalBasis:
    """Enumeration of the full G_n = +1 product basis.

    NOTE: the basis is NOT restricted to a charge sector -- individual Pauli
    strings (e.g. the XX half of a hop) change Q, and only their sum
    cancels; Q = q blocks are exact principal submatrices AFTER summing
    (self.q gives the diagonal charge of each basis state)."""

    def __init__(self, lat: Z2Lattice):
        assert lat.pbc, "gauge-fixed basis implemented for PBC"
        ns = lat.ns
        z = np.arange(1 << ns, dtype=np.int64)
        zbit = (z[:, None] >> np.arange(ns)) & 1          # zbit[c, n]
        s = ((-1) ** np.arange(ns))[None, :] * (1 - 2 * zbit)
        keep = s.prod(axis=1) == 1                         # ring closure
        z = z[keep]
        zbit = zbit[keep]
        s = s[keep]
        stag = ((-1) ** np.arange(ns))[None, :]
        q = ((stag - (1 - 2 * zbit)) // 2).sum(axis=1)
        # link bits for both holonomies: e_n = e_0 * prod_{m<=n, m>=1} s_m
        cum = np.cumprod(np.concatenate(
            [np.ones((len(z), 1), dtype=np.int64), s[:, 1:]], axis=1), axis=1)
        self.lat, self.ns = lat, ns
        self.z = np.concatenate([z, z])                    # index = (z, e0)
        self.q = np.concatenate([q, q])
        self.h = np.concatenate([np.zeros(len(z), np.int64),
                                 np.ones(len(z), np.int64)])
        e0 = 1 - 2 * self.h[:, None]
        self.ebit = np.concatenate([cum, cum]) * e0        # e_n = +-1
        self.xbit = ((1 - self.ebit) // 2).astype(np.int64)
        self.dim = len(self.z)
        # lookup (z, holonomy) -> row
        self.lut = -np.ones(1 << (ns + 1), dtype=np.int64)
        self.lut[(self.z << 1) | self.h] = np.arange(self.dim)

    def _zbit(self, n):
        return (self.z >> n) & 1

    def matrix(self, op) -> sp.csr_matrix:
        """Reduced matrix of a gauge-invariant SparsePauliOp."""
        lat, ns = self.lat, self.ns
        site_of = {lat.site_qubit(n): n for n in range(ns)}
        link_of = {lat.link_qubit(n): n for n in range(lat.n_links)}
        terms = _terms(op)
        data, cols = [], []
        for ops, c in terms:
            phase = np.full(self.dim, c, dtype=complex)
            zmask, lflip = 0, np.zeros(self.dim, dtype=np.int64)
            for q, p in ops.items():
                if q in site_of:
                    n = site_of[q]
                    zn = self._zbit(n)
                    if p == "Z":
                        phase = phase * (1 - 2 * zn)
                    elif p == "X":
                        zmask |= 1 << n
                    else:                                   # Y
                        phase = phase * 1j * (1 - 2 * zn)
                        zmask |= 1 << n
                else:
                    n = link_of[q]
                    en = self.ebit[:, n]
                    if p == "X":
                        phase = phase * en
                    elif p == "Z":
                        lflip = lflip ^ (1 << n)
                    else:                                   # Y
                        phase = phase * 1j * (-en)
                        lflip = lflip ^ (1 << n)
            z2 = self.z ^ zmask
            h2 = self.xbit[:, 0] ^ (lflip & 1)
            col = self.lut[(z2 << 1) | h2]
            if np.any(col < 0):
                raise ValueError("operator leaves the physical sector")
            data.append(phase)
            cols.append(col)
        data = np.concatenate(data)
        cols = np.concatenate(cols)
        rows = np.tile(np.arange(self.dim), len(terms))
        M = sp.coo_matrix((data, (rows, cols)),
                          shape=(self.dim, self.dim)).tocsr()
        return M

    def translation(self) -> sp.csr_matrix:
        """Reduced T2 (one-spatial-site translation), matching
        spectroscopy.translate: optional string twist, then site n+2 -> n."""
        lat, ns = self.lat, self.ns
        phase = np.ones(self.dim, dtype=complex)
        if lat.ns % 4 == 0:
            tw = np.ones(self.dim, dtype=np.int64)
            for n in range(2, ns):
                tw = tw * (1 - 2 * self._zbit(n))
            phase = -phase * tw
        # rotate bits: new bit n = old bit (n+2) mod ns
        def rot(v, nbits):
            return ((v >> 2) | (v << (nbits - 2))) & ((1 << nbits) - 1)
        z2 = rot(self.z, ns)
        # new link n = old link n+2 -> new holonomy = old xbit[2]
        h2 = self.xbit[:, 2 % lat.n_links]
        col = self.lut[(z2 << 1) | h2]
        if np.any(col < 0):
            raise ValueError("translation left the sector (check ns mod 4)")
        # T2|j> = phase(j) |target(j)>: entry at (row=target, col=source)
        return sp.coo_matrix((phase, (col, np.arange(self.dim))),
                             shape=(self.dim, self.dim)).tocsr()


def deep_spectrum(lat: Z2Lattice, m0, g2, eta, k: int | None = None,
                  degeneracy_tol: float = 1e-6):
    """(gaps, T2 phases, energies) in the physical Q=0 sector.

    k=None -> dense eigh, ALL levels; else sparse eigsh lowest k."""
    basis = PhysicalBasis(lat)
    sel = np.flatnonzero(basis.q == 0)
    H = basis.matrix(ham.build_hamiltonian(lat, m0, g2, eta))
    H = H[sel][:, sel]
    assert np.abs(H.imag).max() < 1e-12
    if k is None or len(sel) <= 4096:
        w, v = np.linalg.eigh(H.real.toarray())
        if k is not None:
            w, v = w[:k], v[:, :k]
    else:
        w, v = sp.linalg.eigsh(H.real, k=k, which="SA")
        o = np.argsort(w)
        w, v = w[o], v[:, o]
    T = basis.translation()[sel][:, sel]
    phases = np.empty(len(w))
    i = 0
    while i < len(w):
        j = i + 1
        while j < len(w) and w[j] - w[i] < degeneracy_tol:
            j += 1
        block = v[:, i:j].conj().T @ (T @ v[:, i:j])
        ev = np.linalg.eigvals(block)
        phases[i:j] = np.sort(np.angle(ev))
        i = j
    return w - w[0], phases, w
