"""Exact circuit simulation and spectroscopy in the gauge-fixed basis (WS2-B).

Purpose (2026-09): wavepacket blocks with sigma_x >= 1 spatial site do not
fit the ns <= 12 rings that statevector training can afford (fit condition
ns >= 2 (6 sigma_x + 2 xi), xi ~ 1 spatial site), so training and
certification move to ns = 16-26, where the Gauss-law-resolved, charge-zero
basis has dimension 2 C(ns, ns/2) (ns = 16: 25,740; 20: 369,512; 24: 5.4M;
26: 20.8M) instead of 2^(2 ns).

Everything the training loop touches is simulated EXACTLY in that basis:

  * the vacuum ansatz (stateprep.vacuum_ansatz) and the wavepacket block
    (wavepacket.block_circuit / block_engine.BlockEngine) -- same gate order
    and conventions, validated to 1e-12 against Statevector/BlockEngine at
    ns = 6, 8 through the embedding isometry (tests/test_m6_packets.py);
  * the single-meson band per momentum, obtained by block-diagonalizing H
    in the T2 (one-spatial-site translation) Bloch basis of the gf space,
    so every grid momentum gets its lowest state directly -- no degenerate
    +-k cluster resolution, the failure mode of spectroscopy._t2_phases
    documented in scripts/train_cgkA_packets.py:band_full;
  * band-projected Gaussian packet targets (ports of
    spectroscopy._packet_vector / optimize_interpolator / meson_wavepacket,
    and scripts/train_cgkA_packets.py clean_target / tune_k0_env);
  * observables: energy, energy variance, <T2>, band weights, J0 profile.

Basis conventions follow htensor.gaugefixed.PhysicalBasis (matter z-bits,
link X-eigenvalues e_n = s_n e_{n-1} with s_n = (-1)^n (1 - 2 z_n), holonomy
bit h = x_0, ring closure); the enumeration here is restricted to ONE charge
sector at construction (every gate and H conserve Q; only the Q-changing
halves of individual Pauli strings leave it, and those cancel pairwise),
which is what makes ns = 24-26 affordable.  Operator ACTION convention:
O|j> = phase(j) |col(j)>, i.e. phase(j) = <col(j)|O|j>.

Gate kernel: a bond generator G = 1/4 (XX + YY) sigma^z (hop) or
1/4 (XY - YX) sigma^z (current) satisfies (2G)^2 = P, the projector onto
bond configurations with one occupied and one empty site, and 2G is a
signed pairing M of those configurations (M[I, J] = u sigma, u = 1 for hop,
u = i for cur, sigma = +-1).  Hence exp(-i th G) = cos(th/2) - i sin(th/2) M
on each pair and the identity elsewhere: one gather/scatter over ~dim/4
pairs per gate, precomputed once per bond (two int32 index arrays + two
int8 sign arrays, ~2.5 bytes per basis state per bond).  Site rz / link rx
are diagonal (int8 sign rows).

Measured 2026-09-02 (shared 20-core box, OMP_NUM_THREADS=4, nice 10,
production couplings, L=3 block, numba kernels):
  ns=16  dim 25,740     grad 0.19 s (156 params)   space+pairs    3 MB
  ns=20  dim 369,512    grad 1.9 s (156) / 3.3 s (228); band 21 s;   47 MB
  ns=24  dim 5,408,312  forward 3.8 s, grad 19.8 s (228 params);
                        space 5.9 s + pairs 38 s = 803 MB; band 483 s
                        (12 sectors of dim 450k); peak RSS 2.7 GB
                        (band vectors 86 MB each).
An L-BFGS-B anneal of 350+200+150 iterations at ns=20 is ~10-40 min.
"""

import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .lattice import Z2Lattice
from . import hamiltonian as ham
from . import currents as cur
from .wavepacket import KINDS, _bond_generator_full, window_offsets
from .stateprep import N_PARAMS_PER_LAYER

# ---------------------------------------------------------- fused kernels
# The pair rotation is ~10 numpy passes over dim/4 gathers per gate; the
# fused numba loops below are one pass (3-8x faster at ns = 20-24).  Thread
# count follows OMP_NUM_THREADS (compute etiquette on the shared box);
# pure-numpy fallbacks keep the module usable without numba.
try:
    import os as _os
    import numba as _nb

    _nthreads = int(_os.environ.get("OMP_NUM_THREADS", "4"))
    _nb.set_num_threads(max(1, min(_nthreads, _nb.config.NUMBA_NUM_THREADS)))
    _HAVE_NUMBA = True

    @_nb.njit(cache=True, parallel=True)
    def _k_rot_pairs(psi, I, J, sg, c, bI, bJ):
        """psi[I] <- c a + bI sg b ; psi[J] <- c b + bJ sg a (disjoint pairs)."""
        for p in _nb.prange(I.shape[0]):
            i, j = I[p], J[p]
            sgn = sg[p]
            a, b = psi[i], psi[j]
            psi[i] = c * a + (bI * sgn) * b
            psi[j] = c * b + (bJ * sgn) * a

    @_nb.njit(cache=True, parallel=True)
    def _k_rot_diag(psi, d, c, ms):
        """psi <- (c + ms d) psi."""
        for i in _nb.prange(psi.shape[0]):
            psi[i] = psi[i] * (c + ms * d[i])

    @_nb.njit(cache=True, parallel=True)
    def _k_bracket_pairs(lam, psi, I, J, sg, uI, uJ):
        acc = 0j
        for p in _nb.prange(I.shape[0]):
            i, j = I[p], J[p]
            sgn = sg[p]
            acc += (np.conj(lam[i]) * (uI * sgn) * psi[j]
                    + np.conj(lam[j]) * (uJ * sgn) * psi[i])
        return acc

    @_nb.njit(cache=True, parallel=True)
    def _k_bracket_diag(lam, psi, d):
        acc = 0j
        for i in _nb.prange(psi.shape[0]):
            acc += np.conj(lam[i]) * d[i] * psi[i]
        return acc
except ImportError:                                   # pragma: no cover
    _HAVE_NUMBA = False


def _terms(op):
    """SparsePauliOp -> [({qubit: 'X'|'Y'|'Z'}, complex coeff), ...]
    (same as gaugefixed._terms; duplicated so this module does not depend
    on a private helper of a file owned by another workstream)."""
    op = op.simplify()
    n = op.num_qubits
    out = []
    for label, c in zip(op.paulis.to_labels(), op.coeffs):
        ops = {n - 1 - i: ch for i, ch in enumerate(label) if ch != "I"}
        out.append((ops, complex(c)))
    return out


# ============================================================== the space
class GFSpace:
    """Gauss-law-resolved basis of one charge sector of the PBC ring.

    Arrays (dim = 2 C(ns, ns/2 + q)):
      z    int64  matter occupation bits (bit n = site n occupied)
      h    int8   holonomy bit (x_0: 0 -> link 0 in |+>, 1 -> |->)
      zs   int8 (ns, dim)  Z eigenvalue per site, 1 - 2 z_n
      es   int8 (ns, dim)  X eigenvalue per link, e_n
      lut  int32 (2^(ns+1))  (z << 1 | h) -> row, -1 outside the sector
    Row order: [h = 0 block, h = 1 block], z ascending in each (the same
    order PhysicalBasis uses for its (z, e0) index).
    """

    def __init__(self, lat: Z2Lattice, q: int = 0, chunk: int = 1 << 22):
        if not lat.pbc:
            raise ValueError("gauge-fixed basis implemented for PBC")
        if q % 2:
            raise ValueError("odd charge sectors violate the ring closure")
        ns = lat.ns
        n_occ = ns // 2 + q                      # Q = N_occ - ns/2
        zs = []
        for start in range(0, 1 << ns, chunk):
            blk = np.arange(start, min(start + chunk, 1 << ns), dtype=np.int64)
            pc = np.bitwise_count(blk.astype(np.uint64)).astype(np.int64)
            zs.append(blk[pc == n_occ])
        z = np.concatenate(zs)
        nz = len(z)
        self.lat, self.ns, self.nx, self.q = lat, ns, lat.nx, q
        self.z = np.concatenate([z, z])
        self.h = np.concatenate([np.zeros(nz, np.int8), np.ones(nz, np.int8)])
        self.dim = 2 * nz
        self.zs = np.empty((ns, self.dim), dtype=np.int8)
        for n in range(ns):
            self.zs[n] = 1 - 2 * ((self.z >> n) & 1)
        # e_0 = 1 - 2 h; e_n = s_n e_{n-1}, s_n = (-1)^n zs_n   (n >= 1)
        self.es = np.empty((ns, self.dim), dtype=np.int8)
        self.es[0] = 1 - 2 * self.h
        for n in range(1, ns):
            self.es[n] = self.es[n - 1] * ((-1) ** n * self.zs[n]).astype(np.int8)
        self.lut = np.full(1 << (ns + 1), -1, dtype=np.int32)
        self.lut[(self.z << 1) | self.h] = np.arange(self.dim, dtype=np.int32)
        # diagonal sums used by the vacuum ansatz layers and by H
        stag = ((-1) ** np.arange(ns)).astype(np.int16)
        self.stag_z = (stag[:, None] * self.zs.astype(np.int16)).sum(axis=0,
                                                                     dtype=np.int16)
        self.e_sum = self.es.sum(axis=0, dtype=np.int16)
        self._site_of = {lat.site_qubit(n): n for n in range(ns)}
        self._link_of = {lat.link_qubit(n): n for n in range(lat.n_links)}
        self._pairs = {}
        self._t2 = None
        self._orbits = None

    # ------------------------------------------------------------ indexing
    def index(self, z: int, h: int) -> int:
        r = int(self.lut[(int(z) << 1) | int(h)])
        if r < 0:
            raise ValueError("configuration outside the sector")
        return r

    def product_state(self, link_ref: str = "+") -> np.ndarray:
        """Strong-coupling reference (exact.strong_coupling_vacuum): odd
        sites occupied, all links |+> (h = 0) or |-> (h = 1)."""
        z = sum(1 << n for n in range(1, self.ns, 2))
        psi = np.zeros(self.dim, dtype=complex)
        psi[self.index(z, 0 if link_ref == "+" else 1)] = 1.0
        return psi

    # ------------------------------------------------- Pauli-string action
    def _grouped(self, op, rows=None):
        """Group the Pauli strings of a gauge-invariant op by their flip
        pattern (site mask, link-0 flip) and return
        {(zmask, lflip): phase array over rows}; the target row of each
        group is lut[((z ^ zmask) << 1) | (h ^ lflip)]."""
        idx = slice(None) if rows is None else rows
        nrow = self.dim if rows is None else len(rows)
        groups = {}
        for ops, c in _terms(op):
            phase = np.full(nrow, c, dtype=complex)
            zmask, lflip = 0, 0
            for qb, p in ops.items():
                if qb in self._site_of:
                    n = self._site_of[qb]
                    if p == "Z":
                        phase *= self.zs[n, idx]
                    elif p == "X":
                        zmask |= 1 << n
                    else:                                   # Y
                        phase *= 1j * self.zs[n, idx]
                        zmask |= 1 << n
                else:
                    m = self._link_of[qb]
                    if p == "X":
                        phase *= self.es[m, idx]
                    elif p == "Z":
                        lflip ^= 1 << m
                    else:                                   # Y
                        phase *= -1j * self.es[m, idx]
                        lflip ^= 1 << m
            key = (zmask, lflip & 1)
            groups[key] = groups[key] + phase if key in groups else phase
        return groups

    def _target(self, zmask, hflip, rows=None):
        z = self.z if rows is None else self.z[rows]
        h = self.h if rows is None else self.h[rows]
        return self.lut[((z ^ zmask) << 1) | (h ^ hflip)]

    def apply(self, op, v: np.ndarray) -> np.ndarray:
        """op @ v for any gauge-invariant, Q-conserving SparsePauliOp
        (generic path; the hop/cur generators and H have fast kernels)."""
        out = np.zeros(self.dim, dtype=complex)
        for (zmask, hflip), phase in self._grouped(op).items():
            if zmask == 0 and hflip == 0:
                out += phase * v
                continue
            col = self._target(zmask, hflip)
            m = col >= 0
            out[col[m]] += phase[m] * v[m]
        return out

    def expect(self, op, psi: np.ndarray) -> complex:
        return complex(np.vdot(psi, self.apply(op, psi)))

    # ------------------------------------------------------- bond pairing
    def bond_pairs(self, b: int):
        """(I, J, sig_hop, sig_cur) for bond b: M_hop[I,J] = sig_hop,
        M_cur[I,J] = i sig_cur with M = 2 G (see module docstring), I < J.
        Derived mechanically from wavepacket._bond_generator_full (seam
        sign included) and cached."""
        b = b % self.ns
        if b in self._pairs:
            return self._pairs[b]
        gh = self._grouped(_bond_generator_full(self.lat, b, "hop"))
        gc = self._grouped(_bond_generator_full(self.lat, b, "cur"))
        assert len(gh) == 1 and len(gc) == 1 and set(gh) == set(gc)
        (zmask, hflip), ph_h = next(iter(gh.items()))
        ph_c = gc[(zmask, hflip)]
        col = self._target(zmask, hflip)
        active = (col >= 0) & (np.abs(ph_h) > 1e-12)
        I = np.flatnonzero(active & (col > np.arange(self.dim)))
        J = col[I]
        sh = 2 * ph_h[J]                       # M[I, J] = phase(J)
        sc = -2j * ph_c[J]
        assert np.abs(sh.imag).max() < 1e-12 and np.abs(np.abs(sh.real) - 1).max() < 1e-12
        assert np.abs(sc.imag).max() < 1e-12 and np.abs(np.abs(sc.real) - 1).max() < 1e-12
        pairs = (I.astype(np.int32), J.astype(np.int32),
                 np.round(sh.real).astype(np.int8), np.round(sc.real).astype(np.int8))
        self._pairs[b] = pairs
        return pairs

    def apply_generator(self, kind: str, idx: int, v: np.ndarray) -> np.ndarray:
        """G_kind |v> for the block generators: cur/hop on bond idx (the
        1/4-normalized quadratures of wavepacket._bond_generator), site Z/2,
        link X/2 (the rz/rx generators of BlockEngine)."""
        out = np.zeros(self.dim, dtype=complex)
        if kind in ("hop", "cur"):
            I, J, sh, sc = self.bond_pairs(idx)
            if kind == "hop":
                out[I] = 0.5 * sh * v[J]
                out[J] = 0.5 * sh * v[I]
            else:
                out[I] = 0.5j * sc * v[J]
                out[J] = -0.5j * sc * v[I]
        elif kind == "site":
            out[:] = 0.5 * self.zs[idx % self.ns] * v
        elif kind == "link":
            out[:] = 0.5 * self.es[idx % self.ns] * v
        else:
            raise ValueError(kind)
        return out

    # ------------------------------------------------------- gate kernels
    def rotate_bond(self, psi: np.ndarray, kind: str, b: int, theta: float):
        """In place: psi <- exp(-i theta G_kind(b)) psi.
        hop: M[I,J] = M[J,I] = sh -> (c a - i s sh b, c b - i s sh a);
        cur: M[I,J] = i sc, M[J,I] = -i sc -> (c a + s sc b, c b - s sc a)."""
        I, J, sh, sc = self.bond_pairs(b)
        c, s = np.cos(theta / 2), np.sin(theta / 2)
        if kind == "hop":
            sg, bI, bJ = sh, -1j * s, -1j * s
        else:
            sg, bI, bJ = sc, s + 0j, -s + 0j
        if _HAVE_NUMBA:
            _k_rot_pairs(psi, I, J, sg, float(c), complex(bI), complex(bJ))
            return
        a, bb = psi[I], psi[J]
        psi[I] = c * a + bI * (sg * bb)
        psi[J] = c * bb + bJ * (sg * a)

    def rotate_diag(self, psi: np.ndarray, kind: str, n: int, theta: float):
        """In place: psi <- exp(-i theta d/2) psi, d = Z_site (rz) or
        X_link (rx)."""
        d = self.zs[n % self.ns] if kind == "site" else self.es[n % self.ns]
        c, ms = np.cos(theta / 2), -1j * np.sin(theta / 2)
        if _HAVE_NUMBA:
            _k_rot_diag(psi, d, float(c), complex(ms))
            return
        ph = d * ms
        ph += c
        psi *= ph

    def bracket_generator(self, lam: np.ndarray, kind: str, idx: int,
                          psi: np.ndarray) -> complex:
        """<lam| G_kind(idx) |psi> without forming G psi."""
        if kind in ("hop", "cur"):
            I, J, sh, sc = self.bond_pairs(idx)
            sg, uI, uJ = (sh, 0.5 + 0j, 0.5 + 0j) if kind == "hop" else (sc, 0.5j, -0.5j)
            if _HAVE_NUMBA:
                return complex(_k_bracket_pairs(lam, psi, I, J, sg, uI, uJ))
            return uI * np.vdot(lam[I], sg * psi[J]) + uJ * np.vdot(lam[J], sg * psi[I])
        d = self.zs[idx % self.ns] if kind == "site" else self.es[idx % self.ns]
        if _HAVE_NUMBA:
            return 0.5 * complex(_k_bracket_diag(lam, psi, d))
        return 0.5 * np.vdot(lam, d * psi)

    # ------------------------------------------------------- translation
    def translation(self):
        """(t, phi): T2 |j> = phi[j] |t[j]>, T2 = one-spatial-site
        translation with the ns = 0 mod 4 string twist, matching
        spectroscopy.translate / gaugefixed.PhysicalBasis.translation."""
        if self._t2 is not None:
            return self._t2
        ns = self.ns
        phi = np.ones(self.dim, dtype=np.int8)
        if ns % 4 == 0:
            tw = np.ones(self.dim, dtype=np.int8)
            for n in range(2, ns):
                tw = tw * self.zs[n]
            phi = -tw
        mask = (1 << ns) - 1
        z2 = ((self.z >> 2) | (self.z << (ns - 2))) & mask   # new bit n = old n+2
        h2 = ((1 - self.es[2 % ns]) // 2).astype(np.int64)  # new holonomy = old x_2
        t = self.lut[(z2 << 1) | h2]
        if np.any(t < 0):
            raise ValueError("translation left the sector")
        self._t2 = (t, phi)
        return self._t2

    def translate(self, psi: np.ndarray) -> np.ndarray:
        t, phi = self.translation()
        out = np.empty_like(psi)
        out[t] = phi * psi
        return out

    def t2_expect(self, psi: np.ndarray) -> complex:
        t, phi = self.translation()
        return complex(np.vdot(psi[t], phi * psi))

    def orbits(self):
        """Orbit bookkeeping for the T2 Bloch basis:
        rep[j]  smallest row in j's orbit,
        m[j]    shift with T2^m |rep> = chi[j] |j>,
        ell[j]  orbit length, Phi[j] = phase with T2^ell |rep> = Phi |rep>."""
        if self._orbits is not None:
            return self._orbits
        t, phi = self.translation()
        dim, nx = self.dim, self.nx
        j0 = np.arange(dim, dtype=np.int64)
        cur = j0.copy()
        ph = np.ones(dim, dtype=np.int8)
        rep = j0.copy()
        m_at_rep = np.zeros(dim, dtype=np.int16)
        ph_at_rep = np.ones(dim, dtype=np.int8)
        ell = np.zeros(dim, dtype=np.int16)
        Phi = np.zeros(dim, dtype=np.int8)
        for m in range(1, nx + 1):
            ph = ph * phi[cur]                 # T2^m |j> = ph |t[cur]>
            cur = t[cur]
            better = cur < rep
            rep[better] = cur[better]
            m_at_rep[better] = m
            ph_at_rep[better] = ph[better]
            closed = (cur == j0) & (ell == 0)
            ell[closed] = m
            Phi[closed] = ph[closed]
        assert np.all(ell > 0), "T2^nx != 1 on the sector"
        # T2^m |j> = ph |rep>  ->  |j> = ph T2^{nx-m} |rep>  (T2^nx = 1)
        mrep = (nx - m_at_rep) % nx
        chi = ph_at_rep.copy()
        is_rep = rep == j0
        mrep[is_rep] = 0
        chi[is_rep] = 1
        # orbit length / closure phase belong to the orbit: copy from rep
        ell = ell[rep]
        Phi = Phi[rep]
        self._orbits = (rep.astype(np.int32), mrep.astype(np.int16),
                        chi.astype(np.int8), ell.astype(np.int16), Phi.astype(np.int8))
        return self._orbits

    def momentum_grid(self) -> np.ndarray:
        """Sorted grid momenta 2 pi j / nx wrapped to (-pi, pi] (the
        ordering of spectroscopy.meson_band)."""
        k = 2 * np.pi * np.arange(self.nx) / self.nx
        k = np.where(k > np.pi + 1e-9, k - 2 * np.pi, k)
        return np.array(sorted(set(np.round(k, 12))))

    # ------------------------------------------------------- diagnostics
    def j0_profile(self, psi: np.ndarray) -> np.ndarray:
        """<J0(v)> = ((-1)^v - <Z_v>)/2 per staggered site (currents.py)."""
        p = np.abs(psi) ** 2
        zexp = self.zs.astype(float) @ p
        return 0.5 * ((-1.0) ** np.arange(self.ns) - zexp)

    def efield_profile(self, psi: np.ndarray) -> np.ndarray:
        """<sigma^x_m> per link."""
        p = np.abs(psi) ** 2
        return self.es.astype(float) @ p

    def memory_bytes(self) -> int:
        n = self.z.nbytes + self.h.nbytes + self.zs.nbytes + self.es.nbytes \
            + self.lut.nbytes + self.stag_z.nbytes + self.e_sum.nbytes
        for I, J, sh, sc in self._pairs.values():
            n += I.nbytes + J.nbytes + sh.nbytes + sc.nbytes
        if self._t2 is not None:
            n += self._t2[0].nbytes + self._t2[1].nbytes
        if self._orbits is not None:
            n += sum(a.nbytes for a in self._orbits)
        return n


# ======================================================== Hamiltonian
class GFHamiltonian:
    """H = (g2/2) sum X_link - (m0/2) sum (-1)^n Z_n + eta sum_b G_hop(b) on
    the sector; diagonal precomputed, hops through the bond pairing."""

    def __init__(self, space: GFSpace, m0, g2, eta):
        self.space, self.m0, self.g2, self.eta = space, m0, g2, eta
        self.diag = (g2 / 2) * space.e_sum.astype(float) \
            - (m0 / 2) * space.stag_z.astype(float)

    def apply(self, v: np.ndarray) -> np.ndarray:
        sp_ = self.space
        out = self.diag * v
        half = 0.5 * self.eta
        for b in range(sp_.ns):
            I, J, sh, _ = sp_.bond_pairs(b)
            out[I] += half * (sh * v[J])
            out[J] += half * (sh * v[I])
        return out

    def energy_stats(self, psi: np.ndarray) -> tuple[float, float]:
        """(<H>, sigma_E) with sigma_E from <H^2> - <H>^2."""
        hp = self.apply(psi)
        e = float(np.real(np.vdot(psi, hp)))
        var = float(np.real(np.vdot(hp, hp))) - e * e
        return e, float(np.sqrt(max(var, 0.0)))

    def linear_operator(self) -> spla.LinearOperator:
        return spla.LinearOperator((self.space.dim,) * 2, matvec=self.apply,
                                   dtype=complex)


# ====================================================== vacuum ansatz
def gf_vacuum_state(space: GFSpace, thetas, link_ref: str = "+") -> np.ndarray:
    """stateprep.vacuum_ansatz(lat, thetas, link_ref) simulated in the
    sector.  Layer l: even-bond hops exp(-i th_e H_hop-even(eta=1)), odd-bond
    hops, mass rz(-(-1)^n th_m) on every site, gauge rx(th_g) on every link
    (trotter._hop_layer / _single_qubit_layer conventions)."""
    thetas = np.asarray(thetas, dtype=float).reshape(-1, N_PARAMS_PER_LAYER)
    psi = space.product_state(link_ref)
    ns = space.ns
    for th_e, th_o, th_m, th_g in thetas:
        for b in range(0, ns, 2):
            space.rotate_bond(psi, "hop", b, th_e)
        for b in range(1, ns, 2):
            space.rotate_bond(psi, "hop", b, th_o)
        # rz(-(-1)^n th_m): exp(+i (-1)^n th_m Z/2); rx(th_g): exp(-i th_g X/2)
        psi *= np.exp(0.5j * (th_m * space.stag_z - th_g * space.e_sum))
    return psi


# ============================================================ the engine
class GFEngine:
    """BlockEngine API on the gauge-fixed sector: `keys`, `offsets`,
    state(vac, vec), fidelity_and_grad(vac, target, vec).  Gate order is
    that of wavepacket.block_circuit (layer, KINDS order, ascending
    offset), so trained vectors drop straight into circuits."""

    def __init__(self, space: GFSpace, center: int, n_layers: int,
                 offsets=None, max_offset: int | None = None):
        self.space, self.center, self.n_layers = space, center, n_layers
        self.offsets = window_offsets(space.ns) if offsets is None else list(offsets)
        if max_offset is not None:
            self.offsets = [o for o in self.offsets if abs(o) <= max_offset]
        self.gates = []                       # (key, kind, index)
        for l in range(n_layers):
            for kind in KINDS:
                for off in sorted(self.offsets):
                    self.gates.append(((l, kind, off), kind, (center + off) % space.ns))
        self.keys = [g[0] for g in self.gates]

    def _apply(self, psi, gate, theta):
        _, kind, idx = gate
        if kind in ("hop", "cur"):
            self.space.rotate_bond(psi, kind, idx, theta)
        else:
            self.space.rotate_diag(psi, kind, idx, theta)

    def state(self, vac: np.ndarray, vec) -> np.ndarray:
        psi = np.array(vac, dtype=complex, copy=True)
        for gate, th in zip(self.gates, vec):
            self._apply(psi, gate, th)
        return psi

    def fidelity_and_grad(self, vac: np.ndarray, target: np.ndarray, vec):
        """F = |<target|U(vec)|vac>|^2 and dF/dvec by the constant-memory
        adjoint sweep of block_engine.BlockEngine.fidelity_and_grad (three
        live vectors: post-gate state, adjoint state, target)."""
        post = self.state(vac, vec)
        o = np.vdot(target, post)
        lam = np.array(target, dtype=complex, copy=True)
        grad = np.empty(len(vec))
        for g in range(len(self.gates) - 1, -1, -1):
            gate, th = self.gates[g], vec[g]
            _, kind, idx = gate
            # d/dth <lam|U_g|s_pre> = <lam|(-i G)|s_post>
            br = -1j * self.space.bracket_generator(lam, kind, idx, post)
            grad[g] = 2 * np.real(np.conj(o) * br)
            self._apply(post, gate, -th)           # U_g^dag -> s_{g-1}
            self._apply(lam, gate, -th)
        return float(abs(o) ** 2), grad


# ============================================================== the band
def _sector_hamiltonian(space: GFSpace, H: GFHamiltonian, k: float):
    """H restricted to the T2 = e^{ik} Bloch sector:
      |k, r> = N_r^{-1/2} sum_m e^{-ikm} T2^m |r>   (r orbit representative,
      N_r = nx^2 / ell_r, nonzero iff e^{-ik ell_r} Phi_r = 1),
      <k, r'|H|k, r> = sum_{j in orbit(r')} <j|H|r> chi_j e^{ik m_j}
                       sqrt(ell_r / ell_r').
    Returns (H_k csr, R = representative rows, idx = row -> sector column)."""
    rep, mrep, chi, ell, Phi = space.orbits()
    j0 = np.arange(space.dim)
    reps = np.flatnonzero(rep == j0)
    comp = np.abs(np.exp(-1j * k * ell[reps]) * Phi[reps] - 1) < 1e-9
    R = reps[comp]
    nk = len(R)
    idx = np.full(space.dim, -1, dtype=np.int64)
    idx[R] = np.arange(nk)
    rows, cols, vals = [], [], []
    # diagonal
    rows.append(np.arange(nk)); cols.append(np.arange(nk))
    vals.append(H.diag[R].astype(complex))
    # hops: <j|H|r> = (eta/2) sig for the partner j of r on bond b
    for b in range(space.ns):
        I, J, sh, _ = space.bond_pairs(b)
        # partner map restricted to R via a scatter
        part = np.full(space.dim, -1, dtype=np.int64)
        part[I] = J
        part[J] = I
        sig = np.zeros(space.dim, dtype=np.int8)
        sig[I] = sh
        sig[J] = sh
        j = part[R]
        m = j >= 0
        r, j = R[m], j[m]
        c = 0.5 * H.eta * sig[r]
        rj = rep[j]
        ci = idx[rj]
        mm = ci >= 0
        r, j, c, rj, ci = r[mm], j[mm], c[mm], rj[mm], ci[mm]
        val = c * chi[j] * np.exp(1j * k * mrep[j]) * np.sqrt(ell[r] / ell[rj])
        rows.append(ci); cols.append(idx[r]); vals.append(val)
    Hk = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                       shape=(nk, nk)).tocsr()
    return Hk, R, idx


def _bloch_to_sector(space: GFSpace, v: np.ndarray, k: float, idx: np.ndarray) -> np.ndarray:
    """Sector eigenvector -> gf-basis vector:
    psi[j] = v[idx(rep_j)] e^{-ik m_j} chi_j / sqrt(ell_j)."""
    rep, mrep, chi, ell, _ = space.orbits()
    ci = idx[rep]
    m = ci >= 0
    psi = np.zeros(space.dim, dtype=complex)
    psi[m] = v[ci[m]] * np.exp(-1j * k * mrep[m]) * chi[m] / np.sqrt(ell[m])
    return psi


def gf_band(space: GFSpace, m0, g2, eta, k=None, n_per_k: int = 3,
            tol: float = 1e-9, degeneracy_tol: float = 1e-5, log=None) -> dict:
    """Vacuum + lowest state per grid momentum (the single-meson band when
    the lowest excitation per momentum is one meson, the same assumption
    as spectroscopy.meson_band), from per-sector diagonalization.

    k: None -> all grid momenta; else one momentum (must be on the grid).
    Convention T2|k> = e^{+ik}|k> (meson_band).  Returns
    {vacuum, e0, k, energy (gaps), states, multiplets, degeneracy, next_gap,
     levels (gaps of the n_per_k lowest levels per k), sector_dim}.

    Degenerate band levels: at k = pi the lowest level is EXACTLY two-fold
    degenerate at every volume (CT1 -- charge conjugation times the
    one-staggered-site shift, whose eigenvalues +-i at T2 = -1 are swapped
    by complex conjugation -- a Kramers-type pair; seen as a 6e-14 gap at
    ns = 6, 8).  spectroscopy.meson_band silently keeps one member, so a
    packet's k = pi weight depends on the eigensolver's basis choice.  Here
    `states[i]` is one member (API compatibility) and `multiplets[i]` the
    full list of states within degeneracy_tol of the band level; the
    projections and weights in this module use the multiplets."""
    import warnings

    H = GFHamiltonian(space, m0, g2, eta)
    grid = space.momentum_grid()
    ks = grid if k is None else np.array([k])
    if k is None:
        ks = grid
    if not all(np.min(np.abs(np.angle(np.exp(1j * (grid - kk))))) < 1e-9 for kk in ks):
        raise ValueError("momentum not on the grid 2 pi j / nx")
    t0 = time.time()
    res = {}
    all_k = sorted(set(list(ks) + [0.0]))
    for kk in all_k:
        Hk, R, idx = _sector_hamiltonian(space, H, kk)
        nk = Hk.shape[0]
        herm = abs(Hk - Hk.conj().T).max() if nk else 0.0
        if herm > 1e-9:
            raise RuntimeError(f"sector k={kk:+.4f} Hamiltonian not Hermitian ({herm:.2e})")
        need = n_per_k + (1 if abs(kk) < 1e-12 else 0)
        if nk <= max(400, 2 * need + 2):
            w, v = np.linalg.eigh(Hk.toarray())
            w, v = w[:need], v[:, :need]
        else:
            w, v = spla.eigsh(Hk, k=need, which="SA", tol=tol, ncv=max(2 * need + 1, 20))
            o = np.argsort(w)
            w, v = w[o], v[:, o]
        # convert only what is kept (vacuum, band level + degenerate
        # partners): every gf vector is 16 dim bytes (86 MB at ns = 24)
        i0 = 1 if abs(kk) < 1e-12 else 0
        keep = [0] if i0 else []
        keep.append(i0)
        for m in range(i0 + 1, len(w)):
            if w[m] - w[i0] < degeneracy_tol:
                keep.append(m)
            else:
                break
        vecs = {i: _bloch_to_sector(space, v[:, i], kk, idx) for i in keep}
        del Hk, v
        res[kk] = (w, vecs, nk)
        if log is not None:
            log(f"  sector k={kk:+.4f}: dim {nk}, levels {np.round(w, 5)} "
                f"[{time.time() - t0:.1f}s]")
    w0, v0, _ = res[0.0]
    vacuum, e0 = v0[0], float(w0[0])
    band_k, band_e, states, mults, degs, gaps, levels, sdim = [], [], [], [], [], [], [], []
    for kk in ks:
        w, vs, nk = res[kk]
        i = 1 if abs(kk) < 1e-12 else 0
        mult = [vs[m] for m in sorted(vs) if m >= i]
        if i + len(mult) >= len(w):
            warnings.warn(f"gf_band: k={kk:+.4f} multiplet fills all {len(w)} computed "
                          f"levels; raise n_per_k", RuntimeWarning)
            gap = np.nan
        else:
            gap = float(w[i + len(mult)] - w[i])
        band_k.append(kk)
        band_e.append(float(w[i] - e0))
        states.append(vs[i])
        mults.append(mult)
        degs.append(len(mult))
        gaps.append(gap)
        levels.append(np.asarray(w[i:] - e0))
        sdim.append(nk)
    return {"vacuum": vacuum, "e0": e0, "k": np.array(band_k),
            "energy": np.array(band_e), "states": states, "multiplets": mults,
            "degeneracy": np.array(degs), "next_gap": np.array(gaps),
            "levels": levels, "sector_dim": np.array(sdim)}


def _band_states(band: dict):
    """[(k index, state)] over every multiplet member (falls back to
    `states` for spectroscopy-style band dicts)."""
    if "multiplets" in band:
        return [(i, s) for i, ms in enumerate(band["multiplets"]) for s in ms]
    return list(enumerate(band["states"]))


# ======================================================== packet targets
def _wrap(dk):
    return (dk + np.pi) % (2 * np.pi) - np.pi


def gf_packet_vector(space: GFSpace, vac: np.ndarray, kind: str, parity: int,
                     k0: float, sigma_x: float, x0: int) -> np.ndarray:
    """Port of spectroscopy._packet_vector:
    sum_x f(x) e^{i k0 (x - x0)} O_kind(bond 2x + parity) |vac> with
    O_cur = currents.bond_current(eta=1) = -G_cur, O_hop = hop_term(eta=1)
    = G_hop (generator normalizations of wavepacket._bond_generator)."""
    nx = space.nx
    out = np.zeros(space.dim, dtype=complex)
    for x in range(nx):
        pos = x + (0.25 if parity == 0 else 0.75)
        dd = pos - (x0 + 0.25)
        d = (dd + nx / 2) % nx - nx / 2
        f = np.exp(-d ** 2 / (4 * sigma_x ** 2)) * np.exp(1j * k0 * d)
        gv = space.apply_generator(kind, 2 * x + parity, vac)
        out += (-f if kind == "cur" else f) * gv
    return out


def gf_optimize_interpolator(space: GFSpace, band: dict, k0: float = 0.0,
                             sigma_x: float = 1.0, x0: int | None = None) -> dict:
    """Port of spectroscopy.optimize_interpolator (4x4 generalized Rayleigh
    quotient over {cur, hop} x {even, odd bonds})."""
    import scipy.linalg

    if x0 is None:
        x0 = space.nx // 2
    basis = [("cur", 0), ("cur", 1), ("hop", 0), ("hop", 1)]
    vac = band["vacuum"]
    us = [gf_packet_vector(space, vac, kd, p, k0, sigma_x, x0) for kd, p in basis]
    S = np.array([s for _, s in _band_states(band)])      # (n_states, dim)
    pus = [S.T @ (S.conj() @ u) for u in us]
    A = np.array([[np.vdot(pi, pj) for pj in pus] for pi in pus])
    B = np.array([[np.vdot(ui, uj) for uj in us] for ui in us])
    B = B + 1e-12 * np.eye(len(basis))
    vals, vecs = scipy.linalg.eigh(A, B)
    c = vecs[:, -1]
    return {"mix": dict(zip(basis, c)), "band_fraction": float(vals[-1].real),
            "basis": basis}


def tune_k0_env(k_grid, K: float, sigma_x: float) -> float:
    """Port of scripts/train_cgkA_packets.py:tune_k0_env -- envelope momentum
    such that the band-basis Gaussian target's arg<T2> equals K on the
    discrete grid (identity for on-grid K, brentq detuning otherwise)."""
    from scipy.optimize import brentq

    kg = np.asarray(k_grid)

    def ph(k0e):
        w2 = np.exp(-2 * sigma_x ** 2 * _wrap(kg - k0e) ** 2)
        return np.angle(np.sum(w2 * np.exp(1j * kg)) * np.exp(-1j * K))

    lo, hi = K - 0.45, K + 0.45
    if ph(lo) * ph(hi) > 0:
        return K
    return float(brentq(ph, lo, hi, xtol=1e-6))


def gf_packet_target(space: GFSpace, band: dict, k0: float, sigma_x: float,
                     x0: int | None = None, mix: dict | None = None,
                     clean: bool = False):
    """Band-projected boosted Gaussian meson wavepacket.

    clean=False: spectroscopy.meson_wavepacket semantics -- the mix-weighted
      interpolator packet acting on the vacuum, projected on the band and
      normalized (the production-block target).  Returns (state, band
      fraction of the raw packet).
    clean=True: scripts/train_cgkA_packets.py:clean_target semantics -- exact
      Gaussian momentum amplitudes exp(-sigma_x^2 wrap(k - k0)^2), phases
      anchored by the raw projected packet (phase-convention-safe).  k0 is
      then the ENVELOPE momentum (use tune_k0_env for off-grid K).
    mix: {(kind, parity): coeff}; default the bare even-bond current.
    """
    if x0 is None:
        x0 = space.nx // 2
    if mix is None:
        mix = {("cur", 0): 1.0}
    vac = band["vacuum"]
    raw = np.zeros(space.dim, dtype=complex)
    for (kind, parity), c in mix.items():
        raw += c * gf_packet_vector(space, vac, kind, parity, k0, sigma_x, x0)
    members = _band_states(band)
    amps = np.array([np.vdot(s, raw) for _, s in members])
    proj = sum(a * s for a, (_, s) in zip(amps, members))
    norm = np.linalg.norm(proj)
    if norm < 1e-12:
        raise RuntimeError("interpolator has no overlap with the meson band")
    if not clean:
        return proj / norm, float(norm / np.linalg.norm(raw))
    # per-k direction inside the (possibly degenerate) multiplet, anchored
    # by the raw packet; exact Gaussian weight per k
    tgt = np.zeros(space.dim, dtype=complex)
    for i, k in enumerate(band["k"]):
        comp = sum(a * s for a, (ik, s) in zip(amps, members) if ik == i)
        nk = np.linalg.norm(comp)
        w = np.exp(-sigma_x ** 2 * _wrap(k - k0) ** 2)
        if nk > 1e-9:
            tgt += w * comp / nk
    return tgt / np.linalg.norm(tgt), float(norm / np.linalg.norm(raw))


# ============================================================ observables
def band_weights(band: dict, psi: np.ndarray):
    """(p_k array over band['k'] summed over degenerate multiplet members,
    vacuum weight)."""
    p = np.zeros(len(band["k"]))
    for i, s in _band_states(band):
        p[i] += abs(np.vdot(s, psi)) ** 2
    return p, float(abs(np.vdot(band["vacuum"], psi)) ** 2)


def sigma_k_from_weights(k, p, K: float):
    """(mean momentum, sigma_k) of a discrete momentum distribution,
    unwrapped around K."""
    k, p = np.asarray(k), np.asarray(p)
    dk = _wrap(k - K)
    w = p / p.sum()
    mean = float(K + np.sum(w * dk))
    var = float(np.sum(w * _wrap(k - mean) ** 2))
    return mean, float(np.sqrt(max(var, 0.0)))


def packet_report(space: GFSpace, H: GFHamiltonian, band: dict, psi: np.ndarray,
                  vac: np.ndarray, K: float, sigma_x: float) -> dict:
    """The certification numbers of one prepared state:
    band weights / purity (scripts/train_cgkA_packets.py:eval_gates gate 2),
    arg<T2> of the vacuum-subtracted state (gate 3), dE and sigma_E from
    <H>, <H^2> (gate 4), sigma_k from the per-k composition."""
    p, p_vac = band_weights(band, psi)
    p_band = float(p.sum())
    purity = p_band / max(1.0 - p_vac, 1e-15)
    a0 = np.vdot(band["vacuum"], psi)
    psi_x = psi - a0 * band["vacuum"]
    psi_x /= np.linalg.norm(psi_x)
    t2 = space.t2_expect(psi)
    t2x = space.t2_expect(psi_x)
    e, sig = H.energy_stats(psi)
    e_vac, _ = H.energy_stats(vac)
    kbar, sk = sigma_k_from_weights(band["k"], p, K)
    e_band = float(np.sum(p * band["energy"]) / p_band) if p_band > 0 else np.nan
    return dict(p_k=p, p_band=p_band, p_vac=p_vac, purity=purity,
                phase=float(np.angle(t2)), phase_x=float(np.angle(t2x)),
                absT2=float(abs(t2)), dE=e - e_vac, sigma_E=sig,
                E_band=e_band, kbar=kbar, sigma_k=sk,
                sigma_k_target=0.5 / sigma_x,
                j0=space.j0_profile(psi) - space.j0_profile(vac))


# ============================================================== embedding
def embed_vector(space: GFSpace, psi: np.ndarray) -> np.ndarray:
    """gf-basis vector -> full 2^(2 ns) statevector (qiskit bit order:
    site n = qubit 2n, link m = qubit 2m+1; links |+-> expanded in Z).
    Isometry from scripts/deep_levels_gf.py:embed, vectorized over the
    2^ns link patterns of every basis state (ns <= 12 practical)."""
    ns, lat = space.ns, space.lat
    nq = lat.n_qubits
    full = np.zeros(1 << nq, dtype=complex)
    xs = np.arange(1 << ns, dtype=np.int64)
    xspread = np.zeros(1 << ns, dtype=np.int64)
    for m in range(ns):
        xspread |= ((xs >> m) & 1) << lat.link_qubit(m)
    scale = 2.0 ** (-ns / 2)
    for j in np.flatnonzero(np.abs(psi) > 0):
        zspread = 0
        minus = 0
        for n in range(ns):
            if (int(space.z[j]) >> n) & 1:
                zspread |= 1 << lat.site_qubit(n)
            if space.es[n, j] < 0:
                minus |= 1 << n
        sign = 1.0 - 2.0 * (np.bitwise_count((xs & minus).astype(np.uint64)) % 2)
        full[zspread + xspread] += psi[j] * scale * sign
    return full


def project_vector(space: GFSpace, full: np.ndarray) -> np.ndarray:
    """Adjoint of embed_vector: full statevector -> gf-basis amplitudes
    psi[j] = <prod_j|full>.  The norm loss 1 - |psi|^2 is the weight
    outside the Gauss sector / charge sector (zero for any circuit built
    from gauge-invariant, Q-conserving gates on a physical state)."""
    ns, lat = space.ns, space.lat
    xs = np.arange(1 << ns, dtype=np.int64)
    xspread = np.zeros(1 << ns, dtype=np.int64)
    for m in range(ns):
        xspread |= ((xs >> m) & 1) << lat.link_qubit(m)
    scale = 2.0 ** (-ns / 2)
    zspread = np.zeros(space.dim, dtype=np.int64)
    for n in range(ns):
        zspread |= ((space.z >> n) & 1) << lat.site_qubit(n)
    minus = np.zeros(space.dim, dtype=np.int64)
    for n in range(ns):
        minus |= (space.es[n] < 0).astype(np.int64) << n
    psi = np.empty(space.dim, dtype=complex)
    for j in range(space.dim):
        sign = 1.0 - 2.0 * (np.bitwise_count((xs & minus[j]).astype(np.uint64)) % 2)
        psi[j] = scale * np.dot(sign, full[zspread[j] + xspread])
    return psi


def embed_isometry(space: GFSpace) -> np.ndarray:
    """Dense (2^(2ns), dim) isometry V (tests, ns <= 8)."""
    V = np.zeros((1 << space.lat.n_qubits, space.dim))
    for j in range(space.dim):
        e = np.zeros(space.dim, dtype=complex)
        e[j] = 1.0
        V[:, j] = embed_vector(space, e).real
    return V
