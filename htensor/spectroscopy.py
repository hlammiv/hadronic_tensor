"""Exact spectroscopy: translation quantum numbers, the single-meson band,
and boosted Gaussian meson wavepackets (small-volume ground truth).

The wavepacket construction is phase-convention-safe: a localized
interpolating operator (the vector bond current, i.e. exactly the current
that appears in W^{mu nu}) is applied to the vacuum with a boosted Gaussian
envelope, then projected onto the single-meson band with sum_s |s><s| --
independent of the arbitrary phases eigsh attaches to each |s>.
"""

import numpy as np

from .lattice import Z2Lattice
from . import hamiltonian as ham
from . import currents as cur
from . import exact
from .pauli import pauli_term


# ------------------------------------------------------------- translations
def _qubit_shift(psi: np.ndarray, n_qubits: int, shift: int = 4) -> np.ndarray:
    """Cyclic qubit relabeling: content of qubit (q + shift) moves to qubit q."""
    tensor = psi.reshape((2,) * n_qubits)
    perm_axis = [n_qubits - 1 - ((n_qubits - 1 - ax + shift) % n_qubits)
                 for ax in range(n_qubits)]
    return np.transpose(tensor, axes=perm_axis).reshape(-1)


def translate(psi: np.ndarray, lat: Z2Lattice) -> np.ndarray:
    """One-spatial-site translation T2, exact on the physical (Gauss) sector.

    In the JW representation the seam string is pinned to the qubit origin,
    so the naive qubit shift is the translation symmetry only for
    ns = 2 (mod 4), where the physical-sector seam sign is +1 (periodic
    wrap).  For ns = 0 (mod 4) (antiperiodic-like wrap, seam sign -1) the
    qubit shift must be preceded by the compensating string twist
    V = prod_{n=2}^{ns-1} Z_n, which relocates the wrap sign along with the
    seam.  [T2, H] = 0 holds on physical states only -- this operator is NOT
    a symmetry of the full JW Hilbert space, a point the paper must state.
    """
    if lat.pbc and lat.ns % 4 == 0:
        mask = 0
        for n in range(2, lat.ns):
            mask |= 1 << lat.site_qubit(n)
        signs = 1.0 - 2.0 * (np.bitwise_count(
            np.arange(psi.size, dtype=np.uint64) & np.uint64(mask)) % 2)
        # overall -1 normalizes the twist so the vacuum carries k = 0
        # (V on the strong-coupling vacuum = (-1)^(ns/2 - 1) = -1 here)
        psi = -psi * signs
    return _qubit_shift(psi, lat.n_qubits, 4)


def _t2_phases(states: list[np.ndarray], energies: np.ndarray, lat: Z2Lattice,
               degeneracy_tol: float = 1e-6):
    """Resolve T2 (one-spatial-site translation) eigenphases, diagonalizing
    T2 inside each degenerate energy cluster."""
    resolved_states, phases = [], []
    i = 0
    while i < len(states):
        j = i + 1
        while j < len(states) and abs(energies[j] - energies[i]) < degeneracy_tol:
            j += 1
        block = states[i:j]
        tblock = [translate(s, lat) for s in block]
        m = np.array([[np.vdot(a, tb) for tb in tblock] for a in block])
        vals, vecs = np.linalg.eig(m)
        for col, lam in enumerate(vals):
            s_new = sum(vecs[k, col] * block[k] for k in range(len(block)))
            s_new = s_new / np.linalg.norm(s_new)
            resolved_states.append(s_new)
            phases.append(np.angle(lam))
        i = j
    return resolved_states, np.array(phases)


# ------------------------------------------------------------- meson band
def meson_band(lat: Z2Lattice, m0, g2, eta, n_states: int | None = None) -> dict:
    """Vacuum + lowest physical Q=0 states resolved by momentum.

    Returns {vacuum, e0, k, energy, states}: for each momentum k = 2 pi j / Nx
    the lowest excited state and energy gap; the single-meson band when the
    lowest excitation per momentum is one meson (true at strong-ish coupling).
    """
    nx = lat.nx
    if n_states is None:
        n_states = 2 * nx + 4
    energies, vecs = exact.lowest_physical_states(lat, m0, g2, eta, k=n_states)
    states = [vecs[:, i] for i in range(vecs.shape[1])]
    resolved, phases = _t2_phases(states, energies, lat)
    # recompute energies for resolved combinations (unchanged within clusters)
    H = exact.to_sparse(ham.build_hamiltonian(lat, m0, g2, eta))
    e_res = np.array([np.real(np.vdot(s, H @ s)) for s in resolved])
    order = np.argsort(e_res)
    resolved = [resolved[i] for i in order]
    e_res, phases = e_res[order], phases[order]

    vacuum, e0 = resolved[0], e_res[0]
    k_grid = 2 * np.pi * np.arange(nx) / nx  # in (-pi, pi] after wrap
    k_grid = np.where(k_grid > np.pi + 1e-9, k_grid - 2 * np.pi, k_grid)
    band_k, band_e, band_states = [], [], []
    for k in sorted(set(np.round(k_grid, 12))):
        for s, e, ph in zip(resolved[1:], e_res[1:], phases[1:]):
            # convention: T2|k> = e^{+ik}|k>, aligned so a wavepacket built
            # with envelope e^{+i k0 x} has arg<T2> = +k0 (verified at
            # ns = 6 and 8); absolute direction vs +x is anchored against
            # the group velocity in the analysis layer
            if abs((np.angle(np.exp(1j * (k - ph))))) < 1e-4:
                band_k.append(k)
                band_e.append(e - e0)
                band_states.append(s)
                break
    return {"vacuum": vacuum, "e0": e0, "k": np.array(band_k),
            "energy": np.array(band_e), "states": band_states,
            "all_energies": e_res, "all_phases": phases}


# ------------------------------------------------------------- wavepacket
def meson_wavepacket(lat: Z2Lattice, band: dict, k0: float = 0.0,
                     sigma_x: float = 1.0, x0: int | None = None) -> np.ndarray:
    """Boosted Gaussian meson wavepacket, band-projected (see module docstring).

    k0 in units of inverse spatial sites (2 pi j / Nx), sigma_x / x0 in
    spatial sites; x0 defaults to mid-ring.
    """
    nx = lat.nx
    if x0 is None:
        x0 = nx // 2
    vac = band["vacuum"]
    raw = np.zeros_like(vac)
    for x in range(nx):
        d = min(abs(x - x0), nx - abs(x - x0))  # ring distance
        f = np.exp(-d**2 / (4 * sigma_x**2)) * np.exp(1j * k0 * (x - x0))
        op = exact.to_sparse(cur.bond_current(lat, 2 * x, eta=1.0))
        raw = raw + f * (op @ vac)
    proj = np.zeros_like(raw)
    for s in band["states"]:
        proj = proj + s * np.vdot(s, raw)
    norm = np.linalg.norm(proj)
    if norm < 1e-12:
        raise RuntimeError("interpolator has no overlap with the meson band")
    return proj / norm, float(np.linalg.norm(proj) / np.linalg.norm(raw))


def electric_energy_profile(lat: Z2Lattice, psi: np.ndarray, vacuum: np.ndarray,
                            g2: float = 1.0) -> np.ndarray:
    """Vacuum-subtracted electric (link) energy per spatial site: the string
    of the meson shows up as a localized deficit in <sigma^x_link>."""
    out = np.empty(lat.nx)
    for x in range(lat.nx):
        val = 0.0
        for b in (2 * x, 2 * x + 1):
            op = exact.to_sparse(
                pauli_term(lat.n_qubits, {lat.link_qubit(b): "X"}, g2 / 2))
            val += np.real(np.vdot(psi, op @ psi) - np.vdot(vacuum, op @ vacuum))
        out[x] = -val  # deficit is positive energy
    return out
