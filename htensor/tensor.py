"""All four components of the hadronic tensor from the production grids.

Component naming: C^{mu nu}(x, t) = <psi| J^mu(x, t) J^nu(0, 0) |psi>, so mu
labels the PROBE (later time) and nu the INSERTION (t = 0).  The production
files hold two probe blocks per insertion:

  data/w_meson_ns50_k*_v3.npz   insert J^0 at site CENTER
        corr[:, :ns]  = J^0 probes at sites v      -> C^{00}
        corr[:, ns:]  = J^1 probes at bonds b      -> C^{10}
  data/w_meson_j1_ns50_k*.npz   insert J^1 at bond CENTER
        corr[:, :ns]  = J^0 probes                 -> C^{01}
        corr[:, ns:]  = J^1 probes                 -> C^{11}

Positions are in spatial-site units relative to the insertion (staggered
site v sits at v/2, bond b at (b + 1/2)/2), minimal-image folded on the ring.
The transform is the production one-sided windowed transform
(analysis.onesided_ft).  Lattice Ward identity (exact operator statement,
tests/test_m1_operators.py::test_continuity_equation):

    d/dt J^0(v, t) = -[J^1_{v+1/2}(t) - J^1_{v-1/2}(t)],

whose momentum-space form is q^0 W^{0 nu} = qhat W^{1 nu} with the lattice
momentum qhat = 2 sin(q^1 / 4) (the point-split current shifts by +-1/4 of a
spatial site).  For the finite one-sided windowed transform the q-space form
acquires boundary (t = 0) and window-derivative terms, so the exact check is
the pointwise integral form on the (x, t) grid (continuity_residual_xt);
ward_residual_q reports the size of the q-space systematic.
"""

from dataclasses import dataclass

import numpy as np

from . import analysis
from .lattice import Z2Lattice

COMPONENTS = ("00", "10", "01", "11")
# (source file, probe block, insertion kind)
_LAYOUT = {"00": ("j0", 0, 0), "10": ("j0", 1, 0),
           "01": ("j1", 0, 1), "11": ("j1", 1, 1)}


@dataclass
class TensorGrids:
    """Connected, vacuum-subtracted correlators for the four components."""

    lat: Z2Lattice
    center: int
    k0: float
    times: np.ndarray
    g: dict            # comp -> (nt, ns) complex, connected-minus-connected
    x: dict            # comp -> (ns,) positions (spatial units, ring-folded)
    raw_wp: dict       # comp -> raw wavepacket correlator (for continuity)
    one_pt_wp: dict    # comp -> <J^mu(x,t)>_wp for the probe block
    meta: dict

    @property
    def dt(self):
        return float(self.times[1] - self.times[0])


def x_grid(lat: Z2Lattice, comp: str, center: int, fold: bool = True) -> np.ndarray:
    """Probe positions relative to the insertion, spatial units."""
    ns = lat.ns
    probe_is_bond = _LAYOUT[comp][1] == 1
    ins_is_bond = _LAYOUT[comp][2] == 1
    probe = np.arange(ns) + (0.5 if probe_is_bond else 0.0)
    ins = center + (0.5 if ins_is_bond else 0.0)
    x = (probe - ins) / 2.0
    return analysis.ring_fold(x, lat.nx) if fold else x


def _connected(corr, one_pt, ins):
    return analysis.subtract(corr, None, one_pt, complex(ins))


def load_mps_components(path_j0: str, path_j1: str | None = None) -> TensorGrids:
    """Load the production npz files and form the connected-minus-connected
    combination [C_wp - <J><J>_wp] - [C_vac - <J><J>_vac] for every available
    component (exactly as scripts/analyze_w_meson.py does for W^{00})."""
    files = {"j0": np.load(path_j0)}
    if path_j1 is not None:
        files["j1"] = np.load(path_j1)
    d0 = files["j0"]
    ns, center, k0 = int(d0["ns"]), int(d0["center"]), float(d0["k0"])
    lat = Z2Lattice(ns, pbc=True)
    times = np.asarray(d0["times"], float)
    g, x, raw, one = {}, {}, {}, {}
    for comp, (src, block, _) in _LAYOUT.items():
        if src not in files:
            continue
        d = files[src]
        assert int(d["ns"]) == ns and int(d["center"]) == center
        assert np.allclose(d["times"], times)
        sl = slice(block * ns, (block + 1) * ns)
        g_wp = _connected(d["corr_wp"][:, sl], d["one_pt_wp"][:, sl], d["insert_1pt_wp"])
        g_vac = _connected(d["corr_vac"][:, sl], d["one_pt_vac"][:, sl], d["insert_1pt_vac"])
        g[comp] = g_wp - g_vac
        x[comp] = x_grid(lat, comp, center)
        raw[comp] = np.asarray(d["corr_wp"][:, sl], complex)
        one[comp] = np.asarray(d["one_pt_wp"][:, sl], complex)
    meta = {k: float(d0[k]) for k in ("m0", "g2", "eta", "dt_target", "trunc")
            if k in d0.files}
    return TensorGrids(lat, center, k0, times, g, x, raw, one, meta)


# --------------------------------------------------------------- transforms
def default_windows(grids: TensorGrids, hardware: bool = False):
    """Production windows: sigma_t = t_max/3; sigma_x = nx/6 (MPS) or nx/2
    (hardware assembly, scripts/hw_w00_coarse.py)."""
    return grids.times[-1] / 3.0, grids.lat.nx / (2.0 if hardware else 6.0)


def assemble_W(grids: TensorGrids, q0, q1, sigma_t=None, sigma_x=None,
               factors=(0.75, 1.0, 1.5), comps=None):
    """-> {comp: (W real (nq0, nq1), spread)} via the one-sided transform,
    window widths scanned by `factors` for the systematic (central = the
    middle factor)."""
    st, sx = default_windows(grids)
    sigma_t = st if sigma_t is None else sigma_t
    sigma_x = sx if sigma_x is None else sigma_x
    out = {}
    for comp in (comps or grids.g):
        ws = np.stack([analysis.onesided_ft(grids.times, grids.x[comp], grids.g[comp],
                                            q0, q1, f * sigma_t, f * sigma_x,
                                            grids.dt, 0.5) for f in factors])
        out[comp] = (ws[len(factors) // 2], np.ptp(ws, axis=0))
    return out


def qhat(q1):
    """Lattice momentum of the point-split current: 2 sin(q1/4)."""
    return 2.0 * np.sin(np.asarray(q1, float) / 4.0)


# --------------------------------------------------------------- checks
def _div_bonds(c_bond):
    """div at site v from a bond grid: J1_{v+1/2} - J1_{v-1/2} = c[b=v] - c[b=v-1]."""
    return c_bond - np.roll(c_bond, 1, axis=1)


def continuity_residual_xt(grids: TensorGrids, use_raw: bool = True,
                           form: str = "integral") -> dict:
    """Probe-side continuity on the (x, t) grid, per insertion nu.

    form='integral' (paper convention, scripts/final_ridge_overlay.py):
        C^{0 nu}(t) - C^{0 nu}(0) = -int_0^t div C^{1 nu} dt'  (cumulative Simpson)
        residual = max|lhs - rhs| / max|lhs| over t > 0.
    form='differential': central differences d_t C^{0 nu} + div C^{1 nu};
        at the 0.5 production output spacing this is dominated by the
        O((omega dt)^2) finite-difference error (~45%), so use it only on
        fine grids (tests).  On the raw grids the identity is exact up to
        Trotter/truncation; on the connected grids the one-point terms
        cancel identically as well."""
    from scipy.integrate import cumulative_simpson
    out = {}
    src = grids.raw_wp if use_raw else grids.g
    for nu, (c0, c1) in {"0": ("00", "10"), "1": ("01", "11")}.items():
        if c0 not in src or c1 not in src:
            continue
        dt = grids.dt
        div = _div_bonds(src[c1])
        if form == "integral":
            lhs = src[c0][1:] - src[c0][:1]
            rhs = -cumulative_simpson(div, dx=dt, axis=0, initial=0.0)[1:]
        elif form == "differential":
            lhs = (src[c0][2:] - src[c0][:-2]) / (2 * dt)
            rhs = -div[1:-1]
        else:
            raise ValueError(form)
        out[nu] = float(np.abs(lhs - rhs).max() / np.abs(lhs).max())
    return out


def ward_residual_q(W: dict, q0, q1) -> dict:
    """Momentum-space Ward form q0 W^{0 nu} - qhat W^{1 nu} per nu, normalised
    by max|q0 W^{0 nu}| over the grid.  NOT exact for the windowed one-sided
    transform (t = 0 boundary and window-derivative terms), reported as the
    size of that systematic."""
    q0 = np.asarray(q0, float)[:, None]
    qh = qhat(q1)[None, :]
    out = {}
    for nu, (c0, c1) in {"0": ("00", "10"), "1": ("01", "11")}.items():
        if c0 not in W or c1 not in W:
            continue
        a = q0 * W[c0][0]
        b = qh * W[c1][0]
        out[nu] = float(np.abs(a - b).max() / np.abs(a).max())
    return out


def ward_converted_W11(W00, q0, q1):
    """W^{11} implied by W^{00} through the Ward identity:
    (q0/qhat)^2 W^{00} (scripts/kubo_conductivity.py convention).  Undefined
    at q1 = 0 (returns nan there)."""
    qh = qhat(q1)[None, :]
    q0 = np.asarray(q0, float)[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        out = (q0 / qh) ** 2 * W00
    out[:, np.isclose(qhat(q1), 0.0)] = np.nan
    return out


def parity_residuals(grids: TensorGrids) -> dict:
    """Reflection symmetry of a REST packet, per component, about the
    insertion point: bond insertions (01, 11) sit on the block-window centre
    (staggered center + 1/2), where reflection is the PC symmetry of the
    staggered ring (sublattices swap), and J^0 (PC-odd) / J^1 (PC-even)
    give C^{01} odd, C^{11} even in x -- exact for a symmetric packet.  Site
    insertions (00, 10) are reflected about the insertion SITE (plain P:
    C^{00} even, C^{10} odd), which is 1/4 spatial site off the envelope
    centre, so those residuals are only approximate (packet-offset
    effect).  Returns max|C(x) -+ C(-x)| / max|C|."""
    ns = grids.lat.ns
    out = {}
    for comp, sign in (("00", +1), ("11", +1), ("01", -1), ("10", -1)):
        if comp not in grids.g:
            continue
        probe_is_bond = _LAYOUT[comp][1] == 1
        ins_is_bond = _LAYOUT[comp][2] == 1
        xc = grids.center + (0.5 if ins_is_bond else 0.0)
        pos = np.arange(ns) + (0.5 if probe_is_bond else 0.0)
        refl = (2 * xc - pos) % ns
        idx = np.array([int(np.argmin(np.abs(((pos - r + ns / 2) % ns) - ns / 2)))
                        for r in refl])
        if not np.allclose(pos[idx] % ns, refl):
            out[comp] = float("nan")
            continue
        c = grids.g[comp]
        out[comp] = float(np.abs(c - sign * c[:, idx]).max() / np.abs(c).max())
    return out


def insertion_ward_residual(W: dict, q0, q1) -> dict:
    """Insertion-side Ward form q0 W^{mu 0} - qhat W^{mu 1} per probe mu.
    Holds for ENERGY EIGENSTATES only (time-translation invariance moves the
    derivative from the probe to the insertion); for a wavepacket it is
    violated at O(sigma_E / q0).  Diagnostic of eigenstate-likeness."""
    q0 = np.asarray(q0, float)[:, None]
    qh = qhat(q1)[None, :]
    out = {}
    for mu, (c0, c1) in {"0": ("00", "01"), "1": ("10", "11")}.items():
        if c0 not in W or c1 not in W:
            continue
        a = q0 * W[c0][0]
        b = qh * W[c1][0]
        out[mu] = float(np.abs(a - b).max() / np.abs(a).max())
    return out
