"""Wavepacket-block training and certification chain (WS2-B/C, 2026-09).

Generalizes scripts/train_cgkA_packets.py (CGK-A couplings, sigma_x = 0.75,
ns = 10 statevector training) to arbitrary couplings, packet widths and
volumes.  Momentum-narrow packets (sigma_x >= 1 spatial site) do not fit
ns <= 12 rings, so training/certification run in the gauge-fixed charge-zero
basis (htensor/gf_engine.py: exact circuit simulation, per-momentum band,
targets, observables) at ns = 16-26; `--engine sv` keeps the original
statevector path (block_engine.BlockEngine + the contiguous-GEMM kernel of
train_cgkA_packets.py:57-96) for ns <= 12 cross-checks.

  PYTHONPATH=. .venv/bin/python scripts/train_packets.py MODE [options]

Modes
  vacuum   optimize (ns=6, 2 layers, 2 restarts) or load the vacuum angles
           -> data/vac_<tag>.npz (stateprep.save_vacuum convention).
           GATE 1: F >= 0.999 vs the exact ns=8 vacuum; E/ns at 6/8/10.
  band     E(k), level degeneracies and gaps at --ns-train and --ns-cert.
  train    train a block at --ns-train against the band-projected Gaussian
           target, then verify at --ns-train and every --ns-cert.
  verify   gates for an existing --params file at every --ns-cert.
  summary  table of the `cert` dicts of data/wp_<tag>_*.npz (or --params).

Packet geometry: K = --k0 is the target mean momentum (2 pi j / nx of the
production ring); the envelope momentum k0_env is detuned per volume so
the discrete-grid target has arg<T2> = K exactly (tune_k0_env).  x0 = nx//2
spatial sites, block center = 2 x0 staggered sites (production: ns=10 ->
x0=2, center=4).  --max-offset auto = max(4, ceil(6 sigma_x)) staggered
offsets on each side (+-3 sigma_x in staggered units, floor 4 = the
production window; sigma_x = 0.75 -> 5 under this rule, so the production
block is reproduced with an explicit --max-offset 4).

Results 2026-09-02 (production couplings, K = 2 pi/5, clean target, window
auto, trained at ns=20, certified at ns=20 and 24; see data/wp_prod_*.npz
`cert`):  sigma_x = 0.75 (+-4, L=3, 108 params): F 0.993, 426 CZ at ns=50;
sigma_x = 1.0 (+-6, L=3, 156 params): F 0.984, purity 0.989, 618 CZ (L=2
saturates at F 0.93); sigma_x = 1.5 (+-9, 19 offsets, L=3, 228 params)
saturates at F 0.935 for any iteration budget -- depth is the limit: L=4
(304 params) reaches F 0.978/0.981 (ns=20/24), purity 0.988, 1208 CZ.
The default anneal l2 = 5e-4 is too strong for the wide blocks (drops F by
~0.015 that the short last stage cannot recover); 0,1e-4,5e-6 with
800,300,400 iterations is the schedule that produced the quoted numbers.

Gates (train_cgkA_packets.py:eval_gates, extended):
  2  band purity p_band / (1 - p_vac) >= 0.98
  3  |arg<T2>_excited - K| <= 0.05
  4  E(K) - 0.05 <= dE <= E(K) + 0.10   (E(K) from this volume's band)
  4' |dE - E_target| <= 0.05, E_target = band-mean energy of the exact
     target (dispersion-aware; gate 4 fails a correct K=0 packet on a
     curved band, e.g. relA where E(1.26) - M = 0.26)
  5  sigma_E from <H^2> (reported)
  6  transfer F_train - F_cert <= 0.01
  7  |sigma_k_meas / (1 / 2 sigma_x) - 1| <= 0.15  (exact per-k weights)
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np

from htensor import Z2Lattice, stateprep, wavepacket, block_engine
from htensor import gf_engine as gfe

t0 = time.time()


def log(m):
    print(f"[{time.time() - t0:7.1f}s] {m}", flush=True)


# ------------------------------------------------- fast local-gate kernel
# Verbatim from scripts/train_cgkA_packets.py:57-96 (contiguous-qubit gates
# as one batched GEMM; 3-5x faster BlockEngine gradients).
_apply_local_orig = block_engine._apply_local


def _apply_local_fast(psi, U, qubits, n):
    qs = sorted(qubits)
    if qs != list(range(qs[0], qs[0] + len(qubits))):
        return _apply_local_orig(psi, U, qubits, n)
    k, q0 = len(qubits), qs[0]
    D, B = 1 << k, 1 << q0
    A = psi.size // (D * B)
    p = np.zeros(D, dtype=np.intp)
    for s in range(D):
        loc = 0
        for i, ql in enumerate(qubits):
            if (s >> (ql - q0)) & 1:
                loc |= 1 << i
        p[s] = loc
    Ua = U[np.ix_(p, p)]
    out = np.matmul(Ua, psi.reshape(A, D, B))
    return out.reshape(-1)


def _selftest_fast_apply():
    rng = np.random.default_rng(3)
    lat = Z2Lattice(8, pbc=True)
    eng = block_engine.BlockEngine(lat, 3, 2, max_offset=2)
    psi = rng.standard_normal(2 ** lat.n_qubits) + 1j * rng.standard_normal(
        2 ** lat.n_qubits)
    psi /= np.linalg.norm(psi)
    vec = 0.3 * rng.standard_normal(len(eng.keys))
    ref = eng.state(psi, vec)
    block_engine._apply_local = _apply_local_fast
    new = eng.state(psi, vec)
    block_engine._apply_local = _apply_local_orig
    err = np.abs(new - ref).max()
    assert err < 1e-12, f"fast kernel mismatch: {err}"


_selftest_fast_apply()
block_engine._apply_local = _apply_local_fast


# ------------------------------------------------------------- helpers
def geometry(ns):
    """(x0 spatial, center staggered) -- production convention."""
    nx = ns // 2
    return nx // 2, 2 * (nx // 2)


def auto_offset(sigma_x):
    return max(4, math.ceil(6 * sigma_x))


def packet_file(tag, K, sigma_x, L):
    return f"data/wp_{tag}_k{K:+.2f}_s{sigma_x:.2f}_L{L}.npz"


def couplings(a):
    return float(a.m0), float(a.g2), float(a.eta)


class BandCache:
    """gf spaces and bands per volume (bands also cached on disk: the ns=24
    band is ~6 min of sector eigsh)."""

    def __init__(self, a):
        self.a = a
        self.spaces, self.bands = {}, {}

    def space(self, ns):
        if ns not in self.spaces:
            t = time.time()
            self.spaces[ns] = gfe.GFSpace(Z2Lattice(ns, pbc=True))
            log(f"ns={ns}: gf space dim {self.spaces[ns].dim} ({time.time() - t:.1f}s)")
        return self.spaces[ns]

    def band(self, ns):
        if ns in self.bands:
            return self.bands[ns]
        m0, g2, eta = couplings(self.a)
        fn = os.path.join(self.a.cache_dir, f"band_{self.a.tag}_ns{ns}.npz")
        sp_ = self.space(ns)
        if os.path.exists(fn) and not self.a.no_cache:
            z = np.load(fn)
            if np.allclose(z["couplings"], [m0, g2, eta]) and z["vacuum"].size == sp_.dim:
                # NpzFile re-reads an array on EVERY __getitem__: pull the
                # (n_states, dim) block out exactly once (1.1 GB at ns=24)
                # and hand out views (a per-access comprehension held 13
                # full copies -> 10 GB RSS, 2026-09-02)
                ks = z["mult_k"]
                ms = z["mult_states"]
                band = {"vacuum": z["vacuum"], "e0": float(z["e0"]), "k": z["k"],
                        "energy": z["energy"], "degeneracy": z["degeneracy"],
                        "next_gap": z["next_gap"], "levels": list(z["levels"]),
                        "multiplets": [[ms[i] for i in np.flatnonzero(ks == j)]
                                       for j in range(len(z["k"]))]}
                del z
                band["states"] = [m[0] for m in band["multiplets"]]
                log(f"ns={ns}: band loaded from {fn}")
                self.bands[ns] = band
                return band
        t = time.time()
        band = gfe.gf_band(sp_, m0, g2, eta, log=log if ns >= 20 else None)
        log(f"ns={ns}: band in {time.time() - t:.1f}s")
        if not self.a.no_cache:
            os.makedirs(self.a.cache_dir, exist_ok=True)
            mult_k = np.concatenate([[j] * len(m) for j, m in enumerate(band["multiplets"])])
            lv = np.full((len(band["k"]), max(len(l) for l in band["levels"])), np.nan)
            for j, l in enumerate(band["levels"]):
                lv[j, :len(l)] = l
            np.savez(fn, couplings=[m0, g2, eta], vacuum=band["vacuum"], e0=band["e0"],
                     k=band["k"], energy=band["energy"], degeneracy=band["degeneracy"],
                     next_gap=band["next_gap"], levels=lv, mult_k=mult_k,
                     mult_states=np.array([s for m in band["multiplets"] for s in m]))
        self.bands[ns] = band
        return band


def band_table(band):
    return " ".join(f"{k:+.3f}:{e:.4f}{'*' * (d - 1)}"
                    for k, e, d in zip(band["k"], band["energy"], band["degeneracy"]))


def band_energy_at(band, K):
    """E(K) from this volume's band (exact on-grid; cubic in |k| off-grid)."""
    from scipy.interpolate import CubicSpline
    k, e = np.abs(band["k"]), band["energy"]
    o = np.argsort(k)
    ku, idx = np.unique(np.round(k[o], 9), return_index=True)
    eu = e[o][idx]
    if len(ku) < 3:
        return float(np.interp(abs(K), ku, eu))
    return float(CubicSpline(ku, eu, bc_type=((1, 0.0), (1, 0.0)))(abs(K)))


def load_vac(a):
    d = stateprep.load_vacuum(a.vac)
    if d["m0"] is not None and not np.allclose([d["m0"], d["g2"], d["eta"]], couplings(a)):
        raise SystemExit(f"{a.vac} was optimized for {d['m0'], d['g2'], d['eta']}, "
                         f"not {couplings(a)}")
    return d


def make_target(space, band, K, sigma_x, x0, kind):
    """Band-projected Gaussian target at this volume: (target, info)."""
    k0_env = gfe.tune_k0_env(band["k"], K, sigma_x) if kind == "clean" else K
    mix = gfe.gf_optimize_interpolator(space, band, k0=k0_env, sigma_x=sigma_x, x0=x0)
    tgt, frac = gfe.gf_packet_target(space, band, k0_env, sigma_x, x0, mix["mix"],
                                     clean=(kind == "clean"))
    return tgt, dict(k0_env=float(k0_env), band_fraction=frac, mix=mix["mix"])


def gf_lsq_init(space, vac, target, center, offsets, n_layers):
    """Port of wavepacket.lsq_init on the gf basis (layer-0 cur/hop seed)."""
    import scipy.linalg

    cols, keys = [], []
    for kind in ("cur", "hop"):
        for off in offsets:
            m = -1j * space.apply_generator(kind, (center + off) % space.ns, vac)
            m = m - vac * np.vdot(vac, m)
            cols.append(m)
            keys.append((0, kind, off))
    a = np.array([np.vdot(target, m) for m in cols])
    G = np.array([[np.vdot(mi, mj) for mj in cols] for mi in cols])
    A = np.real(np.outer(np.conj(a), a))
    vals, vecs = scipy.linalg.eigh((A + A.T) / 2, np.real(G) + 1e-10 * np.eye(len(a)))
    theta = np.real(vecs[:, -1])
    theta = theta / (np.abs(theta).max() + 1e-12)
    full = np.zeros(wavepacket.n_params(offsets, n_layers))
    lut = {k: i for i, k in enumerate(
        [(l, kind, off) for l in range(n_layers) for kind in wavepacket.KINDS
         for off in offsets])}
    for k, th in zip(keys, theta):
        full[lut[k]] = th
    return full


def conj_map(vec, offsets, n_layers):
    """train_cgkA_packets.py:conj_map -- +K solution -> -K warm start."""
    out = vec.copy()
    i = 0
    for _l in range(n_layers):
        for kind in wavepacket.KINDS:
            for _o in offsets:
                if kind != "cur":
                    out[i] = -out[i]
                i += 1
    return out


def remap_vector(vec, offsets_from, L_from, offsets_to, L_to):
    """Zero-padded re-keying of a block vector onto another window/depth."""
    src = wavepacket.params_from_vector(vec, offsets_from, L_from)
    out = np.zeros(wavepacket.n_params(offsets_to, L_to))
    i = 0
    for l in range(L_to):
        for kind in wavepacket.KINDS:
            for off in offsets_to:
                out[i] = src.get((l, kind, off), 0.0)
                i += 1
    return out


def prepare_state(a, ns, vacd, vec, offsets, L, space):
    """Prepared packet state in the gf basis (engine gf, or sv for ns <= 12:
    full-space circuit projected into the sector; returns the leak)."""
    x0, center = geometry(ns)
    if a.cert_engine == "sv" and ns <= 12:
        lat = Z2Lattice(ns, pbc=True)
        vac = stateprep.ansatz_state(lat, vacd["thetas"], link_ref=vacd["link_ref"])
        eng = block_engine.BlockEngine(lat, center, L, offsets=list(offsets))
        psi = gfe.project_vector(space, eng.state(vac, vec))
        vacg = gfe.project_vector(space, vac)
        leak = 1.0 - float(np.vdot(psi, psi).real)
        return psi / np.linalg.norm(psi), vacg / np.linalg.norm(vacg), leak
    vacg = gfe.gf_vacuum_state(space, vacd["thetas"], vacd["link_ref"])
    eng = gfe.GFEngine(space, center, L, offsets=list(offsets))
    return eng.state(vacg, vec), vacg, 0.0


def cz_count(vec, offsets, L, ns=50):
    """Two-qubit gate count of the block (and the vacuum ansatz) at ns=50,
    basis rz/sx/x/cz, optimization_level=1."""
    from qiskit import transpile
    lat = Z2Lattice(ns, pbc=True)
    params = wavepacket.params_from_vector(vec, list(offsets), L)
    _, center = geometry(ns)
    qc = wavepacket.block_circuit(lat, center, params)
    tq = transpile(qc, basis_gates=["rz", "sx", "x", "cz"], optimization_level=1)
    ops = tq.count_ops()
    return {"block": dict(cz=int(ops.get("cz", 0)), depth=int(tq.depth()),
                          ops={k: int(v) for k, v in ops.items()})}


# ----------------------------------------------------------- vacuum mode
def mode_vacuum(a):
    m0, g2, eta = couplings(a)
    if os.path.exists(a.vac) and not a.retrain:
        d = stateprep.load_vacuum(a.vac)
        log(f"loaded {a.vac}: {d['n_layers']} layers, link_ref {d['link_ref']}")
    else:
        log(f"optimizing vacuum at ns={a.ns_vac}, {a.L_vac} layers, restarts {a.restarts}, "
            f"link_ref {a.link_ref}")
        r = stateprep.optimize_vacuum(Z2Lattice(a.ns_vac, pbc=True), m0, g2, eta,
                                      n_layers=a.L_vac, restarts=a.restarts,
                                      link_ref=a.link_ref)
        log(f"  thetas {np.round(r['thetas'], 6)}  F(ns={a.ns_vac}) = {r['fidelity']:.6f}")
        stateprep.save_vacuum(a.vac, r["thetas"], a.L_vac, a.link_ref, m0, g2, eta,
                              extra=dict(ns_opt=a.ns_vac, restarts=a.restarts,
                                         energy=r["energy"], exact_energy=r["exact_energy"],
                                         fidelity=r["fidelity"], tag=a.tag))
        log(f"saved {a.vac}")
        d = stateprep.load_vacuum(a.vac)
    bc = BandCache(a)
    dens = {}
    for ns in (6, 8, 10):
        sp_ = bc.space(ns)
        band = bc.band(ns)
        vac = gfe.gf_vacuum_state(sp_, d["thetas"], d["link_ref"])
        H = gfe.GFHamiltonian(sp_, m0, g2, eta)
        e, sig = H.energy_stats(vac)
        F = abs(np.vdot(band["vacuum"], vac)) ** 2
        dens[ns] = e / ns
        log(f"ns={ns}: ansatz E/ns {e / ns:.6f} (exact {band['e0'] / ns:.6f}), sigma_E {sig:.4f}, "
            f"F vs exact vacuum {F:.6f}; band {band_table(band)}")
        if ns == 8:
            F8 = F
    log(f"GATE 1: F(ns=8) = {F8:.6f} ({'PASS' if F8 >= 0.999 else 'FAIL'}); "
        f"density drift 6->10 = {abs(dens[10] - dens[6]):.2e}")


# ------------------------------------------------------------- band mode
def mode_band(a):
    bc = BandCache(a)
    for ns in [a.ns_train] + list(a.ns_cert):
        band = bc.band(ns)
        log(f"ns={ns}: e0 {band['e0']:.6f}; E(k)-e0 [* = degenerate]: {band_table(band)}")
        log(f"      next level gap per k: {np.round(band['next_gap'], 3)}; "
            f"E(K={a.k0:+.4f}) = {band_energy_at(band, a.k0):.4f}")


# ------------------------------------------------------------ verify
def certify(a, ns, vec, offsets, L, K, sigma_x, F_train, target_kind, bc, vacd):
    m0, g2, eta = couplings(a)
    sp_ = bc.space(ns)
    band = bc.band(ns)
    x0, center = geometry(ns)
    t = time.time()
    tgt, info = make_target(sp_, band, K, sigma_x, x0, target_kind)
    psi, vac, leak = prepare_state(a, ns, vacd, vec, offsets, L, sp_)
    H = gfe.GFHamiltonian(sp_, m0, g2, eta)
    r = gfe.packet_report(sp_, H, band, psi, vac, K, sigma_x)
    F = float(abs(np.vdot(tgt, psi)) ** 2)
    EK = band_energy_at(band, K)
    pt, _ = gfe.band_weights(band, tgt)
    _, sk_t = gfe.sigma_k_from_weights(band["k"], pt, K)
    # band-mean energy of the exact target: a packet with sigma_k on a
    # curved dispersion sits ABOVE E(K) by construction (relA K=0: +0.065),
    # so besides the legacy window around E(K) (train_cgkA gate 4, tuned on
    # the flat CGK-A band) the dispersion-aware gate |dE - E_tgt| <= 0.05
    # is reported (energy_tgt)
    E_tgt = float(np.sum(pt * band["energy"]) / pt.sum())
    g = dict(purity=r["purity"] >= 0.98,
             phase=abs(r["phase_x"] - K) <= 0.05,
             energy=(EK - 0.05) <= r["dE"] <= (EK + 0.10),
             energy_tgt=abs(r["dE"] - E_tgt) <= 0.05,
             transfer=(F_train - F) <= 0.01,
             sigma_k=abs(r["sigma_k"] / r["sigma_k_target"] - 1) <= 0.15)
    res = dict(ns=ns, F=F, F_train=F_train, dF=F_train - F, purity=r["purity"],
               p_band=r["p_band"], p_vac=r["p_vac"], phase=r["phase"], phase_x=r["phase_x"],
               absT2=r["absT2"], dE=r["dE"], EK=EK, E_band=r["E_band"], E_target=E_tgt,
               sigma_E=r["sigma_E"],
               kbar=r["kbar"], sigma_k=r["sigma_k"], sigma_k_target=r["sigma_k_target"],
               sigma_k_grid_target=sk_t, k0_env=info["k0_env"],
               band_fraction=info["band_fraction"], leak=leak,
               p_k=dict(zip([f"{k:+.4f}" for k in band["k"]], map(float, r["p_k"]))),
               j0=list(map(float, r["j0"])), gates=g,
               engine=a.cert_engine if ns <= 12 else "gf",
               seconds=time.time() - t)
    log(f"ns={ns} [{res['engine']}] F {F:.4f} (train {F_train:.4f}, dF {F_train - F:+.4f}) | "
        f"purity {r['purity']:.4f} (band {r['p_band']:.4f}, vac {r['p_vac']:.4f}) | "
        f"arg<T2> {r['phase']:+.4f} excited-only {r['phase_x']:+.4f} (K {K:+.4f}) | "
        f"dE {r['dE']:.4f} vs E(K) {EK:.4f} [band-mean {r['E_band']:.4f}, target {E_tgt:.4f}] "
        f"sigma_E {r['sigma_E']:.4f} | "
        f"sigma_k {r['sigma_k']:.4f} (target {r['sigma_k_target']:.4f}, grid target {sk_t:.4f}) "
        f"kbar {r['kbar']:+.4f}" + (f" | leak {leak:.1e}" if leak else ""))
    log("      per-k weight " + " ".join(f"{k:+.2f}:{w:.3f}" for k, w in zip(band["k"], r["p_k"])))
    log("      GATES " + " ".join(f"{k} {'PASS' if v else 'FAIL'}" for k, v in g.items()))
    return res


def write_cert(fn, certs, extra=None):
    z = dict(np.load(fn, allow_pickle=True))
    old = json.loads(str(z["cert"])) if "cert" in z else {}
    for c in certs:
        old[str(c["ns"])] = c
    z["cert"] = json.dumps(old)
    for k, v in (extra or {}).items():
        z[k] = v
    np.savez(fn, **z)


def mode_verify(a, fn=None, bc=None, vacd=None):
    fn = fn or a.params
    z = np.load(fn, allow_pickle=True)
    vec, offsets, L = np.asarray(z["vec"]), [int(o) for o in z["offsets"]], int(z["L"])
    K, sigma_x, F_train = float(z["k0"]), float(z["sigma"]), float(z["F"])
    target_kind = str(z["target_kind"]) if "target_kind" in z.files else a.target
    log(f"{fn}: L={L}, window {min(offsets)}..{max(offsets)} ({len(vec)} params), "
        f"K {K:+.4f}, sigma_x {sigma_x}, F_train {F_train:.4f}, target {target_kind}, "
        f"|theta| max {np.abs(vec).max():.2f}")
    bc = bc or BandCache(a)
    vacd = vacd or load_vac(a)
    certs = [certify(a, ns, vec, offsets, L, K, sigma_x, F_train, target_kind, bc, vacd)
             for ns in a.ns_cert]
    extra = {}
    if a.cz:
        cz = cz_count(vec, offsets, L)
        log(f"ns=50 block transpiled (rz,sx,x,cz; O1): {cz['block']['cz']} CZ, depth {cz['block']['depth']}")
        extra["cz50"] = json.dumps(cz)
    write_cert(fn, certs, extra)
    log(f"cert dict updated in {fn}")
    return certs


# ------------------------------------------------------------- train
def mode_train(a):
    m0, g2, eta = couplings(a)
    K, sigma_x, L, ns = float(a.k0), float(a.sigma_x), int(a.L), int(a.ns_train)
    w = auto_offset(sigma_x) if a.max_offset == "auto" else int(a.max_offset)
    if 2 * w + 1 > ns:
        raise SystemExit(f"window +-{w} does not fit ns={ns}")
    offsets = list(range(-w, w + 1))
    x0, center = geometry(ns)
    out = a.out or packet_file(a.tag, K, sigma_x, L)
    vacd = load_vac(a)
    bc = BandCache(a)
    sp_ = bc.space(ns)
    band = bc.band(ns)
    log(f"ns={ns} band: {band_table(band)}")
    tgt, info = make_target(sp_, band, K, sigma_x, x0, a.target)
    p, _ = gfe.band_weights(band, tgt)
    _, sk = gfe.sigma_k_from_weights(band["k"], p, K)
    log(f"target ({a.target}): k0_env {info['k0_env']:+.4f}, interpolator band fraction "
        f"{info['band_fraction']:.4f}, arg<T2> {np.angle(sp_.t2_expect(tgt)):+.4f}, "
        f"sigma_k {sk:.4f} (continuum {0.5 / sigma_x:.4f}); weights "
        + " ".join(f"{k:+.2f}:{x:.3f}" for k, x in zip(band["k"], p)))
    vac = gfe.gf_vacuum_state(sp_, vacd["thetas"], vacd["link_ref"])
    log(f"window +-{w} ({len(offsets)} offsets), L={L}: {wavepacket.n_params(offsets, L)} params, "
        f"center {center}, x0 {x0}, engine {a.engine}")

    # ---- inits: warm file (conj-mapped / re-keyed) or lsq amplitudes
    inits = []
    if a.warm:
        zw = np.load(a.warm, allow_pickle=True)
        wv, wo, wL = np.asarray(zw["vec"]), [int(o) for o in zw["offsets"]], int(zw["L"])
        if float(zw["k0"]) * K < 0:
            wv = conj_map(wv, wo, wL)
            log(f"warm start from {a.warm} (conj-mapped K -> -K)")
        else:
            log(f"warm start from {a.warm}")
        inits.append(remap_vector(wv, wo, wL, offsets, L))
    if not inits or a.warm_plus_lsq:
        d = gf_lsq_init(sp_, vac, tgt, center, offsets, L)
        inits += [float(amp) * d for amp in a.lsq_amps.split(",")]

    l2s = [float(x) for x in a.l2_anneal.split(",")]
    its = [int(x) for x in a.maxiters.split(",")]
    assert len(l2s) == len(its)

    if a.engine == "sv":
        lat = Z2Lattice(ns, pbc=True)
        vac_sv = stateprep.ansatz_state(lat, vacd["thetas"], link_ref=vacd["link_ref"])
        tgt_sv = gfe.embed_vector(sp_, tgt)
        eng = block_engine.BlockEngine(lat, center, L, offsets=offsets)
        fg = lambda v: eng.fidelity_and_grad(vac_sv, tgt_sv, v)
    else:
        eng = gfe.GFEngine(sp_, center, L, offsets=offsets)
        fg = lambda v: eng.fidelity_and_grad(vac, tgt, v)

    def run(x0v, l2, maxiter, label):
        import scipy.optimize
        state = {"n": 0, "t": time.time(), "best": 0.0}

        def cost_grad(v):
            f, g = fg(v)
            state["n"] += 1
            state["best"] = max(state["best"], f)
            if state["n"] % a.log_every == 0:
                log(f"    {label} eval {state['n']}: F {f:.5f} (best {state['best']:.5f}), "
                    f"{(time.time() - state['t']) / state['n']:.2f} s/eval")
            return 1.0 - f + l2 * np.dot(v, v), -g + 2 * l2 * v

        res = scipy.optimize.minimize(cost_grad, x0v, jac=True, method="L-BFGS-B",
                                      options={"maxiter": maxiter})
        F = fg(res.x)[0]
        log(f"  {label}: F {F:.5f}, |theta| max {np.abs(res.x).max():.2f}, norm "
            f"{np.linalg.norm(res.x):.2f}, {res.nit} iters / {state['n']} evals, "
            f"{time.time() - state['t']:.0f}s ({res.message})")
        return res.x, F

    best = None
    for i, x0v in enumerate(inits):
        x, F = run(x0v, l2s[0], its[0], f"stage0 init{i} l2={l2s[0]:g}")
        if best is None or F > best[1]:
            best = (x, F)
    for l2, mi in zip(l2s[1:], its[1:]):
        best = run(best[0], l2, mi, f"anneal l2={l2:g}")
    vec, F = best
    np.savez(out, vec=vec, offsets=np.array(offsets), L=L, F=F, k0=K, sigma=sigma_x,
             k0_env=info["k0_env"], center=center, x0=x0, ns_train=ns, engine=a.engine,
             couplings=np.array([m0, g2, eta]), target_kind=a.target,
             l2_anneal=np.array(l2s), maxiters=np.array(its), vac_file=a.vac,
             vac_thetas=vacd["thetas"], link_ref=vacd["link_ref"],
             band_fraction=info["band_fraction"],
             mix_keys=np.array([f"{k}{p}" for k, p in info["mix"]]),
             mix_vals=np.array(list(info["mix"].values())), sigma_k_target_grid=sk,
             seconds=time.time() - t0, cert=json.dumps({}))
    log(f"saved {out} (F = {F:.5f})")
    a.ns_cert = [ns] + [n for n in a.ns_cert if n != ns]
    mode_verify(a, out, bc, vacd)


# ------------------------------------------------------------- summary
def mode_summary(a):
    import glob
    files = a.params_list or sorted(glob.glob(f"data/wp_{a.tag}_*.npz"))
    for fn in files:
        z = np.load(fn, allow_pickle=True)
        cert = json.loads(str(z["cert"])) if "cert" in z.files else {}
        cz = json.loads(str(z["cz50"]))["block"]["cz"] if "cz50" in z.files else None
        print(f"\n{fn}: K {float(z['k0']):+.4f} sigma_x {float(z['sigma'])} L {int(z['L'])} "
              f"window {int(min(z['offsets']))}..{int(max(z['offsets']))} ({len(z['vec'])} params) "
              f"F_train {float(z['F']):.4f}" + (f"  CZ(ns=50) {cz}" if cz else ""))
        print(f"  {'ns':>3} {'eng':>3} {'F':>7} {'dF':>7} {'purity':>7} {'argT2x':>7} {'dE':>7} "
              f"{'E(K)':>7} {'sigE':>6} {'sig_k':>6} {'target':>6} gates")
        for ns, c in sorted(cert.items(), key=lambda kv: int(kv[0])):
            g = c["gates"]
            print(f"  {ns:>3} {c['engine']:>3} {c['F']:7.4f} {c['dF']:+7.4f} {c['purity']:7.4f} "
                  f"{c['phase_x']:+7.4f} {c['dE']:7.4f} {c['EK']:7.4f} {c['sigma_E']:6.3f} "
                  f"{c['sigma_k']:6.3f} {c['sigma_k_target']:6.3f} "
                  + " ".join(f"{k}:{'P' if v else 'F'}" for k, v in g.items()))


# ----------------------------------------------------------------- main
def parse(argv):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["vacuum", "band", "train", "verify", "summary"])
    p.add_argument("--m0", type=float, default=0.7)
    p.add_argument("--g2", type=float, default=1.1)
    p.add_argument("--eta", type=float, default=1.3)
    p.add_argument("--tag", default="prod")
    p.add_argument("--vac", default=None, help="data/vac_<tag>.npz")
    p.add_argument("--link-ref", default="+", choices=["+", "-"], help="vacuum mode")
    p.add_argument("--ns-vac", type=int, default=6)
    p.add_argument("--L-vac", type=int, default=2)
    p.add_argument("--restarts", type=int, default=2)
    p.add_argument("--retrain", action="store_true", help="vacuum mode: ignore existing file")
    p.add_argument("--k0", type=float, default=2 * np.pi / 5, help="target mean momentum K")
    p.add_argument("--sigma-x", type=float, default=0.75)
    p.add_argument("--ns-train", type=int, default=10)
    p.add_argument("--ns-cert", type=int, nargs="*", default=[])
    p.add_argument("--L", type=int, default=3)
    p.add_argument("--max-offset", default="auto")
    p.add_argument("--engine", choices=["sv", "gf"], default="gf",
                   help="training engine (sv: BlockEngine statevector, ns <= 12)")
    p.add_argument("--cert-engine", choices=["sv", "gf"], default=None,
                   help="state preparation for certification volumes <= 12 "
                        "(default: same as --engine); sv prepares the full "
                        "statevector and projects it into the gf sector")
    p.add_argument("--target", choices=["raw", "clean"], default="clean",
                   help="raw = spectroscopy.meson_wavepacket (production blocks); "
                        "clean = exact Gaussian momentum weights (train_cgkA)")
    p.add_argument("--l2-anneal", default="0,5e-4,5e-6")
    p.add_argument("--maxiters", default="350,200,150")
    p.add_argument("--out", default=None)
    p.add_argument("--params", default=None, help="verify: packet file")
    p.add_argument("--params-list", nargs="*", default=None, help="summary: files")
    p.add_argument("--warm", default=None, help="train: warm-start packet file")
    p.add_argument("--warm-plus-lsq", action="store_true",
                   help="train: add the lsq inits to a warm start")
    p.add_argument("--lsq-amps", default="1.0,1.6",
                   help="train: lsq-direction amplitudes tried at stage 0 "
                        "(train_cgkA_packets.py default 1.0,1.6)")
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--cz", action="store_true", help="verify: transpile the block at ns=50")
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--cache-dir", default=os.environ.get(
        "HT_CACHE", "/tmp/claude-1000/-home-hlamm-Desktop-QC-hadronic-tensor/"
        "4a179643-b4b8-42be-b43f-ffcd49ec9721/scratchpad/ws2b/cache"),
        help="band cache (ns=24 bands are ~1 GB; default: session scratchpad, "
             "override with HT_CACHE or this flag)")
    a = p.parse_args(argv)
    a.vac = a.vac or f"data/vac_{a.tag}.npz"
    a.cert_engine = a.cert_engine or a.engine
    a.cache_dir = os.path.abspath(a.cache_dir)
    return a


if __name__ == "__main__":
    a = parse(sys.argv[1:])
    {"vacuum": mode_vacuum, "band": mode_band, "train": mode_train,
     "verify": mode_verify, "summary": mode_summary}[a.mode](a)
    log("done")
