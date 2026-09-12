"""Stage A: meson-wavepacket preparation chain at CGK-A couplings
(m0, g2, eta) = (0.1, 0.4, 1.0), for Stage B two-packet collisions at ns=30.

Packets at kbar = +-K with K on the ns=30 grid (nx=15, K = 2 pi j' / 15):
  j=2  ->  K = 4 pi 2 / 30 = 0.8378   (off-grid at any training nx <= 6)
  j=3  ->  K = 4 pi 3 / 30 = 1.2566   (= 2 pi / 5, on the ns=10 grid)

Production pattern (lenore_batch.py cgktrain): vacuum angles at ns=6,
band + optimized interpolator + band-projected Gaussian target at ns=10
(sigma_x = 0.75, x0 = 2, center = 4), L=3 adjoint training with L2
annealing, saved as data/wpCGKA_params_k{K:.2f}_L3.npz.

Modes (argv[1]):
  vacuum    verify (and retrain if needed) data/cgkA_vacuum_thetas.npz
            [gate 1: F >= 0.999 at ns=8; energy-density stability 6/8/10]
  bandE     E(K) targets from the deep_levels_cgkinA band interpolation
  verify K [ns_eval]   gates 2-4 for data/wpCGKA_params_k{K:.2f}_L3.npz
  train  K [ns_eval]   train packet at K (ns=10), then run gates
  summary   assemble data/stageA_summary.npz from the verify caches
"""

import os
import sys
import time

import numpy as np
from scipy.interpolate import CubicSpline

from htensor import Z2Lattice, stateprep, spectroscopy, wavepacket, block_engine
from htensor import hamiltonian as ham
from htensor import exact

M0, G2, ETA = 0.1, 0.4, 1.0
K_J2 = 4 * np.pi * 2 / 30          # 0.837758
K_J3 = 4 * np.pi * 3 / 30          # 1.256637  = 2 pi / 5
SIGMA_X, X0, CENTER = 0.75, 2, 4   # production packet shape (cgktrain)
VAC_FILE = "data/cgkA_vacuum_thetas.npz"
SCRATCH = "data"                   # verify caches live next to the params

t0 = time.time()


def log(m):
    print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)


# ------------------------------------------------- fast local-gate kernel
# block_engine._apply_local moveaxes the state tensor twice per gate (four
# full-vector copies).  Every gate in our block acts on qubits that are
# CONTIGUOUS in qubit index (bond b -> qubits 2b, 2b+1, 2b+2; single-qubit
# gates trivially), so the state reshapes copy-free to (A, 2^k, B) and the
# gate is one batched GEMM.  ~3-5x faster gradient evals; verified against
# the original kernel at import (see _selftest_fast_apply).
_apply_local_orig = block_engine._apply_local


def _apply_local_fast(psi, U, qubits, n):
    qs = sorted(qubits)
    if qs != list(range(qs[0], qs[0] + len(qubits))):
        return _apply_local_orig(psi, U, qubits, n)
    k, q0 = len(qubits), qs[0]
    D, B = 1 << k, 1 << q0
    A = psi.size // (D * B)
    # axis bit j of the middle index <-> physical qubit q0+j; the gate
    # matrix is indexed by local states (bit i <-> qubits[i], qubits[0]=LSB)
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
    psi = rng.standard_normal(2**lat.n_qubits) + 1j * rng.standard_normal(
        2**lat.n_qubits)
    psi /= np.linalg.norm(psi)
    vec = 0.3 * rng.standard_normal(len(eng.keys))
    ref = eng.state(psi, vec)
    block_engine._apply_local = _apply_local_fast
    new = eng.state(psi, vec)
    block_engine._apply_local = _apply_local_orig
    err = np.abs(new - ref).max()
    assert err < 1e-12, f"fast kernel mismatch: {err}"
    return err


_selftest_fast_apply()
block_engine._apply_local = _apply_local_fast


# ------------------------------------------------------------------ band E(K)
def band_dispersion():
    """Single-meson dispersion spline from the ns=20 CGK-A deep levels
    (min gap per |T2 phase| inside the 0.1 < gap < 1.5 band window --
    identical to scripts/factorization.py)."""
    d = np.load("data/deep_levels_cgkinA_ns20.npz")
    g, phv = d["gaps"], d["phases"]
    ks, es = [], []
    for kk in np.unique(np.round(np.abs(phv), 6)):
        m = np.isclose(np.abs(phv), kk) & (g > 0.1) & (g < 1.5)
        if m.any():
            ks.append(kk)
            es.append(g[m].min())
    return CubicSpline(np.array(ks), np.array(es), bc_type=((1, 0.0), (1, 0.0)))


# ------------------------------------------------------------------ vacuum
def vac_state(ns):
    th = np.load(VAC_FILE)["thetas"]
    return stateprep.ansatz_state(Z2Lattice(ns, pbc=True), th), th


def energy(lat, psi):
    H = ham.build_hamiltonian(lat, M0, G2, ETA)
    return float(np.real(np.vdot(psi, exact.apply_pauli_sum(H, psi))))


def mode_vacuum():
    th = np.load(VAC_FILE)["thetas"] if os.path.exists(VAC_FILE) else None
    lat8 = Z2Lattice(8, pbc=True)
    e8, v8 = exact.lowest_physical_states(lat8, M0, G2, ETA, k=1)
    best = {"thetas": th} if th is not None else None
    if th is not None:
        F = abs(np.vdot(v8[:, 0], stateprep.ansatz_state(lat8, th))) ** 2
        log(f"stored thetas: ns=8 exact-vacuum fidelity F = {F:.6f}")
        best["F8"] = float(F)
    # attempt ladder: deepen (L=3, then 4) at ns=6, then refine at ns=8,
    # warm-started from the previous best (zero-padded to the new depth)
    attempts = [(6, 3), (8, 3), (8, 4)]
    for ns_opt, L in attempts:
        if best is not None and best["F8"] >= 0.999:
            break
        x0 = None
        if th is not None:
            x0 = np.zeros(4 * L)
            x0[:min(len(th), 4 * L)] = th[:4 * L]
        log(f"retraining vacuum at ns={ns_opt} (n_layers={L}) ...")
        r = stateprep.optimize_vacuum(Z2Lattice(ns_opt, pbc=True), M0, G2,
                                      ETA, n_layers=L, restarts=3, x0=x0)
        F = abs(np.vdot(v8[:, 0], stateprep.ansatz_state(lat8, r["thetas"]))) ** 2
        log(f"retrained: ns={ns_opt} F = {r['fidelity']:.6f}, ns=8 F = {F:.6f}")
        if best is None or F > best["F8"]:
            best = {"thetas": r["thetas"], "F8": float(F)}
            th = r["thetas"]
            np.savez(VAC_FILE, thetas=best["thetas"])
            log(f"updated {VAC_FILE}")
    th = best["thetas"]
    dens = {}
    for ns in (6, 8, 10):
        lat = Z2Lattice(ns, pbc=True)
        psi = stateprep.ansatz_state(lat, th)
        dens[ns] = energy(lat, psi) / ns
        log(f"ns={ns}: ansatz energy density {dens[ns]:.6f}")
    e6 = exact.lowest_physical_states(Z2Lattice(6, pbc=True), M0, G2, ETA,
                                      k=1)[0][0]
    e10 = exact.lowest_physical_states(Z2Lattice(10, pbc=True), M0, G2, ETA,
                                       k=1, matrix_free=True)[0][0]
    log(f"exact densities: ns=6 {e6/6:.6f}  ns=8 {e8[0]/8:.6f}  "
        f"ns=10 {e10/10:.6f}")
    np.savez(f"{SCRATCH}/stageA_vacuum_gate.npz", thetas=th, F8=best["F8"],
             dens=np.array([dens[6], dens[8], dens[10]]),
             exact_dens=np.array([e6 / 6, e8[0] / 8, e10 / 10]))
    log(f"GATE 1: F(ns=8) = {best['F8']:.6f} "
        f"({'PASS' if best['F8'] >= 0.999 else 'FAIL'}); "
        f"density drift 6->10 = {abs(dens[10]-dens[6]):.6f}")


# ------------------------------------------------------------------ band states
SCRATCHPAD = ("/tmp/claude-1000/-home-hlamm-Desktop-QC-hadronic-tensor/"
              "fb5de832-5f2a-4241-8c29-2ea520bb7131/scratchpad")


def band_full(ns, n_states=16, tol=1e-5):
    """Single-meson band at ns: raw eigsh pairs cached to scratch, then
    T2-resolved with a cluster tolerance loose enough for eigsh's numerical
    splitting of degenerate +-k pairs.

    NOTE: spectroscopy.meson_band's hardcoded degeneracy_tol=1e-6 silently
    DROPS a +-k pair when eigsh splits it by more than that (the pair's
    real eigenvectors then sit in 1-dim 'clusters' whose <T2> phase matches
    no grid momentum).  At CGK-A couplings, ns=10, the k=+-1.2566 pair is
    lost exactly this way -- the root cause of the earlier contaminated
    k1.26 packet."""
    lat = Z2Lattice(ns, pbc=True)
    fn = f"{SCRATCHPAD}/cgkA_eigs_ns{ns}.npz"
    if os.path.exists(fn):
        z = np.load(fn)
        energies, vecs = z["energies"], z["vecs"]
    else:
        energies, vecs = exact.lowest_physical_states(
            lat, M0, G2, ETA, k=n_states, matrix_free=True)
        os.makedirs(SCRATCHPAD, exist_ok=True)
        np.savez(fn, energies=energies, vecs=vecs)
        log(f"ns={ns}: cached {n_states} raw eigenpairs, gaps "
            f"{np.round(energies - energies[0], 4)}")
    states = [vecs[:, i] for i in range(vecs.shape[1])]
    resolved, phases = spectroscopy._t2_phases(states, energies, lat,
                                               degeneracy_tol=tol)
    H_op = ham.build_hamiltonian(lat, M0, G2, ETA)
    e_res = np.array([np.real(np.vdot(s, exact.apply_pauli_sum(H_op, s)))
                      for s in resolved])
    order = np.argsort(e_res)
    resolved = [resolved[i] for i in order]
    e_res, phases = e_res[order], phases[order]
    vacuum, e0 = resolved[0], e_res[0]
    nx = lat.nx
    k_grid = 2 * np.pi * np.arange(nx) / nx
    k_grid = np.where(k_grid > np.pi + 1e-9, k_grid - 2 * np.pi, k_grid)
    band_k, band_e, band_states = [], [], []
    for k in sorted(set(np.round(k_grid, 12))):
        for s, e, phv in zip(resolved[1:], e_res[1:], phases[1:]):
            if abs(np.angle(np.exp(1j * (k - phv)))) < 1e-3:
                band_k.append(k)
                band_e.append(e - e0)
                band_states.append(s)
                break
    band = {"vacuum": vacuum, "e0": e0, "k": np.array(band_k),
            "energy": np.array(band_e), "states": band_states}
    log(f"ns={ns} band: E(k)-E0 = {np.round(band['energy'], 4)} at "
        f"k = {np.round(band['k'], 4)}")
    if len(band_k) < nx:
        log(f"WARNING: only {len(band_k)} of {nx} band momenta resolved")
    return lat, band


# ------------------------------------------------------------ clean targets
def _wrap(dk):
    return (dk + np.pi) % (2 * np.pi) - np.pi


def tune_k0_env(band, K, sigma_x=SIGMA_X):
    """Envelope momentum k0_env such that the band-basis Gaussian target's
    arg<T2> equals K exactly on this volume's momentum grid (the coarse
    grid biases the packet phase for off-grid K)."""
    from scipy.optimize import brentq

    kg = band["k"]

    def ph(k0e):
        w2 = np.exp(-2 * sigma_x**2 * _wrap(kg - k0e) ** 2)
        return np.angle(np.sum(w2 * np.exp(1j * kg))) - K

    lo, hi = K - 0.45, K + 0.45
    if ph(lo) * ph(hi) > 0:
        return K
    return brentq(ph, lo, hi, xtol=1e-6)


def clean_target(lat, band, k0_env, sigma_x=SIGMA_X, x0=X0):
    """Band-basis Gaussian wavepacket: momentum weights imposed exactly
    (exp(-sigma_x^2 dk^2) amplitudes), per-state phases anchored by the
    production envelope-smeared interpolator packet (phase-convention-safe).
    Fixes the within-band distortion of the raw construction, where the
    interpolator's k-dependent matrix elements skew the distribution toward
    the band top."""
    mix = spectroscopy.optimize_interpolator(lat, band, k0=k0_env,
                                             sigma_x=sigma_x, x0=x0)
    anchor, _ = spectroscopy.meson_wavepacket(lat, band, k0=k0_env,
                                              sigma_x=sigma_x, x0=x0,
                                              mix=mix["mix"])
    tgt = np.zeros_like(anchor)
    for k, s in zip(band["k"], band["states"]):
        a = np.vdot(s, anchor)
        w = np.exp(-sigma_x**2 * _wrap(k - k0_env) ** 2)
        if abs(a) > 1e-9:
            tgt = tgt + w * (a / abs(a)) * s
    return tgt / np.linalg.norm(tgt), mix


def composition(band, psi, label):
    amps = np.array([np.vdot(s, psi) for s in band["states"]])
    p = np.abs(amps) ** 2
    t2 = np.sum(p * np.exp(1j * band["k"]))
    log(f"  {label}: per-k weight "
        + " ".join(f"{k:+.2f}:{w:.3f}" for k, w in zip(band["k"], p))
        + f" | band total {p.sum():.4f} | band arg<T2> {np.angle(t2):+.4f}"
        + f" | band <E> {np.sum(p*band['energy'])/p.sum():.4f}")
    return p


def packet_state(lat, th, K, npzfile):
    z = np.load(npzfile, allow_pickle=True)
    eng = block_engine.BlockEngine(lat, CENTER, int(z["L"]),
                                   offsets=list(z["offsets"]))
    vac = stateprep.ansatz_state(lat, th)
    return eng.state(vac, z["vec"]), vac, float(z["F"])


def gates_pass(r):
    return (r["purity"] >= 0.98 and abs(r["phase_x"] - r["K"]) <= 0.05
            and (r["EK"] - 0.05) <= r["dE"] <= (r["EK"] + 0.10))


def eval_gates(K, npzfile, ns_eval, band=None):
    """Gates 2-4 for a trained packet file, evaluated at ns_eval."""
    E = band_dispersion()
    EK = float(E(abs(K)))
    if band is None:
        lat, band = band_full(ns_eval)
    else:
        lat = Z2Lattice(ns_eval, pbc=True)
    th = np.load(VAC_FILE)["thetas"]
    psi, vac, Ftrain = packet_state(lat, th, K, npzfile)
    # gate 2: band-1 weight
    p_band = sum(abs(np.vdot(s, psi)) ** 2 for s in band["states"])
    p_vac = abs(np.vdot(band["vacuum"], psi)) ** 2
    purity = p_band / (1.0 - p_vac)
    # gate 3: <T2> phase
    t2 = np.vdot(psi, spectroscopy.translate(psi, lat))
    ph = float(np.angle(t2))
    # subtracting the coherent vacuum leakage isolates the packet's phase
    a0 = np.vdot(band["vacuum"], psi)
    psi_x = psi - a0 * band["vacuum"]
    psi_x = psi_x / np.linalg.norm(psi_x)
    t2x = np.vdot(psi_x, spectroscopy.translate(psi_x, lat))
    phx = float(np.angle(t2x))
    # gate 4: on-circuit packet energy
    dE = energy(lat, psi) - energy(lat, vac)
    res = dict(K=K, ns_eval=ns_eval, Ftrain=Ftrain, p_band=p_band,
               p_vac=p_vac, purity=purity, phase=ph, phase_x=phx,
               dE=dE, EK=EK, absT2=abs(t2))
    log(f"K={K:+.4f} ns={ns_eval}: train-F {Ftrain:.4f} | band weight "
        f"{p_band:.4f} (vac leak {p_vac:.4f}, purity {purity:.4f}) | "
        f"arg<T2> {ph:+.4f} (excited-only {phx:+.4f}; target {K:+.4f}) | "
        f"dE {dE:.4f} vs E(K) {EK:.4f} window [{EK-0.05:.4f}, {EK+0.10:.4f}]")
    g2 = purity >= 0.98
    g3 = abs(phx - K) <= 0.05
    g4 = (EK - 0.05) <= dE <= (EK + 0.10)
    log(f"  GATE 2 {'PASS' if g2 else 'FAIL'} | GATE 3 "
        f"{'PASS' if g3 else 'FAIL'} | GATE 4 {'PASS' if g4 else 'FAIL'}")
    np.savez(f"{SCRATCH}/stageA_gates_k{K:.2f}.npz", **res)
    return res


# ------------------------------------------------------------------ training
def conj_map(vec, offsets, n_layers):
    """+K solution -> -K warm start: conj of every generator's rotation.
    Real generators (hop, site Z, link X) flip their angle; the purely
    imaginary cur generator keeps it.  Exact when the vacuum is real (ours
    is 0.999-fidelity-close to the real exact vacuum)."""
    out = vec.copy()
    i = 0
    for _l in range(n_layers):
        for kind in KINDS_ORDER:
            for _o in offsets:
                if kind != "cur":
                    out[i] = -out[i]
                i += 1
    return out


KINDS_ORDER = wavepacket.KINDS


def train_packet(K, lat, band, sigma_x=SIGMA_X, n_layers=3, max_offset=4,
                 l2_anneal=(0.0, 5e-4, 5e-6), maxiters=(350, 200, 150),
                 warm=None, seed=11):
    """cgktrain pattern (L=3 adjoint training + L2 annealing) against the
    cleaned band-basis Gaussian target with grid-detuned envelope k0."""
    k0_env = tune_k0_env(band, K, sigma_x)
    log(f"K={K:+.4f}: envelope k0 detuned to {k0_env:+.4f} for the "
        f"ns={lat.ns} grid")
    th = np.load(VAC_FILE)["thetas"]
    vac = stateprep.ansatz_state(lat, th)
    target, mix = clean_target(lat, band, k0_env, sigma_x)
    log(f"K={K:+.4f}: interpolator band fraction {mix['band_fraction']:.4f}")
    composition(band, target, f"target(K={K:+.2f})")
    if warm is not None:
        inits = [warm]
    else:
        # two lsq amplitudes (the production default's best performers);
        # halves stage-0 wall time vs the 4-start default
        offs = [o for o in wavepacket.window_offsets(lat.ns)
                if abs(o) <= max_offset]
        d = wavepacket.lsq_init(lat, vac, target, CENTER, offs, n_layers)
        inits = [1.0 * d, 1.6 * d]
    r = block_engine.train_adjoint(lat, vac, target, center=CENTER,
                                   n_layers=n_layers, maxiter=maxiters[0],
                                   max_offset=max_offset, seed=seed,
                                   l2=l2_anneal[0], inits=inits)
    log(f"K={K:+.4f}: stage-0 F = {r['fidelity']:.4f}")
    if warm is not None and r["fidelity"] < 0.90:
        log(f"K={K:+.4f}: warm start underperformed, retrying cold")
        r2 = block_engine.train_adjoint(lat, vac, target, center=CENTER,
                                        n_layers=n_layers,
                                        maxiter=maxiters[0],
                                        max_offset=max_offset, seed=seed,
                                        l2=l2_anneal[0])
        if r2["fidelity"] > r["fidelity"]:
            r = r2
    for l2, mi in zip(l2_anneal[1:], maxiters[1:]):
        r = block_engine.train_adjoint(lat, vac, target, center=CENTER,
                                       n_layers=n_layers, maxiter=mi,
                                       max_offset=max_offset, l2=l2,
                                       inits=[r["vector"]])
        log(f"K={K:+.4f}: l2={l2:g} F = {r['fidelity']:.4f}")
    fn = f"data/wpCGKA_params_k{K:.2f}_L{n_layers}.npz"
    np.savez(fn, vec=r["vector"], offsets=r["offsets"], L=int(n_layers),
             F=r["fidelity"], k0=K, k0_env=k0_env, sigma_x=sigma_x,
             center=CENTER, ns_train=lat.ns)
    log(f"saved {fn} (F = {r['fidelity']:.4f})")
    return fn, r


# ------------------------------------------------------------------ modes
if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "vacuum":
        mode_vacuum()
    elif mode == "bandE":
        E = band_dispersion()
        log(f"M = {float(E(0)):.4f}; E({K_J2:.4f}) = {float(E(K_J2)):.4f}; "
            f"E({K_J3:.4f}) = {float(E(K_J3)):.4f}")
    elif mode == "verify":
        K = float(sys.argv[2])
        ns_eval = int(sys.argv[3]) if len(sys.argv) > 3 else 10
        eval_gates(K, f"data/wpCGKA_params_k{K:.2f}_L3.npz", ns_eval)
    elif mode == "diag":
        # composition of the production-style target vs the cleaned target
        # vs the stored packet state, at ns=10
        lat, band = band_full(10)
        th = np.load(VAC_FILE)["thetas"]
        for K in (K_J3, K_J2):
            mix = spectroscopy.optimize_interpolator(lat, band, k0=K,
                                                     sigma_x=SIGMA_X, x0=X0)
            old_t, _ = spectroscopy.meson_wavepacket(lat, band, k0=K,
                                                     sigma_x=SIGMA_X, x0=X0,
                                                     mix=mix["mix"])
            log(f"K={K:+.4f} production-style target "
                f"(band fraction {mix['band_fraction']:.4f}):")
            composition(band, old_t, "envelope+mix target")
            k0e = tune_k0_env(band, K)
            tgt, _ = clean_target(lat, band, k0e)
            log(f"K={K:+.4f} cleaned target (k0_env {k0e:+.4f}):")
            composition(band, tgt, "clean target")
        fn = f"data/wpCGKA_params_k{K_J3:.2f}_L3.npz"
        if os.path.exists(fn):
            psi, vac, _ = packet_state(lat, th, K_J3, fn)
            log("stored k1.26 packet on the current vacuum:")
            composition(band, psi, "stored packet")
    elif mode == "train":
        K = float(sys.argv[2])
        lat, band = band_full(10)
        warm = None
        if len(sys.argv) > 3 and sys.argv[3] == "warm":
            zw = np.load(f"data/wpCGKA_params_k{-K:.2f}_L3.npz",
                         allow_pickle=True)
            warm = conj_map(zw["vec"], list(zw["offsets"]), int(zw["L"]))
        fn, _ = train_packet(K, lat, band, warm=warm)
        eval_gates(K, fn, 10, band=band)
    elif mode == "trainall":
        lat, band = band_full(10)
        results = {}
        for K in (K_J3, -K_J3, K_J2, -K_J2):
            fn = f"data/wpCGKA_params_k{K:.2f}_L3.npz"
            done = False
            if os.path.exists(fn):
                z = np.load(fn, allow_pickle=True)
                if "k0_env" in z.files:       # produced by this pipeline
                    r = eval_gates(K, fn, 10, band=band)
                    done = gates_pass(r)
                    if done:
                        results[K] = r
                        log(f"K={K:+.4f}: existing file passes, skipping")
            if not done:
                warm = None
                pair = f"data/wpCGKA_params_k{-K:.2f}_L3.npz"
                if K < 0 and os.path.exists(pair):
                    zw = np.load(pair, allow_pickle=True)
                    if "k0_env" in zw.files:  # only warm-start from cleaned
                        warm = conj_map(zw["vec"], list(zw["offsets"]),
                                        int(zw["L"]))
                fn, _ = train_packet(K, lat, band, warm=warm)
                results[K] = eval_gates(K, fn, 10, band=band)
        for K, r in results.items():
            log(f"K={K:+.4f}: purity {r['purity']:.4f} phase {r['phase_x']:+.4f} "
                f"dE {r['dE']:.4f} (E(K) {r['EK']:.4f})")
    elif mode == "summary":
        out = {}
        vg = np.load(f"{SCRATCH}/stageA_vacuum_gate.npz")
        out["vac_F8"] = vg["F8"]
        out["vac_dens"] = vg["dens"]
        out["vac_exact_dens"] = vg["exact_dens"]
        for K in (K_J2, -K_J2, K_J3, -K_J3):
            g = np.load(f"{SCRATCH}/stageA_gates_k{K:.2f}.npz")
            tag = f"k{K:+.2f}"
            for f in g.files:
                out[f"{tag}_{f}"] = g[f]
        np.savez("data/stageA_summary.npz", **out)
        log("wrote data/stageA_summary.npz")
    else:
        raise SystemExit(f"unknown mode {mode}")
