"""Self-contained batch modes for spare lenore capacity (argv[1] = mode).

  chiscanA / chiscanB : MPS systematics for the error-budget table -- rest
      packet, reduced grid (t <= 6, J0 probes), varying evolution bond cap,
      truncation, and Trotter dt around the production anchor.
  warddt : boosted packet at dt_target = 0.25 with J0+J1 probes (finer-dt
      integral Ward check -- paper itodo).
  cgktrain : full state-prep chain at CGK inelastic-A couplings
      (0.1, 0.4, 1.0) for the factorization test (task 17): vacuum angles,
      band, optimized interpolator, L=3 adjoint training + L2 regularization,
      rest + boosted k0 = 2pi/5.
"""

import os
import sys
import time

import numpy as np

t0 = time.time()


def log(m):
    print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)


mode = sys.argv[1]

if mode.startswith("chiscan") or mode == "warddt":
    from htensor import Z2Lattice, stateprep, wavepacket, backends
    from htensor import currents as cur
    from htensor.measure import split_current

    M0, G2, ETA = 0.7, 1.1, 1.3
    NS, CENTER = 50, 24
    lat = Z2Lattice(NS, pbc=True)
    TH = stateprep.optimize_vacuum(Z2Lattice(6, pbc=True), M0, G2, ETA,
                                   n_layers=2, restarts=2)["thetas"]
    k0tag = "k0.00" if mode.startswith("chiscan") else "k1.26"
    z = np.load(f"data/wp10reg_params_{k0tag}_L3.npz", allow_pickle=True)
    params = wavepacket.params_from_vector(z["vec"], list(z["offsets"]),
                                           int(z["L"]))
    prep = stateprep.vacuum_ansatz(lat, TH)
    prep.compose(wavepacket.block_circuit(lat, CENTER, params), inplace=True)
    insert = cur.charge_density(lat, CENTER)
    anc_site = min(split_current(insert)[1][0][0])
    mps, perm = backends.prepare_state_mps(lat, prep, anc_site,
                                           cap=512, trunc=1e-10)
    log(f"{mode}: stored prep ready")
    TIMES = np.arange(0.0, 6.01, 0.5)

    if mode == "warddt":
        probes = [cur.charge_density(lat, v) for v in range(NS)] + \
                 [cur.bond_current(lat, b, ETA) for b in range(NS)]
        # dt025/tr10 gave IDENTICAL residuals -> neither Trotter nor
        # truncation: the Simpson integral is limited by the OUTPUT
        # sampling grid (0.5-spaced), which those configs kept fixed.
        # samp025 halves the output grid: residual should drop ~16x.
        TIMES = np.arange(0.0, 6.01, 0.25)
        configs = [(None, 1e-8, 0.25, "samp025")]
    else:
        probes = [cur.charge_density(lat, v) for v in range(NS)]
        configs = ([(None, 1e-6, 0.5, "tr1e6"), (256, 1e-8, 0.5, "cap256"),
                    (None, 1e-8, 0.25, "dt025_rest")]
                   if mode == "chiscanA" else
                   [(None, 1e-8, 0.5, "anchor"), (None, 1e-10, 0.5, "tr1e10"),
                    (64, 1e-8, 0.5, "cap64")])

    for cap, trunc, dt, tag in configs:
        if os.path.exists(f"data/sysscan_{mode}_{tag}.npz"):
            log(f"{mode}/{tag}: exists, skipping")
            continue
        rows = []
        for t in TIMES:
            d = backends.hadamard_correlator_aer(
                lat, None, insert, probes, M0, G2, ETA, [t], dt_target=dt,
                method="matrix_product_state", mps_max_bond=cap,
                mps_trunc=trunc, initial_mps=mps, initial_perm=perm)
            rows.append(d)
        np.savez(f"data/sysscan_{mode}_{tag}.npz", times=TIMES,
                 corr=np.concatenate([r.correlator for r in rows]),
                 one_pt=np.concatenate([r.probe_expect for r in rows]),
                 insert_1pt=rows[0].insert_expect,
                 cap=0 if cap is None else cap, trunc=trunc, dt=dt)
        log(f"{mode}/{tag} saved")
    log("done")

elif mode == "cgktrain":
    from qiskit.quantum_info import Statevector
    from htensor import Z2Lattice, stateprep, spectroscopy, block_engine

    M0, G2, ETA = 0.1, 0.4, 1.0  # CGK inelastic-A in our conventions
    TH = stateprep.optimize_vacuum(Z2Lattice(6, pbc=True), M0, G2, ETA,
                                   n_layers=2, restarts=2)["thetas"]
    np.savez("data/cgkA_vacuum_thetas.npz", thetas=TH)
    log("CGK-A vacuum angles done")
    lat = Z2Lattice(10, pbc=True)
    band = spectroscopy.meson_band(lat, M0, G2, ETA, n_states=14,
                                   matrix_free=True)
    log(f"CGK-A band: E(k) = {np.round(band['energy'], 4)}")
    vac = np.asarray(Statevector.from_instruction(
        stateprep.vacuum_ansatz(lat, TH)))
    for k0 in (0.0, 2 * np.pi / 5):
        mix = spectroscopy.optimize_interpolator(lat, band, k0=k0,
                                                 sigma_x=0.75, x0=2)
        log(f"k0={k0:.2f}: band fraction {mix['band_fraction']:.4f}")
        target, _ = spectroscopy.meson_wavepacket(lat, band, k0=k0,
                                                  sigma_x=0.75, x0=2,
                                                  mix=mix["mix"])
        r = block_engine.train_adjoint(lat, vac, target, center=4,
                                       n_layers=3, maxiter=600, max_offset=4)
        r = block_engine.train_adjoint(lat, vac, target, center=4,
                                       n_layers=3, maxiter=300, max_offset=4,
                                       l2=5e-4, inits=[r["vector"]])
        r = block_engine.train_adjoint(lat, vac, target, center=4,
                                       n_layers=3, maxiter=200, max_offset=4,
                                       l2=5e-6, inits=[r["vector"]])
        np.savez(f"data/wpCGKA_params_k{k0:.2f}_L3.npz", vec=r["vector"],
                 offsets=r["offsets"], L=3, F=r["fidelity"], k0=k0)
        log(f"k0={k0:.2f}: trained F = {r['fidelity']:.4f}")
    log("done")
else:
    raise SystemExit(f"unknown mode {mode}")
