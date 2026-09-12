"""Shared state-preparation helpers for the N_s-parametric production and
calibration scripts (run_w_ns.py, hw_cal_grids.py).

A 'state' = couplings + vacuum angles (+ link reference) + optional
wavepacket block parameters.  The expensive preparation MPS is built once
per (state, ancilla site) and pickled under data/work/prep/ so that every
downstream grid starts from `set_matrix_product_state`.

Vacuum-angle file (--vac): npz with `thetas` (+ optional `n_layers`,
`link_ref`, couplings) as written by stateprep.save_vacuum; without it the
angles are re-optimized at ns=6 (2 layers, 2 restarts) as the legacy
scripts did.  Block-parameter file (--params): legacy keys vec / offsets /
L (/ F / sigma / k0), e.g. data/wp10reg_params_k1.26_L3.npz.
"""

import argparse
import inspect
import os
import pickle
import time

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

from htensor import Z2Lattice, stateprep, wavepacket, backends
from htensor import hamiltonian as ham

_t0 = time.time()


def log(msg):
    print(f"[{time.time()-_t0:6.0f}s] {msg}", flush=True)


def add_state_args(p: argparse.ArgumentParser):
    p.add_argument("--m0", type=float, default=0.7)
    p.add_argument("--g2", type=float, default=1.1)
    p.add_argument("--eta", type=float, default=1.3)
    p.add_argument("--ns", type=int, default=50)
    p.add_argument("--center", type=int, default=None,
                   help="insertion / block centre (staggered site); default ns//2 - 1")
    p.add_argument("--vac", default=None, help="vacuum angles npz (stateprep.save_vacuum)")
    p.add_argument("--params", default=None, help="wavepacket block npz (vec, offsets, L)")
    p.add_argument("--state", choices=["packet", "vac"], default="packet")
    p.add_argument("--k0", type=float, default=None, help="label only; default from --params")
    p.add_argument("--tag", default="prod", help="coupling tag used in file names")
    p.add_argument("--prep-pkl", default=None, help="pickle cache for the prepared MPS")
    p.add_argument("--cap-prep", type=int, default=512)
    p.add_argument("--trunc-prep", type=float, default=1e-10)
    p.add_argument("--threads", type=int, default=4)
    return p


def resolve(args):
    if args.center is None:
        args.center = args.ns // 2 - 1
    if args.state == "packet" and args.params is None:
        raise SystemExit("--params is required for --state packet")
    return args


def vacuum_thetas(args):
    """-> (thetas, link_ref)."""
    if args.vac:
        z = np.load(args.vac, allow_pickle=True)
        link_ref = str(z["link_ref"]) if "link_ref" in z.files else "+"
        for k in ("m0", "g2", "eta"):
            if k in z.files and not np.isclose(float(z[k]), getattr(args, k)):
                raise SystemExit(f"--vac {args.vac} has {k}={float(z[k])} != {getattr(args, k)}")
        log(f"vacuum angles from {args.vac} (link_ref {link_ref})")
        return np.asarray(z["thetas"], float), link_ref
    log("optimizing vacuum angles at ns=6 (2 layers, 2 restarts)")
    th = stateprep.optimize_vacuum(Z2Lattice(6, pbc=True), args.m0, args.g2, args.eta,
                                   n_layers=2, restarts=2)["thetas"]
    return np.asarray(th, float), "+"


def _vacuum_ansatz(lat, thetas, link_ref):
    if link_ref == "+" or "link_ref" not in inspect.signature(stateprep.vacuum_ansatz).parameters:
        if link_ref != "+":
            raise SystemExit("this stateprep.vacuum_ansatz has no link_ref support yet")
        return stateprep.vacuum_ansatz(lat, thetas)
    return stateprep.vacuum_ansatz(lat, thetas, link_ref=link_ref)


def load_params(path):
    z = np.load(path, allow_pickle=True)
    info = {"k0": float(z["k0"]) if "k0" in z.files else None,
            "sigma": float(z["sigma"]) if "sigma" in z.files else
            (float(z["sigma_x"]) if "sigma_x" in z.files else None),
            "F": float(z["F"]) if "F" in z.files else None,
            "offsets": [int(o) for o in z["offsets"]], "L": int(z["L"])}
    params = wavepacket.params_from_vector(z["vec"], info["offsets"], info["L"])
    return params, info


def state_label(args, info=None):
    if args.state == "vac":
        return "vac"
    k0 = args.k0 if args.k0 is not None else (info or {}).get("k0")
    sig = (info or {}).get("sigma")
    lab = f"k{k0:.2f}" if k0 is not None else "kNA"
    if sig is not None:
        lab += f"_s{sig:.2f}"
    return lab


def build_prep(args):
    """-> (lat, prep_circuit, info)."""
    lat = Z2Lattice(args.ns, pbc=True)
    thetas, link_ref = vacuum_thetas(args)
    prep = _vacuum_ansatz(lat, thetas, link_ref)
    info = {"thetas": thetas, "link_ref": link_ref}
    if args.state == "packet":
        params, pinfo = load_params(args.params)
        info.update(pinfo)
        prep.compose(wavepacket.block_circuit(lat, args.center, params), inplace=True)
        log(f"block: {args.params} (L={pinfo['L']}, offsets {pinfo['offsets'][0]}..{pinfo['offsets'][-1]}, "
            f"F={pinfo['F']})")
    info["label"] = state_label(args, info)
    return lat, prep, info


def stored_prep(args, lat, prep, anc_site, info=None):
    """Prepare once (Aer MPS, cap/trunc from args), certify <H>, pickle.
    -> (mps, perm, H)."""
    pkl = args.prep_pkl or f"data/work/prep/{args.tag}_ns{args.ns}_{state_label(args, info)}_anc{anc_site}.pkl"
    if os.path.exists(pkl):
        with open(pkl, "rb") as f:
            ck = pickle.load(f)
        if ck["ns"] == args.ns and ck["anc_site"] == anc_site:
            log(f"prep MPS from {pkl}  <H> = {ck['H']:.6f}")
            return ck["mps"], ck["perm"], ck["H"]
        log(f"{pkl} does not match (ns/anc_site); recomputing")
    os.makedirs(os.path.dirname(pkl), exist_ok=True)
    t1 = time.time()
    mps, perm = backends.prepare_state_mps(lat, prep, anc_site, cap=args.cap_prep,
                                           trunc=args.trunc_prep, max_threads=args.threads)
    n_tot = lat.n_qubits + 1
    qc = QuantumCircuit(n_tot)
    qc.set_matrix_product_state(mps)
    H = ham.build_hamiltonian(lat, args.m0, args.g2, args.eta)
    qc.save_expectation_value(backends.permute_pauli(H, perm, n_tot), list(range(n_tot)), label="H")
    sim = AerSimulator(method="matrix_product_state",
                       matrix_product_state_truncation_threshold=1e-8,
                       max_parallel_threads=args.threads)
    Hval = float(np.real(sim.run(qc).result().data()["H"]))
    log(f"prep+store {time.time()-t1:.0f}s  <H> = {Hval:.6f}  -> {pkl}")
    with open(pkl, "wb") as f:
        pickle.dump({"mps": mps, "perm": perm, "H": Hval, "ns": args.ns, "anc_site": anc_site,
                     "couplings": (args.m0, args.g2, args.eta), "state": state_label(args, info),
                     "cap": args.cap_prep, "trunc": args.trunc_prep}, f)
    return mps, perm, Hval
