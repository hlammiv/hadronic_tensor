"""Ideal (noiseless, hardware-Trotterized, dt = 0.5) calibration grids for
every insertion family and probe type of the four-component campaign --
the t = 0 truth the mirror calibration divides by (kappa_v = sx_mirror /
sx_ideal) and the full-grid reference for the dress rehearsal.

Generalizes scripts/hw_t3_dryrun.py `ideal50` (J0 insertion, J0 probes) to

  insertion families   j0   : CZ(anc, site CENTER)          c_a = -1/2, id_a = (-1)^c/2
                       j1p1 : Y_a Z_l X_b on bond CENTER    c_a = +eta/4, id_a = 0
                       j1p2 : X_a Z_l Y_b on bond CENTER    c_a = -eta/4, id_a = 0
  probe observables    J0 sites v : XB_v, YB_v, B_v   (X/Y/I ancilla x J0(v) incl. identity)
                       J1 bonds b : XT1_b, YT1_b, T1_b, XT2_b, YT2_b, T2_b
                                    (ancilla x the Pauli TERM, seam string included)
  plus X, Y (bare ancilla), one_pt_J0 / one_pt_J1 (no ancilla), insert_1pt, H_prep.

The j0 file keeps the legacy key names (times, probes, id_a, c_a, H_prep,
one_pt, XB_v, YB_v, B_v, X, Y) so it is a drop-in for data/hw_t3_ideal50_*.

  PYTHONPATH=. .venv/bin/python scripts/hw_cal_grids.py --params data/wp10reg_params_k1.26_L3.npz \
      --insert j1 --probes j0,j1 --times 0 0.5 1 1.5 2 2.5 3
  -> data/hw_cal_<tag>_ns<ns>_<family>_<state>.npz  (one file per family)
"""

import argparse
import sys
import os

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prep_common import add_state_args, resolve, build_prep, stored_prep, log  # noqa: E402
from htensor import backends, trotter  # noqa: E402
from htensor import currents as cur  # noqa: E402
from htensor.measure import split_current, controlled_pauli  # noqa: E402
from htensor.pauli import pauli_term  # noqa: E402

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
add_state_args(p)
p.add_argument("--insert", choices=["j0", "j1"], default="j0")
p.add_argument("--insert-site", type=int, default=None,
               help="insertion site/bond (default = --center, the block centre); e.g. center+1 for the "
                    "dither family j0d that calibrates the packet-region bias on the physics circuit")
p.add_argument("--probes", default="j0,j1", help="comma list of j0, j1")
p.add_argument("--times", type=float, nargs="+", default=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
p.add_argument("--dt", type=float, default=0.5, help="hardware Trotter step")
p.add_argument("--trunc", type=float, default=1e-8)
p.add_argument("--cap-evol", type=int, default=None,
               help="bond-dimension cap for the evolution circuits (default none; the relA point needs one "
                    "at t > 2 -- check convergence against a larger cap on one slice)")
p.add_argument("--out", default=None, help="output template; '{family}' is substituted")
p.add_argument("--mirror", action="store_true",
               help="also run the depth-matched mirror (same Trotter skeleton, total angle 1e-8) "
                    "for every t > 0; keys prefixed m_ (noiseless: equals the t=0 row)")
p.add_argument("--noise", type=float, nargs=3, metavar=("P2", "P1", "SEED"), default=None,
               help="Pauli-trajectory noise (per-2q, per-1q rate, seed) on the transpiled circuits: "
                    "one trajectory per run, for the noisy dress rehearsal (hw_t3_dryrun.noise_transform)")
args = resolve(p.parse_args())
rng = np.random.default_rng(int(args.noise[2])) if args.noise else None
probe_kinds = [s.strip() for s in args.probes.split(",") if s.strip()]

lat, prep, info = build_prep(args)
n_sys, n_tot, anc = lat.n_qubits, lat.n_qubits + 1, lat.n_qubits
C = args.insert_site if args.insert_site is not None else args.center
DITHER = args.insert_site is not None and args.insert_site != args.center

# ---- insertion families
if args.insert == "j0":
    id_a, terms = split_current(cur.charge_density(lat, C))
    families = {("j0d" if DITHER else "j0"): (terms[0][0], float(np.real(terms[0][1])), float(np.real(id_a)))}
    insert_op = cur.charge_density(lat, C)
else:
    if lat.is_seam(C):
        raise SystemExit("J1 insertion on the seam bond is not supported (JW string)")
    insert_op = cur.bond_current(lat, C, args.eta)
    id_a, terms = split_current(insert_op)
    families = {}
    for ops, c in terms:
        name = ("j1p1" if ops[lat.site_qubit(C)] == "Y" else "j1p2") + ("d" if DITHER else "")
        families[name] = (ops, float(np.real(c)), 0.0)
anc_site = min(next(iter(families.values()))[0])
assert anc_site == lat.site_qubit(C), "all families anchor the ancilla at the insertion site"

mps, perm, Hval = stored_prep(args, lat, prep, anc_site, info)
_sim_opts = {"method": "matrix_product_state",
             "matrix_product_state_truncation_threshold": args.trunc,
             "max_parallel_threads": args.threads}
if args.cap_evol:
    _sim_opts["matrix_product_state_max_bond_dimension"] = args.cap_evol
sim = AerSimulator(**_sim_opts)


def with_anc(op, ap):
    return SparsePauliOp([ap + l for l in op.paulis.to_labels()], op.coeffs)


def j1_terms(b):
    a, l, bb = lat.site_qubit(b), lat.link_qubit(b), lat.site_qubit(b + 1)
    string = {q: "Z" for q in lat.seam_string_qubits()} if lat.is_seam(b) else {}
    return {1: {a: "Y", l: "Z", bb: "X", **string}, 2: {a: "X", l: "Z", bb: "Y", **string}}


def probe_observables():
    obs = {}
    if "j0" in probe_kinds:
        for v in range(lat.ns):
            B = cur.charge_density(lat, v)
            for tag, ap in (("XB", "X"), ("YB", "Y"), ("B", "I")):
                obs[f"{tag}_{v}"] = with_anc(B, ap)
    if "j1" in probe_kinds:
        for b in range(lat.ns):
            for k, ops in j1_terms(b).items():
                T = pauli_term(n_sys, ops, 1.0)
                for tag, ap in ((f"XT{k}", "X"), (f"YT{k}", "Y"), (f"T{k}", "I")):
                    obs[f"{tag}_{b}"] = with_anc(T, ap)
    for ap in ("X", "Y"):
        obs[ap] = SparsePauliOp([ap + "I" * n_sys])
    return obs


def noise_transform(circ, p2, p1):
    """One Pauli trajectory: after each 2q gate, with prob p2 a random Pauli on
    each of its qubits; after each 1q gate, with prob p1 a random non-identity
    Pauli (scripts/hw_t3_dryrun.py:66-81, the g*=1 model calibrated by the
    S(q) self-consistency study)."""
    out = QuantumCircuit(circ.num_qubits)
    for inst in circ.data:
        qs = [circ.find_bit(b).index for b in inst.qubits]
        out.append(inst.operation, qs)
        nn = inst.operation.num_qubits
        if nn == 2 and rng.random() < p2:
            for q in qs:
                k = rng.integers(0, 4)
                if k:
                    (out.x, out.y, out.z)[k - 1](q)
        elif nn == 1 and rng.random() < p1:
            (out.x, out.y, out.z)[rng.integers(0, 3)](qs[0])
    return out


def run(circ_logical, obs):
    """obs: {label: SparsePauliOp on the LOGICAL register (n_tot qubits)}."""
    transform = (lambda c: noise_transform(c, args.noise[0], args.noise[1])) if rng is not None else None
    tqc, perm_obs = backends.chain_transpile(circ_logical, perm, n_tot, transform)
    full = QuantumCircuit(n_tot)
    full.set_matrix_product_state(mps)
    full.compose(tqc, inplace=True)
    for lbl, op in obs.items():
        full.save_expectation_value(backends.permute_pauli(op, perm_obs, n_tot), list(range(n_tot)), label=lbl)
    d = sim.run(full).result().data()
    return {k: float(np.real(d[k])) for k in obs}


MIRROR_EPS = 1e-8


def evolution(t, mirror=False):
    n = int(round(t / args.dt))
    return trotter.trotter_circuit(lat, args.m0, args.g2, args.eta, MIRROR_EPS if mirror else t, n)


# ---- one-point functions (no ancilla; ancilla idle in |0>)
one_obs = {}
for v in range(lat.ns):
    one_obs[f"J0_{v}"] = with_anc(cur.charge_density(lat, v), "I")
    one_obs[f"J1_{v}"] = with_anc(cur.bond_current(lat, v, args.eta), "I")
one_obs["ins"] = with_anc(insert_op, "I")
one_pt_J0 = np.empty((len(args.times), lat.ns))
one_pt_J1 = np.empty((len(args.times), lat.ns))
insert_1pt = None
for i, t in enumerate(args.times):
    qc = QuantumCircuit(n_tot)
    qc.compose(evolution(t), qubits=range(n_sys), inplace=True)
    d = run(qc, one_obs)
    one_pt_J0[i] = [d[f"J0_{v}"] for v in range(lat.ns)]
    one_pt_J1[i] = [d[f"J1_{v}"] for v in range(lat.ns)]
    if i == 0:
        insert_1pt = d["ins"]
    log(f"one-point t={t:3.1f} done  <J0({C})> = {one_pt_J0[i, C]:+.5f}  <ins> = {insert_1pt:+.5f}")

# ---- ancilla grids per family
obs = probe_observables()
for fam, (ins_ops, c_a, id_a_f) in families.items():
    rows = {k: [] for k in obs}
    mrows = {f"m_{k}": [] for k in obs}
    key = f"XB_{C}" if "j0" in probe_kinds else f"XT1_{C}"
    for t in args.times:
        qc = QuantumCircuit(n_tot)
        qc.h(anc)
        controlled_pauli(qc, anc, ins_ops)      # unitarily == the parity-ladder gadget
        qc.compose(evolution(t), qubits=range(n_sys), inplace=True)
        d = run(qc, obs)
        for k in obs:
            rows[k].append(d[k])
        msg = f"{fam} t={t:3.1f} done  <{key}> = {rows[key][-1]:+.5f}"
        if args.mirror:
            qm = QuantumCircuit(n_tot)
            qm.h(anc)
            controlled_pauli(qm, anc, ins_ops)
            if t > 0:
                qm.compose(evolution(t, mirror=True), qubits=range(n_sys), inplace=True)
            dm = run(qm, obs)
            for k in obs:
                mrows[f"m_{k}"].append(dm[k])
            msg += f"  mirror <{key}> = {mrows['m_' + key][-1]:+.5f}"
        log(msg)
    if args.mirror:
        rows.update(mrows)
    suffix = f"_noise{int(args.noise[2])}" if args.noise else ""
    out = (args.out or f"data/hw_cal_{args.tag}_ns{args.ns}_{{family}}_{info['label']}{suffix}.npz").format(family=fam)
    np.savez(out, times=np.array(args.times), probes=np.arange(lat.ns), bonds=np.arange(lat.ns),
             id_a=id_a_f, c_a=c_a, H_prep=Hval, one_pt=one_pt_J0, one_pt_J0=one_pt_J0,
             one_pt_J1=one_pt_J1, insert_1pt=insert_1pt, family=fam, insert=args.insert,
             m0=args.m0, g2=args.g2, eta=args.eta, ns=args.ns, center=C, block_center=args.center, dt=args.dt,
             k0=(info.get("k0") if info.get("k0") is not None else np.nan),
             sigma_x=(info.get("sigma") if info.get("sigma") is not None else np.nan),
             state=info["label"], noise=(np.array(args.noise) if args.noise else np.zeros(3)),
             cap_evol=(args.cap_evol or 0), trunc=args.trunc,
             **{k: np.array(v) for k, v in rows.items()})
    log(f"saved {out}")
