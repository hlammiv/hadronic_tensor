"""The prepared state's error, propagated: prep factors per probe per time.

The mirror calibration divides a probe's signal at time t by the mirror's
signal at the same depth, which is the circuit damping N(t) times the
hardware's EQUAL-TIME correlator over the ideal one (note of 2026-10-22,
sec. 14).  That second factor, the prep factor f_v(0), is what the prepared
state's error does to the probe, and it is only right for the dynamical
signal at t > 0 if the error acts on it the same way, which nothing
guarantees.  This script asks the question directly: it takes the card's own
preparation circuit at the block's training size (Ns = 10, where the state
is exactly the trained one), damages it with a Pauli-trajectory noise model,
evolves the damaged state under the ideal Trotter dynamics, and reads off

    f_v(t) = S_noisy(v, t) / S_ideal(v, t),    S(v, t) = Re <J0_v(t) A>

for the J0 probes (A = Z_c, the j0 insertion) and the two J1 Pauli terms,
together with the one-points.  The equal-time factors f_v(0) are what the
hardware mirrors measure (kappa_v / kappa_centre), so they fit the noise
strength with no free normalisation; the t > 0 factors are the prediction,
and f_v(t) / f_v(0) is the residual bias of the mirror calibration on probe
v at time t.

Two noise models, to bound the model dependence:

  gate   a random Pauli after each 2q gate of the preparation with
         probability p2 (p1 = p2 / 10 after each 1q gate), the dress
         rehearsal's model (htq_hw.sim.noise_transform)
  post   single-qubit depolarising of strength p on every qubit after a
         perfect preparation
  zgate  dephasing during the preparation: a Z on each qubit of a 2q gate
         with probability p (p / 10 after a 1q gate); Gauss-law preserving
         on matter, so it damages the ZZ correlators only through the gates
         that follow, while hitting the coherence the imaginary branch
         measures directly (note sec. 15f)

Stages (``--stage``):

  fit    t = 0 only, a scan over the noise strength, chi^2 against the
         hardware equal-time factors given with ``--target`` (json written by
         scripts/prep_factor_targets.py)
  run    the dynamics at one noise strength (``--p``), all times

  PYTHONPATH=. python scripts/prep_factor_model.py --card prod --model gate --stage fit --p 0.005 0.01 0.02 0.03 --ntraj 200
  PYTHONPATH=. python scripts/prep_factor_model.py --card prod --model gate --stage run --p 0.02 --ntraj 300 --out data/work/prep_factor/prod_gate.npz
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from htq_hw import DT  # noqa: E402
from htq_hw import circuits as C  # noqa: E402
from htq_hw import sim as S  # noqa: E402
from htq_hw.model import Lattice, charge_terms, current_terms, to_sparse_pauli_op  # noqa: E402

CARDS = {"prod": "prod_k1.26_s0.75_ns50", "relA": "relA_k1.26_s0.75_ns50"}
# the j0 insertion A = Z_c and the two J1 insertion terms on bond c
INSERTIONS = ("j0", "j1p1", "j1p2")


def pauli_matrix(lat, terms):
    """Sparse matrix of a term table on the system register."""
    return to_sparse_pauli_op(lat.n_qubits, terms).to_matrix(sparse=True).tocsr()


class Probes:
    """J0 probes (every site), J1 Pauli terms (every bond) and the insertions,
    as sparse matrices, built once."""

    def __init__(self, lat: Lattice, center: int, eta: float):
        self.lat, self.center = lat, center
        self.J0 = [pauli_matrix(lat, charge_terms(lat, v)) for v in range(lat.ns)]
        self.T1, self.T2 = [], []
        for b in range(lat.ns):
            t1, t2 = current_terms(lat, b, 1.0)
            # the bare Pauli TERM with its seam sign, coefficient stripped (eta/4)
            self.T1.append(pauli_matrix(lat, [(t1[0], np.sign(t1[1]))]))
            self.T2.append(pauli_matrix(lat, [(t2[0], np.sign(t2[1]))]))
        zc = {lat.site_qubit(center): "Z"}
        self.A = {"j0": pauli_matrix(lat, [(zc, 1.0)]),
                  "j1p1": self.T1[center], "j1p2": self.T2[center]}

    def read(self, psi, phis: dict) -> dict:
        """-> {'b': (ns,), 'one1': (ns,), 'one2': (ns,), 'S_<fam>': (3, ns)}
        S rows: J0 probes, T1 terms, T2 terms; S = Re <psi| P |phi_fam>."""
        out = {"b": np.array([np.vdot(psi, m @ psi).real for m in self.J0]),
               "one1": np.array([np.vdot(psi, m @ psi).real for m in self.T1]),
               "one2": np.array([np.vdot(psi, m @ psi).real for m in self.T2])}
        for fam, phi in phis.items():
            z = np.array([[np.vdot(psi, m @ phi) for m in group] for group in (self.J0, self.T1, self.T2)])
            out["S_" + fam] = z.real          # the real branch (X ancilla): Re <P(t) A>
            # the imaginary branch (Y ancilla): Im <P(t) A> = (1/2i) Tr([A, rho] P(t)), zero for
            # any state that commutes with the insertion, so first-order sensitive to the
            # prepared state's coherence across the insertion's eigensectors, where the
            # real branch's Z^2 = 1 anchor is blind to it
            out["Simag_" + fam] = z.imag
        return out


def noisy_prep(prep_isa, model: str, p: float, rng, lat: Lattice):
    """One trajectory of the damaged preparation -> (circuit, number of Paulis inserted)."""
    if p <= 0:
        return prep_isa, 0
    if model == "gate":
        out = S.noise_transform(prep_isa, rng, p, p / 10.0)
        return out, len(out.data) - len(prep_isa.data)
    if model == "post":
        out, n = prep_isa.copy(), 0
        for q in range(lat.n_qubits):
            if rng.random() < p:
                (out.x, out.y, out.z)[rng.integers(0, 3)](q)
                n += 1
        return out, n
    if model == "zgate":
        # dephasing during the preparation: a Z on each qubit of a 2q gate with
        # probability p, on a 1q gate's qubit with probability p / 10.  Z on
        # matter commutes with Gauss's law and with every Z probe, so the damage
        # to the ZZ correlators comes only through the gates that follow a kick,
        # while the coherence across the insertion's eigensectors (the
        # imaginary branch, note sec. 15f) is hit directly.
        from qiskit import QuantumCircuit
        out, n = QuantumCircuit(prep_isa.num_qubits), 0
        for inst in prep_isa.data:
            qs = [prep_isa.find_bit(b).index for b in inst.qubits]
            out.append(inst.operation, qs)
            pr = p if len(qs) == 2 else p / 10.0
            for q in qs:
                if rng.random() < pr:
                    out.z(q)
                    n += 1
        return out, n
    raise ValueError(model)


def p_clean(prep_isa, model: str, p: float, lat: Lattice) -> float:
    """Probability that a trajectory inserts no Pauli at all.  Sampling is
    stratified on it: the clean stratum is the ideal state exactly, and the
    Monte Carlo runs over damaged trajectories only (rejection), which takes
    the variance down by (1 - p_clean)^2 and makes the small-p limit exact."""
    if p <= 0:
        return 1.0
    if model == "gate":
        n2 = sum(1 for inst in prep_isa.data if inst.operation.num_qubits == 2)
        n1 = sum(1 for inst in prep_isa.data if inst.operation.num_qubits == 1)
        # a 2q event inserts nothing with probability 1/16 (both Paulis drawn as I)
        return float((1 - p * 15 / 16) ** n2 * (1 - p / 10.0) ** n1)
    if model == "zgate":
        n2 = sum(1 for inst in prep_isa.data if inst.operation.num_qubits == 2)
        n1 = sum(1 for inst in prep_isa.data if inst.operation.num_qubits == 1)
        return float((1 - p) ** (2 * n2) * (1 - p / 10.0) ** n1)
    return float((1 - p) ** lat.n_qubits)


def statevector(sim, qc):
    # the circuits are already in Aer's native gate set: no transpile pass
    c = qc.copy()
    c.save_statevector()
    return np.asarray(sim.run(c).result().get_statevector().data)


def evolve_all(sim, step, states: dict, n_steps: int) -> list[dict]:
    """-> per time k = 0..n_steps, {name: statevector}.  One Aer run per
    initial state, every step's statevector saved."""
    from qiskit import QuantumCircuit
    saved = {}
    if n_steps == 0:
        return [dict(states)]
    for name, v in states.items():
        qc = QuantumCircuit(step.num_qubits)
        qc.set_statevector(v)
        for k in range(1, n_steps + 1):
            qc.compose(step, inplace=True)
            qc.save_statevector(label=f"t{k}")
        data = sim.run(qc).result().data(0)
        saved[name] = [v] + [np.asarray(data[f"t{k}"].data) for k in range(1, n_steps + 1)]
    return [{name: saved[name][k] for name in states} for k in range(n_steps + 1)]


def trajectory(sim, prep_isa, step, probes: Probes, model, p, rng, lat, n_steps, insertions):
    qc, n = noisy_prep(prep_isa, model, p, rng, lat)
    while p > 0 and n == 0:                     # damaged stratum only (see p_clean)
        qc, n = noisy_prep(prep_isa, model, p, rng, lat)
    psi = statevector(sim, qc)
    states = {"psi": psi}
    for fam in insertions:
        states[fam] = probes.A[fam] @ psi
    rows = []
    for st in evolve_all(sim, step, states, n_steps):
        rows.append(probes.read(st["psi"], {f: st[f] for f in insertions}))
    return rows


def accumulate(acc, rows):
    for k, r in enumerate(rows):
        for key, val in r.items():
            acc.setdefault(key, [np.zeros_like(val) for _ in rows])
            acc[key][k] += val
            acc.setdefault(key + "_sq", [np.zeros_like(val) for _ in rows])
            acc[key + "_sq"][k] += val ** 2


def finish(acc, n, ideal=None, p0=0.0):
    """Trajectory means and errors; with ``ideal`` and ``p0`` the clean stratum
    is added exactly: result = p0 ideal + (1 - p0) <damaged>."""
    out = {}
    for key, lst in acc.items():
        if key.endswith("_sq"):
            continue
        mean = np.array(lst) / n
        var = np.array(acc[key + "_sq"]) / n - mean ** 2
        err = np.sqrt(np.maximum(var, 0.0) / max(n - 1, 1))
        if ideal is not None:
            mean = p0 * ideal[key] + (1 - p0) * mean
            err = (1 - p0) * err
        out[key] = mean
        out[key + "_err"] = err
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--card", choices=sorted(CARDS), default="prod")
    ap.add_argument("--ns", type=int, default=10)
    ap.add_argument("--model", choices=("gate", "post", "zgate"), default="gate")
    ap.add_argument("--stage", choices=("fit", "run"), default="run")
    ap.add_argument("--p", type=float, nargs="+", default=[0.02])
    ap.add_argument("--ntraj", type=int, default=200)
    ap.add_argument("--tmax", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--precision", choices=("double", "single"), default="double",
                    help="Aer statevector precision; single is about twice as fast and good to 1e-7")
    ap.add_argument("--insertions", default="j0,j1p1,j1p2")
    ap.add_argument("--target", default=None, help="json of hardware equal-time factors (fit stage)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    card = C.load_card(CARDS[args.card])
    m0, g2, eta = C.card_couplings(card)
    lat = Lattice(args.ns)
    center = args.ns // 2 - 1
    insertions = tuple(args.insertions.split(","))
    prep = C.prep_circuit(card, args.ns, center)
    prep_isa = S._aer_ready(prep)
    n2q = sum(1 for inst in prep_isa.data if inst.operation.num_qubits == 2)
    n1q = sum(1 for inst in prep_isa.data if inst.operation.num_qubits == 1)
    step = C.trotter_step(lat, DT, m0, g2, eta)
    n_steps = int(round(args.tmax / DT)) if args.stage == "run" else 0
    from qiskit_aer import AerSimulator
    sim = AerSimulator(method="statevector", max_parallel_threads=args.threads, precision=args.precision)
    probes = Probes(lat, center, eta)
    print(f"{args.card}: ns={args.ns} centre={center} couplings {m0},{g2},{eta}; prep {n2q} 2q + {n1q} 1q gates "
          f"(AER basis); model {args.model}; {n_steps} steps of dt={DT}", flush=True)

    t0 = time.time()
    ideal = finish((lambda a: (accumulate(a, trajectory(sim, prep_isa, step, probes, args.model, 0.0, None,
                                                        lat, n_steps, insertions)), a)[1])({}), 1)
    print(f"  ideal pass {time.time() - t0:.1f} s", flush=True)
    results = {}
    for p in args.p:
        rng = np.random.default_rng(args.seed)
        acc, t1 = {}, time.time()
        for i in range(args.ntraj):
            accumulate(acc, trajectory(sim, prep_isa, step, probes, args.model, p, rng, lat, n_steps, insertions))
            if i == 0:
                print(f"  p={p}: first trajectory {time.time() - t1:.1f} s", flush=True)
        p0 = p_clean(prep_isa, args.model, p, lat)
        results[p] = finish(acc, args.ntraj, ideal, p0)
        print(f"  p={p}: {args.ntraj} damaged trajectories in {time.time() - t1:.0f} s; clean stratum weight "
              f"{p0:.3f} added exactly", flush=True)
        # equal-time factors on the J0 probes and J1 terms, j0 insertion
        s0, si = results[p]["S_j0"][0], ideal["S_j0"][0]
        f0 = np.where(np.abs(si) > 0.02, s0 / np.where(np.abs(si) > 0.02, si, 1.0), np.nan)
        # the hardware measures kappa_v / kappa_c, so normalise by the insertion site
        f0 = f0 / f0[0, center]
        rel = lambda row: " ".join(f"{x:5.2f}" for x in row)  # noqa: E731
        print(f"    f_c(0) at the insertion site (Z^2 = 1 up to <Z_c>/2): {s0[0, center] / si[0, center]:.3f}")
        print(f"    f_v(0)/f_c(0) J0 sites 0..{args.ns - 1}: {rel(f0[0])}")
        print(f"    f_b(0) T1 bonds       : {rel(f0[1])}")
        print(f"    f_b(0) T2 bonds       : {rel(f0[2])}", flush=True)
        if args.target:
            tg = json.load(open(args.target))[args.card]
            chi, n = 0.0, 0
            for kind, row in (("J0", 0), ("T1", 1), ("T2", 2)):
                for off, (val, err) in tg[kind].items():
                    v = (center + int(off)) % args.ns
                    if np.isfinite(f0[row, v]):
                        chi += ((f0[row, v] - val) / err) ** 2
                        n += 1
            print(f"    chi2/n against hardware: {chi:.1f}/{n}", flush=True)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        save = {"times": np.arange(n_steps + 1) * DT, "center": center, "ns": args.ns, "model": args.model,
                "card": args.card, "ps": np.array(args.p), "ntraj": args.ntraj, "n2q_prep": n2q,
                "p_clean": np.array([p_clean(prep_isa, args.model, p, lat) for p in args.p]),
                "couplings": np.array([m0, g2, eta])}
        for key, val in ideal.items():
            save["ideal_" + key] = val
        for p, res in results.items():
            for key, val in res.items():
                save[f"p{p:g}_" + key] = val
        np.savez(args.out, **save)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
