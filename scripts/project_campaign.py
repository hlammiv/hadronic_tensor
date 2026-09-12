"""Projected outcome of the phoenix campaign: what W^{mu nu} and its error
bars will look like, from the MEASURED noise model and the costed shot plan.

This is a Monte Carlo of the campaign, not a gate-level simulation.  For each
campaign time slice and component we take the exact hardware-Trotterized
correlator from the ideal grid, damp it by the mirror factor kappa(t) fitted
to the kingston campaign, add shot noise scaled from the end-to-end rehearsal,
estimate kappa from a mirror that carries its OWN shot noise (the
kappa-estimation floor is a real error source, cf. the 7k-shot pass-10 lesson),
run the same calibration and assembly the real analysis uses, and repeat.

Noise model, all three pieces empirical:
  kappa(t)   = exp(-0.265 - 1.01e-4 * n2q(t))      kingston mirror-damping fit,
               n2q from the target embedding (ladder: prep + 300/step)
  sigma_C    = sigma_reh * sqrt(N_reh/N) * kappa_reh/kappa(t)
               sigma_reh from the Ns=50 rehearsal slices at known shot count
  kappa_hat  = kappa * (1 + eps),  eps ~ N(0, sigma_kappa/kappa)  per site
               with sigma_kappa from the mirror's own shots

  PYTHONPATH=. .venv/bin/python scripts/project_campaign.py CARD [--nmc 200]
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_style import OI  # noqa: E402
from htensor import Z2Lattice, analysis, tensor  # noqa: E402

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--tag", default="prod", help="card tag: prod or relA")
p.add_argument("--ideal", default="data/hw_cal_{tag}_ns50_{family}_k1.26_s0.75{suffix}.npz")
p.add_argument("--suffix", default="_t8")
p.add_argument("--ref-j0", default="data/w_prodR_ns50_j0_k1.26_s0.75.npz")
p.add_argument("--ref-j1", default="data/w_prodR_ns50_j1_k1.26_s0.75.npz")
p.add_argument("--times", type=float, nargs="+", default=None, help="campaign slices (default 0.5..6 by 0.5 + t=0)")
p.add_argument("--shots", type=float, default=6e4, help="shots per physics pub")
p.add_argument("--mirror-shots", type=float, default=6e4)
p.add_argument("--embedding", choices=["ladder", "ring"], default="ladder")
p.add_argument("--prep-2q", type=int, default=None, help="prep 2q gates (default: 603 ladder / 872 ring, +208 for relA)")
p.add_argument("--step-2q", type=int, default=None, help="2q gates per Trotter step (default 300 ladder / 458 ring)")
p.add_argument("--signal", choices=["ideal", "correlator"], default="ideal",
               help="where the noiseless C(t,x) comes from: the dt=0.5 ideal grids (exact hardware "
                    "Trotterization, needs grids covering every campaign time) or the MPS correlator "
                    "grids (dt_target=0.1, so it omits the dt=0.5 Trotter error, which is a systematic "
                    "the projection does not claim to model anyway)")
p.add_argument("--nmc", type=int, default=200)
p.add_argument("--seed", type=int, default=11)
p.add_argument("--out", default=None)
args = p.parse_args()

NS, CENTER, DT = 50, 24, 0.5
lat = Z2Lattice(NS, pbc=True)
id_b = np.array([(-1) ** v / 2 for v in range(NS)])
times = np.array(args.times if args.times else [0.0] + list(np.arange(0.5, 6.001, 0.5)))
rng = np.random.default_rng(args.seed)

# ---- gate counts -> kappa(t), the kingston mirror-damping fit
prep2q = args.prep_2q if args.prep_2q else ((603 if args.embedding == "ladder" else 872) + (208 if args.tag == "relA" else 0))
step2q = args.step_2q if args.step_2q else (300 if args.embedding == "ladder" else 458)
n2q = prep2q + 1064 - 1024 + step2q * np.round(times / DT).astype(int)   # + gadget overhead
kappa_t = np.exp(-0.265 - 1.01e-4 * n2q)
kappa_t[times == 0] = np.exp(-0.265 - 1.01e-4 * (prep2q + 40))

# ---- shot-noise scale, calibrated on the Ns=50 rehearsal
SIG_REH, N_REH, KAP_REH = 0.0093, 2.0e4, 0.945     # median C_err, shots, kappa at t=0.5
sig_C = SIG_REH * np.sqrt(N_REH / args.shots) * (KAP_REH / kappa_t)

print(f"card {args.tag}, {args.embedding} embedding: prep {prep2q} 2q, {step2q} 2q/step")
print(f"{'t':>5} {'2q':>6} {'kappa':>7} {'sigma_C':>9}")
for i, t in enumerate(times):
    print(f"{t:>5.1f} {n2q[i]:>6d} {kappa_t[i]:>7.3f} {sig_C[i]:>9.4f}")

# ---- exact hardware-Trotterized correlator per component, from the ideal grids
def ideal_C(family, comp):
    f = args.ideal.format(tag=args.tag, family=family, suffix=args.suffix)
    if not os.path.exists(f):
        f = args.ideal.format(tag=args.tag, family=family, suffix="")
    d = np.load(f, allow_pickle=True)
    tg = np.asarray(d["times"], float)
    rows = [int(np.argmin(np.abs(tg - t))) for t in times]
    if max(abs(tg[r] - t) for r, t in zip(rows, times)) > 1e-6:
        raise SystemExit(f"{f} lacks campaign times (has {tg[0]}..{tg[-1]}); extend the ideal grid")
    id_a, c_a = float(d["id_a"]), float(d["c_a"])
    A0 = float(d["insert_1pt"])
    if comp[0] == "0":      # J0 probes
        sx = np.array([[d[f"XB_{v}"][r] - id_b[v] * d["X"][r] for v in range(NS)] for r in rows])
        b = np.array([[d[f"B_{v}"][r] for v in range(NS)] for r in rows])
        idb_v = id_b
    else:                    # J1 probes: the two Pauli terms
        eta = 1.3 if args.tag == "prod" else 2.3
        sx = np.array([[eta / 4 * (d[f"XT1_{v}"][r] - d[f"XT2_{v}"][r]) for v in range(NS)] for r in rows])
        b = np.array([[eta / 4 * (d[f"T1_{v}"][r] - d[f"T2_{v}"][r]) for v in range(NS)] for r in rows])
        idb_v = np.zeros(NS)
    C = c_a * sx + id_a * (b - idb_v) + idb_v * (A0 - id_a) + id_a * idb_v
    return C, sx, b, id_a, c_a, A0, idb_v

FAM = {"00": "j0", "10": "j0", "01": "j1p1", "11": "j1p1"}

# ---- reference (MPS truth) on the same times, for the comparison panel
g = tensor.load_mps_components(args.ref_j0, args.ref_j1)


def correlator_C(comp):
    """Signal from the MPS correlator grids: split the full C into the ancilla
    part A = c_a*sx (which the device damps by kappa and the mirror calibrates)
    and the probe one-point b (damped toward its identity value)."""
    src, block, _ = tensor._LAYOUT[comp]
    d = np.load(args.ref_j0 if src == "j0" else args.ref_j1)
    rows = [int(np.argmin(np.abs(np.asarray(d["times"], float) - t))) for t in times]
    sl = slice(block * NS, (block + 1) * NS)
    C = np.asarray(d["corr_wp"][rows, sl], float)
    b = np.asarray(d["one_pt_wp"][rows, sl], float)
    A0 = float(np.real(d["insert_1pt_wp"]))
    idb_v = id_b if comp[0] == "0" else np.zeros(NS)
    id_a = float((-1) ** CENTER / 2) if comp[1] == "0" else 0.0
    c_a = -0.5 if comp[1] == "0" else (1.3 if args.tag == "prod" else 2.3) / 4
    sx = (C - id_a * (b - idb_v) - idb_v * (A0 - id_a) - id_a * idb_v) / c_a
    return C, sx, b, id_a, c_a, A0, idb_v


ideal = {c: (ideal_C(FAM[c], c) if args.signal == "ideal" else correlator_C(c))
         for c in tensor.COMPONENTS}
q0 = np.arange(-1.0, 6.001, 0.04)
ks = np.arange(-(lat.nx // 2), lat.nx // 2 + 1)
q1 = 2 * np.pi * ks / lat.nx
sig_x = lat.nx / 2.0
sig_t = max(0.45, times[-1] / 2)

def transform(G, x):
    return analysis.onesided_ft(times, x, G, q0, q1, sig_t, sig_x, DT, 0.5)

# ---- Monte Carlo of the campaign
odd = lambda W: 0.5 * (W - W[:, ::-1])
reg = (q0 >= -1) & (q0 <= 2.5)
W_mc = {c: [] for c in tensor.COMPONENTS}
A_mc, A_win = [], []
ti = [int(np.argmin(np.abs(g.times - t))) for t in times]
W_ref = {c: transform(g.g[c][ti].real, g.x[c]) for c in tensor.COMPONENTS}
T_template = odd(W_ref["00"])[reg]
den = float((T_template * T_template).sum())

for _ in range(args.nmc):
    Wm = {}
    for c in tensor.COMPONENTS:
        C, sx, b, id_a, c_a, A0, idb_v = ideal[c]
        # device: damp, add shot noise; mirror: same damping, its own shot noise
        sx_meas = kappa_t[:, None] * sx + rng.normal(0, sig_C[:, None], sx.shape)
        sx_mirr = kappa_t[:, None] * sx[0][None, :] + rng.normal(
            0, sig_C[:, None] * np.sqrt(args.shots / args.mirror_shots), sx.shape)
        vis = np.abs(sx[0]) > 0.02
        kap_hat = np.where(vis[None, :], sx_mirr / np.where(vis, sx[0], 1.0)[None, :], np.nan)
        kap_hat = np.where(np.isnan(kap_hat), np.nanmedian(kap_hat, axis=1)[:, None], kap_hat)
        ok = kap_hat > 0.05
        sx_cal = np.where(ok, sx_meas / np.where(ok, kap_hat, 1.0), 0.0)
        b_meas = idb_v[None, :] + kappa_t[:, None] * (b - idb_v[None, :]) + rng.normal(0, sig_C[:, None], b.shape)
        b_cal = idb_v[None, :] + (b_meas - idb_v[None, :]) / kappa_t[:, None]
        C_cal = c_a * sx_cal + id_a * (b_cal - idb_v) + idb_v * (A0 - id_a) + id_a * idb_v
        G = (C_cal - b_cal * A0) - (C[0] - b[0] * A0)[None, :] * 0   # connected pieces cancel in the diff below
        G = C_cal - C + g.g[c][ti].real                              # device deviation added to the truth
        if c in ("00", "01"):                                        # charge-conservation projection
            for i in range(len(times)):
                G[i] -= G[i].sum() / NS
        Wm[c] = transform(G, g.x[c])
        W_mc[c].append(Wm[c])
        if c == "00":
            G00_last = G
    A_mc.append(float((T_template * odd(Wm["00"])[reg]).sum()) / den)
    # window systematic: the published analysis scans sigma_t by 0.75/1.5 and
    # quotes half the spread; it does NOT shrink with shots
    a_w = [float((T_template * odd(analysis.onesided_ft(times, g.x["00"], G00_last, q0, q1,
                                                        f * sig_t, sig_x, DT, 0.5))[reg]).sum()) / den
           for f in (0.75, 1.5)]
    A_win.append(0.5 * abs(a_w[1] - a_w[0]))

A_mc = np.array(A_mc)
print(f"\nprojected campaign ({args.nmc} Monte Carlo replicas, {args.shots:.0e} shots/pub, "
      f"{len(times)} slices to t={times[-1]})")
print(f"{'comp':>5} {'max|W_ref|':>11} {'bias':>9} {'scatter':>9} {'corr':>7}")
out = {}
for c in tensor.COMPONENTS:
    M = np.array(W_mc[c]); mean, sd = M.mean(axis=0), M.std(axis=0)
    corr = np.corrcoef(mean.ravel(), W_ref[c].ravel())[0, 1]
    print(f"{c:>5} {np.abs(W_ref[c]).max():>11.4f} {np.abs(mean-W_ref[c]).max():>9.4f} "
          f"{sd.max():>9.4f} {corr:>7.4f}")
    out[f"W_mean_{c}"], out[f"W_sd_{c}"], out[f"W_ref_{c}"] = mean, sd, W_ref[c]
A_win = np.array(A_win)
tot = np.hypot(A_mc.std(), A_win.mean())
print(f"boost asymmetry A = {A_mc.mean():.3f} +- {A_mc.std():.3f}(shot) +- {A_win.mean():.3f}(window)"
      f"  -> {A_mc.mean()/tot:.1f} sigma (published: 0.85 +- 0.34 +- 0.20, 2.2 sigma)")
print("NOT modelled: the packet-region state-dependent bias that limited the published forward region.")
print("It is a systematic, not a statistical error, and measuring it is exactly what the dither pubs are for.")

tag = args.out or f"data/projection_{args.tag}_{args.embedding}"
np.savez(tag + ".npz", q0=q0, q1=q1, times=times, kappa=kappa_t, sigma_C=sig_C, n2q=n2q,
         A=A_mc, A_window=A_win, shots=args.shots, **out)

fig, axes = plt.subplots(2, 4, figsize=(13.2, 5.6), constrained_layout=True)
for j, c in enumerate(tensor.COMPONENTS):
    mean, sd = out[f"W_mean_{c}"], out[f"W_sd_{c}"]
    v = max(np.abs(W_ref[c]).max(), np.abs(mean).max())
    pm = axes[0, j].pcolormesh(q1, q0, mean, cmap="RdBu_r", vmin=-v, vmax=v, shading="nearest", rasterized=True)
    axes[0, j].set_title(rf"$W^{{{c}}}$ projected ({args.tag})", fontsize=9)
    sn = np.abs(mean) / np.maximum(sd, 1e-12)
    axes[1, j].pcolormesh(q1, q0, np.clip(sn, 0, 10), cmap="viridis", shading="nearest", rasterized=True)
    axes[1, j].set_title(rf"$|W|/\sigma$ (cap 10)", fontsize=9)
    for a in axes[:, j]:
        a.set_xlabel(r"$q^1$"); a.grid(False)
    fig.colorbar(pm, ax=axes[0, j], pad=0.02)
axes[0, 0].set_ylabel(r"$q^0$"); axes[1, 0].set_ylabel(r"$q^0$")
fig.suptitle(f"projected {args.tag} campaign: {len(times)} slices, {args.shots:.0e} shots/pub, "
             f"{args.embedding} embedding, A = {A_mc.mean():.2f} ± {A_mc.std():.2f}(shot) ± {A_win.mean():.2f}(win)", fontsize=10)
fig.savefig(tag + ".pdf", dpi=200)
print("wrote", tag + ".npz", tag + ".pdf")
