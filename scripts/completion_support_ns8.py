"""What each step of the W^{mu nu} pipeline costs the q0 support, exactly.

The hadronic tensor is supposed to have weight only at q0 = E_n - E_psi >=
-dE.  The published pipeline reaches it through three approximations, and
exact states on a small ring price them one at a time, with no device and no
reference input:

  two-sided    the true complex G(x, t) on |t| <= T, windowed.  The only
               error is truncation plus the window itself.
  completed    G(x, -t) = [G(-x, t)]* applied to the complex t >= 0 half
               instead of measuring it.  Exact for a momentum eigenstate,
               O(sigma_p^2) for a packet.
  Re only      what the device actually delivers: the real part of the
               connected correlator on t >= 0, through analysis.onesided_ft.
               For real G the transform satisfies W(-q0, q1) = W(q0, -q1)
               identically, so this step is what forces the map to be a
               reflection rather than a one-sided response.

and the sampling step dt is priced separately, since a transform over a grid
of spacing dt is periodic in q0 with period 2 pi / dt and folds everything
above pi / dt back into the window.

  PYTHONPATH=. python scripts/completion_support_ns8.py [--ns 8] [--sigma-x 1.0]
"""

import argparse
import os
import sys
import time

import numpy as np
import scipy.sparse.linalg as spla

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from htensor import Z2Lattice, analysis, exact, spectroscopy, tensor  # noqa: E402
from htensor import currents as cur  # noqa: E402
from htensor import hamiltonian as ham  # noqa: E402
from run_outputs import in_place  # noqa: E402

p = argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--ns", type=int, default=8)
p.add_argument("--m0", type=float, default=0.7)
p.add_argument("--g2", type=float, default=1.1)
p.add_argument("--eta", type=float, default=1.3)
p.add_argument("--sigma-x", type=float, nargs="+", default=[1.0],
               help="packet widths; the completion error is O(sigma_p^2) = O(1/sigma_x^2)")
p.add_argument("--tmax", type=float, default=8.0, help="production time range")
p.add_argument("--tlong", type=float, default=32.0, help="stand-in for the spectral limit")
p.add_argument("--dt", type=float, default=0.1, help="fine sampling step")
p.add_argument("--dt-coarse", type=float, default=0.5, help="the published sampling step")
p.add_argument("--reconstruct", action="store_true",
               help="also ask whether imposing the support bound recovers Im G "
                    "from Re G alone, which would make the Re-only measurement "
                    "lossless and the imaginary branch unnecessary")
args = p.parse_args()

lat = Z2Lattice(args.ns, pbc=True)
CENTER = lat.ns // 2
Q0 = np.arange(-12.0, 12.001, 0.05)
x = tensor.x_grid(lat, "00", CENTER)
ks = np.arange(-(lat.nx // 2), lat.nx // 2 + 1)
Q1 = 2 * np.pi * ks / lat.nx
print(f"ns = {lat.ns} ({lat.n_qubits} qubits, dim {1 << lat.n_qubits}), "
      f"nx = {lat.nx}, centre {CENTER}", flush=True)

H = exact.to_sparse(ham.build_hamiltonian(lat, args.m0, args.g2, args.eta)).tocsc()
band = spectroscopy.meson_band(lat, args.m0, args.g2, args.eta)
vac, e0 = np.asarray(band["vacuum"]), float(band["e0"])
probes = [exact.to_sparse(cur.charge_density(lat, v)) for v in range(lat.ns)]
insert = probes[CENTER]
print(f"vacuum E0 = {e0:+.6f}; band k = {np.round(band['k'], 3).tolist()}", flush=True)


def walk(state, mags, sign):
    """-> (C, P) at times sign * mags, with `mags` ascending from 0.

    C[i, v] = <state| J0(v, t) J0(centre, 0) |state>,  P[i, v] = <J0(v, t)>.
    sign = -1 evolves backwards, which is what makes the two-sided
    correlator available with no completion assumption.  The generator
    carries the sign and the grid stays ascending: expm_multiply's interval
    form returns garbage on a descending grid (norm 16 instead of 1), which
    the unitarity guard below would catch.
    """
    assert mags[0] == 0.0 and np.all(np.diff(mags) > 0), "mags must ascend from 0"
    pair = np.column_stack([state, insert @ state])
    n0 = np.linalg.norm(pair, axis=0)
    C = np.empty((len(mags), lat.ns), dtype=complex)
    P = np.empty((len(mags), lat.ns), dtype=complex)
    # one small step at a time.  expm_multiply's interval form is several
    # times faster but loses accuracy whenever the block does not start at
    # t = 0 (1e-2 in the norm at t = 8, and outright garbage on a descending
    # grid), so the guard below is what decides the method.
    for i, t in enumerate(mags):
        if i:
            pair = spla.expm_multiply(-1j * sign * (t - mags[i - 1]) * H, pair)
        bra, ket = pair[:, 0], pair[:, 1]
        for v, op in enumerate(probes):
            C[i, v] = np.vdot(bra, op @ ket)
            P[i, v] = np.vdot(bra, op @ bra)
    drift = float(np.abs(np.linalg.norm(pair, axis=0) - n0).max())
    assert drift < 1e-8, f"evolution is not unitary at t = {sign * mags[-1]}: {drift:.2e}"
    return C, P


def integrand(state, mags):
    """Connected, vacuum-subtracted G on the full signed grid, exactly."""
    out = {}
    for tag, st in (("psi", state), ("vac", vac)):
        a0 = complex(np.vdot(st, insert @ st))
        Cp, Pp = walk(st, mags, +1)
        Cn, Pn = walk(st, mags, -1)
        C = np.concatenate([Cn[:0:-1], Cp])          # drop the duplicated t = 0
        P = np.concatenate([Pn[:0:-1], Pp])
        out[tag] = C - P * a0
    return out["psi"] - out["vac"]


def reconstruct_imag(G, times, dt, sig_t, sig_x, edge, W_two):
    """Can demanding W = 0 below `edge` recover Im G from Re G alone?

    The forward map is linear in G, so this is a linear least-squares
    problem in the nt x nx unknowns of Im G, constrained on the q0 rows
    below the edge.  Tikhonov-regularized over eight decades, scored
    against the Im G we actually know.  Printed, not returned: the answer
    is a decision about hardware, not a number anything else consumes.
    """
    wt = analysis.time_window(times, sig_t, "gauss").copy()
    wt[0] *= 0.5
    wx = np.exp(-x ** 2 / (2 * sig_x ** 2))
    et = np.exp(1j * np.outer(Q0, times))
    ex = np.exp(-1j * np.outer(x, Q1))
    taper = wt[:, None] * wx[None, :]
    fwd = lambda g: 2 * np.real(dt * 0.5 * (et @ (taper * g) @ ex))

    low, phys = Q0 < edge, Q0 >= edge
    nt, nx = G.shape
    cols = np.empty((low.sum() * len(Q1), nt * nx))
    for k in range(nt * nx):
        e = np.zeros((nt, nx), complex)
        e.flat[k] = 1j
        cols[:, k] = fwd(e)[low].ravel()
    re_only = fwd(np.real(G) + 0j)
    rhs = -re_only[low].ravel()
    U, S, Vt = np.linalg.svd(cols, full_matrices=False)
    pk = np.abs(np.real(W_two)).max()
    dev = lambda W: float(np.abs(W - np.real(W_two))[phys].max() / pk * 100)
    under = lambda W: float(np.abs(W)[low].max() / pk * 100)

    print(f"      constraint matrix {cols.shape}, condition "
          f"{S.max() / max(S.min(), 1e-300):.1e}")
    print(f"      {'lambda':>10}{'under-edge%':>13}{'corr(Im)':>10}"
          f"{'|Im| ratio':>12}{'phys dev%':>11}")
    print(f"      {'truth':>10}{under(np.real(W_two)):13.2f}{1.0:10.3f}"
          f"{1.0:12.3f}{0.0:11.2f}")
    print(f"      {'Re only':>10}{under(re_only):13.2f}{0.0:10.3f}{0.0:12.3f}"
          f"{dev(re_only):11.2f}")
    best = None
    for lam in np.logspace(-6, 2, 17):
        f = S / (S ** 2 + lam ** 2)
        im = (Vt.T @ (f * (U.T @ rhs))).reshape(G.shape)
        W = fwd(np.real(G) + 1j * im)
        corr = float(np.corrcoef(im.ravel(), np.imag(G).ravel())[0, 1])
        ratio = float(np.linalg.norm(im) / np.linalg.norm(np.imag(G)))
        print(f"      {lam:10.1e}{under(W):13.2f}{corr:10.3f}{ratio:12.3f}"
              f"{dev(W):11.2f}")
        if best is None or dev(W) < best[1]:
            best = (lam, dev(W), corr)
    print(f"      best physical-region deviation {best[1]:.2f}% at lambda = "
          f"{best[0]:.1e} (correlation with the true Im G {best[2]:.3f}), "
          f"against {dev(re_only):.2f}% for not trying")


def profile(W):
    return np.sqrt((np.real(W) ** 2).sum(axis=1))


def below(W, edge):
    """max of the q0 profile under `edge`, as a fraction of its peak."""
    pr = profile(W)
    sel = Q0 < edge
    return float(pr[sel].max() / pr.max()) if sel.any() else float("nan")


rows = []
for sig_x in args.sigma_x:
    k0 = 2 * np.pi * max(1, lat.nx // 4) / lat.nx
    mix = spectroscopy.optimize_interpolator(lat, band, k0=k0, sigma_x=sig_x,
                                             x0=CENTER // 2)
    psi, frac = spectroscopy.meson_wavepacket(lat, band, k0=k0, sigma_x=sig_x,
                                              x0=CENTER // 2, mix=mix["mix"])
    psi = np.asarray(psi)
    e_psi = float(np.real(np.vdot(psi, H @ psi)))
    h2 = float(np.real(np.vdot(psi, H @ (H @ psi))))
    spread = float(np.sqrt(max(h2 - e_psi ** 2, 0.0)))
    dE = e_psi - e0
    edge = -(dE + spread)
    print(f"\n-- sigma_x = {sig_x:g}: k0 = {k0:.4f}, band fraction {frac:.4f}, "
          f"dE = {dE:.4f}, energy spread {spread:.4f}", flush=True)
    print(f"   kinematic edge q0 >= -(dE + spread) = {edge:+.4f}", flush=True)

    t0 = time.time()
    keep = {}
    for dt, tmax, label in ((args.dt, args.tmax, "production T, fine dt"),
                            (args.dt, args.tlong, "long T, fine dt"),
                            (args.dt_coarse, args.tmax, "production T, published dt")):
        tp = np.arange(0.0, tmax + 1e-9, dt)
        G = integrand(psi, tp)
        t_full = np.concatenate([-tp[:0:-1], tp])
        sig_t = tmax / 2.0
        npos = len(tp)

        W_two = analysis.windowed_ft(t_full, x, G, Q0, Q1, sig_t, lat.nx / 2.0)
        grid = analysis.CorrelatorGrid(tp, x, G[npos - 1:])
        tc, Gc = analysis.complete_time(grid)
        W_cmp = analysis.windowed_ft(tc, x, Gc, Q0, Q1, sig_t, lat.nx / 2.0)
        W_re = analysis.onesided_ft(tp, x, np.real(G[npos - 1:]), Q0, Q1,
                                    sig_t, lat.nx / 2.0, dt, 0.5)

        phys = Q0 >= edge
        pk = profile(W_two).max()
        print(f"   {label:<26} dt={dt:g} T={tmax:g}  "
              f"period 2pi/dt = {2 * np.pi / dt:.2f}", flush=True)
        for tag, Wm in (("two-sided", W_two), ("completed", W_cmp), ("Re only", W_re)):
            d_tru = float(np.abs(np.real(Wm - W_two))[phys].max() / pk)
            print(f"      {tag:<11} under-edge {100 * below(Wm, edge):6.2f}%   "
                  f"max|Im W| {np.abs(np.imag(Wm)).max() / pk * 100:6.2f}%   "
                  f"delta vs two-sided {100 * d_tru:6.2f}%")
        keep[(dt, tmax)] = np.real(W_two)
        if args.reconstruct and dt == args.dt and tmax == args.tmax:
            print("   does imposing the support bound recover Im G from Re G?",
                  flush=True)
            reconstruct_imag(G[npos - 1:], tp, dt, sig_t, lat.nx / 2.0, edge, W_two)
        rows.append(dict(sigma_x=sig_x, dt=dt, tmax=tmax, dE=dE, spread=spread,
                         two=below(W_two, edge), cmp=below(W_cmp, edge),
                         re=below(W_re, edge),
                         cmp_dev=float(np.abs(np.real(W_cmp - W_two))[phys].max() / pk),
                         re_dev=float(np.abs(np.real(W_re - W_two))[phys].max() / pk)))
    fine, coarse = (args.dt, args.tmax), (args.dt_coarse, args.tmax)
    if fine in keep and coarse in keep:
        phys = Q0 >= edge
        pk = np.abs(keep[fine]).max()
        d = float(np.abs(keep[coarse] - keep[fine])[phys].max() / pk * 100)
        print(f"   aliasing alone, exact two-sided maps at dt = {args.dt_coarse:g} "
              f"against dt = {args.dt:g}: {d:.2f}% of the peak in the physical "
              f"region", flush=True)
    print(f"   ({time.time() - t0:.1f} s)", flush=True)

np.savez(in_place(f"data/completion_support_ns{lat.ns}"),
         **{k: np.array([r[k] for r in rows]) for k in rows[0]},
         ns=lat.ns, m0=args.m0, g2=args.g2, eta=args.eta, q0=Q0, q1=Q1)

print("\nsigma_x   dt   T     dE      under-edge: two / completed / Re only"
      "      deviation in the physical region: completed / Re")
for r in rows:
    print(f"{r['sigma_x']:6g} {r['dt']:5g} {r['tmax']:4g} {r['dE']:7.3f}   "
          f"{100*r['two']:6.2f}% {100*r['cmp']:7.2f}% {100*r['re']:7.2f}%        "
          f"{100*r['cmp_dev']:9.2f}% {100*r['re_dev']:7.2f}%")
