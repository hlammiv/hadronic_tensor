"""Vacuum-transfer bias correction (pass 8) for the tier-3 C(t,x) slices.

The pass-8 vacuum pubs run the identical prep-class circuit (vacuum
ansatz + gadget + U(t)) whose noiseless truth is known at every site
(data/hw_t3_idealVAC50_k1.26.npz).  Pushing the vacuum data through the
IDENTICAL estimator (packet-mirror kappa/beta) yields the residual map
  R_C(t,x) = C_cal[vacuum data] - C_ideal[vacuum circuit],
the site-resolved bias of the calibration chain on a known state.

VALIDATION: the parity-mean wings of R_C must match the packet slices'
measured wing-anchor offsets (x id_a) -- the region where both are
measured.  Disagreement > TOL aborts the transfer.

APPLICATION: the stored slices already subtract parity-mean wing
anchors, so only the parity-demeaned map  Rt = R_C - <R_C>_wing,parity
is applied (site-resolved shape, no double counting):
  C_corr = C_stored - Rt(t,x) * s(t),   s = two-qubit-count ratio for
slices without their own vacuum pub (50% of the scaled part enters the
error), s=1 where measured (t=1.0 CZ, t=2.0 rzz).  Vacuum shot noise
and a 10% session-transfer systematic are added in quadrature.

  PYTHONPATH=. .venv/bin/python scripts/vacuum_transfer.py
"""
import sys
import numpy as np

sys.path.insert(0, "scripts")
from htensor import Z2Lattice
from htensor.measure import split_current
from htensor import currents as cur

NS, NSYS, C = 50, 100, 24
TOL = 0.5                     # abort if wing parity means disagree worse
WING = np.abs(np.arange(NS) - C) >= 6
lat = Z2Lattice(NS, pbc=True)
id_b = np.array([split_current(cur.charge_density(lat, v))[0]
                 for v in range(NS)]).real
CZC = -0.5
id_a, c_a = 0.5, -0.5

idl = np.load("data/hw_t3_ideal50_k1.26.npz")        # packet ideal
vgl = np.load("data/hw_t3_idealVAC50_k1.26.npz")     # vacuum ideal
it0 = int(np.argmin(np.abs(idl["times"])))
sx_i0 = np.array([idl[f"XB_{v}"][it0] for v in range(NS)]) \
    - id_b * idl["X"][it0]
b_i0 = np.array([idl[f"B_{v}"][it0] for v in range(NS)])
vt0 = int(np.argmin(np.abs(vgl["times"])))
vb_i0 = np.array([vgl[f"B_{v}"][vt0] for v in range(NS)])
A0v = vb_i0[C]


def vac_ideal_C(t):
    r = int(np.argmin(np.abs(vgl["times"] - t)))
    sxi = np.array([vgl[f"XB_{v}"][r] for v in range(NS)]) \
        - id_b * vgl["X"][r]
    bi = np.array([vgl[f"B_{v}"][r] for v in range(NS)])
    return (c_a * sxi + id_a * (bi - id_b)
            + id_b * (A0v - id_a) + id_a * id_b)


def obs(key, d):
    b = d[key][:, ::-1]
    z = 1.0 - 2.0 * b[:, 0:NSYS:2].astype(np.float64)
    xa = 1.0 - 2.0 * b[:, NSYS].astype(np.float64)
    Bs = id_b[None, :] + CZC * z
    N = len(b)
    return dict(sx=(xa[:, None] * Bs).mean(0) - id_b * xa.mean(),
                sxe=(xa[:, None] * Bs
                     - id_b[None, :] * xa[:, None]).std(0) / np.sqrt(N),
                B=Bs.mean(0), Be=Bs.std(0) / np.sqrt(N))


def mirror_cal(m):
    kap = np.where(np.abs(sx_i0) > 0.02, m["sx"] / sx_i0, np.nan)
    kap = np.where(np.isnan(kap), np.nanmedian(kap), kap)
    bet = np.where(np.abs(b_i0 - 0.5) > 0.02,
                   (m["B"] - 0.5) / (b_i0 - 0.5), 1.0)
    return kap, bet


def residual_map(vkey, vfile, mobs, t):
    v = obs(vkey, vfile)
    kap, bet = mirror_cal(mobs)
    b_cal = 0.5 + (v["B"] - 0.5) / bet
    C_cal = (c_a * v["sx"] / kap + id_a * (b_cal - id_b)
             + id_b * (A0v - id_a) + id_a * id_b)
    R = C_cal - vac_ideal_C(t)
    Re_err = np.sqrt((c_a / kap) ** 2 * v["sxe"] ** 2
                     + (id_a * v["Be"] / np.abs(bet)) ** 2)
    return R, Re_err, kap


dA = np.load("data/hw/losch_bits_d9nqt4mij12s73fu7f00.npz")   # v1.0 CZ
dB = np.load("data/hw/losch_bits_d9nqtfeij12s73fu7fbg.npz")   # rzz job
d12 = [np.load("data/hw/losch_bits_d9lsndnbupns73e7uobg.npz"),
       np.load("data/hw/losch_bits_d9not7csfqic73ara4m0.npz")]


def pooled_m10():
    """Pooled passes-1/2 m1.0 (the CZ-twirled calibrator sessions)."""
    parts = []
    for d in d12:
        b = d["m1.0"][:, ::-1]
        parts.append(b)
    b = np.concatenate(parts)
    z = 1.0 - 2.0 * b[:, 0:NSYS:2].astype(np.float64)
    xa = 1.0 - 2.0 * b[:, NSYS].astype(np.float64)
    Bs = id_b[None, :] + CZC * z
    N = len(b)
    return dict(sx=(xa[:, None] * Bs).mean(0) - id_b * xa.mean(),
                sxe=(xa[:, None] * Bs
                     - id_b[None, :] * xa[:, None]).std(0) / np.sqrt(N),
                B=Bs.mean(0), Be=Bs.std(0) / np.sqrt(N))


# residual maps: t=1.0 (CZ, cross-session pooled mirror) and t=2.0
# (rzz, in-session pass-8 mirror)
R10, R10e, _ = residual_map("v1.0", dA, pooled_m10(), 1.0)
R20, R20e, _ = residual_map("v2.0", dB, obs("m2.0", dB), 2.0)

# validation against the packet wing anchors (b-sector deltas x id_a)
PACK_ANCH = {1.0: (0.073, 0.005), 2.0: (0.1216, -0.0613)}  # even, odd
ok_all = True
for t, R in [(1.0, R10), (2.0, R20)]:
    for par, nm in ((0, "even"), (1, "odd ")):
        s = WING & (np.arange(NS) % 2 == par)
        rw = np.mean(R[s])
        pk = id_a * PACK_ANCH[t][par]
        rel = abs(rw - pk) / max(abs(pk), 0.02)
        flag = "OK" if rel < TOL else "FAIL"
        ok_all &= rel < TOL
        print(f"t={t} {nm}: R_C wing {rw:+.4f} vs packet anchor "
              f"{pk:+.4f}  ({rel:.0%} {flag})")
if not ok_all:
    raise SystemExit("TRANSFER VALIDATION FAILED — not applied")

# parity-demeaned maps (anchors already applied in the stored slices)
def demean(R):
    out = R.copy()
    for par in (0, 1):
        s = WING & (np.arange(NS) % 2 == par)
        out[np.arange(NS) % 2 == par] -= np.mean(R[s])
    return out


Rt10, Rt20 = demean(R10), demean(R20)
print(f"shape correction |Rt| center region: t=1.0 "
      f"{np.abs(Rt10[~WING]).mean():.4f}  t=2.0 "
      f"{np.abs(Rt20[~WING]).mean():.4f}")

# apply: (slice file, base map, err, depth-scale, extra syst frac)
PLAN = [("data/hw/job_tier3a_POOLED.npz", Rt10, R10e, 1958 / 2714, 0.5),
        ("data/hw/job_tier3b_POOLED.npz", Rt10, R10e, 1.0, 0.1),
        ("data/hw/job_tier3h_T15RZZ.npz", Rt20, R20e, 1950 / 3522, 0.5),
        ("data/hw/job_tier3f_T20RZZ.npz", Rt20, R20e, 1.0, 0.1),
        ("data/hw/job_tier3g_T30RZZ.npz", Rt20, R20e, 4728 / 3522, 0.5)]
for path, Rt, Re, s, fsys in PLAN:
    d = np.load(path)
    out = {k: d[k] for k in d.files}
    corr = s * Rt
    out["C_cal"] = (d["C_cal"][0] - corr)[None, :]
    err = np.sqrt(d["C_err"][0] ** 2 + (s * Re) ** 2
                  + (fsys * corr) ** 2 + (0.1 * corr) ** 2)
    out["C_err"] = err[None, :]
    new = path.replace(".npz", "_VT.npz")
    np.savez(new, **out)
    print(f"{path.split('/')[-1]}: scale {s:.2f}, median |corr| "
          f"{np.median(np.abs(corr)):.4f} -> {new.split('/')[-1]}")
