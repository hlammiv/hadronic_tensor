"""Smith dwell time from the published density rasters of Chai et al.
(arXiv:2505.21240): time-integrated central-region density excess of the
collision relative to a free-propagation model,

    tau_dwell = int dt [ P_C(t) - P_C^free(t) ] / P_norm ,

with P_C the fermion density integrated over the central region C
normalized to the TWO-packet total (so the integral returns the
per-packet delay directly), and the free model
built from the pre-collision data alone: per-packet centroid lines and
Gaussian profiles with linearly growing width, fit on the incoming leg.

Cases: the four (m, kbar) columns of their Fig. scattering_m0.1 raster
plus the elastic eps=1.0 collision (predicted time ADVANCE, the sign
check).  Produces a diagnostic figure (NOT for the paper): raster with
the free-model trajectories and region C, P_C(t) vs the model, and the
running dwell integral.

  PYTHONPATH=. .venv/bin/python scripts/chai_dwell_time.py <scratchdir>
"""
import sys

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl

SP = sys.argv[1]
NS = 30
LUT = {}


def raster(png, cmap):
    if cmap not in LUT:
        LUT[cmap] = mpl.colormaps[cmap](np.linspace(0, 1, 256))[:, :3]
    img = np.asarray(Image.open(png).convert("RGB"),
                     dtype=np.float32) / 255.0
    H, W = img.shape[:2]
    flat = img.reshape(-1, 3)
    lut = LUT[cmap].astype(np.float32)
    idx = np.empty(len(flat), dtype=np.int16)
    for a in range(0, len(flat), 50_000):        # chunked: bounded memory
        blk = flat[a:a + 50_000]
        d2 = ((blk[:, None, :] - lut[None]) ** 2).sum(-1)
        idx[a:a + 50_000] = np.argmin(d2, axis=1)
    val = (idx / 255.0).reshape(H, W)
    val = val[::-1]                       # row 0 = t = 0
    # resample W pixels -> NS sites by block mean
    edges = np.linspace(0, W, NS + 1)
    dens = np.stack([val[:, int(a):max(int(a) + 1, int(b))].mean(1)
                     for a, b in zip(edges[:-1], edges[1:])], axis=1)
    return dens


def ring_d(x, c):
    return (x - c + NS / 2) % NS - NS / 2


CASES = [
    ("$\\bar k=4$ ($m{=}0.1$)", "sws-000.png", "sws-001.png",
     40.0, 20.0, None),
    ("$\\bar k=6$ ($m{=}0.1$)", "sws-002.png", "sws-003.png",
     50.0, 20.0, "exp"),
    ("$\\bar k=8$ ($m{=}0.1$)", "sws-004.png", "sws-005.png",
     80.0, 20.0, "lin"),
    ("$\\bar k=8$ ($m{=}0.2$)", "sws-006.png", "sws-007.png",
     80.0, 20.0, "lin"),
    ("elastic $\\varepsilon{=}1.0$, $\\bar k=6$", "chai_img-000.png",
     None, 80.0, 36.0, None),
]
XBARS = (8.0, 23.0)                     # their packet launch sites

fig, axes = plt.subplots(len(CASES), 3, figsize=(11, 3.0 * len(CASES)))
print(f"{'case':34s}  {'t_int':>5s}  {'tau_dwell by region w':>28s}")
results = {}
for row, (name, png, fpng, T, t_guess, tail) in enumerate(CASES):
    cmap = "Purples"
    dens = raster(f"{SP}/{png}", cmap)
    satfrac = float((dens > 0.97).mean())
    fdens = raster(f"{SP}/{fpng}", "Oranges") if fpng else None
    H = dens.shape[0]
    times = np.linspace(0, T, H)
    x = np.arange(NS, dtype=float)

    # ---------- free-propagation model from the incoming leg ----------
    # fit while the packets stay separated by > 8 cells
    early = np.zeros(len(times), bool)
    g1, g2 = XBARS
    for i in range(len(times)):
        for k, g in ((0, g1), (1, g2)):
            d = ring_d(x, g)
            m = np.abs(d) <= 5
            wgt = np.clip(dens[i, m], 0, None)
            if wgt.sum() > 0:
                if k == 0:
                    g1 = g + (wgt * d[m]).sum() / wgt.sum()
                else:
                    g2 = g + (wgt * d[m]).sum() / wgt.sum()
        early[i] = np.abs(ring_d(np.array([g1]), g2))[0] > 8
        if not early[i]:
            break
    packs = []
    for x0 in XBARS:
        cs, ws, amps, ts = [], [], [], []
        g = x0
        for i in np.flatnonzero(early):
            d = ring_d(x, g)
            m = np.abs(d) <= 5
            wgt = np.clip(dens[i, m], 0, None)
            if wgt.sum() <= 0:
                continue
            c = g + (wgt * d[m]).sum() / wgt.sum()
            w2 = (wgt * (d[m] - (c - g)) ** 2).sum() / wgt.sum()
            cs.append(c); ws.append(np.sqrt(max(w2, 0.3)))
            amps.append(wgt.sum()); ts.append(times[i])
            g = c
        cs = np.unwrap(np.array(cs), period=NS)
        ts, ws, amps = np.array(ts), np.array(ws), np.array(amps)
        v, b = np.polyfit(ts, cs, 1)
        # free dispersion law: sigma(t)^2 = s0^2 + (beta t)^2
        bet2, s02 = np.polyfit(ts ** 2, ws ** 2, 1)
        bet2, s02 = max(bet2, 0.0), max(s02, 0.25)
        area = np.median(amps)            # conserved column area
        packs.append((v, b, s02, bet2, area))

    def free_density(t):
        out = np.zeros(NS)
        for v, b, s02, bet2, area in packs:
            c = v * t + b
            s = np.sqrt(s02 + bet2 * t * t)
            prof = np.exp(-0.5 * (ring_d(x, c) / s) ** 2)
            out += area * prof / prof.sum()   # area-conserving advection
        return out
    # normalization: early-time total density (matter converts to string
    # at the collision, so per-slice normalization would be wrong)
    N0 = np.mean([np.clip(dens[i], 0, None).sum()
                  for i in np.flatnonzero(early)[2:]])

    # integration cutoff: stop when the free packets re-approach within
    # 2 cells across the seam (wrap re-encounter)
    (v1, b1, *_), (v2, b2, *_) = packs
    sep = np.abs(ring_d(v1 * times + b1, v2 * times + b2))
    half = times < 0.5 * T
    t_cross = times[half][int(np.argmin(sep[half]))]
    after = times > t_cross + 0.3 * (NS / 2) / max(abs(v1), 1e-3)
    twrap = times[after][sep[after] < 3.5][0] if (sep[after] < 3.5).any() \
        else T
    ti = times <= twrap

    if fdens is not None:
        NF = np.mean([np.clip(fdens[i], 0, None).sum()
                      for i in np.flatnonzero(early)[2:]])
    taus, curves, traps = [], [], []
    for w in (3, 4, 5):
        inC = np.abs(ring_d(x, (XBARS[0] + XBARS[1]) / 2)) <= w
        PF = np.array([free_density(t)[inC].sum() for t in times]) / N0
        Pm = np.clip(dens, 0, None)[:, inC].sum(1) / N0
        if fdens is not None:
            Pf = np.clip(fdens, 0, None)[:, inC].sum(1) / NF
            PC = 0.5 * (Pm + Pf)      # conversion moves between channels
        else:
            Pf = None
            PC = Pm
        # N0 normalizes to the TWO-packet total, so this integral already
        # equals the per-packet delay (synthetic-delay validated); no 1/2
        run = np.cumsum((PC - PF) * np.gradient(times))
        fitc = None
        if tail == "exp":
            from scipy.optimize import curve_fit as _cf
            fm = (times >= t_cross + 6) & ti
            def _exp(t, ti_, A, g):
                return ti_ - A * np.exp(-g * (t - times[fm][0]))
            p0 = [run[ti][-1] * 1.3, run[ti][-1], 0.05]
            popt, pcov = _cf(_exp, times[fm], run[fm], p0=p0, maxfev=20000)
            taus.append(popt[0])
            if w == 4:
                print(f"    [exp fit w=4] release rate Gamma_eff = "
                      f"{popt[2]:.3f}  ->  lifetime {1/popt[2]:.1f}")
            fitc = ("exp", fm, _exp(times[fm], *popt), popt)
            traps.append(float(np.mean((PC - PF)[fm][-10:])))
        elif tail == "lin":
            fm = (times >= times[ti][-1] - 25) & ti
            (sl, ic), pcov = np.polyfit(times[fm], run[fm], 1, cov=True)
            # prompt dwell referenced to the free crossing time
            taus.append(ic + sl * t_cross)
            traps.append(sl)              # P_trap = late-time excess
            fitc = ("lin", fm, sl * times[fm] + ic, (sl, ic))
        else:
            taus.append(run[ti][-1])
            late = times >= times[ti][-1] - 8.0
            traps.append(float(np.mean((PC - PF)[late & ti])))
        curves.append((w, Pm, Pf, PC, PF, run, fitc))
    results[name] = (np.mean(taus), np.std(taus), twrap,
                     np.mean(traps), np.std(traps), satfrac, tail)
    print(f"{name:34s}  {twrap:5.1f}  "
          + "  ".join(f"w={w}: {t:+.2f}" for (w, *_), t in zip(curves, taus))
          + f"   Ptrap = {np.mean(traps):+.2f}({np.std(traps):.2f})"
          + f"   sat {100*satfrac:.0f}%")

    # ---------------- diagnostics ----------------
    a0, a1, a2 = axes[row]
    a0.pcolormesh(x, times, dens, cmap=cmap, shading="nearest")
    for v, b, *_ in packs:
        a0.plot((v * times + b) % NS, times, "r--", lw=0.8)
    a0.axvspan((XBARS[0] + XBARS[1]) / 2 - 4,
               (XBARS[0] + XBARS[1]) / 2 + 4, color="g", alpha=0.12, lw=0)
    a0.axhline(twrap, color="0.3", lw=0.8, ls=":")
    a0.set_ylabel(f"{name}\n$t$")
    if row == len(CASES) - 1:
        a0.set_xlabel("site $n$")
    w, Pm, Pf, PC, PF, run, fitc = curves[1]
    a1.plot(times, Pm, "-", color="#5e3c99", lw=0.9, label="matter")
    if Pf is not None:
        a1.plot(times, Pf, "-", color="#e66101", lw=0.9, label="field")
        a1.plot(times, PC, "k-", lw=1.3, label="mean")
    a1.plot(times, PF, "r--", lw=1.0, label="free model")
    a1.axvline(twrap, color="0.3", lw=0.8, ls=":")
    a1.legend(fontsize=7)
    a1.set_ylabel("$P_C$ ($w=%d$)" % w)
    for w2, *mid, run2, fc2 in curves:
        a2.plot(times, run2, lw=1.0, label=f"$w={w2}$")
        if fc2 is not None:
            kind, fm2, yfit, pp = fc2
            a2.plot(times[fm2], yfit, "k:", lw=1.2)
            if kind == "exp":
                a2.axhline(pp[0], color="0.4", lw=0.6, ls="--")
    a2.axvline(twrap, color="0.3", lw=0.8, ls=":")
    a2.axhline(0, color="0.6", lw=0.6)
    a2.legend(fontsize=7)
    a2.set_ylabel(r"running $\tau_{\rm dwell}$")
    if row == len(CASES) - 1:
        a1.set_xlabel("$t$")
        a2.set_xlabel("$t$")

fig.suptitle("Smith dwell-time extraction diagnostics "
             "(Chai et al. published rasters)", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.985))
fig.savefig("data/chai_dwell_diagnostics.pdf", dpi=150)
fig.savefig("data/chai_dwell_diagnostics.png", dpi=150)
print("\nwrote data/chai_dwell_diagnostics.{pdf,png}")
for k, (m, s_, tw, pt, pte, sf, tail) in results.items():
    kind = {"exp": "asymptotic(exp fit)", "lin": "prompt(lin fit)",
            None: "at cutoff"}[tail]
    print(f"  {k:34s} tau_dwell = {m:+.2f} +- {s_:.2f} [{kind}]"
          f"   P_trap = {pt:+.2f}({pte:.2f})  sat {100*sf:.0f}%")
