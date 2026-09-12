"""Tensor-derived dispersions at the collision studies' couplings.
Reads data/rt_wvac_<tag>_ns<NS>.npz (vacuum-polarization W^{00} by MPS, same
protocol as the production run), extracts the ridge E(q1) per momentum with the
paper's analysis chain, and compares with (i) the ED bands used in Fig. 10 and
(ii) the published CGK points / DHK R(t).  Writes data/rt_wvac_ridge.json.
Usage: PYTHONPATH=. python scripts/rt_wvac_ridge.py
"""
import json, sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from collections import defaultdict
from htensor import Z2Lattice, analysis
from postdictions_figure import bands, dhk_curve  # ED references used in Fig. 10

def ridge(path):
    d = np.load(path); ns, vc = int(d["ns"]), int(d["vc"]); lat = Z2Lattice(ns, pbc=True)
    times = d["times"]
    c_conn = analysis.subtract(d["corr"], None, d["probe_1pt"], complex(d["insert_1pt"]))
    x = analysis.ring_fold((np.arange(ns) - vc) / 2, lat.nx)
    grid = analysis.CorrelatorGrid(times, x, c_conn)
    t_full, c_full = analysis.complete_time(grid)
    q0 = np.arange(-0.5, 7.0, 0.01)
    ks = np.arange(0, lat.nx // 2 + 1); q1 = 2 * np.pi * ks / lat.nx
    sig_t = times[-1] / 3.0
    sig_x = lat.nx / 6.0 if os.environ.get("RT_XWIN", "paper") == "paper" else 1e6   # RT_XWIN=full -> exact ring momentum projection, no spatial window
    out = {}
    for f in (0.75, 1.0, 1.5):
        W, _ = analysis.window_scan(t_full, x, c_full, q0, q1, sig_t, sig_x, factors=(f,))
        Wr = W.real
        for j, k in enumerate(ks):
            w = Wr[:, j]; thr = 0.005 * Wr.max()
            peaks = [i for i in range(1, len(q0) - 1) if w[i] > w[i-1] and w[i] >= w[i+1] and w[i] > thr and q0[i] > 0.3]
            peaks = sorted(peaks, key=lambda i: -w[i])            # highest first: the meson band dominates
            if peaks:
                i1 = peaks[0]
                higher = [i for i in peaks[1:] if q0[i] > q0[i1] and w[i] > 0.1 * w[i1]]
                peaks = [i1] + ([max(higher, key=lambda i: w[i])] if higher else [])
            pk = []
            for i in peaks[:2]:
                a, b, c = w[i-1], w[i], w[i+1]; den = a - 2*b + c
                pk.append(q0[i] + (0.5 * (a - c) / den * 0.01 if den != 0 else 0.0))
            out.setdefault(int(k), {})[f] = pk
    res = []
    for k in ks:
        p = out[int(k)]; e1 = [p[f][0] for f in p if p[f]]; e2 = [p[f][1] for f in p if len(p[f]) > 1]
        res.append({"j": int(k), "q1": float(2*np.pi*k/lat.nx),
                    "E1": float(np.mean(e1)) if e1 else None, "E1_win": float(np.ptp(e1)/2) if e1 else None,
                    "E2": float(np.mean(e2)) if len(e2) == 3 else None, "E2_win": float(np.ptp(e2)/2) if len(e2) == 3 else None})
    return lat.nx, float(d["m0"]), float(d["g2"]), float(d["eta"]), res

report = {}
cmp = np.load("data/cgk_spectrum_compare.npz")
pub = {"cgkel": cmp["panel_a"], "cgkinA": cmp["panel_c"]}   # (k*L/2pi label, DeltaE) published, L = 30
import glob
runs = [(os.path.basename(f)[8:].split("_ns")[0], int(os.path.basename(f).split("_ns")[1].split("_")[0].split(".")[0]), f) for f in sorted(glob.glob("data/rt_wvac_*.npz"))]
for tag, ns, path in runs:
    nx, m0, g2, eta, res = ridge(path)
    key = os.path.basename(path)[8:-4]
    rows = []
    if tag.startswith("cgk"):
        (k1, e1), (k2, e2) = bands(tag[3:])
        for r in res:
            ed1 = float(np.interp(r["q1"], k1, e1)); ed2 = float(np.interp(r["q1"], k2, e2)) if len(k2) else None
            jl = r["j"]  # tensor physical momentum 2pi j/15 == published label j' = j (L=30, k' = 2pi j'/30, physical 2k' = 2pi j'/15)
            pubv = None
            if tag in pub:
                m = np.isclose(np.abs(pub[tag][:, 0]), jl, atol=1e-6)
                if m.any(): pubv = float(np.mean(pub[tag][m, 1]))
            rows.append({**r, "ED_band1": ed1, "ED_band2": ed2, "published": pubv})
        d1 = [r["E1"] - r["ED_band1"] for r in rows if r["E1"] is not None]
        dp = [r["E1"] - r["published"] for r in rows if r["E1"] is not None and r["published"] is not None]
        report[key] = {"nx": nx, "couplings": (m0, g2, eta), "rows": rows,
                       "rms_vs_ED_band1": float(np.sqrt(np.mean(np.square(d1)))),
                       "rms_vs_published": float(np.sqrt(np.mean(np.square(dp)))) if dp else None, "n_published": len(dp)}
    else:
        # DHK: R(t) from the tensor dispersion instead of the ED dispersion
        true = np.load("data/dhk_Rt_true.npz"); t = true["t"].astype(float); R = true["R_ideal"]
        ks_t = np.array([r["q1"] for r in res if r["E1"] is not None]); Es_t = np.array([r["E1"] for r in res if r["E1"] is not None])
        NP, SIGMA, KBAR = 13, 3*np.pi/13, 2*np.pi/13
        k = np.pi/NP*np.arange(-NP, NP); k = k[(k >= -np.pi/2-1e-9) & (k < np.pi/2-1e-9)]
        w1 = np.exp(-((k-KBAR)**2)/(2*SIGMA**2)); w2 = np.exp(-((k+KBAR)**2)/(2*SIGMA**2)); w1, w2 = w1/w1.sum(), w2/w2.sum()
        Kf = (2*k+np.pi) % (2*np.pi) - np.pi
        # E(q1) is even and periodic; W^{00}(q1=0) vanishes by charge conservation, so fit a
        # three-term cosine series to the tensor points (j>=1) and use it at all momenta
        Acos = np.stack([np.ones_like(ks_t), np.cos(ks_t), np.cos(2*ks_t)], 1)
        cfit, *_ = np.linalg.lstsq(Acos, Es_t, rcond=None)
        Efit = lambda q: cfit[0] + cfit[1]*np.cos(q) + cfit[2]*np.cos(2*q)
        E = Efit(np.abs(Kf))
        fit_resid = float(np.sqrt(np.mean((Acos @ cfit - Es_t)**2))); E0_fit = float(Efit(0.0))
        A1 = np.array([(w1*np.exp(-1j*E*tt)).sum() for tt in t]); A2 = np.array([(w2*np.exp(-1j*E*tt)).sum() for tt in t])
        R_t = np.abs(A1*A2)**2; R_ed = dhk_curve(t)
        from collections import defaultdict as _dd
        kp = {}
        for nsl in (12, 16, 20):
            dl = np.load(f"data/deep_levels_dhk_ns{nsl}.npz"); b = _dd(list)
            for gg, pp in zip(dl["gaps"], dl["phases"]):
                if gg > 0.1: b[abs(round(float(pp), 4))].append(float(gg))
            for kk in b: kp[kk] = min(kp.get(kk, 99), min(b[kk]))
        kk = np.array(sorted(kp)); ee = np.array([kp[q] for q in kk])
        for r in res:
            r["ED_band1"] = float(np.interp(r["q1"], kk, ee))
        d1 = [r["E1"] - r["ED_band1"] for r in res if r["E1"] is not None]
        report[key] = {"nx": nx, "couplings": (m0, g2, eta), "rows": res, "rms_vs_ED_band1": float(np.sqrt(np.mean(np.square(d1)))),
                       "rms_Rt_tensor_vs_published": float(np.sqrt(np.mean((R_t-R)**2))),
                       "rms_Rt_ED_vs_published": float(np.sqrt(np.mean((R_ed-R)**2))),
                       "rms_Rt_tensor_vs_ED": float(np.sqrt(np.mean((R_t-R_ed)**2))), "cos_fit_resid": fit_resid, "E0_from_fit": E0_fit, "E0_ED": float(np.interp(0.0, kk, ee))}
json.dump(report, open("data/rt_wvac_ridge.json" if os.environ.get("RT_XWIN","paper")=="paper" else "data/rt_wvac_ridge_fullx.json", "w"), indent=1)
for tag, r in report.items():
    print(f"\n{tag}: N_x={r['nx']} couplings={r['couplings']}")
    for row in r["rows"]:
        print(f"  j={row['j']:2d} q1={row['q1']:.3f}  E1={row['E1'] if row['E1'] is None else round(row['E1'],4)}  ED1={None if row.get('ED_band1') is None else round(row['ED_band1'],4)}  pub={None if row.get('published') is None else round(row['published'],4)}  E2={row['E2'] if row['E2'] is None else round(row['E2'],4)}")
    print("  " + ", ".join(f"{k}={v}" for k, v in r.items() if k.startswith("rms") or k == "n_published"))
