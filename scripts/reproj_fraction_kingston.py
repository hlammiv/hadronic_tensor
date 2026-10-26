"""Re-projection fraction at the kingston configuration (prod card, t <= 3,
six slices, sigma_t = 1.5, wedge [0, 2.5], published mask): matched filter
of the Re-only reference map against the one built from the complex G."""
import numpy as np
from htensor import analysis, asymmetry
from htensor.lattice import Z2Lattice

NS, VC = 50, 24
lat = Z2Lattice(NS, pbc=True)
tru = np.load("data/w_meson_ns50_k1.26_v3.npz")
g_vac = analysis.subtract(tru["corr_vac"], None, tru["one_pt_vac"],
                          complex(tru["insert_1pt_vac"]))[:, :NS]
g_wp = analysis.subtract(tru["corr_wp"], None, tru["one_pt_wp"],
                         complex(tru["insert_1pt_wp"]))[:, :NS]
G_mps = g_wp - g_vac
TIER3 = {0.5: "data/hw/job_tier3k_T05MERGED.npz", 1.0: "data/hw/job_tier3i_T10RZZ.npz",
         1.5: "data/hw/job_tier3h_T15RZZ.npz", 2.0: "data/hw/job_tier3f_T20RZZ.npz",
         3.0: "data/hw/job_tier3g_T30RZZ.npz"}
times = np.array([0.0] + sorted(TIER3))
mask = [np.ones(NS, bool)]
for t in sorted(TIER3):
    d = np.load(TIER3[t])
    bet, kap = np.asarray(d["beta_v"][0]).real, np.asarray(d["kappa_v"][0]).real
    mask.append((kap > 0.05) & (np.abs(bet) > 0.01))
mask = np.array(mask)
ti = [int(np.argmin(np.abs(tru["times"] - t))) for t in times]
G = G_mps[ti]
x0 = analysis.ring_fold((np.arange(NS) - VC) / 2, lat.nx)
q0 = np.arange(-1.0, 6.001, 0.04)
ks = np.arange(-(lat.nx // 2), lat.nx // 2 + 1)
q1 = 2 * np.pi * ks / lat.nx
DT, DX, sig_x, SIG_T = 0.5, 0.5, lat.nx / 2.0, 1.5

def onesided(G, msk, sig_t):
    wt = analysis.time_window(times, sig_t, "gauss").copy(); wt[0] *= 0.5
    wx = np.exp(-x0 ** 2 / (2 * sig_x ** 2))[None, :] * msk
    et = np.exp(1j * np.outer(q0, times)); ex = np.exp(-1j * np.outer(x0, q1))
    return 2 * np.real(DT * DX * (et @ ((wt[:, None] * wx) * G) @ ex))

pub = np.load("data/hw_w00_coarse_t3.npz")
W_re = onesided(G.real, mask, SIG_T)
print("reproduces published W_ref:", np.allclose(W_re, pub["W_ref"]), np.abs(W_re - pub["W_ref"]).max())
for label, m in (("published mask", mask), ("no mask", np.ones_like(mask))):
    W_re = onesided(G.real, m, SIG_T); W_cx = onesided(G, m, SIG_T)
    reg = asymmetry.region_mask(q0)
    dev = np.abs(W_re - W_cx)[reg].max() / np.abs(W_cx)[reg].max()
    print(f"{label}: A(Re-only vs complex) = {asymmetry.amplitude(W_re, W_cx, q0):.3f}, "
          f"A(complex vs Re-only) = {asymmetry.amplitude(W_cx, W_re, q0):.3f}, "
          f"max |dW|/max|W_cx| on wedge = {dev:.3f}")
    for st in (0.75 * SIG_T, SIG_T, 1.5 * SIG_T):
        print(f"   sigma_t={st:.3f}: A = {asymmetry.amplitude(onesided(G.real, m, st), onesided(G, m, st), q0):.3f}")
r = np.sqrt((np.abs(G.imag) ** 2).sum(1) / (np.abs(G.real) ** 2).sum(1))
print("rms |Im G|/|Re G| per slice:", np.round(r, 3), " all:", np.round(np.sqrt((G.imag**2).sum()/(G.real**2).sum()), 3))
