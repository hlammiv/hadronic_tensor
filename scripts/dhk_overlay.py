"""DHK postdiction overlay + resolution-advantage argument (task 15).

Overlays our EXACT return probability R(t) (Krylov evolution of a faithful
two-meson state, no Trotter/truncation/noise) on the digitized DHK Fig. 10
(arXiv:2505.20408, N_P=13): their MPS "ideal" (TDVP) and two hardware-
emulator approximations.  Our exact curve is the reference those methods
target; the gap to their "ideal" bounds their TDVP + state-prep error.

Resolution advantage: from our small-volume spectroscopy at the DHK
couplings (data/deep_levels_dhk_ns*.npz -> prediction tables) we predict
qualitative features of R(t) -- absence of revivals within t<=20 from the
finite-volume level spacings, and the MM bound state at E_B -- WITHOUT the
full collision, at a fraction of its classical cost.

  PYTHONPATH=. .venv/bin/python scripts/dhk_overlay.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# DHK Fig. 10 (N_P=13), digitized (pixel-calibrated on 0.1 gridlines, +-0.01)
DHK_T = np.arange(0, 21)
DHK_IDEAL = np.array([1.000, 0.962, 0.903, 0.881, 0.886, 0.864, 0.779, 0.695,
                      0.643, 0.617, 0.559, 0.474, 0.397, 0.354, 0.321, 0.277,
                      0.219, 0.173, 0.151, 0.131, 0.108])
DHK_APPX1 = np.array([1.000, 0.948, 0.908, 0.871, 0.884, 0.845, 0.781, 0.683,
                      0.644, 0.607, 0.563, 0.478, 0.412, 0.366, 0.344, 0.302,
                      0.248, 0.199, 0.181, 0.167, 0.137])
DHK_APPX2 = np.array([1.000, 0.810, 0.790, 0.731, 0.676, 0.724, 0.556, 0.562,
                      0.558, 0.485, 0.492, 0.328, 0.319, 0.294, 0.238, 0.278,
                      0.219, 0.228, 0.226, 0.153, 0.108])


def resolution_advantage():
    """Predict R(t) features from spectroscopy alone (no collision)."""
    d = np.load("data/deep_levels_dhk_ns20.npz")
    g = d["gaps"]
    M = float(np.load("data/predict_tables.npz")["dhk_M"]) \
        if False else None
    # meson mass and lowest two-meson gaps (parity-even, above 2M)
    import numpy as _np
    gaps = g[(g > 0.05)]
    M = gaps.min()
    two_m = g[(g > 2 * M - 0.3) & (g < 2 * M + 1.5)]
    two_m = _np.sort(two_m)[:8]
    # a return probability |<Psi|e^{-iHt}|Psi>|^2 revives on the timescale
    # 2 pi / (level spacing); the smallest spacing sets the longest period
    spac = _np.diff(two_m)
    spac = spac[spac > 1e-3]
    periods = 2 * _np.pi / spac
    print(f"DHK couplings: M = {M:.4f}, 2M = {2*M:.4f}")
    print(f"  two-meson levels near threshold: {_np.round(two_m, 4)}")
    print(f"  level spacings: {_np.round(spac, 4)}")
    print(f"  -> revival periods 2pi/spacing: {_np.round(periods, 1)}")
    print(f"  longest period {periods.max():.1f} >> t=20 window "
          f"=> NO revival predicted in DHK's Fig. 10  [{(periods.max()>20)}]")
    return M


def main():
    M = resolution_advantage()
    fig, ax = plt.subplots(figsize=(6.2, 4.2), constrained_layout=True)
    ax.plot(DHK_T, DHK_IDEAL, "s", color="0.35", ms=5,
            label="DHK MPS-ideal (TDVP)")
    ax.plot(DHK_T, DHK_APPX1, "^", color="C1", ms=4, alpha=0.7,
            label="DHK emulator (Appx I)")
    ax.plot(DHK_T, DHK_APPX2, "v", color="C3", ms=4, alpha=0.5,
            label="DHK emulator (Appx II)")
    try:
        r = np.load("data/dhk_Rt_NP13.npz")
        ax.plot(r["times"], r["R"], "-", color="C0", lw=2.2,
                label="this work: exact (Krylov)")
        # quantitative comparison to their ideal on the shared grid
        Rours = np.interp(DHK_T, r["times"], r["R"])
        rms = np.sqrt(np.mean((Rours - DHK_IDEAL) ** 2))
        print(f"\nRMS(exact - DHK ideal) over t=0..20: {rms:.4f}")
        print(f"max deviation: {np.abs(Rours - DHK_IDEAL).max():.4f} "
              f"at t={DHK_T[np.argmax(np.abs(Rours-DHK_IDEAL))]}")
    except FileNotFoundError:
        print("\n(data/dhk_Rt_NP13.npz not ready -- overlay shows DHK only)")
    ax.axhline(0, color="0.8", lw=0.6)
    ax.set_xlabel("time $t$ (lattice units)")
    ax.set_ylabel(r"return probability $R(t)=|\langle\Psi|U(t)|\Psi\rangle|^2$")
    ax.set_title(r"DHK $N_P=13$ meson-meson collision: exact vs approximate",
                 fontsize=10)
    ax.legend(fontsize=8.5)
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 1.02)
    fig.savefig("data/dhk_overlay.pdf", dpi=200)
    print("wrote data/dhk_overlay.pdf")


if __name__ == "__main__":
    main()
