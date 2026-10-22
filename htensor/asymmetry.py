"""The boost asymmetry: definition, extraction and its two errors.

A boosted packet's response is not symmetric under q^1 -> -q^1, and that
asymmetry is the kinematic signature the measurement is after.  It is
isolated by the odd projection

    W_odd(q^0, q^1) = [W(q^0, q^1) - W(q^0, -q^1)] / 2,

which kills everything symmetric in the momentum transfer, the static
structure included.

Reporting a two-dimensional map is not a measurement, so the map is reduced
to one number by a **matched filter**: the least-squares amplitude of the
measured odd map along the odd map of the classical (MPS) reference,

    A = sum_R T .  W_odd^meas / sum_R T . T,      T = W_odd^ref,

over a wedge R of the (q^0, q^1) plane.  A = 1 means the device reproduces
the reference asymmetry in full; A = 0 means no asymmetry survived.  The
template fixes the shape, so A tests the amplitude of a predicted structure
rather than fitting a free one -- and because the filter is linear in the
measured slices, the shot error propagates exactly through the transform
(`shot_error` below) instead of being bootstrapped.

Two errors are quoted and they do not combine into one:

- **shot**, from the per-slice, per-site errors pushed through the linear
  filter.  Falls as 1/sqrt(N).
- **window**, from the arbitrariness of the one-sided transform's time
  window sigma_t.  Half the peak-to-peak of A over a scan of sigma_t.  This
  is a systematic: more shots do not shrink it, and on the published data it
  is 60% as large as the shot error.

A third effect is NOT an error on A and is not quoted as one.  The device
measures only the real part of the connected correlator, so A is the
amplitude of the **Re-projected** asymmetry; the reference goes through the
same projection, so it cancels in the ratio, but the Re-projected asymmetry
is 0.67 of the full one at the production point and 0.30 at relA.

Published (ibm_kingston, t <= 3, 6 slices, sigma_t = 1.5, wedge q0 in
[0, 2.5]): see data/hw_w00_coarse_t3.npz for the current A_hat, A_err and
A_sys; the script prints them.
Extracted by scripts/hw_w00_coarse.py, stored as `A_hat`, `A_err`, `A_sys`
in data/hw_w00_coarse_t3.npz, and reproducible with

    PYTHONPATH=. python scripts/boost_asymmetry.py data/hw_w00_coarse_t3.npz
"""

import numpy as np

# q^0 range of the filter.  It starts at 0, not at -1, because for a real
# integrand the one-sided transform satisfies W(-q0, q1) = W(q0, -q1)
# identically, so the odd projection is exactly odd in q0 and every bin
# below zero is the sign-flipped mirror of one above, in the template as
# much as in the data.  Those bins are redundant rather than wrong: the old
# [-1, 2.5] wedge gives 0.9305 on the published kingston map against 0.9705
# here.  See scripts/w_map_symmetry.py.
WEDGE = (0.0, 2.5)


def odd(W) -> np.ndarray:
    """Odd part in q^1, assuming the q^1 axis is symmetric about zero and
    is the LAST axis of ``W`` (shape (n_q0, n_q1))."""
    W = np.asarray(W)
    return 0.5 * (W - W[..., ::-1])


def region_mask(q0, wedge=WEDGE) -> np.ndarray:
    q0 = np.asarray(q0)
    return (q0 >= wedge[0]) & (q0 <= wedge[1])


def amplitude(W_meas, W_ref, q0, wedge=WEDGE) -> float:
    """The matched-filter amplitude A.  Both maps must come from the SAME
    transform (same times, same window, same mask): the template carries the
    window's own distortion, and comparing across windows would fold that
    into A."""
    reg = region_mask(q0, wedge)
    T = odd(W_ref)[reg]
    den = float((T * T).sum())
    if den <= 0:
        raise ValueError("empty template: the reference has no odd part on this wedge")
    return float((T * odd(W_meas)[reg]).sum()) / den


def shot_error(G_err, template, transform, mask=None) -> float:
    """Exact propagation of the per-slice, per-site shot errors.

    ``transform(G)`` -> W applies the same one-sided Fourier transform used
    for the data; ``template`` is the (already masked) T of `amplitude`.
    Each measured point is pushed through the filter on its own to get its
    kernel coefficient K, and the errors add in quadrature.  This is exact
    because the filter is linear, so no resampling is needed.
    """
    G_err = np.asarray(G_err)
    den = float((template * template).sum())
    var = 0.0
    for i in range(G_err.shape[0]):
        for x in range(G_err.shape[1]):
            if mask is not None and not mask[i][x]:
                continue
            basis = np.zeros_like(G_err)
            basis[i, x] = 1.0
            K = float((template * odd(transform(basis))).sum()) / den
            var += (K * G_err[i][x]) ** 2
    return float(np.sqrt(var))


def window_systematic(G, template, transform_sigma, sigmas=(0.75, 1.0, 1.5)) -> float:
    """Half the peak-to-peak of A as the transform window is scanned.

    ``transform_sigma(f)`` -> W applies the transform with sigma_t scaled by
    f.  The template is held fixed, so this measures the window's effect on
    the data rather than on the comparison.
    """
    den = float((template * template).sum())
    vals = [float((template * odd(transform_sigma(f))).sum()) / den for f in sigmas]
    return float(0.5 * np.ptp(vals))


def significance(A: float, shot: float, window: float) -> tuple[float, float]:
    """-> (total error, sigma).  The two errors are independent and added in
    quadrature; the window part does not shrink with shots."""
    tot = float(np.hypot(shot, window))
    return tot, float(A / tot) if tot > 0 else float("inf")
