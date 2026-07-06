# Mock referee report — PRD/PRX style

**Manuscript:** "Hadronic tensor of Z2 gauge theory from meson wavepackets on 100 qubits"
(FERMILAB-PUB-26-XXX-T, draft `paper/main_v2.tex`, 2026-07-04 redraft)
**Report prepared:** 2026-07-06 (internal adversarial exercise; written as a demanding but fair PRD referee would)

---

## 1. Summary of the manuscript's claims

The authors present a pipeline for extracting the hadronic tensor W^{mu nu}(q) from
real-time two-current correlators in 1+1d Z2 lattice gauge theory with staggered
matter, explicit dynamical link qubits, and periodic boundary conditions, at
couplings (m0, g2, eta) = (0.7, 1.1, 1.3). The specific claims are:

1. A volume-transferable, Gauss-law-preserving variational state-preparation
   chain: a 2-layer vacuum ansatz optimized at N_s = 6 and reused at N_s = 50
   (energy density reproduced to 5e-6; on-circuit <H> = -52.2254 vs
   "extrapolated exact" -52.227), plus an L = 3 wavepacket block trained at
   N_s = 10 with fidelity quoted as 0.9973, certified by on-circuit packet
   energy landing in the single-meson band.
2. A conserved point-split current with Z_V = 1 and an exact discrete Ward
   identity; a plain-ordered (non-time-ordered) definition of W; pointwise
   vacuum and disconnected subtractions.
3. A single-ancilla Hadamard-test measurement with spatial multiplexing,
   claimed to need ~1e5 shots for the full W^{00} grid at the few-percent
   level — roughly six orders of magnitude below the numerical-differentiation
   protocol of Lamm-Lawrence-Yamauchi — with an ancilla-free
   eigenbasis-projection cross-check.
4. Production MPS (Aer, truncation 1e-8) execution of the true 101-qubit
   circuits: vacuum-polarization tensor reproducing the meson dispersion to
   ~1%, an optical conductivity / timelike-HVP extraction via the Ward
   identity, and (promised, currently placeholder) W^{00} on rest and boosted
   (k0 = 2pi/5) meson wavepackets.
5. A four-part "certification protocol" (volume transfer, on-circuit energy,
   Gauss stabilizers, full-width Clifford-point execution) advertised as a
   general standard for trusting quantum simulations beyond exact classical
   reference.
6. A bridge section converting W peaks to finite-volume two-meson energies and
   elastic phase shifts via the 1+1d quantization condition
   p N_x + 2 delta(p) = 2 pi n, plus form-factor and Omnes-type consistency
   claims.

Supporting material in the repository (`notes/`, `data/`, `scripts/`) contains
additional results evidently intended for the final version: a parameter-free
two-band ridge model matching the wavepacket data at 0.7-2.2% RMS, phase-shift
values delta = +0.57 (p = 1.03) and +0.15 (p = 1.50) with a third point
quarantined, and a rotation-synthesis error study (angle snapping plus a true
gridsynth tier at N_s = 8). I comment on these where the draft text commits to
them, since they will be load-bearing in the submitted version.

The physics program is interesting and timely, the current construction and the
PBC-seam/Gauss-law engineering are genuinely nice, and the shot-cost analysis is
a real service to the community. However, the manuscript in its present form has
several problems that range from incompleteness to internal inconsistency and
overclaiming. I cannot recommend publication without major revisions.

---

## 2. Major concerns

### M1. The manuscript is incomplete, and the abstract claims results the paper does not contain

The abstract states that the authors "measure W^{00}(q^0,q^1) on meson
wavepackets at rest and in motion." Section VI.C — the headline result of the
paper, per the title — is an `\itodo` placeholder. The phase-shift table
(Sec. VII), the boosted-W numbers, the N_s = 12 certification, the
acknowledgments, and the data-availability statement are likewise placeholders.
A referee cannot assess the central claim of a paper that does not yet contain
it. Everything below should therefore be read as conditional on the final
production section; several of the concerns (M2, M3, M4) are about
infrastructure that the placeholder results will sit on, and must be fixed
*before* those results are inserted.

### M2. Wavepacket-vs-eigenstate: the measured object is not the hadronic tensor of Eq. (6), and the text's positivity claim is contradicted by the authors' own figures

Equation (6) defines W^{mu nu}(q) with an eigenstate |P>, and the text asserts
that "for mu = nu it is real and positive semidefinite." Both statements fail
for what is actually measured:

- **The state is a wavepacket, not an eigenstate.** For an eigenstate,
  C(t) = sum_n |<n|J|P>|^2 e^{-i(E_n - E_P) t} has manifestly positive spectral
  weights and W^{00} >= 0. For a wavepacket, cross terms
  c*_{k'} c_k <k'|J|n><n|J|k> with k' != k carry arbitrary phases. The
  low-omega ridge that (per the supporting notes and `w_ridge_final.pdf`) will
  be the paper's headline is *precisely* such a packet-coherence beat structure
  (intra-band transitions at omega = E(k+q1) - E(k)), i.e., a feature that is
  strictly absent for a true momentum eigenstate, for which W^{00} at these
  sub-gap omega would essentially vanish (elastic line only). The ridge figures
  visibly go *negative* near q^0 ~ 1.3-1.6, in direct contradiction with the
  positivity sentence in Sec. III. The authors know this — their own two-band
  model reproduces the negativity — but the formal section does not: the paper
  must define the measured object honestly as a wavepacket response function
  W_Phi(q; k0, sigma_k), state its relation to the eigenstate W (diagonal terms
  + O(sigma_k) coherences, not O(sigma_k^2)), and only then discuss which
  features survive the sigma_k -> 0 limit.
- **The O(sigma_k^2) systematic is not small and is never quantified.** With
  sigma_x = 0.75 (spatial sites), sigma_k = 1/(2 sigma_x) ≈ 0.67 — this is not
  a small parameter (compare the band momentum spacing 2pi/25 ≈ 0.25 at
  N_s = 50: the packet spans several momentum modes by construction). The claim
  that fixing J^nu at the packet center is "justified by translation invariance
  up to corrections that enter W at O(sigma_k^2), which we quote as a
  systematic" is (a) parametrically misleading — the *coherence* contributions
  are O(sigma_k^0) relative to the beat structure itself — and (b) never
  actually quoted anywhere. Provide the smeared-W expansion
  W_Phi = W(q; kbar) + (sigma_k^2/2) d^2W/dP^2 + ... with numbers, or drop the
  claim.
- **Normalization is undefined.** Eigenstate W carries relativistic/volume
  normalization (2E V or delta-function); the wavepacket is unit-normalized.
  In what units is the W^{00} of Figs. 1-2 and the promised headline plot? The
  reader cannot compare it across volumes, to Ref. [Zou et al.], or to any
  continuum expression. Define the normalization convention explicitly.
- **The elastic form-factor claim in Sec. VII is wrong at these quantum
  numbers.** Sec. VII(i) promises "the meson charge form factor |F(q^2)|, the
  cleanest apples-to-apples quantity across methods" from the elastic ridge of
  the wavepacket W^{00}. But the lightest meson here is a rho^0 analog: charge
  conjugation forces <k'|J^mu_V|k> = 0 within the band (the authors' own
  supporting analysis states this, and the repository figure
  `w_elastic_vs_prediction.pdf` shows the ED matrix-element prediction for the
  elastic window consistent with ~zero while the data ridge is large and
  axial/beat-dominated). As written, Sec. VII(i) advertises an observable that
  vanishes identically in this theory. Either move to a charged-state analog, a
  different current, or rewrite this bridge around the axial channel — and in
  all cases reconcile with the C-parity statement that must appear in the
  results section to explain the ridge.

### M3. The t < 0 hermiticity completion advertised in Secs. III and IV.D is invalid for wavepackets — and the authors' own analysis code agrees

Two places in the draft commit to reconstructing t < 0 data "from hermiticity
and translation invariance" (Sec. III, "halving the required circuit depth";
Sec. IV.D, "After the pointwise subtractions ... and hermiticity completion to
t < 0, the windowed transform ..."). Both steps are unjustified for the packet
states:

- The relation <P|J(x,-t)J(0)|P> = <P|J(-x,t)J(0)|P>* requires |P> to be a
  Hamiltonian eigenstate (for the time shift) *and* translation invariant (for
  the spatial flip). A localized wavepacket is neither.
- The repository's own validation record and data-release README state
  explicitly that for wavepacket states the completion "is *not* valid at our
  precision — it scatters higher-sector content into the low-omega window," and
  the production ridge analysis (`final_ridge_overlay.py`,
  `analyze_w_meson.py` in its current form) uses one-sided transforms.

So the submitted text would describe an analysis convention that the actual
analysis (correctly) abandoned. This is not cosmetic: it changes the claimed
circuit-depth budget (no factor-of-2 saving for wavepacket runs), the shot
accounting of Sec. IV.C, and the definition of every wavepacket W plot. The
paper must (i) present the one-sided convention as the definition for
wavepacket states, with the t = 0 half-weight and 2Re assembly spelled out;
(ii) retain the completion *only* for vacuum-state products, where it is exact;
and (iii) quantify the error of the completion on wavepacket data (the notes
suggest it is fatal in the ridge window — show this, it is instructive).
Relatedly: a one-sided Gaussian-windowed transform of a truncated series is not
a spectral function; the Dawson-tail leakage and window-ordering caveats
currently buried in Eq. (10)'s parenthetical need a real paragraph with the
window-scan systematic defined quantitatively (the {0.75, 1.0, 1.5} rescaling
convention in the data release is fine — put it in the paper).

### M4. The vacuum-subtraction convention in the text is not the one implemented, and the interpretation of the subtracted (partly negative) W is never given

Sec. III prescribes subtracting (a) the raw vacuum correlator
<Omega|J J|Omega> and (b) the packet disconnected product <Phi|J|Phi><Phi|J|Phi>.
The production code subtracts *connected-minus-connected*:
[C_wp - <J>_wp<J>_wp] - [C_vac - <J>_vac<J>_vac], i.e., it additionally adds
back the staggered rho^2 term (rho = 0.0868). The internal audit quantified the
difference as inert at the quoted precision at q1 = ±2pi/5 (and 0.2% of peak at
±4pi/5) — good, but then the paper must *state the implemented scheme* and the
bound, not a different scheme. Beyond bookkeeping, two physics points are
unaddressed:

1. **What is the subtracted W, spectrally?** Subtracting the vacuum correlator
   from the packet correlator removes the vacuum-polarization background but
   also introduces packet -> vacuum-sector and packet -> M* cross terms that
   are neither pure "hadron structure" nor removed by the subtraction. The
   audit bounds these in-window at the sub-percent level; the paper should
   quote that bound and state where it degrades (near the q^0 ~ 1.6 window
   edge, per the audit).
2. **Negative regions.** After subtraction and one-sided windowing, Re W^{00}
   dips below zero (visible in `w_ridge_final.pdf` and
   `w_elastic_vs_prediction.pdf`). A definition-level discussion is required:
   which negativity is packet coherence (physical, survives window widening),
   which is window ringing (shrinks under the window scan), and which is the
   subtraction convention. A referee — and any experimental reader who thinks
   "W^{00} is a cross section" — will stumble here immediately. One clarifying
   subsection with a worked example (the two-band model is perfect for this)
   would resolve it.

### M5. Volume-transfer and "certification" claims outrun the evidence; quoted fidelities are internally inconsistent

- **Energy is a weak certificate.** All three production certificates are
  energies: vacuum energy density, <H> on the packet "inside the band," and
  small-volume-extrapolated spectra. Energy expectation values are famously
  insensitive certificates: a state can have the right <H> and badly wrong
  long-distance correlations (exactly what matters for a correlator at
  separations up to 25 sites). The variance <H^2> - <H>^2 is mentioned
  parenthetically ("where informative") but never reported — report it, for
  both vacuum and packet, at N_s = 50. State clearly that the certification
  battery provides *necessary* conditions, and temper "we suggest as a general
  standard for trusting quantum-simulation results" accordingly.
- **No uncertainties on the flagship certificate.** "<H> = -52.2254 versus the
  extrapolated exact value -52.227" — no error bar on either number, no
  description of the extrapolation (fit form? volumes? correlated?), no MPS
  truncation systematic. A 1.6e-3 absolute agreement on a ~52 number is 3e-5
  relative — fine, but meaningless without an error budget. Similarly "energy
  density constant to 1e-6" and "reproduce the exact energy density to 5e-6 at
  N_s = 50" (abstract) need stated uncertainties and definitions (per site? per
  link?).
- **Fidelity numbers disagree across the paper and the data.** The draft quotes
  F = 0.9973 (L = 3, production packet) and "fidelity > 0.997". The production
  v3 data files record F = 0.99463 (k = 0) and 0.99357 (k = 2pi/5) for the
  regularized parameter sets actually used, and 0.99882 for the *retired*
  unregularized run. Which packet is in the paper? Every quoted fidelity must
  be traceable to the parameter file used in production. (Also specify: fidelity
  against what — the band-projected ED target at the training volume — and at
  which N_s.)
- **The MPS itself needs a systematic section.** Truncation 1e-8 and
  "bond-dimension and truncation scans providing the MPS systematic" is a
  promise, not a result. For a paper whose entire dataset is MPS, the chi/trunc
  scan table is mandatory, including at the largest evolution times where
  entanglement growth is worst, and including the Trotter step scan
  (dt_target = 0.1 is asserted, never varied in the text).

### M6. Positioning and "first at 100 qubits": the qualification against arXiv:2606.17003 is present but the framing contradicts itself, and the title implies hardware that does not exist in the paper

Credit where due: the introduction *does* cite Zou et al. (arXiv:2606.17003)
and devotes a paragraph to differentiation (wavepackets vs momentum
eigenstates; 100 qubits with explicit links/PBC vs 20-22 with links integrated
out; shot-cost analysis; Clifford-point validation). Three problems remain:

1. **The title.** "…from meson wavepackets on 100 qubits" will be read as a
   hardware result. Every number in the paper is a classical MPS simulation of
   101-qubit *circuits*. The community has been burned by this ambiguity
   before; PRD referees now flag it routinely. Add the qualifier in the title
   or abstract's first sentence ("classically simulated", "circuit-level", or
   similar), and fix 100 vs 101 (title says 100; abstract says 101; N_s = 50
   gives 100 system qubits + 1 ancilla).
2. **A self-undermining hardness claim.** The introduction dismisses 20-22
   qubit emulator work and stakes the claim "at the scale where quantum
   simulation is actually interesting" — while Sec. V boasts that the folded
   MPS ordering makes "a full 101-qubit correlator grid a matter of minutes at
   truncation 1e-8." Both cannot be load-bearing. If the production physics is
   minutes of MPS, the honest framing is: *this is a protocol, economics, and
   certification paper executed at hardware-realistic width in a classically
   tractable regime*, with quantum advantage expected only at parameters/dims
   where entanglement growth defeats MPS (say which: higher d, longer t,
   weaker confinement). Say that in the introduction, not implicitly.
3. **"What has not been demonstrated…" needs a date-stamped scope.** Zou et
   al. appeared 15 June 2026 and has a GPD follow-up; the differentiating
   sentence should be precise about what remains unclaimed (hadronic-tensor
   measurements on *hardware* at any scale; wavepacket-based W at any scale
   above ~20 qubits; explicit-link/PBC Z2 response), so that the claim survives
   even if the Zou group posts again during review. Also note their Hann window
   vs your Gaussian window when comparing methodology, since you cite your
   window systematics as a differentiator.

### M7. Phase shifts: the mod-pi ambiguity, channel identification, and the "cross-volume" certificate must be confronted in the text

Eq. (12) determines delta(p) only **mod pi**; nothing in the draft says so. The
internal audit (which I largely endorse and which the final text must absorb)
found concretely:

- **n-assignment bias:** the implementation forbids n = 0, so near-threshold
  levels can never be read as repulsive-shifted, and the acceptance window
  (-pi/2, pi] has width 3pi/2 > pi, producing duplicate (delta, delta ± pi)
  emissions for single levels. The branch must be fixed by level counting from
  the free tower and a symmetric branch (-pi/2, pi/2], with the procedure
  stated in the paper.
- **Channel contamination:** the M + M' threshold (2.745 + 3.155 ≈ 5.90) lies
  *inside* the naive candidate window; the erstwhile third point
  delta(p = 2.81) = +0.67 at E2 = 6.062 sits 0.03 above the M(0) + M'(0)
  combination and has a competing assignment with delta ≈ 0. The current
  internal status ("third point quarantined as MM'") is correct — but the draft
  fragment intended for the paper still lists all three points and even calls
  them "uniformly positive and hence attractive." The published table must be
  restricted to E < M + M' = 5.900 (two points: +0.567 at p = 1.03, +0.149 at
  p = 1.50) unless a C-parity or interpolator-overlap channel diagnostic is
  added; and the M*-as-bound-state alternative (Levinson-type branch anchoring
  shifting all n by one) deserves the one-paragraph level-counting argument the
  audit sketched.
- **The "cross-volume collapse" certificate is currently rhetorical:** one
  point per volume at non-overlapping momenta is not a collapse test, and the
  excluded N_s = 6 levels look repulsive. Either add volumes/momenta (N_s = 12
  ED, or the promised production-volume levels) or describe the check honestly
  as a consistency of two isolated points with a smooth attractive delta(E).
- With two (not three) surviving points, any Wigner time-delay derivative
  2 d delta/dE quoted downstream is a two-point finite difference — label it as
  such or drop it.

### M8. Synthesis-error study: a useful pilot, but as designed it cannot support the advertised conclusion

The study (repository: `htensor/synthesis.py`, `scripts/synthesis_pilot_ns8.py`,
`synthesis_ladder_ns50.py`, figure `synthesis_pilot_ns8.pdf`) is not yet in the
draft; if the conclusions circulating internally ("physics survives at ~20
T/rotation; stochastic beats deterministic rounding by 2-6x") are to appear,
the methodology needs hardening:

1. **Proxy vs real compilation.** Angle snapping models only the *diagonal*
   (Rz-angle) component of synthesis error. True Ross-Selinger Clifford+T
   approximants err in operator norm with an axis-tilt component that snapping
   cannot represent, and with angle-dependent, effectively pseudo-random signs
   — which matters precisely for the deterministic-vs-stochastic accumulation
   comparison that is the study's headline. The true-gridsynth tier exists
   (good — this is the right control) but only at N_s = 8, three epsilon
   values, statevector, and single-qubit-unitary substitution. The paper
   version must show, at least at pilot scale, that snapped-delta and
   gridsynth-epsilon ladders agree on the *physics* error once delta and
   epsilon are put on a common footing — and must state that footing: snapping
   at grid delta gives per-gate diamond-ish error up to |theta - k delta| <=
   delta/2 in *angle*, which is not the same as gridsynth's operator-norm
   epsilon; the top axis "T gates per rotation (Ross-Selinger)" on the pilot
   figure silently conflates the two.
2. **No scaling argument to production size.** "Survives at ~20 T/rotation" is
   an N_s = 8 statement with ~a few hundred rotations. The N_s = 50 pipeline
   has ~40x more rotations and deeper evolution; coherent (deterministic)
   errors accumulate ~ N_rot * delta and stochastic ones ~ sqrt(N_rot) * delta,
   so the per-rotation tolerance — hence T-count — that keeps physics fixed
   *grows with system size and depth*. Either report the N_s = 50 ladder
   (`synthesis_ladder_ns50.py` exists — where are its results?) or present the
   pilot with an explicit N_rot-scaling extrapolation and error model. A
   per-rotation T-count quoted without the accompanying (N_rot, depth) is not a
   fault-tolerant cost statement.
3. **Statistics of the stochastic claim.** The stochastic mode is run with one
   or two seeds per delta. A "2-6x better" claim between two rounding modes
   needs seed ensembles with error bars (the stochastic curve is a random
   variable; its *mean* physics error and its variance both matter — an
   unbiased-in-expectation angle does not imply unbiased W, which is quadratic
   in the state).
4. **Missing interactions.** No shot noise, no Trotter-synthesis error
   interplay (the pilot reuses dt_target = 0.1 at snapped angles — snapping
   interacts with step size since smaller steps mean smaller angles, i.e.,
   *worse* relative snapping error; indeed the ladder script quietly moves to
   dt = 0.5 "the larger-angle, FT-friendlier Trotter regime," a choice that
   must be surfaced and justified in the text, not in a code comment).

---

## 3. Minor issues

1. **Eq. (6):** mixes a discrete sum over x with \int dt for data on a discrete
   t grid; Eq. (10) then has a stray lone "Delta t" but no Delta x. Write the
   discrete transform you evaluate, with both measures, and give W's units.
2. **Eq. (10) limit ordering:** "windows that must widen as data extend (the
   all-data limit precedes the window removal)" — the parenthetical says the
   opposite of the intended epsilon-after-L ordering. State
   lim_{window -> infinity} after lim_{T,L -> infinity} explicitly.
3. **"real to 5e-5" / "real to 0.1%":** relative to which norm (peak |W|,
   pointwise, L2)? Define once, use throughout.
4. **Shot-cost claims:** "~1e5 shots" for the grid, "six orders of magnitude
   less" than LLY, and the abstract's "nearly two orders of magnitude" for
   multiplexing — give the arithmetic in one footnote (points x shots/point,
   target precision, and the N^{-1/4} derivation for the LLY comparison), and
   state whether "before mitigation overheads" multiplies all quoted numbers.
5. **Clifford-point validation:** the stabilizer points require
   eta*eps_T in 2 pi Z etc., i.e., parameters far from the physics point.
   Say explicitly that this validates *circuit generation and symmetry
   plumbing*, not the physics values; as written a casual reader may take
   "validated at full production width" to cover the production couplings.
6. **Optical conductivity:** Re sigma(omega) = lim_{q->0} omega W^{00}/q^2 is
   evaluated at q1 = 0.25, 0.50, 0.75 with no extrapolation; the three curves
   differ by ~40% at peak (Fig. `sigma_vac_ns50.pdf`). Either extrapolate or
   present sigma at fixed q and drop the lim symbol. Also "excitonic
   insulator" language: the "exciton" is the confined meson — one clause
   noting this avoids condensed-matter readers importing the wrong picture.
7. **HVP / R-ratio rhetoric:** "measured rather than reconstructed" oversells —
   the spectral function is still window-broadened (a smearing kernel is a
   choice, not an inversion, granted — cite the smeared-spectral-function
   literature, e.g. Hansen-Lupo-Tantalo, to make this precise and defensible).
   Quote the 0.07% sum-rule closure with its definition of S(q1) in the text.
8. **Meson mass error:** "a_s M = 2.7451(1) (from exact diagonalization,
   already volume converged between N_s = 8 and 10)" — the (1) is an
   ED-convergence statement, not a statistical error; say so, and give the
   e^{-ML} estimate justifying "volume converged."
9. **Interpolator optimization:** "99.9% single-meson-band purity" from a 4x4
   GEVP at small volume — at which volume, and how does band purity transfer
   with volume? One sentence + number.
10. **Ward identity:** Sec. III advertises q_mu W^{mu nu} = 0 as "a nontrivial
    check of the entire pipeline," but no number appears in the draft; the
    internal fragment quotes 6.6% closure dominated by dt = 0.5 integration
    error. 6.6% next to "0.7-2.2% RMS" headline agreement will draw fire —
    either improve the time quadrature (finer dt or higher-order stencil in
    the check itself) or explain the budget carefully.
11. **Chiral sweep asymmetry:** the internal fragment discloses a ~13%
    reflection asymmetry from the sequential gate sweep, mitigated by odd-in-q1
    constructions with the rest packet as a null test. This *must* appear in
    the paper (it is a systematic of the state-prep circuit itself), with the
    null-test figure.
12. **Boost bookkeeping:** for the boosted packet the insertion stays at the
    packet's t = 0 center while the packet drifts (measured v_g = +0.093);
    Sec. III's "scanning J^mu(x)" discussion should state whether x is
    measured relative to the drifted center (the plan document says it should
    be) and what error the choice induces.
13. **Figure `w00_vac_ns50.pdf`:** the q1 = 3.02 peak sits visibly below the
    plotted ED marker at E(pi) = 3.078 (quoted as 3.06 vs 3.078, i.e. 0.6%);
    the "percent-level" abstract claim is fine but quote the worst case, and
    explain the window-induced downward pull.
14. **Citation hygiene:** Chen et al. is "10+1 qubits," not "ten-qubit scale";
    the 1+1d quantization condition should cite Luscher-Wolff and the
    massive-Schwinger scattering literature (hep-lat/9709158, hep-lat/9904015,
    2112.15228); if the synthesis study enters, cite Ross-Selinger and the
    pygridsynth implementation; Mitarai-Fujii and Wang et al. citations for the
    ancilla-free protocol are present — good.
15. **Couplings context:** none of the cited Z2 papers (Davoudi et al., Chai et
    al.) simulates your coupling point, and momentum-unit conventions differ
    (staggered vs physical sites — factor-of-2 trap). When Sec. VII promises
    "predictions at the couplings of existing Z2 collision experiments," add
    the translation table (the internal `couplings_translation.txt` is exactly
    this — publish it as an appendix).
16. **Reproducibility:** the data-availability paragraph (drafted internally)
    promising raw grids + scripts is strong — include it; PRD increasingly
    expects it, and in this paper it does real epistemic work given M5.
17. **Typography/consistency:** three distinct epsilons still lurk (Trotter
    eps_T, FT window, synthesis delta/eps) — the draft fixed two; the synthesis
    section will reintroduce the third, name them now. "100 qubits" (title) vs
    "101-qubit circuits" (abstract) vs "17-33 qubits" (Sec. VIII) — audit all
    counts.

---

## 4. Recommendation

**Major revision.** The underlying work — the current construction, the seam
trick, the shot-cost economics, the two-band understanding of the wavepacket
response, and the certification mindset — is substantial and, if the production
section lands with honest systematics, this will be a valuable paper that I
would expect to recommend for publication. But the present draft (i) is missing
its headline results, (ii) describes an analysis convention (hermiticity
completion, subtraction scheme) that its own production code correctly does not
use, (iii) asserts a positivity property its own figures violate, (iv) promises
an observable (elastic vector form factor) that vanishes by C-parity in this
theory, (v) rests certification on unquantified energy comparisons with
internally inconsistent fidelity numbers, and (vi) carries a title and framing
that a hardware-free, classically-tractable MPS study cannot support without
qualification. Each of these is fixable; none is optional.

### Itemized requests to the authors (condensed)

R1. Complete Sec. VI.C and all placeholders; align abstract claims with content.
R2. Reformulate Sec. III around the wavepacket response function: definition,
    normalization, relation to eigenstate W, coherence terms, negative regions,
    quantified sigma_k systematics; fix the positivity sentence.
R3. Replace the hermiticity-completion prescription with the one-sided
    convention for wavepacket states; keep completion for vacuum only; quantify
    the difference.
R4. State the implemented (connected-minus-connected) subtraction with its
    quantified in-window bound.
R5. Rewrite Sec. VII(i): no elastic vector FF for a C-odd neutral meson; recast
    via the axial channel or a charged analog.
R6. Add error budgets: MPS chi/truncation/Trotter scans, <H> uncertainties and
    extrapolation description, variance certificates, correct fidelity
    provenance (0.9946/0.9936, not 0.9973).
R7. Phase shifts: state mod-pi ambiguity and branch-fixing procedure, restrict
    to E < M + M' = 5.900 (two points) or add a channel diagnostic; present the
    volume test honestly.
R8. Title/abstract: qualify as classically simulated circuits; resolve the
    "interesting scale" vs "minutes on MPS" tension; date-stamped novelty
    statement vs arXiv:2606.17003.
R9. If the synthesis study is included: unify snapping-delta with gridsynth-eps
    on a common error metric, provide seed ensembles, and either run the
    N_s = 50 ladder or give an explicit N_rot-scaling model; disclose the
    dt = 0.5 regime change.
R10. Publish the Ward-identity closure number with its quadrature budget, the
    chiral-asymmetry systematic and null test, and the couplings-translation
    appendix.
