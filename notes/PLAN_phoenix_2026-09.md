# Phoenix campaign plan (current as of 2026-09-10)

Single source for the next hardware run of the Z₂ hadronic tensor. Supersedes the 36-minute manifest in
the 2026-08-17 IBM workshop deck.

## 1. Goal and device

- **Goal.** All four components W^{μν}(q⁰,q¹) of a meson on ~100 qubits, at a coupling point where the
  hadron is relativistic, with every calibration control on the device; plus a fully hardware-supplied
  vacuum subtraction, the quasi-PDF plane-wave extrapolation, and one new technique.
- **Device.** ibm_phoenix (IBM Nighthawk r2): 120 qubits, square lattice, 218 couplers, dissipative
  reset elements, Heron-class two-qubit error (~3e-3), 7,500+-gate circuits demonstrated. Per-shot time
  assumed Heron-like, ~250 µs; the 100k circuits/s headline does not apply at our depth. Our Open Plan
  instances do not see phoenix; access comes with the award.
- **Embedding.** Ladder: matter sites on a 50-cycle, each link qubit pendant to the right site of its
  bond, ancilla adjacent to the insertion site; 101 of 120 qubits. Trotter step 300 two-qubit gates,
  layout-preserving, identical in CZ and fractional bases. Ns = 50 only (Ns = 58 fits only as a ring
  and belongs to Heron r3).
- **Budget.** 3 h QPU ≈ 4.3e7 shots. Unit cost: one pub (one circuit × one readout basis) at 6e4 shots
  = 15 s. Mirrors never below 3e4 shots (κ-estimation floor learned on kingston).

## 2. The two coupling points

| | production | relA |
|---|---|---|
| (m₀, g², η) | (0.7, 1.1, 1.3) | (0.4, 1.4, 2.3) |
| meson mass M | 2.745 | 2.775 |
| band width W | 0.33 | 1.01 |
| E″(0)·M | 0.46 (m_eff = 2.2 M) | 0.997 (relativistic to 0.3%) |
| packet speed at k = 2π/5 | 0.51 c | 0.97 c |
| second band above ridge | 0.41–0.54 | 1.3 |
| vacuum ansatz | 2 layers, links \|+⟩, 603 gates prep | 3 layers, links \|−⟩, 811 gates prep |
| dominant component | W⁰⁰ (1.14) over W¹⁰ (0.64) | **W¹⁰ (1.38) over W⁰⁰ (0.64)** |
| hardware data | kingston W⁰⁰ (paper) | none |

The reversed component hierarchy is a campaign argument in itself: at relA the spatial-current response
is the larger one, so measuring all four components matters more there than at the published point, where
W⁰⁰ dominates. Selection criterion: E″(0) = 1/M, which needs η ≈ 2.3 (see §7, item 4). All six top
candidates passed every gate (band identified at all k, two-meson margin ≥ 0.3, confinement, M′ gap,
vacuum transfer).

## 3. Campaign package (committed 129 min, stretch 18, contingency 33)

| # | preset | card | slices | pubs/slice | pubs | QPU | purpose |
|---|---|---|---|---|---|---|---|
| 1 | relA-core (flagship) | relA k=2π/5 σ_x=0.75 | 12 (t ≤ 6) | 18 | 261 | 65.2 min | complete tensor at the relativistic point |
| 1s | relA stretch | same | t = 6.5–8 | 18 | 72 | 18.0 min | run last; drop if κ(t=6) < 0.4 |
| 2 | prod-bridge | prod k=2π/5 σ_x=0.75 | 16 (t ≤ 8) | 9 | 150 | 37.5 min | fix the published forward region (dither); validate controls where hardware data exist |
| 3 | vac-w00 ×2 | vacuum, both couplings | 12 | 2 | 50 | 12.4 min | hardware vacuum subtraction; on-device dispersion |
| 4 | qpdf-scan | 8 cards, prep only | – | 21 | 168 | 14.4 min | ⟨x⟩ extrapolation to the plane-wave limit |
| 5 | gauss-midcircuit | prod, j0×Z | 8 | ~4 | ~30 | 8 min | new technique; risk-gated |

Per-slice content of relA-core: 9 physics pubs (j0 × {Z, XYA, XYB}; j1p1 and j1p2 × {Z, XYA, XYB}),
3 j0 mirrors, 3 j1p1 mirrors, 3 dither pubs (J⁰ inserted at site 25 instead of 24), plus 9 t = 0
references and a half-step Trotter control (dt = 0.25) at t = 0.5 and 1.0. Readout bases: Z = matter
Z / links X (J⁰ probes and the 50 Gauss checks for free); XYA / XYB = alternating X/Y on matter, links Z
(the two J¹ Pauli terms on all 50 bonds). Re part only (ancilla X), as published.

Depth: relA committed 811 + 12×300 = 4.4k gates (κ ≈ 0.49 extrapolated from kingston's mirror damping);
stretch 5.6k (κ ≈ 0.43, 20% deeper than anything run). Production bridge 603 + 16×300 = 5.4k.

**Built and priced** (`python -m htq_hw --ns 50 plan --preset relA-core prod-bridge vac-w00 qpdf-scan
--budget 180 --target fake:nighthawk`): 701 pubs, 112 jobs, 3.5×10⁷ shots, committed 129.2 min, stretch
18.0 min, contingency 32.8 min at 250 µs per shot (39 h if the 4 ms rate were to apply). Shots are equal
per pub with mirrors floored at 3×10⁴; `plan` re-pins everything from the measured per-shot time once
phoenix is visible.

Decisions taken:

- **Contingency (~33 min) is held unallocated** — reserve for re-runs, drift and whatever the first
  slices reveal, to be spent after data exist, not before. Costed candidates if it survives: a full-tensor
  width card at σ_x = 1.5 (12 slices, 38 min), or extra Im-tier pubs.
- **Dither on the J¹ families is deferred**, revisited once the J⁰ dither has data. The production bridge
  runs the J⁰ dither over 16 slices and is itself the test of whether relocating a gadget changes the
  packet-region bias. Trigger to revisit: the J⁰ dither resolves (or fails to resolve) the forward region
  AND the J¹ components show an anomalous packet-region residual. Cost if adopted: ≈ 9 min of QPU plus
  its own relA ideal grid (hours of classical time at the expensive coupling, so start the grid when the
  trigger fires). Motivation: the J¹ gadget is a five-gate parity ladder against J⁰'s single controlled-Z,
  so its bias may differ in size and character, and the κ study found weight-3 and weight-1 observables
  differing by up to 25%.

Open: the Im tier (ancilla Y) is omitted for cost, since it would double every physics pub. Adding it for
a subset of slices remains a possibility.

## 4. What each piece delivers

1. **Complete tensor + Ward test.** q⁰W^{0ν} = q̂ W^{1ν} with q̂ = 2 sin(q¹/4) is an over-determination
   check on the device. On the fully routed production reference the integral continuity form closes to
   2.6% (J⁰ insertion) and 14% (J¹ insertion) at the output spacing dt = 0.5. That residual is time
   quadrature, not physics: halving the sampling density inflates it to 43% and 387%, the Simpson dt⁴ law,
   so the relA grids' dt = 0.25 should reach ~1% on the J¹ insertion. The J¹ residual is the larger of the
   two because C^{11} carries higher-frequency content.
2. **Resolution.** δq⁰ ≈ 3/t_max = 0.5 at 12 slices resolves relA's second band (1.3 above the ridge);
   16 slices (0.38) are what the production point needs for its 0.4–0.5 gap, hence the bridge depth.
3. **Boost asymmetry.** Published A = 0.85 ± 0.39 (2.2σ). With 6e4 shots on every slice, the dither
   calibration in the packet region and 12 slices: ~4–5σ expected (shot error alone would allow ~8σ).
4. **Forward region.** The paper names the relocated gadget as the designed fix; preset 2 delivers it.
5. **Vacuum on device.** Removes the "assembly is not purely hardware-supplied" caveat.
6. **Quasi-PDF.** The width scan extrapolates in σ_k² onto the connected eigenstate moment:

   | | eigenstate target | width scan σ_x = 0.75 / 1.0 / 1.5 | σ_k² → 0 intercept |
   |---|---|---|---|
   | production | 0.320 (0.25% volume-independent) | 0.389 / 0.373 / 0.346 | 0.332 |
   | relA | 0.264 | 0.350 / 0.330 / 0.299 | 0.282 |

   4% (production) and 7% (relA) agreement, which is what the σ_k² argument predicts, so the preset stands
   as designed. The published measurement 0.389 is the σ_x = 0.75 point, biased high by the packet's
   momentum spread. Both quantities are *connected* (vacuum-subtracted, as the measurement is) and use
   A(z) = ⟨O_R⟩/2 for z ≠ 0 with A(0) = ⟨n⟩; the connected moment is volume-independent while the raw one
   moves 15% between volumes. Nothing here touches main_v4.tex, which has no quasi-PDF section.
7. **Gauss mid-circuit.** Five patch checks extracted after every 2 steps on reset-and-reused ancillas:
   acceptance versus depth and a post-selected slice. Gate: Aer dry run + FakeNighthawk dynamic-circuit
   support; dropped without regret if it fails.

## 5. Calibration and analysis design

- Mirror self-mitigation per pub: κ per probe and readout basis from the same-job mirror (physics and
  mirror always in one job); β for J⁰ probes with the published +0.5 anchor (re-checked against MPS
  truth: the depolarizing anchor is worse); wing anchor per parity; χ²-inflated inverse-variance merges.
- J¹ probes have only 3–9 sites with a usable t = 0 reference, so their κ is effectively a global factor.
  Measured at Ns = 16 with ~93 Pauli trajectories per family (`scripts/kappa_ratio.py`, `data/kappa16/`):
  weight-3 probes damp faster than weight-1 in the same circuit, consistently across all three insertion
  families —

  | t | κ(J¹ probe) / κ(J⁰ probe) |
  |---|---|
  | 0.5 | 0.93–0.97 |
  | 1.0 | 0.83–0.96 |
  | 2.0 | 0.72–0.82 |

  errors 0.02–0.07, equivalent to κ_w ≈ κ_1 · exp(−0.05 (w−1) t). Using a J⁰ mirror's κ on J¹ probes
  would inflate W⁰¹/W¹¹ by ~5% at t = 0.5 and ~25% at t = 2. The manifest already avoids this: every
  mirror pub is read in all three bases, so κ is always measured on the probe type it calibrates, and
  relA-core carries per-family J¹ mirrors. Caveat: measured at Ns = 16, where the light cone wraps; the
  ratio is a local operator-weight effect and should transfer, but the first hardware slice is the check.
- Dither: same physics with the insertion moved one site; the difference map after wing anchoring is the
  packet-region bias estimate. Analysis method still to be finalized on the rehearsal data.
- Sum-rule projection (Σ_x G = 0) for J⁰-probe components only; seam bond masked; κ < 0.05 masked.
- Assembly: one-sided windowed transform (σ_t = t_max/3, σ_x = N_x/2), MPS reference windowed identically
  (scripts/hw_w_tensor.py, exact on synthetic slices).

## 6. Classical pipeline (complete for the committed run)

- Routed MPS backend (explicit Sabre routing onto the chain): preps in 9–43 s, energies converged
  (production packet ⟨H⟩ = −49.3690, relA gap 3.108 = certification).
- **Reference grids installed for every card the campaign needs.** Production j0 and j0d to t = 8;
  prod_vac j0 to t = 6; relA j0, j0d, j1p1, j1p2 to t = 3; relA_vac j0 to t = 2. The acceptance gate
  checks this card by card and the package README carries the coverage table.
- **The relA deep-time gap is closed by the wing surrogate**, not by weeks of MPS. The relA grids stop at
  t = 3 while the flagship runs to t = 6; only the *wing anchor* needs the deeper rows, and the wing
  signal is bulk staggered vacuum breathing, so an exact small-ring calculation supplies it
  (`scripts/wing_surrogate.py`, shipped as `htq_hw/refs/wing_surrogate_{prod,relA}.npz`). Validated
  against the Ns = 50 production wings to 2e-5 against a 2e-3 gate. The surrogate carries its couplings
  and is refused for a card with a different η. Without one, affected slices degrade to
  `wing_applied=False` and are recorded as such.
- Packets certified: production σ_x = 0.75 (426 CZ), 1.0 (618), 1.5 (L = 4, 1208); relA σ_x = 0.75 at
  rest and k = 2π/5, 1.0, 1.5 (provisional). Wide packets are depth-limited (L = 3 saturates at σ_x = 1.5).
- Absolute κ comes from the empirical kingston damping law (ln κ = −0.265 − 1.01e-4 × 2q-gates); the
  weight-3 versus weight-1 ratio comes from the Ns = 16 trajectory ensemble (§5). A full Ns = 50
  trajectory rehearsal is not worth it: one trajectory is a deterministic Pauli frame giving κ = ±1, and
  ~400 would be needed at hours each.

## 7. Handoff package (htq_hw/)

Self-contained Qiskit package, no htensor import: circuits, both embeddings, parametric-O3 transpile with
skeleton assertions, manifest / shots plan / submit / fetch, analysis to slice files, check and rehearse
against ideal grids, cards with their grids, the shipped references, README with knobs versus invariants
and device asks, bundle. 118 package tests + 128 repo tests green. Tagged `htq_hw-0.1.0`.

Validated: `check` at t = 2/4/6, all three insertion families × three readouts, on both topologies (worst
1.97e-4 ring, 1.56e-4 ladder; gate 5e-3); the package's grids agree with the independently written
main-repo routed grids to 1.8e-4 over all 13 times, preparation energies to 4.9e-7.

That second pipeline earned its keep once already: the package built every Trotter circuit from
module-level couplings rather than the card's, so a relA card was prepared correctly but *evolved under
the production Hamiltonian*. Nothing inside the package could see it — the production card's couplings
are the defaults — and it surfaced only because the independent relA grids disagreed by 0.30 while the
preparation energy agreed to 1.3e-6. Uncaught, the 65-minute flagship would have run the wrong physics.
Fixed with a regression test.

### 7.1 What the pre-handoff audit changed (2026-09-05 … 09-09)

Eleven defects, four able to destroy the allocation or the data silently. Each is now something a command
refuses rather than something a reader has to notice:

- **`submit` safety.** Without `--real`, a live backend name used to submit to hardware while printing
  "local testing mode"; it now refuses outright. `--real` requires `--confirm <backend>` and rejects
  stand-ins, twins, simulators, name mismatches and `--basis rzz` on a device without `rzz`.
- **Budget.** The whole campaign is priced against `usage_remaining_seconds` before job 0 (112 jobs that
  each fit can still overrun together). The guard is tri-state: an unreadable allocation stops a real
  submission instead of passing. A cancelled job stops the loop instead of letting the other 111 go out,
  and its metadata is written anyway so no spent job id is lost. The jobs run inside one `Batch`.
- **Embedding.** The ladder is searched on the *operational* graph (dead qubits and bad couplers removed)
  by beam search over cycles rather than idealized rectangles; `allow_transpiler=False` is the default, so
  a failed embedding raises with diagnostics instead of silently doubling the gate count and voiding the
  mirror argument. On FakeNighthawk it finds redundancy 7 with 19 spare qubits. The search is bounded by
  work rather than wall clock, so a busy machine and an idle one choose the same layout, and a search that
  runs out of budget raises instead of falling back to a 1.9×-deeper grid cycle.
- **Acceptance gate** (`python -m htq_hw acceptance`): environment pins, target reality, embedding
  redundancy, per-card grid coverage, all 701 pubs, the shot plan against the budget, `check`, and a
  sampled `rehearse` + `analyze` round trip, written to a machine-readable record. `bundle` refuses to
  build without a passing full-level record whose file hashes still match the tree, and ships
  `ACCEPTANCE.json` and `MANIFEST.sha256`. The bundle is verified self-contained: 52 files, hashes check,
  and its own tests pass from an unpacked copy with no repository data present.
- **The quasi-PDF half became runnable.** The gate's rehearsal found that no qpdf pub could be sampled at
  Ns = 50 at all (`KeyError: 'qpdf'` — those pubs have no insertion gadget). Beyond that, h(0) had no
  honest measurement (it was to come from a Hadamard-test reference pub, whose marginal density is an
  ancilla-average, and four of the eight width cards have no such pub), and there was no reduction from
  bits to ⟨x⟩. Each card now carries a preparation-only `qZ` pub (701 pubs total) and `analyze --qpdf`
  does the reduction in the reference convention, pinned to the ideal references to 1e-12. End to end at
  Ns = 50: 21 + 21 rehearsed pubs at 2e4 shots give ⟨x⟩ = 0.3913 ± 0.0042 against the ideal 0.3891 (0.5σ).

## 8. Paper fixes (applied and reviewed 2026-09-04; verified against paper/main_v4.tex 2026-09-11)

1. `scattering.gauge_fixed_system` overwrote the single-meson entry with higher levels; the "exact
   eigenstate ⟨x⟩ = 0.61 / 0.85" was computed on the wrong state. True band-1: 0.332 (Ns = 10),
   0.423 (Ns = 20), consistent with the measured 0.389.
2. MPS prep truncation: published packet ⟨H⟩ −49.354 is really −49.369 (gap 2.856, the Ns = 12
   certification value). Impact on W⁰⁰: 0.2% of max, W¹⁰ 0.7%, vacuum grids 3e-4, continuity unchanged:
   the tensors stand. The quoted ⟨H⟩ and the tab:certs truth column are fixed.
3. The β anchor is fine as published (checked against MPS truth).
4. W·M → 2 in the IBM one-pager is the wrong relativistic criterion: it is met at η ≈ 1.95 without
   relativistic dispersion. The curvature criterion E″(0) = 1/M needs η ≈ 2.3, which is what relA uses.

Checked in the current draft (`paper/main_v4.tex`, gitignored, not in the public repo): ⟨H⟩ is quoted as
−49.37; the wrong-eigenstate ⟨x⟩ = 0.61 does not appear anywhere, and neither does any quasi-PDF result
(the parton mentions are all citations), so item 1 has nothing to correct in the manuscript; item 4 was
the one-pager, not the paper. The Gauss-law post-selection section has been rewritten since v3: the loose
"∼30% to ∼80%" claim is gone, replaced by the patch table, whose entries (0.25 / 0.46 / 0.60 / 0.67 at
30000 / 11653 / 6415 / 3613 shots) reproduce the independent measurement in `docs/METHODS.md` exactly.

## 9. Live risks

- Access to phoenix and its measured per-shot time (re-pin the shot plan from the real target).
- Stretch slices: drop if κ(t = 6) < 0.4.
- Gauss mid-circuit: dynamic-circuit support and latency on phoenix are unknown.
- relA σ_x = 1.5 packet is provisional (needs Ns = 24 training, ~8 h/packet) — not in the committed run.
- No IBM job is ever submitted from here; `submit` refuses a live backend without `--real --confirm`.

Closed: dead couplers (the operational-graph search reports redundancy and the gate requires ≥ 2, and a
failure raises with `largest_feasible_ns`, making a smaller Ns a physics decision rather than a transpiler
one); MPS convergence of the relA grids (cap 512 versus 1024 is bit-identical across all 452 observables
at t = 0 and 2, and the truncation threshold rather than the cap sets the 6.6e-6 error).

## 10. What is left

IBM awarded **180 minutes**; the acceptance letter is filled and waiting on signatures, and phoenix is not
visible on any saved account, so the instance will arrive with the award.

**Done, 2026-09-10:** the full-level acceptance run against the FakeNighthawk stand-in, on lenore
(101 min, record in `data/hw/acceptance_lenore_fakenighthawk.json`). Every step passes; the only warning
is the offline stand-in itself, which only phoenix access can clear. `check` 1.09e-4 over 18 circuits
against a 5e-3 gate; the rehearsal produced 20 slices from 234 sampled pubs with worst |κ−1| = 0.12; the
quasi-PDF reduction recovered all six width cards' ⟨x⟩ within 1.6σ of ideal at 4000 shots. Its record is
superseded by the `--require-real` run below, so it is a rehearsal of the gate, not the shipping artifact.

It also earned its keep three times over, each on the path between "the device finished" and "we have
numbers", and none of it reachable by the unit suite: no qpdf pub could be sampled at Ns = 50 at all; the
gate crashed outside a git checkout (an unpacked bundle, i.e. the recipient's situation); analysis
demanded J¹ grids from vacuum cards that can never have them; and a card belonging to two presets lost
half its pubs to a single-preset name filter. All fixed, all now regression-tested.

**Blocked on access, in order, ~1 day once the instance exists:**

1. letter signed, instance saved as a named account, `service.backend("ibm_phoenix")` resolves;
2. snapshot the real device (coupling map, operational qubits and edges, calibration errors, whether
   `rzz` is exposed, `estimate_duration` on the deepest pub);
3. re-run the embedding on the real graph and confirm redundancy ≥ 2 at Ns = 50;
4. re-pin the shot plan from the measured per-shot time — the 250 µs assumption is the one number that
   cannot be checked offline; the committed/stretch/contingency split re-derives automatically;
5. `acceptance --level full --require-real`, then `bundle`;
6. pilot: submit the t = 0 reference group and analyze it end to end — a proof that submit, fetch and
   analyze work against the real instance, not a physics pilot;
7. run the campaign: prod-bridge, relA-core, vacuum, qpdf, stretch slices last and droppable, ~33 min
   contingency held unallocated.

Lüscher and width-tensor items remain options, independent of access.
