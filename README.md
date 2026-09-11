# hadronic_tensor

Quantum-simulation code for extracting the hadronic tensor of 1+1d Z₂ lattice
gauge theory with staggered fermions on digital quantum computers.

Three things live here.

## 1. The code base

| directory | what |
|---|---|
| `htensor/` | the physics library: lattice, currents, state preparation, Trotter, MPS backends, analysis, and the two reductions documented in `docs/METHODS.md` |
| `scripts/` | everything that produces a result or a figure; each names its inputs and outputs in its docstring |
| `tests/` | the test suite, including the cross-check that the handoff package reproduces this library |
| `docs/` | `METHODS.md`: how the boost asymmetry and the gauge post-selection are defined and extracted |

```
python -m venv .venv && .venv/bin/pip install -r requirements.txt
PYTHONPATH=. .venv/bin/python -m pytest tests -q
```

## 2. The results

`data/` holds the published arrays and the figures rendered from them; see
`data/README.md` for conventions and file groups, and `data/hw/README.md` for
the hardware runs. Per-shot device data, the reduced slices and the
primitive-level results retrieved from IBM are all there, with
`data/hw/primitive/README.md` explaining what exists for which job and why.

`data/work/` is working state — preparation caches, trajectory ensembles,
checkpoints — regenerable and not part of the release.

## 3. The next hardware campaign

`htq_hw/` is a self-contained Qiskit package for running the four-component
W^{μν} measurement on IBM hardware: circuits, embeddings, the shot plan,
submission, fetch and analysis, with cards and reference grids included and no
dependency on `htensor`. It is what gets handed to the device team.

```
PYTHONPATH=. .venv/bin/python -m htq_hw --ns 50 acceptance --level full --target <backend>
```

One command validates the environment, the target, the embedding, every card's
reference grids, all 701 circuits, the shot plan, and a sampled end-to-end
rehearsal, and writes a machine-readable record; `bundle` refuses to ship
without a passing one. See `htq_hw/README.md`.
