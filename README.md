# hadronic_tensor

Quantum-simulation code for extracting the hadronic tensor of 1+1d Z₂ lattice
gauge theory with staggered fermions on digital quantum computers.

The `htensor` package builds Jordan–Wigner-mapped Hamiltonians and circuits for
lattices of N_s staggered sites (2·N_s qubits with periodic boundary
conditions), prepares localized meson wavepackets, Trotter-evolves them, and
measures two-current correlators ⟨P|J^μ(x,t)J^ν(0,0)|P⟩ whose windowed Fourier
transform yields W^{μν}(q). It targets Qiskit ≥ 2.x with

- exact-diagonalization validation at small volume,
- Qiskit Aer `statevector`, `matrix_product_state`, and `stabilizer`
  (Clifford-point) simulation at up to 100+ qubits,
- IBM Quantum hardware via Qiskit Runtime V2 primitives.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest tests/
```

Research code under active development.
