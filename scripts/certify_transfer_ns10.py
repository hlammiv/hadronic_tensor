"""Volume-transfer certification: ns=8-trained wavepacket block evaluated
against the exact band-projected wavepacket at ns=10.

  PYTHONPATH=. .venv/bin/python scripts/certify_transfer_ns10.py [k0]
"""

import sys
import time

import numpy as np
from qiskit.quantum_info import Statevector

from htensor import Z2Lattice, stateprep, spectroscopy, wavepacket, block_engine

M0, G2, ETA = 0.7, 1.1, 1.3
K0 = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
t0 = time.time()

lat6 = Z2Lattice(6, pbc=True)
TH = stateprep.optimize_vacuum(lat6, M0, G2, ETA, n_layers=2, restarts=2)["thetas"]

z = np.load(f"data/wp_params_k{K0:.2f}_L3.npz", allow_pickle=True)
params = wavepacket.params_from_vector(z["vec"], list(z["offsets"]), int(z["L"]))
print(f"[{time.time()-t0:.0f}s] ns=8 training fidelity was {float(z['F']):.4f}",
      flush=True)

lat10 = Z2Lattice(10, pbc=True)
band10 = spectroscopy.meson_band(lat10, M0, G2, ETA)
print(f"[{time.time()-t0:.0f}s] ns=10 band: E(k) = "
      f"{np.round(band10['energy'], 4)} at k = {np.round(band10['k'], 3)}",
      flush=True)
mix10 = spectroscopy.optimize_interpolator(lat10, band10, k0=K0, sigma_x=1.0)
target10, _ = spectroscopy.meson_wavepacket(lat10, band10, k0=K0, sigma_x=1.0,
                                            mix=mix10["mix"])

# ns=8-trained block, translated to the ns=10 center (spatial x0=2 -> site 4;
# use center 4 so offsets land identically)
prep = stateprep.vacuum_ansatz(lat10, TH)
prep.compose(wavepacket.block_circuit(lat10, 4, params), inplace=True)
psi = np.asarray(Statevector.from_instruction(prep))
# target centered at x0=2 to match
target10c, _ = spectroscopy.meson_wavepacket(lat10, band10, k0=K0, sigma_x=1.0,
                                             x0=2, mix=mix10["mix"])
F = abs(np.vdot(target10c, psi)) ** 2
print(f"[{time.time()-t0:.0f}s] TRANSFER ns=8 -> ns=10 fidelity: {F:.4f}",
      flush=True)
