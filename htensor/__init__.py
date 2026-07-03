"""htensor: hadronic tensor of 1+1d Z2 lattice gauge theory on quantum computers.

Conventions (fixed package-wide; see PLAN.md sec. 2):
  - N_s staggered sites, 0-based index n. Even n = positron components,
    odd n = electron components. N_s must be even under PBC.
  - Qubit layout: matter site n -> qubit 2n; link (n, n+1) -> qubit 2n+1.
    PBC seam link (N_s-1, 0) -> qubit 2*N_s - 1. OBC: 2*N_s - 1 qubits.
  - Occupation n_f = (1 - Z)/2  (|1> = occupied).
  - H = (g2/2) sum_links sigma^x - (m0/2) sum_n (-1)^n Z_n
        + (eta/4) sum_bonds (X_n X_{n+1} + Y_n Y_{n+1}) sigma^z_{n,n+1},
    with the PBC seam hop carrying the Jordan-Wigner string over interior
    matter qubits (equal to -P_f (XX+YY) sigma^z on the seam pair).
  - Gauss law: G_n = (-1)^n Z_n sigma^x_{n-1,n} sigma^x_{n,n+1} = +1 on
    physical states.
"""

from .lattice import Z2Lattice
from . import hamiltonian, currents, exact, spectroscopy, stateprep
