"""Qubit indexing and geometry for the 1+1d Z2 gauge theory lattice."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Z2Lattice:
    """N_s staggered sites with interleaved gauge links.

    Matter site n lives on qubit 2n, link (n, n+1) on qubit 2n+1.
    With PBC the seam link (N_s-1, 0) is qubit 2*N_s - 1 and N_s must be
    even for a consistent staggered assignment.
    """

    ns: int
    pbc: bool = True

    def __post_init__(self):
        if self.ns < 2:
            raise ValueError("need at least 2 staggered sites")
        if self.pbc and self.ns % 2:
            raise ValueError("PBC requires an even number of staggered sites")

    @property
    def n_qubits(self) -> int:
        return 2 * self.ns if self.pbc else 2 * self.ns - 1

    @property
    def n_links(self) -> int:
        return self.ns if self.pbc else self.ns - 1

    @property
    def nx(self) -> int:
        """Number of spatial sites (2 staggered sites per spatial site)."""
        return self.ns // 2

    def site_qubit(self, n: int) -> int:
        return 2 * (n % self.ns)

    def link_qubit(self, n: int) -> int:
        """Qubit of the link on bond (n, n+1); n = ns-1 is the PBC seam."""
        n = n % self.ns
        if not self.pbc and n == self.ns - 1:
            raise ValueError("bond (ns-1, 0) does not exist with OBC")
        return 2 * n + 1

    @property
    def bonds(self) -> list[int]:
        """Bond labels n for hops (n, n+1); includes the seam under PBC."""
        return list(range(self.n_links))

    @property
    def matter_qubits(self) -> list[int]:
        return [2 * n for n in range(self.ns)]

    @property
    def link_qubits(self) -> list[int]:
        return [2 * n + 1 for n in range(self.n_links)]

    def seam_string_qubits(self) -> list[int]:
        """Matter qubits carrying the JW string of the PBC seam hop
        (interior sites 1 .. ns-2)."""
        return [2 * n for n in range(1, self.ns - 1)]

    def is_seam(self, bond: int) -> bool:
        return self.pbc and (bond % self.ns) == self.ns - 1
