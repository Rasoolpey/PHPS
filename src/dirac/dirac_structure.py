"""
Dirac Structure for Port-Hamiltonian Power System Networks.

The Dirac structure 𝒟 formalises the power-conserving interconnection
of a network — Kirchhoff's laws become geometric constraints on the
joint (effort, flow) space rather than algebraic equations solved
iteratively.

This module builds the composed Dirac structure from:
  1. The network incidence matrix B (from ``IncidenceBuilder``)
  2. Per-component port information (from ``PowerComponent.port_schema``)

It provides:
  - Construction of the Dirac subspace matrices (F, E)
  - Power conservation verification
  - Casimir function extraction (conserved quantities)
  - Constraint subspace analysis

Mathematical Background
-----------------------
The Kirchhoff Dirac structure for a network with incidence matrix B is:

    𝒟_KCL/KVL = { (f, e) | F·f + E·e = 0 }

where F and E encode KCL and KVL respectively.  For the standard
telltale representation:

    KCL (flow constraint):    B · i_branch = i_inj
    KVL (effort constraint):  v_branch = Bᵀ · v_bus

The composed Dirac structure of the full interconnected PHS is:

    𝒟_total = compose(𝒟_component_1, ..., 𝒟_component_n, 𝒟_network)

References
----------
- van der Schaft & Cervera (2007), "Interconnection of Port-Hamiltonian
  Systems and Composition of Dirac Structures", Automatica.
- van der Schaft (2024), "Port-Hamiltonian Nonlinear Systems",
  arXiv:2412.19673.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.linalg import matrix_rank, svd

from src.system_graph import SystemGraph
from src.dirac.incidence import IncidenceBuilder, IncidenceResult


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class DiracSubspace:
    """Kernel representation of a Dirac structure.

    The Dirac structure is the set:
        𝒟 = { (f, e) ∈ ℱ × 𝒢 | F·f + E·e = 0 }

    Attributes
    ----------
    F : ndarray (k, dim_f)
        Flow coefficient matrix.
    E : ndarray (k, dim_e)
        Effort coefficient matrix.
    dim_f : int
        Dimension of the flow space.
    dim_e : int
        Dimension of the effort space.
    labels_f : list[str]
        Labels for flow variables.
    labels_e : list[str]
        Labels for effort variables.
    """
    F: np.ndarray
    E: np.ndarray
    dim_f: int
    dim_e: int
    labels_f: List[str]
    labels_e: List[str]


# ---------------------------------------------------------------------------
# DiracStructure
# ---------------------------------------------------------------------------

class DiracStructure:
    """Dirac structure for a power system network.

    Reads the ``SystemGraph`` topology and constructs the power-conserving
    interconnection structure encoding KCL and KVL as geometric constraints.

    Usage::

        from src.system_graph import build_system_graph
        from src.dirac import DiracStructure

        graph = build_system_graph("cases/SMIB/system.json")
        D = DiracStructure(graph)

        # Inspect the Dirac subspace
        print(D.subspace.F)
        print(D.subspace.E)

        # Verify power conservation
        ok, residual = D.verify_power_conservation(v_bus, i_branch, i_inj)

        # Extract Casimir functions
        casimirs = D.casimir_functions()
    """

    def __init__(self, graph: SystemGraph,
                 incidence_result: Optional[IncidenceResult] = None):
        """
        Parameters
        ----------
        graph : SystemGraph
            The system topology and component data.
        incidence_result : IncidenceResult, optional
            Pre-computed incidence data.  If None, it is built automatically.
        """
        self.graph = graph

        if incidence_result is not None:
            self.incidence = incidence_result
        else:
            self.incidence = IncidenceBuilder(graph).build()

        self.subspace = self._build_kirchhoff_dirac()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _build_kirchhoff_dirac(self) -> DiracSubspace:
        """Build the Kirchhoff Dirac structure from the incidence matrix.

        The constraint system is:

            [ B   0 ] [ i_branch ]   [ I  ] [ i_inj  ]
            [       ] [          ] = [    ] [         ]
            [ 0  -I ] [ v_branch ]   [ Bᵀ ] [ v_bus   ]

        Rearranged into F·f + E·e = 0 form:

            F · [i_branch; i_inj] + E · [v_branch; v_bus] = 0

        where:
            F = [ B, -I ]       (KCL: B·i_branch = i_inj)
            E = [ -I, Bᵀ ]     (KVL: v_branch = Bᵀ·v_bus)
        """
        B = self.incidence.B
        n_bus, n_branch = B.shape

        # Flow variables: [i_branch (n_branch), i_inj (n_bus)]
        # Effort variables: [v_branch (n_branch), v_bus (n_bus)]
        dim_f = n_branch + n_bus
        dim_e = n_branch + n_bus

        # KCL constraint rows: B · i_branch − I · i_inj = 0
        # → F_kcl = [B, −I_n],  E_kcl = [0, 0]
        F_kcl = np.hstack([B, -np.eye(n_bus)])
        E_kcl = np.zeros((n_bus, dim_e))

        # KVL constraint rows: −I · v_branch + Bᵀ · v_bus = 0
        # → F_kvl = [0, 0],  E_kvl = [−I_m, Bᵀ]
        F_kvl = np.zeros((n_branch, dim_f))
        E_kvl = np.hstack([-np.eye(n_branch), B.T])

        # Combined: [F_kcl; F_kvl] · f + [E_kcl; E_kvl] · e = 0
        F = np.vstack([F_kcl, F_kvl])
        E = np.vstack([E_kcl, E_kvl])

        # Labels
        bus_ids = self.incidence.bus_indices
        labels_f = (
            [f"i_br_{br.label}" for br in self.incidence.branches]
            + [f"i_inj_bus{bid}" for bid in bus_ids]
        )
        labels_e = (
            [f"v_br_{br.label}" for br in self.incidence.branches]
            + [f"v_bus{bid}" for bid in bus_ids]
        )

        return DiracSubspace(
            F=F, E=E,
            dim_f=dim_f, dim_e=dim_e,
            labels_f=labels_f, labels_e=labels_e,
        )

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------

    def verify_power_conservation(
        self,
        v_bus: np.ndarray,
        i_branch: np.ndarray,
        i_inj: np.ndarray,
        tol: float = 1e-6,
    ) -> Tuple[bool, float]:
        """Verify that the Dirac structure conserves power.

        Power conservation requires:
            ⟨e, f⟩ = Σ v_bus · i_inj = Σ v_branch · i_branch

        i.e. the total power entering through bus injections equals
        the total power flowing through branches.

        Parameters
        ----------
        v_bus : ndarray (n_bus,), complex
            Bus voltage phasors.
        i_branch : ndarray (n_branch,), complex
            Branch currents (from → to).
        i_inj : ndarray (n_bus,), complex
            Net current injection at each bus.
        tol : float
            Tolerance for power balance check.

        Returns
        -------
        ok : bool
            True if power is conserved within tolerance.
        residual : float
            Absolute power mismatch.
        """
        # Power at bus injections: P_inj = Re(v_bus · conj(i_inj))
        p_inj = np.real(np.sum(v_bus * np.conj(i_inj)))

        # Power through branches via KVL: v_branch = Bᵀ · v_bus
        v_branch = self.incidence.B.T @ v_bus
        p_branch = np.real(np.sum(v_branch * np.conj(i_branch)))

        residual = abs(p_inj - p_branch)
        return residual < tol, residual

    def verify_dirac_membership(
        self,
        f: np.ndarray,
        e: np.ndarray,
        tol: float = 1e-6,
    ) -> Tuple[bool, np.ndarray]:
        """Check if a (flow, effort) pair belongs to the Dirac structure.

        Tests whether F·f + E·e ≈ 0.

        Parameters
        ----------
        f : ndarray (dim_f,)
            Flow vector: [i_branch; i_inj].
        e : ndarray (dim_e,)
            Effort vector: [v_branch; v_bus].
        tol : float

        Returns
        -------
        ok : bool
        residual : ndarray
            The constraint violation F·f + E·e.
        """
        residual = self.subspace.F @ f + self.subspace.E @ e
        return bool(np.max(np.abs(residual)) < tol), residual

    # ------------------------------------------------------------------
    # Structural analysis
    # ------------------------------------------------------------------

    def constraint_rank(self) -> int:
        """Rank of the combined constraint matrix [F | E]."""
        M = np.hstack([self.subspace.F, self.subspace.E])
        return int(matrix_rank(M))

    def casimir_functions(self) -> np.ndarray:
        """Extract Casimir functions from the Dirac structure.

        Casimirs are conserved quantities C(x) such that {H, C} = 0 for
        any Hamiltonian H.  They live in the kernel of the Dirac structure's
        effort constraint: ker(Eᵀ) ∩ ker(Fᵀ).

        For the Kirchhoff Dirac structure, Casimirs correspond to
        conserved current-sum relations arising from the network topology
        (e.g. total current into an isolated sub-network).

        Returns
        -------
        casimirs : ndarray (n_casimir, dim_f + dim_e)
            Each row is a Casimir direction in the joint (f, e) space.
            Empty array if no Casimirs exist.
        """
        # The Casimir directions are the null space of [F; E]ᵀ,
        # i.e. vectors c such that Fᵀ c = 0 AND Eᵀ c = 0.
        # Equivalently, ker([F; E]) in the joint space.
        M = np.vstack([
            np.hstack([self.subspace.F, self.subspace.E]),
        ])
        # SVD to find null space
        U, s, Vt = svd(M, full_matrices=True)
        # Null space: rows of Vt corresponding to near-zero singular values
        rank = np.sum(s > 1e-10)
        null_space = Vt[rank:, :]
        return null_space

    def n_buses(self) -> int:
        return len(self.incidence.bus_indices)

    def n_branches(self) -> int:
        return len(self.incidence.branches)

    def summary(self) -> str:
        """Human-readable summary of the Dirac structure."""
        n_bus = self.n_buses()
        n_br = self.n_branches()
        rank = self.constraint_rank()
        n_casimir = self.casimir_functions().shape[0]

        lines = [
            "Dirac Structure Summary",
            "=" * 40,
            f"Buses:            {n_bus}",
            f"Branches:         {n_br}",
            f"Flow dimension:   {self.subspace.dim_f}",
            f"Effort dimension: {self.subspace.dim_e}",
            f"Constraint rank:  {rank}",
            f"Casimir functions: {n_casimir}",
            "",
            "Generator buses:",
        ]
        for name, bus in self.incidence.generator_buses.items():
            lines.append(f"  {name} → Bus {bus}")

        lines.append("")
        lines.append("Slack buses:")
        for bus in self.incidence.slack_buses:
            lines.append(f"  Bus {bus}")

        lines.append("")
        lines.append("Branches:")
        for br in self.incidence.branches:
            xfmr = " (transformer)" if br.is_transformer else ""
            lines.append(
                f"  {br.label}: Bus {br.from_bus} → Bus {br.to_bus}"
                f"  (r={br.r:.4f}, x={br.x:.4f}){xfmr}"
            )

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Branch admittance helpers
    # ------------------------------------------------------------------

    def branch_admittance_matrix(self) -> np.ndarray:
        """Diagonal branch admittance matrix Y_b (complex).

        Y_b[k, k] = 1 / (r_k + j·x_k) for branch k.
        """
        return IncidenceBuilder(self.graph).build_admittance_from_branches(
            self.incidence
        )

    def y_bus_from_incidence(self) -> np.ndarray:
        """Reconstruct Y-bus from incidence matrix and branch admittances.

        Y_bus = B · Y_branch · Bᵀ

        This is the standard construction and should match the Y-bus
        built by ``src/ybus.py`` (ignoring shunts and taps).
        """
        Y_branch = self.branch_admittance_matrix()
        B = self.incidence.B
        return B @ Y_branch @ B.T
