"""
Incidence Matrix Builder for Power System Networks.

Constructs the network incidence matrix B from a ``SystemGraph``, encoding
the topology of buses and branches.  The incidence matrix is the foundation
of the Dirac-structure formulation: KCL ↔ B·i = 0, KVL ↔ v = Bᵀ·e.

This module does **not** modify any existing code — it reads the
``SystemGraph`` topology (buses, lines, generators) non-destructively.

Mathematical Background
-----------------------
For a network with n buses and m branches, the incidence matrix
B ∈ ℝⁿˣᵐ is defined as:

    B[i, k] = +1  if branch k leaves bus i  (bus i is the "from" end)
    B[i, k] = −1  if branch k enters bus i  (bus i is the "to" end)
    B[i, k] =  0  otherwise

KCL at every bus:  B · i_branch = i_injection
KVL on branches:   v_branch = Bᵀ · v_bus
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.system_graph import SystemGraph


# ---------------------------------------------------------------------------
# Helper data classes
# ---------------------------------------------------------------------------

@dataclass
class BranchInfo:
    """Metadata about a single branch in the incidence matrix."""
    idx: int                # column index in B
    label: str              # human-readable name (e.g. "Line_0")
    from_bus: int           # bus index (original system index)
    to_bus: int             # bus index
    r: float = 0.0         # series resistance [pu]
    x: float = 0.01        # series reactance  [pu]
    b: float = 0.0         # total shunt susceptance [pu]
    tap: float = 1.0       # off-nominal turns ratio
    phi: float = 0.0       # phase shift [rad]
    is_transformer: bool = False


@dataclass
class IncidenceResult:
    """Complete result from the incidence builder.

    Attributes
    ----------
    B : ndarray (n_bus, n_branch)
        Oriented incidence matrix.
    bus_indices : list[int]
        Ordered list of original bus indices (row order of B).
    bus_map : dict[int, int]
        Maps original bus index → row index in B.
    branches : list[BranchInfo]
        Branch metadata (column order of B).
    generator_buses : dict[str, int]
        Maps generator component name → bus index.
    slack_buses : list[int]
        Original bus indices of slack buses.
    load_buses : dict[int, complex]
        Maps bus index → complex load S = P + jQ.
    """
    B: np.ndarray
    bus_indices: List[int]
    bus_map: Dict[int, int]
    branches: List[BranchInfo]
    generator_buses: Dict[str, int]
    slack_buses: List[int]
    load_buses: Dict[int, complex]


# ---------------------------------------------------------------------------
# IncidenceBuilder
# ---------------------------------------------------------------------------

class IncidenceBuilder:
    """Build the network incidence matrix B from a SystemGraph.

    Usage::

        from src.system_graph import build_system_graph
        from src.dirac.incidence import IncidenceBuilder

        graph = build_system_graph("cases/SMIB/system.json")
        result = IncidenceBuilder(graph).build()

        print(result.B)            # incidence matrix
        print(result.branches)     # branch metadata
    """

    def __init__(self, graph: SystemGraph):
        self.graph = graph

    def build(self) -> IncidenceResult:
        """Construct the incidence matrix and supporting data.

        Returns
        -------
        IncidenceResult
            Contains B matrix, bus/branch ordering, generator/load info.
        """
        g = self.graph

        # --- Ordered bus list ---
        bus_indices = sorted(g.buses.keys())
        bus_map = {bus_id: i for i, bus_id in enumerate(bus_indices)}
        n_bus = len(bus_indices)

        # --- Branches from Line section ---
        branches: List[BranchInfo] = []
        for line in g.lines:
            branches.append(BranchInfo(
                idx=len(branches),
                label=str(line.idx),
                from_bus=line.bus1,
                to_bus=line.bus2,
                r=line.r,
                x=line.x,
                b=line.b,
                tap=line.tap,
                phi=line.phi,
                is_transformer=bool(line.trans),
            ))

        n_branch = len(branches)

        # --- Build B matrix ---
        B = np.zeros((n_bus, n_branch))
        for br in branches:
            i_from = bus_map.get(br.from_bus)
            i_to = bus_map.get(br.to_bus)
            if i_from is not None:
                B[i_from, br.idx] = +1.0
            if i_to is not None:
                B[i_to, br.idx] = -1.0

        # --- Generator-bus mapping ---
        generator_buses: Dict[str, int] = {}
        for comp_name, comp in g.components.items():
            if comp.component_role == 'generator':
                bus_id = comp.params.get('bus')
                if bus_id is not None:
                    generator_buses[comp_name] = int(bus_id)

        # --- Slack buses ---
        slack_buses = [sb.bus for sb in g.slack_buses]

        # --- PQ loads ---
        load_buses: Dict[int, complex] = {}
        for load in g.pq_loads:
            s = complex(load.p0, load.q0)
            if load.bus in load_buses:
                load_buses[load.bus] += s
            else:
                load_buses[load.bus] = s

        return IncidenceResult(
            B=B,
            bus_indices=bus_indices,
            bus_map=bus_map,
            branches=branches,
            generator_buses=generator_buses,
            slack_buses=slack_buses,
            load_buses=load_buses,
        )

    def build_admittance_from_branches(
        self, result: IncidenceResult
    ) -> np.ndarray:
        """Build the branch admittance matrix Y_branch (diagonal) from branch data.

        Returns
        -------
        Y_branch : ndarray (n_branch, n_branch), complex
            Diagonal matrix with y_k = 1 / (r_k + j·x_k) for each branch.

        Note: tap ratio and shunt susceptance are not included in this
        simple diagonal form. For a full Y-bus, use ``src/ybus.py``.
        """
        n = len(result.branches)
        Y = np.zeros((n, n), dtype=complex)
        for br in result.branches:
            z = complex(br.r, br.x)
            if abs(z) > 1e-12:
                Y[br.idx, br.idx] = 1.0 / z
            else:
                Y[br.idx, br.idx] = complex(1e6, 0)  # near-zero impedance
        return Y

    def verify_kcl(
        self,
        result: IncidenceResult,
        i_branch: np.ndarray,
        i_injection: np.ndarray,
        tol: float = 1e-6,
    ) -> Tuple[bool, np.ndarray]:
        """Verify KCL: B · i_branch ≈ i_injection at every bus.

        Parameters
        ----------
        i_branch : ndarray (n_branch,), complex
            Branch current phasors (from → to convention).
        i_injection : ndarray (n_bus,), complex
            Net current injection at each bus (generation − load).
        tol : float
            Tolerance for the mismatch norm.

        Returns
        -------
        ok : bool
            True if KCL is satisfied within tolerance.
        mismatch : ndarray (n_bus,), complex
            KCL residual at each bus: B·i_branch − i_injection.
        """
        mismatch = result.B @ i_branch - i_injection
        return bool(np.max(np.abs(mismatch)) < tol), mismatch

    def verify_kvl(
        self,
        result: IncidenceResult,
        v_bus: np.ndarray,
        v_branch: np.ndarray,
        tol: float = 1e-6,
    ) -> Tuple[bool, np.ndarray]:
        """Verify KVL: v_branch ≈ Bᵀ · v_bus.

        Parameters
        ----------
        v_bus : ndarray (n_bus,), complex
            Bus voltage phasors.
        v_branch : ndarray (n_branch,), complex
            Branch voltage drops (from terminal − to terminal).
        tol : float
            Tolerance.

        Returns
        -------
        ok : bool
        mismatch : ndarray (n_branch,), complex
        """
        expected = result.B.T @ v_bus
        mismatch = v_branch - expected
        return bool(np.max(np.abs(mismatch)) < tol), mismatch
