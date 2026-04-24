"""
Hamiltonian Assembler for Port-Hamiltonian Power Systems.

Computes the **total system Hamiltonian** H_total(x) = Σ Hᵢ(xᵢ) by
querying each component's ``hamiltonian(x_slice)`` method.  This is a
pure analysis tool — it reads simulation results (CSV or state vectors)
and evaluates the energy function without modifying any component code.

For components that do not override ``hamiltonian()`` (controllers,
exciters, governors), the contribution is zero.  The Hamiltonian is
only meaningful for energy-storing elements (generators, lines, etc.).

Usage::

    from src.system_graph import build_system_graph
    from src.dirac.hamiltonian import HamiltonianAssembler

    graph = build_system_graph("cases/SMIB/system.json")
    asm   = HamiltonianAssembler(graph)

    # From a full state vector x (same layout as the compiled kernel)
    H_total = asm.total_hamiltonian(x, state_offsets)

    # From a CSV results file
    H_timeseries = asm.from_csv("outputs/SMIB/results.csv")
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from src.system_graph import SystemGraph


# ---------------------------------------------------------------------------
# HamiltonianAssembler
# ---------------------------------------------------------------------------

class HamiltonianAssembler:
    """Evaluate the total system Hamiltonian from component contributions.

    Parameters
    ----------
    graph : SystemGraph
        The system graph with instantiated components.
    """

    def __init__(self, graph: SystemGraph):
        self.graph = graph
        self._ph_components = self._find_ph_components()

    def _find_ph_components(self) -> List[Tuple[str, object]]:
        """Identify components that contribute a non-trivial Hamiltonian.

        A component contributes if it overrides ``hamiltonian()`` from
        the base ``PowerComponent`` class (i.e. returns non-zero energy).
        """
        ph_comps = []
        for name, comp in self.graph.components.items():
            # Check if the component has overridden hamiltonian()
            # by testing with a zero state vector
            try:
                n_states = len(comp.state_schema)
                if n_states > 0:
                    # Test with a non-trivial state to detect override
                    test_x = np.ones(n_states) * 0.01
                    h_val = comp.hamiltonian(test_x)
                    if abs(h_val) > 1e-15:
                        ph_comps.append((name, comp))
            except (NotImplementedError, TypeError, AttributeError):
                pass
        return ph_comps

    @property
    def ph_component_names(self) -> List[str]:
        """Names of components that contribute to the total Hamiltonian."""
        return [name for name, _ in self._ph_components]

    def total_hamiltonian(
        self,
        x: np.ndarray,
        state_offsets: Dict[str, int],
    ) -> float:
        """Evaluate H_total = Σ Hᵢ(xᵢ) at full state vector x.

        Parameters
        ----------
        x : ndarray
            Full system state vector (same layout as compiled kernel).
        state_offsets : dict[str, int]
            Maps component name → starting index in the state vector x.

        Returns
        -------
        H_total : float
            Total system Hamiltonian.
        """
        H_total = 0.0
        for name, comp in self._ph_components:
            offset = state_offsets.get(name)
            if offset is None:
                continue
            n_states = len(comp.state_schema)
            x_slice = x[offset:offset + n_states]
            H_total += comp.hamiltonian(x_slice)
        return H_total

    def component_hamiltonians(
        self,
        x: np.ndarray,
        state_offsets: Dict[str, int],
    ) -> Dict[str, float]:
        """Evaluate each component's Hamiltonian separately.

        Returns
        -------
        dict[str, float]
            Maps component name → Hᵢ(xᵢ).
        """
        result = {}
        for name, comp in self._ph_components:
            offset = state_offsets.get(name)
            if offset is None:
                continue
            n_states = len(comp.state_schema)
            x_slice = x[offset:offset + n_states]
            result[name] = comp.hamiltonian(x_slice)
        return result

    def total_grad_hamiltonian(
        self,
        x: np.ndarray,
        state_offsets: Dict[str, int],
    ) -> np.ndarray:
        """Evaluate ∂H_total/∂x at full state vector x.

        Returns
        -------
        grad : ndarray (len(x),)
            Gradient of the total Hamiltonian w.r.t. the full state vector.
            Non-PH components contribute zero.
        """
        grad = np.zeros_like(x)
        for name, comp in self._ph_components:
            offset = state_offsets.get(name)
            if offset is None:
                continue
            n_states = len(comp.state_schema)
            x_slice = x[offset:offset + n_states]
            grad[offset:offset + n_states] = comp.grad_hamiltonian(x_slice)
        return grad

    def from_csv(
        self,
        csv_path: str,
        state_offsets: Optional[Dict[str, int]] = None,
        state_columns: Optional[Dict[str, List[str]]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute H_total(t) from a simulation CSV file.

        Parameters
        ----------
        csv_path : str
            Path to CSV with columns: time, state1, state2, ...
        state_offsets : dict[str, int], optional
            Maps component name → starting column index (0-based, after time).
            Auto-detected from column headers if not provided.
        state_columns : dict[str, list[str]], optional
            Maps component name → list of column names for its states.
            Alternative to state_offsets for header-based CSV lookup.

        Returns
        -------
        t : ndarray (N,)
            Time vector.
        H : ndarray (N,)
            Total Hamiltonian at each time step.
        """
        import csv

        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            headers = next(reader)
            data = np.array([[float(v) for v in row] for row in reader])

        t = data[:, 0]
        N = len(t)
        H = np.zeros(N)

        if state_columns is not None:
            # Header-based lookup
            col_map = {h: i for i, h in enumerate(headers)}
            for step in range(N):
                for name, comp in self._ph_components:
                    cols = state_columns.get(name)
                    if cols is None:
                        continue
                    x_slice = np.array([data[step, col_map[c]] for c in cols
                                        if c in col_map])
                    if len(x_slice) == len(comp.state_schema):
                        H[step] += comp.hamiltonian(x_slice)
        elif state_offsets is not None:
            # Offset-based lookup (column indices, 0-based after time)
            for step in range(N):
                for name, comp in self._ph_components:
                    offset = state_offsets.get(name)
                    if offset is None:
                        continue
                    n_states = len(comp.state_schema)
                    x_slice = data[step, 1 + offset:1 + offset + n_states]
                    if len(x_slice) == n_states:
                        H[step] += comp.hamiltonian(x_slice)
        else:
            # Try auto-detection from column headers
            # Look for patterns like "GENROU_1.delta", "GENROU_1.omega", etc.
            col_map = {h: i for i, h in enumerate(headers)}
            for step in range(N):
                for name, comp in self._ph_components:
                    cols = [f"{name}.{s}" for s in comp.state_schema]
                    indices = [col_map.get(c) for c in cols]
                    if all(i is not None for i in indices):
                        x_slice = np.array([data[step, i] for i in indices])
                        H[step] += comp.hamiltonian(x_slice)

        return t, H

    def summary(self) -> str:
        """Human-readable summary of PH components and their Hamiltonians."""
        lines = [
            "Hamiltonian Assembler Summary",
            "=" * 40,
            f"Total components:     {len(self.graph.components)}",
            f"PH components:        {len(self._ph_components)}",
            "",
            "PH Component Details:",
        ]
        for name, comp in self._ph_components:
            n_states = len(comp.state_schema)
            lines.append(f"  {name} ({comp.__class__.__name__}): "
                         f"{n_states} states {comp.state_schema}")

        non_ph = [n for n in self.graph.components
                  if n not in self.ph_component_names]
        if non_ph:
            lines.append("")
            lines.append("Non-PH Components (zero Hamiltonian):")
            for name in non_ph:
                comp = self.graph.components[name]
                lines.append(f"  {name} ({comp.__class__.__name__})")

        return "\n".join(lines)
