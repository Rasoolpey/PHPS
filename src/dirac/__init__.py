"""
Port-Hamiltonian Dirac Structure Core.

This package provides a **parallel** simulation pipeline based on the
Dirac-structure formulation of Port-Hamiltonian Systems (PHS).  It
does **not** replace or modify the existing compiler/runner pipeline
(``src/compiler.py``, ``src/runner.py``); instead it offers:

  1. ``IncidenceBuilder``       — network incidence matrix from SystemGraph
  2. ``DiracStructure``         — KCL/KVL as a power-conserving Dirac subspace
  3. ``HamiltonianAssembler``   — total Hamiltonian from component PHS data
  4. ``DiracCompiler``          — DAE residual C++ code generation
  5. ``DiracRunner``            — build, compile, and run the DAE C++ simulation

All classes consume the existing ``SystemGraph`` and ``PowerComponent``
objects so that the same JSON system files work with both pipelines.

Usage (analysis only — Steps 1–2 from DIRAC_STRUCTURE_README.md)::

    from src.system_graph import build_system_graph
    from src.dirac import IncidenceBuilder, DiracStructure, HamiltonianAssembler

    graph = build_system_graph("cases/SMIB/system.json")
    B     = IncidenceBuilder(graph).build()
    D     = DiracStructure(graph, B)
    H_asm = HamiltonianAssembler(graph)

    # Verify power conservation
    assert D.verify_power_conservation()

    # Compute total Hamiltonian at a state vector
    H_total = H_asm.total_hamiltonian(x0)

Usage (DAE simulation — Steps 3–4)::

    from src.dirac import DiracRunner

    runner = DiracRunner("cases/SMIB/system.json")
    runner.build(dt=0.0005, duration=10.0)
    runner.run()
"""

from src.dirac.incidence import IncidenceBuilder
from src.dirac.dirac_structure import DiracStructure
from src.dirac.hamiltonian import HamiltonianAssembler
from src.dirac.dae_compiler import DiracCompiler
from src.dirac.dae_runner import DiracRunner

__all__ = [
    "IncidenceBuilder",
    "DiracStructure",
    "HamiltonianAssembler",
    "DiracCompiler",
    "DiracRunner",
]
