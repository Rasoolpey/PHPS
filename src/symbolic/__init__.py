"""
Symbolic Port-Hamiltonian System (PHS) layer.

Provides SymPy-based symbolic representations of (J, R, H, g) structure
matrices, automatic verification of PHS properties, LaTeX export, and
code generation that bridges the symbolic layer to the simulation pipeline.

The symbolic layer is the **single source of truth** for the PHS structure.
All downstream artefacts — Python ``hamiltonian()`` / ``grad_hamiltonian()``
callables, numerical ``get_phs_matrices()`` evaluations, C++ Hamiltonian
expressions, and (optionally) C++ dynamics code — are derived from it
automatically via the ``codegen`` module.

Usage:
    from src.symbolic import SymbolicPHS, validate_phs_structure, phs_to_latex
    from src.symbolic.codegen import generate_phs_cpp_dynamics

    sphs = component.get_symbolic_phs()
    report = validate_phs_structure(sphs)
    tex = phs_to_latex(sphs)
    cpp = generate_phs_cpp_dynamics(sphs, state_names, input_names)
"""

from src.symbolic.core import SymbolicPHS, InitSpec
from src.symbolic.validation import validate_phs_structure, validate_phs_passivity, PHSValidationReport
def phs_to_latex(*args, **kwargs):
    from src.symbolic.latex_export import phs_to_latex as _f
    return _f(*args, **kwargs)

def phs_collection_to_tex_document(*args, **kwargs):
    from src.symbolic.latex_export import phs_collection_to_tex_document as _f
    return _f(*args, **kwargs)
from src.symbolic.codegen import (
    generate_phs_cpp_dynamics,
    generate_hamiltonian_cpp_expr,
    generate_dissipation_cpp_expr,
    generate_supply_rate_cpp_expr,
    make_hamiltonian_func,
    make_grad_hamiltonian_func,
    evaluate_phs_matrices,
    solve_equilibrium,
)

__all__ = [
    "SymbolicPHS",
    "InitSpec",
    "validate_phs_structure",
    "validate_phs_passivity",
    "PHSValidationReport",
    "phs_to_latex",
    "phs_collection_to_tex_document",
    "generate_phs_cpp_dynamics",
    "generate_hamiltonian_cpp_expr",
    "generate_dissipation_cpp_expr",
    "generate_supply_rate_cpp_expr",
    "make_hamiltonian_func",
    "make_grad_hamiltonian_func",
    "evaluate_phs_matrices",
    "solve_equilibrium",
]
