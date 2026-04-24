"""Signal-flow graph descriptors and code generation.

This package complements the symbolic PHS layer for controller-style block
diagrams whose runtime behavior is better described as ordered signal-flow
than as port-Hamiltonian dynamics.
"""

from src.signal_flow.core import SignalExpr, SignalFlowGraph
from src.signal_flow.codegen import (
    generate_signal_flow_cpp_dynamics,
    generate_signal_flow_cpp_outputs,
)

__all__ = [
    "SignalExpr",
    "SignalFlowGraph",
    "generate_signal_flow_cpp_dynamics",
    "generate_signal_flow_cpp_outputs",
]