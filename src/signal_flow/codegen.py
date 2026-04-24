from __future__ import annotations

from typing import Dict, List, Set

from src.signal_flow.core import (
    ClampBlock,
    LagBlock,
    LeadLagBlock,
    ModeSelectBlock,
    SignalAssignment,
    SignalExpr,
    SignalFlowGraph,
    WashoutBlock,
)


def generate_signal_flow_cpp_dynamics(
    graph: SignalFlowGraph,
    state_c_names: List[str],
    input_c_names: List[str],
) -> str:
    """Generate C++ dynamics code from a SignalFlowGraph."""
    _validate_signal_flow_graph(graph, state_c_names, input_c_names)
    return _generate_signal_flow_cpp(
        graph,
        state_c_names,
        input_c_names,
        include_dynamics=True,
        include_outputs=False,
    )


def generate_signal_flow_cpp_outputs(
    graph: SignalFlowGraph,
    state_c_names: List[str],
    input_c_names: List[str],
) -> str:
    """Generate C++ output equations from a SignalFlowGraph."""
    if not graph.output_map:
        return ""
    _validate_signal_flow_graph(graph, state_c_names, input_c_names)
    return _generate_signal_flow_cpp(
        graph,
        state_c_names,
        input_c_names,
        include_dynamics=False,
        include_outputs=True,
    )


def _generate_signal_flow_cpp(
    graph: SignalFlowGraph,
    state_c_names: List[str],
    input_c_names: List[str],
    include_dynamics: bool,
    include_outputs: bool,
) -> str:
    state_indices = {name: idx for idx, name in enumerate(state_c_names)}
    used_states: Set[str] = set()
    lines: List[str] = []

    lines.append(f"// Signal-flow code — auto-generated from '{graph.name}'")
    lines.append("")

    for idx, name in enumerate(state_c_names):
        lines.append(f"double {name} = x[{idx}];")
    lines.append("")

    for idx, name in enumerate(input_c_names):
        lines.append(f"double {name} = inputs[{idx}];")
    lines.append("")

    for block in graph.blocks:
        lines.extend(_emit_block(block, state_indices, include_dynamics))
        if isinstance(block, (LagBlock, LeadLagBlock, WashoutBlock)):
            used_states.add(block.state)

    if include_dynamics:
        for idx, state_name in enumerate(state_c_names):
            if state_name not in used_states:
                lines.append(f"dxdt[{idx}] = 0.0;")

    if include_outputs:
        if lines and lines[-1] != "":
            lines.append("")
        for out_idx in sorted(graph.output_map):
            expr_cpp = _expr_to_cpp(graph.output_map[out_idx])
            lines.append(f"outputs[{out_idx}] = {expr_cpp};")

    return "\n".join(lines)


def _emit_block(block, state_indices: Dict[str, int], include_dynamics: bool) -> List[str]:
    if isinstance(block, SignalAssignment):
        return [f"double {block.name} = {_expr_to_cpp(block.expr)};"]

    if isinstance(block, ModeSelectBlock):
        lines = [f"double {block.name};"]
        mode_var = f"{block.name}_mode"
        lines.append(f"int {mode_var} = (int){block.mode_param};")
        first = True
        for mode_value in sorted(block.cases):
            prefix = "if" if first else "else if"
            lines.append(f"{prefix} ({mode_var} == {mode_value}) {{")
            lines.append(f"    {block.name} = {_expr_to_cpp(block.cases[mode_value])};")
            lines.append("}")
            first = False
        lines.append("else {")
        lines.append(f"    {block.name} = {_expr_to_cpp(block.default)};")
        lines.append("}")
        return lines

    if isinstance(block, LagBlock):
        idx = state_indices[block.state]
        input_cpp = _expr_to_cpp(block.input_expr)
        lines = [f"double {block.output};"]
        lines.append(f"if ({block.time_const} > {_format_float(block.bypass_eps)}) {{")
        if include_dynamics:
            lines.append(f"    dxdt[{idx}] = ({input_cpp} - {block.state}) / {block.time_const};")
        lines.append(f"    {block.output} = {block.state};")
        lines.append("} else {")
        if include_dynamics:
            lines.append(f"    dxdt[{idx}] = 0.0;")
        lines.append(f"    {block.output} = {input_cpp};")
        lines.append("}")
        return lines

    if isinstance(block, LeadLagBlock):
        idx = state_indices[block.state]
        input_cpp = _expr_to_cpp(block.input_expr)
        lines = [f"double {block.output};"]
        lines.append(f"if ({block.denominator_param} > {_format_float(block.bypass_eps)}) {{")
        if include_dynamics:
            lines.append(
                f"    dxdt[{idx}] = ({input_cpp} - {block.state}) / {block.denominator_param};"
            )
        lines.append(
            f"    {block.output} = {block.state} + ({block.numerator_param} / {block.denominator_param}) * ({input_cpp} - {block.state});"
        )
        lines.append("} else {")
        if include_dynamics:
            lines.append(f"    dxdt[{idx}] = 0.0;")
        lines.append(f"    {block.output} = {input_cpp};")
        lines.append("}")
        return lines

    if isinstance(block, WashoutBlock):
        idx = state_indices[block.state]
        input_cpp = _expr_to_cpp(block.input_expr)
        lines = [f"double {block.output};"]
        lines.append(f"if ({block.denominator_param} > {_format_float(block.bypass_eps)}) {{")
        if include_dynamics:
            lines.append(
                f"    dxdt[{idx}] = ({input_cpp} - {block.state}) / {block.denominator_param};"
            )
        lines.append(
            f"    {block.output} = {block.numerator_param} * ({input_cpp} - {block.state}) / {block.denominator_param};"
        )
        lines.append("} else {")
        if include_dynamics:
            lines.append(f"    dxdt[{idx}] = 0.0;")
        lines.append(f"    {block.output} = {block.numerator_param} * ({input_cpp});")
        lines.append("}")
        return lines

    if isinstance(block, ClampBlock):
        lines = [f"double {block.name} = {_expr_to_cpp(block.input_expr)};"]
        upper = _bound_to_cpp(block.upper_bound)
        lower = _bound_to_cpp(block.lower_bound)
        lines.append(f"if ({block.name} > {upper}) {block.name} = {upper};")
        lines.append(f"if ({block.name} < {lower}) {block.name} = {lower};")
        return lines

    raise TypeError(f"Unsupported signal-flow block: {type(block).__name__}")


def _expr_to_cpp(expr: SignalExpr) -> str:
    if expr.op in ("ref", "param"):
        return str(expr.value)
    if expr.op == "const":
        return _format_float(float(expr.value))
    if expr.op == "neg":
        return f"(-{_expr_to_cpp(expr.args[0])})"
    if expr.op == "add":
        return f"({_expr_to_cpp(expr.args[0])} + {_expr_to_cpp(expr.args[1])})"
    if expr.op == "sub":
        return f"({_expr_to_cpp(expr.args[0])} - {_expr_to_cpp(expr.args[1])})"
    if expr.op == "mul":
        return f"({_expr_to_cpp(expr.args[0])} * {_expr_to_cpp(expr.args[1])})"
    if expr.op == "div":
        return f"({_expr_to_cpp(expr.args[0])} / {_expr_to_cpp(expr.args[1])})"
    raise ValueError(f"Unsupported SignalExpr op '{expr.op}'")


def _bound_to_cpp(bound) -> str:
    if isinstance(bound, SignalExpr):
        return _expr_to_cpp(bound)
    if isinstance(bound, str):
        return bound
    return _format_float(float(bound))


def _format_float(value: float) -> str:
    text = repr(float(value))
    if "e" not in text and "." not in text:
        text += ".0"
    return text


def _validate_signal_flow_graph(
    graph: SignalFlowGraph,
    state_c_names: List[str],
    input_c_names: List[str],
) -> None:
    available_signals = set(state_c_names) | set(input_c_names)
    dynamic_states: Set[str] = set()

    for block in graph.blocks:
        if isinstance(block, SignalAssignment):
            _require_known_refs(block.expr, available_signals, graph.name, block.name)
            _declare_signal(block.name, available_signals, state_c_names, input_c_names, graph.name)
            continue

        if isinstance(block, ModeSelectBlock):
            for expr in block.cases.values():
                _require_known_refs(expr, available_signals, graph.name, block.name)
            _require_known_refs(block.default, available_signals, graph.name, block.name)
            _declare_signal(block.name, available_signals, state_c_names, input_c_names, graph.name)
            continue

        if isinstance(block, (LagBlock, LeadLagBlock, WashoutBlock)):
            if block.state not in state_c_names:
                raise ValueError(
                    f"SignalFlowGraph '{graph.name}' references unknown state '{block.state}'"
                )
            if block.state in dynamic_states:
                raise ValueError(
                    f"SignalFlowGraph '{graph.name}' uses state '{block.state}' in multiple dynamic blocks"
                )
            dynamic_states.add(block.state)
            _require_known_refs(block.input_expr, available_signals, graph.name, block.output)
            _declare_signal(block.output, available_signals, state_c_names, input_c_names, graph.name)
            continue

        if isinstance(block, ClampBlock):
            _require_known_refs(block.input_expr, available_signals, graph.name, block.name)
            if isinstance(block.lower_bound, SignalExpr):
                _require_known_refs(block.lower_bound, available_signals, graph.name, block.name)
            if isinstance(block.upper_bound, SignalExpr):
                _require_known_refs(block.upper_bound, available_signals, graph.name, block.name)
            _declare_signal(block.name, available_signals, state_c_names, input_c_names, graph.name)
            continue

        raise TypeError(f"Unsupported signal-flow block: {type(block).__name__}")

    for out_idx, expr in graph.output_map.items():
        _require_known_refs(expr, available_signals, graph.name, f"outputs[{out_idx}]")


def _declare_signal(
    name: str,
    available_signals: Set[str],
    state_c_names: List[str],
    input_c_names: List[str],
    graph_name: str,
) -> None:
    if name in state_c_names or name in input_c_names:
        raise ValueError(
            f"SignalFlowGraph '{graph_name}' tries to reuse reserved name '{name}'"
        )
    if name in available_signals:
        raise ValueError(
            f"SignalFlowGraph '{graph_name}' defines signal '{name}' more than once"
        )
    available_signals.add(name)


def _require_known_refs(
    expr: SignalExpr,
    available_signals: Set[str],
    graph_name: str,
    block_name: str,
) -> None:
    missing = sorted(_collect_signal_refs(expr) - available_signals)
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(
            f"SignalFlowGraph '{graph_name}' block '{block_name}' references unknown signals: {missing_text}"
        )


def _collect_signal_refs(expr: SignalExpr) -> Set[str]:
    if expr.op == "ref":
        return {str(expr.value)}
    if expr.op in ("param", "const"):
        return set()
    refs: Set[str] = set()
    for arg in expr.args:
        refs |= _collect_signal_refs(arg)
    return refs