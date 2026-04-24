"""
Structured error types for the PH PowerSystem Framework.

Every error carries:
  - The exact location (component, port, phase, field) where the problem occurred.
  - A clear human-readable description of what went wrong.
  - One or more concrete, copy-pasteable suggestions for how to fix it.

Design goal: a user who sees only the error message (not the Python traceback)
should understand what to change and where.
"""
from typing import Optional, List


class FrameworkError(Exception):
    """Base class for all framework-level errors."""
    pass


# ---------------------------------------------------------------------------
# Connection / wiring errors
# ---------------------------------------------------------------------------

class PortUnconnectedError(FrameworkError):
    """Raised when a required input port has no source wire."""

    def __init__(self, component: str, port: str,
                 suggestion: str = "",
                 available_sources: Optional[List[str]] = None):
        self.component = component
        self.port = port
        self.suggestion = suggestion
        msg = (
            f"\n[Wiring Error] Component '{component}' input port '{port}' has no source.\n"
            f"\n  Add a wire to your 'connections' list in system.json:\n"
            f"    {{\"from\": \"<source>\", \"to\": \"{component}.{port}\"}}\n"
            f"\n  Common sources for '{port}':\n"
        )
        hints = _port_hints(port)
        for h in hints:
            msg += f"    {h}\n"
        if available_sources:
            msg += f"\n  Components with output ports you can wire from:\n"
            for s in available_sources[:8]:
                msg += f"    {s}\n"
        if suggestion:
            msg += f"\n  Additional hint: {suggestion}"
        super().__init__(msg)


class PortTypeMismatchError(FrameworkError):
    """Raised when a wire connects ports whose kind (effort/flow/signal) disagrees."""

    def __init__(self, src: str, dst: str, src_kind: str, dst_kind: str):
        self.src = src
        self.dst = dst
        self.src_kind = src_kind
        self.dst_kind = dst_kind
        super().__init__(
            f"\n[Wiring Error] Port kind mismatch on wire '{src}' → '{dst}'.\n"
            f"  Source port kind : '{src_kind}'\n"
            f"  Destination kind : '{dst_kind}'\n"
            f"\n  Suggestion: check port_schema in both component model files.\n"
            f"  Kinds must match: 'effort' (voltages), 'flow' (currents), 'signal' (control)."
        )


class PortNotFoundError(FrameworkError):
    """Raised when a wire references a port that does not exist on a component."""

    def __init__(self, component: str, port: str, direction: str, available: list):
        self.component = component
        self.port = port
        self.direction = direction
        self.available = available
        super().__init__(
            f"\n[Wiring Error] Component '{component}' has no {direction}put port '{port}'.\n"
            f"  Available {direction}put ports: {available}\n"
            f"\n  Suggestion: check the spelling in your 'connections' block.\n"
            f"  To see all ports for a component, inspect its port_schema property\n"
            f"  or read src/components/<type>/<model>.py."
        )


class ComponentNotFoundError(FrameworkError):
    """Raised when a wire references a component name that does not exist."""

    def __init__(self, name: str, direction: str, available: list):
        self.name = name
        super().__init__(
            f"\n[Wiring Error] Wire {direction} references unknown component '{name}'.\n"
            f"  Registered components: {available}\n"
            f"\n  Suggestions:\n"
            f"    - Check the spelling of the component name in your 'connections' block.\n"
            f"    - Make sure the component is declared under 'components' in system.json.\n"
            f"    - Component names are case-sensitive (e.g. 'GENROU_1', not 'genrou_1')."
        )


class WireSourceError(FrameworkError):
    """Raised when a wire 'from' field uses an unrecognised format."""

    def __init__(self, src: str, suggestion: str = ""):
        self.src = src
        msg = (
            f"\n[Wiring Error] Wire source '{src}' is not a recognised format.\n"
            f"\n  Valid 'from' formats:\n"
            f"    BUS_<id>.Vd            — d-axis network voltage at bus <id>\n"
            f"    BUS_<id>.Vq            — q-axis network voltage at bus <id>\n"
            f"    BUS_<id>.Vterm         — terminal voltage magnitude at bus <id>\n"
            f"    DQ_<GenName>.Vd_dq     — Park-transformed Vd from generator <GenName>\n"
            f"    DQ_<GenName>.Vq_dq     — Park-transformed Vq from generator <GenName>\n"
            f"    <ComponentName>.<port> — output port of another component\n"
            f"    CONST:<float>          — compile-time constant (e.g. CONST:1.0)\n"
            f"    PARAM:<Comp>.<key>     — parameter value from a component\n"
            f"\n  Examples:\n"
            f"    {{\"from\": \"BUS_1.Vterm\",        \"to\": \"ESST3A_1.Vterm\"}}\n"
            f"    {{\"from\": \"GENROU_1.omega\",     \"to\": \"TGOV1_1.omega\"}}\n"
            f"    {{\"from\": \"ESST3A_1.Efd\",       \"to\": \"GENROU_1.Efd\"}}\n"
            f"    {{\"from\": \"CONST:1.0\",           \"to\": \"ESST3A_1.Vref\"}}"
        )
        if suggestion:
            msg += f"\n\n  Additional: {suggestion}"
        super().__init__(msg)


# ---------------------------------------------------------------------------
# Initialization / convergence errors
# ---------------------------------------------------------------------------

class InitializationError(FrameworkError):
    """Raised for general initialization failures."""

    def __init__(self, message: str, component: Optional[str] = None,
                 suggestion: str = ""):
        self.component = component
        self.suggestion = suggestion
        msg = f"\n[Initialization Error]"
        if component:
            msg += f" (component: '{component}')"
        msg += f"\n  {message}\n"
        if suggestion:
            msg += f"\n  Suggestion: {suggestion}\n"
        else:
            msg += (
                f"\n  Suggestions:\n"
                f"    - Verify that PV.p0 is within [PV.pmin, PV.pmax].\n"
                f"    - Check that the generator's operating point is physically feasible\n"
                f"      (not overloaded, not saturated at Efd_max).\n"
                f"    - Confirm line impedances and bus voltages are in per-unit on the\n"
                f"      correct MVA and kV base (system 'config.mva_base')."
            )
        super().__init__(msg)


class InitializationConvergenceError(FrameworkError):
    """Raised when an iterative refinement phase does not converge."""

    def __init__(self, phase: str, n_iter: int, residual: float,
                 component: Optional[str] = None, suggestion: str = ""):
        self.phase = phase
        self.n_iter = n_iter
        self.residual = residual
        self.component = component
        msg = (
            f"\n[Initialization Convergence Error]\n"
            f"  Phase     : '{phase}'\n"
            f"  Iterations: {n_iter} (did not converge)\n"
            f"  Residual  : {residual:.4e}\n"
        )
        if component:
            msg += f"  Component : '{component}'\n"
        if suggestion:
            msg += f"\n  Suggestion: {suggestion}\n"
        else:
            msg += (
                f"\n  Suggestions:\n"
                f"    - Check that the active power setpoint (PV.p0) is within the\n"
                f"      stability region (rotor angle δ should be well below 90°).\n"
                f"    - Verify governor droop R and turbine time constants are realistic.\n"
                f"    - Ensure Efd_max is large enough to sustain the operating voltage\n"
                f"      (if Efd is clipping, the exciter cannot reach equilibrium).\n"
                f"    - For the 'refine_delta_for_torque_balance' phase: increase n_iter\n"
                f"      in runner.py or reduce the operating point (PV.p0).\n"
                f"    - For 'refine_kron_equilibrium': the Kron-reduced network may be\n"
                f"      ill-conditioned — check that all generator buses are connected."
            )
        super().__init__(msg)


# ---------------------------------------------------------------------------
# Power flow errors
# ---------------------------------------------------------------------------

class PowerFlowError(FrameworkError):
    """Raised when the Newton-Raphson power flow does not converge."""

    def __init__(self, n_iter: int, residual: float, suggestion: str = ""):
        self.n_iter = n_iter
        self.residual = residual
        msg = (
            f"\n[Power Flow Error]\n"
            f"  Newton-Raphson did not converge after {n_iter} iterations.\n"
            f"  Final residual: {residual:.4e}  (target: < 1e-8)\n"
            f"\n  Suggestions (check in this order):\n"
            f"    1. PV bus reactive limits: ensure qmax/qmin in 'PV' are wide enough\n"
            f"       to support the required voltage (qmax ≥ actual Q demand).\n"
            f"    2. Operating point: PV.p0 may be too large — reduce and retry.\n"
            f"    3. Slack bus: verify v0=1.0 and a0=0.0 (reference bus).\n"
            f"    4. Network connectivity: every non-slack bus must have at least one line.\n"
            f"    5. Per-unit consistency: r/x/b on lines must be in pu on the same\n"
            f"       MVA/kV base as the system ('config.mva_base', bus 'Vn').\n"
            f"    6. Voltage initialisation: set 'v0' on each bus close to 1.0 pu.\n"
            f"    7. Isolated bus: a bus with only a PQ load and no generator or line\n"
            f"       will cause singular Y-bus — add a line or remove the bus."
        )
        if suggestion:
            msg += f"\n\n  Additional: {suggestion}"
        super().__init__(msg)


# ---------------------------------------------------------------------------
# Compilation errors
# ---------------------------------------------------------------------------

class CompilerError(FrameworkError):
    """Raised when C++ code generation or compilation fails."""

    def __init__(self, message: str, component: Optional[str] = None,
                 port: Optional[str] = None, suggestion: str = ""):
        self.component = component
        self.port = port
        loc = "Compiler error"
        if component:
            loc += f" [{component}"
            if port:
                loc += f".{port}"
            loc += "]"
        msg = f"\n[{loc}]\n  {message}\n"
        if suggestion:
            msg += f"\n  Suggestion: {suggestion}\n"
        else:
            msg += (
                f"\n  Suggestions:\n"
                f"    - Verify the component appears in the 'connections' block with all\n"
                f"      required input ports wired.\n"
                f"    - Run graph.validate() to catch wiring issues before compilation.\n"
                f"    - Check get_cpp_step_code() in the component model for syntax errors.\n"
                f"    - If the g++ error message is shown above, look for undefined variable\n"
                f"      names — these usually indicate a missing or misspelled port wire."
            )
        super().__init__(msg)


class NetworkNotFinalizedError(FrameworkError):
    """Raised when generate_cpp() is called before finalize_network()."""

    def __init__(self):
        super().__init__(
            "\n[Compiler Error] Network has not been finalized.\n"
            "  Call finalize_network(pf_V, pf_theta) after running the power flow\n"
            "  and before calling generate_cpp().\n"
            "\n  This is handled automatically by SimulationRunner.build().\n"
            "  If you are calling the compiler directly, ensure the sequence:\n"
            "    1. compiler.build_structure()\n"
            "    2. init.run()   (runs power flow internally)\n"
            "    3. compiler.finalize_network(pf_V=init.pf.V, pf_theta=init.pf.theta)\n"
            "    4. compiler.generate_cpp()"
        )


# ---------------------------------------------------------------------------
# JSON / schema errors
# ---------------------------------------------------------------------------

class SchemaError(FrameworkError):
    """Raised when the system JSON does not match the expected schema."""

    def __init__(self, message: str, field: Optional[str] = None,
                 suggestion: str = ""):
        self.field = field
        msg = f"\n[Schema Error]"
        if field:
            msg += f" (field: '{field}')"
        msg += f"\n  {message}\n"
        if suggestion:
            msg += f"\n  Suggestion: {suggestion}\n"
        else:
            msg += (
                f"\n  Suggestions:\n"
                f"    - Compare your system.json against the template in cases/SMIB/system.json.\n"
                f"    - Required top-level sections: 'Bus', 'PV' or 'Slack', 'components', 'connections'.\n"
                f"    - If using the old format (ANDES-style with 'GENROU', 'ESST3A' as top-level keys),\n"
                f"      run: python3 tools/migrate_system_json.py <your_file> --dry-run\n"
                f"      to preview the auto-converted new format."
            )
        super().__init__(msg)


class UnknownComponentTypeError(FrameworkError):
    """Raised when a 'type' field in the components dict is not registered."""

    def __init__(self, type_name: str, known_types: list):
        self.type_name = type_name
        super().__init__(
            f"\n[Schema Error] Unknown component type '{type_name}'.\n"
            f"  Registered types: {sorted(known_types)}\n"
            f"\n  Suggestions:\n"
            f"    - Check the spelling of 'type' in your 'components' block.\n"
            f"    - Type names are case-sensitive (e.g. 'GENROU', not 'GenRou').\n"
            f"    - If this is a new model, create src/components/<category>/<name>.py\n"
            f"      and register it in src/json_compat.py _build_registry()."
        )


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _port_hints(port: str) -> List[str]:
    """Return copy-pasteable wire source suggestions for a given port name."""
    _hints = {
        "Vterm":  ['{"from": "BUS_<id>.Vterm", "to": "<comp>.Vterm"}  ← terminal voltage magnitude'],
        "Vd":     ['{"from": "BUS_<id>.Vd",    "to": "<comp>.Vd"}     ← d-axis bus voltage (RI frame)',
                   '{"from": "DQ_<gen>.Vd_dq", "to": "<comp>.Vd"}     ← Park-transformed Vd (dq frame)'],
        "Vq":     ['{"from": "BUS_<id>.Vq",    "to": "<comp>.Vq"}     ← q-axis bus voltage (RI frame)',
                   '{"from": "DQ_<gen>.Vq_dq", "to": "<comp>.Vq"}     ← Park-transformed Vq (dq frame)'],
        "Vref":   ['{"from": "CONST:1.0",       "to": "<comp>.Vref"}   ← voltage reference (initializer sets actual value)'],
        "Efd":    ['{"from": "<ExciterName>.Efd", "to": "<gen>.Efd"}   ← field voltage from exciter'],
        "Tm":     ['{"from": "<GovName>.Tm",    "to": "<gen>.Tm"}      ← mechanical torque from governor'],
        "omega":  ['{"from": "<gen>.omega",     "to": "<comp>.omega"}  ← rotor speed from generator'],
        "Pref":   ['{"from": "CONST:1.0",       "to": "<comp>.Pref"}   ← power reference (initializer sets actual value)'],
        "Pe":     ['{"from": "<gen>.Pe",         "to": "<comp>.Pe"}     ← electrical power from generator'],
        "id_dq":  ['{"from": "<gen>.id_dq",     "to": "<comp>.id_dq"}  ← d-axis stator current (dq frame)'],
        "iq_dq":  ['{"from": "<gen>.iq_dq",     "to": "<comp>.iq_dq"}  ← q-axis stator current (dq frame)'],
        "Vss":    ['{"from": "<PssName>.Vss",   "to": "<exciter>.Vss"} ← PSS stabilising signal'],
    }
    return _hints.get(port, [f'Check the port_schema of the component model for valid source types.'])
