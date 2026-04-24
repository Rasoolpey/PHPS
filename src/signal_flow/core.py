from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union


SignalOperand = Union["SignalExpr", str, int, float]


@dataclass(frozen=True)
class SignalExpr:
    """Small expression tree for signal-flow algebra.

    The signal-flow path only needs light-weight static expressions over:

    - named intermediate signals / input ports
    - named parameters
    - numeric constants
    - basic arithmetic (+, -, *, /)
    """

    op: str
    value: Any = None
    args: Tuple["SignalExpr", ...] = ()

    @staticmethod
    def ref(name: str) -> "SignalExpr":
        return SignalExpr("ref", value=name)

    @staticmethod
    def param(name: str) -> "SignalExpr":
        return SignalExpr("param", value=name)

    @staticmethod
    def const(value: Union[int, float]) -> "SignalExpr":
        return SignalExpr("const", value=float(value))

    def _binary(self, op: str, other: SignalOperand) -> "SignalExpr":
        return SignalExpr(op, args=(self, as_signal_expr(other)))

    def __add__(self, other: SignalOperand) -> "SignalExpr":
        return self._binary("add", other)

    def __radd__(self, other: SignalOperand) -> "SignalExpr":
        return as_signal_expr(other)._binary("add", self)

    def __sub__(self, other: SignalOperand) -> "SignalExpr":
        return self._binary("sub", other)

    def __rsub__(self, other: SignalOperand) -> "SignalExpr":
        return as_signal_expr(other)._binary("sub", self)

    def __mul__(self, other: SignalOperand) -> "SignalExpr":
        return self._binary("mul", other)

    def __rmul__(self, other: SignalOperand) -> "SignalExpr":
        return as_signal_expr(other)._binary("mul", self)

    def __truediv__(self, other: SignalOperand) -> "SignalExpr":
        return self._binary("div", other)

    def __rtruediv__(self, other: SignalOperand) -> "SignalExpr":
        return as_signal_expr(other)._binary("div", self)

    def __neg__(self) -> "SignalExpr":
        return SignalExpr("neg", args=(self,))


def as_signal_expr(value: SignalOperand) -> SignalExpr:
    if isinstance(value, SignalExpr):
        return value
    if isinstance(value, str):
        return SignalExpr.ref(value)
    return SignalExpr.const(value)


@dataclass
class SignalAssignment:
    name: str
    expr: SignalExpr


@dataclass
class ModeSelectBlock:
    name: str
    mode_param: str
    cases: Dict[int, SignalExpr]
    default: SignalExpr


@dataclass
class LagBlock:
    state: str
    output: str
    input_expr: SignalExpr
    time_const: str
    bypass_eps: float = 1e-6


@dataclass
class LeadLagBlock:
    state: str
    output: str
    input_expr: SignalExpr
    numerator_param: str
    denominator_param: str
    bypass_eps: float = 1e-6


@dataclass
class WashoutBlock:
    state: str
    output: str
    input_expr: SignalExpr
    numerator_param: str
    denominator_param: str
    bypass_eps: float = 1e-6


@dataclass
class ClampBlock:
    name: str
    input_expr: SignalExpr
    lower_bound: Union[SignalExpr, str, float, int]
    upper_bound: Union[SignalExpr, str, float, int]


SignalFlowBlock = Union[
    SignalAssignment,
    ModeSelectBlock,
    LagBlock,
    LeadLagBlock,
    WashoutBlock,
    ClampBlock,
]


@dataclass
class SignalFlowGraph:
    """Ordered signal-flow descriptor for controller-style block diagrams."""

    name: str
    blocks: List[SignalFlowBlock] = field(default_factory=list)
    output_map: Dict[int, SignalExpr] = field(default_factory=dict)

    def ref(self, name: str) -> SignalExpr:
        return SignalExpr.ref(name)

    def param(self, name: str) -> SignalExpr:
        return SignalExpr.param(name)

    def const(self, value: Union[int, float]) -> SignalExpr:
        return SignalExpr.const(value)

    def add_expr(self, name: str, expr: SignalOperand) -> "SignalFlowGraph":
        self.blocks.append(SignalAssignment(name=name, expr=as_signal_expr(expr)))
        return self

    def add_gain(
        self,
        name: str,
        input_signal: SignalOperand,
        gain: Union[SignalExpr, str, float, int],
    ) -> "SignalFlowGraph":
        gain_expr = SignalExpr.param(gain) if isinstance(gain, str) else as_signal_expr(gain)
        return self.add_expr(name, gain_expr * as_signal_expr(input_signal))

    def add_sum(self, name: str, terms: List[SignalOperand]) -> "SignalFlowGraph":
        expr = SignalExpr.const(0.0)
        for term in terms:
            expr = expr + as_signal_expr(term)
        return self.add_expr(name, expr)

    def add_mode_select(
        self,
        name: str,
        mode_param: str,
        cases: Dict[int, SignalOperand],
        default: SignalOperand,
    ) -> "SignalFlowGraph":
        self.blocks.append(
            ModeSelectBlock(
                name=name,
                mode_param=mode_param,
                cases={mode: as_signal_expr(expr) for mode, expr in cases.items()},
                default=as_signal_expr(default),
            )
        )
        return self

    def add_lag(
        self,
        state: str,
        input_expr: SignalOperand,
        time_const: str,
        output_name: Optional[str] = None,
        bypass_eps: float = 1e-6,
    ) -> "SignalFlowGraph":
        self.blocks.append(
            LagBlock(
                state=state,
                output=output_name or f"{state}_out",
                input_expr=as_signal_expr(input_expr),
                time_const=time_const,
                bypass_eps=bypass_eps,
            )
        )
        return self

    def add_lead_lag(
        self,
        state: str,
        input_expr: SignalOperand,
        numerator_param: str,
        denominator_param: str,
        output_name: Optional[str] = None,
        bypass_eps: float = 1e-6,
    ) -> "SignalFlowGraph":
        self.blocks.append(
            LeadLagBlock(
                state=state,
                output=output_name or f"{state}_out",
                input_expr=as_signal_expr(input_expr),
                numerator_param=numerator_param,
                denominator_param=denominator_param,
                bypass_eps=bypass_eps,
            )
        )
        return self

    def add_washout(
        self,
        state: str,
        input_expr: SignalOperand,
        numerator_param: str,
        denominator_param: str,
        output_name: Optional[str] = None,
        bypass_eps: float = 1e-6,
    ) -> "SignalFlowGraph":
        self.blocks.append(
            WashoutBlock(
                state=state,
                output=output_name or f"{state}_out",
                input_expr=as_signal_expr(input_expr),
                numerator_param=numerator_param,
                denominator_param=denominator_param,
                bypass_eps=bypass_eps,
            )
        )
        return self

    def add_clamp(
        self,
        name: str,
        input_expr: SignalOperand,
        lower_bound: Union[SignalExpr, str, float, int],
        upper_bound: Union[SignalExpr, str, float, int],
    ) -> "SignalFlowGraph":
        self.blocks.append(
            ClampBlock(
                name=name,
                input_expr=as_signal_expr(input_expr),
                lower_bound=lower_bound,
                upper_bound=upper_bound,
            )
        )
        return self

    def add_output(self, output_idx: int, expr: SignalOperand) -> "SignalFlowGraph":
        self.output_map[output_idx] = as_signal_expr(expr)
        return self

    def set_output_map(self, mapping: Dict[int, SignalOperand]) -> "SignalFlowGraph":
        self.output_map = {idx: as_signal_expr(expr) for idx, expr in mapping.items()}
        return self