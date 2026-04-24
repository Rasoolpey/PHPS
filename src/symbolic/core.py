"""
SymbolicPHS — core symbolic representation of a Port-Hamiltonian system.

A SymbolicPHS encapsulates the symbolic (J, R, Q, g, H) structure for a
single component, where all matrices are SymPy Matrix objects and H is a
SymPy expression.

The dynamics are:
    ẋ = (J − R) Q ∂H/∂x + g · u

Attributes:
    name       : str           — component name (e.g. 'GENROU_PHS')
    states     : list[Symbol]  — state variable symbols [δ, ω, E'q, ...]
    inputs     : list[Symbol]  — input symbols [Tm, Efd, ...]
    params     : dict[str, Symbol] — parameter symbols {H: H, D: D, ...}
    J          : Matrix (n×n) — skew-symmetric interconnection matrix
    R          : Matrix (n×n) — positive semi-definite dissipation matrix
    Q          : Matrix (n×n) — energy-variable transformation (often I)
    g          : Matrix (n×m) — input/port coupling matrix
    H          : Expr         — Hamiltonian (scalar energy function)
"""

import sympy as sp
from sympy import Matrix, Symbol, symbols, diff, simplify, latex
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field


@dataclass
class Limiter:
    """Anti-windup limiter on a state derivative.

    When ``x[state_idx] >= upper_bound`` and ``dxdt > 0``, clamps ``dxdt`` to 0.
    When ``x[state_idx] <= lower_bound`` and ``dxdt < 0``, clamps ``dxdt`` to 0.

    Bound expressions can be:
    - A string: interpreted as a C parameter name (e.g. ``'VRMAX'``)
    - A SymPy expression: codegen will render it as C++ (for dynamic ceilings)
    """
    state_idx: int
    upper_bound: Any   # str (param name) or sp.Expr
    lower_bound: Any   # str (param name) or sp.Expr

@dataclass
class Saturation:
    """Quadratic saturation function on a state.

    SE(|x_i|) = B * (|x_i| - A)^2 / |x_i|   when |x_i| > A, else 0.

    The saturation adds to the *effective* dissipation for the state:
        dxdt[i] = ... - (KE + SE) * x_i / TE   (instead of just KE * x_i / TE)

    ``base_coeff`` is the existing linear dissipation coefficient param name
    (e.g. 'KE') — SE is added to it.
    ``time_const`` is the time constant param name (e.g. 'TE').
    """
    state_idx: int
    sat_A: str          # param name for threshold A
    sat_B: str          # param name for coefficient B
    base_coeff: str     # param name for KE  (SE adds to this)
    time_const: str     # param name for TE  (denominator)


@dataclass
class LeadLag:
    """Lead-lag compensator block on a state.

    Transfer function: (1 + s*TC) / (1 + s*TB)

    When TB > eps:
        dxdt[state_idx] = (input_expr - x_i) / TB
        output variable  = x_i + (TC/TB) * (input_expr - x_i)
    When TB <= eps (bypass):
        dxdt[state_idx] = 0
        output variable  = input_expr

    ``input_expr`` is a SymPy expression for the signal entering the block.
    ``output_var`` is the C variable name for the block's output.
    ``tc_param`` / ``tb_param`` are parameter name strings.
    ``downstream_state_idx`` and ``downstream_gain`` describe how the
    lead-lag output feeds into the next state's dynamics (optional).
    """
    state_idx: int
    input_expr: sp.Expr        # SymPy expression for block input
    output_var: str             # C variable name for LLx_out
    tc_param: str               # param name for TC (numerator time const)
    tb_param: str               # param name for TB (denominator time const)

@dataclass
class InitSpec:
    """Specification for auto-deriving ``init_from_targets()``.

    The generic equilibrium solver uses this to solve *dynamics_expr = 0*
    for the unknown states and free reference parameters given target
    constraints from the initialization pipeline.

    Attributes
    ----------
    target_states : dict
        Maps a *targets-dict key* to the state index that must equal
        that target value at equilibrium.  E.g. ``{'Efd': 2}`` means
        ``x[2] = targets['Efd']`` at steady state.
    input_bindings : dict
        Maps each input-symbol name (matching ``str(input_sym)``) to its
        source at init time:
        - ``str``  → key into the *targets* dict
        - ``float``/``int`` → literal value (e.g. ``u_agc = 0``)
        - ``None`` → this input is **free** (solved for)
    free_param_map : dict
        Maps each free input-symbol name to the component-parameter name
        that should be set to its solved value.
        E.g. ``{'V_{ref}': 'Vref'}`` means ``self.params['Vref'] = <solved>``.
    state_defaults : dict, optional
        Default values for states that are underdetermined at equilibrium
        (e.g. an integrator with zero error).  ``{state_idx: float}``.
    target_transforms : dict, optional
        ``{target_key: callable(val, targets) → state_val}`` for cases
        where the target value must be transformed before constraining
        the state (e.g. ``Vp = Efd / omega``).
    target_bounds : dict, optional
        ``{target_key: (lower_param, upper_param)}`` — clamp the target
        value to ``[params[lower], params[upper]]`` before solving.
    solver_func : callable, optional
        Custom equilibrium callback for components whose steady state is
        not practical to derive directly from ``dynamics_expr``. The
        callback signature is ``solver_func(targets, param_values)`` and
        must return ``(x0, free_params)``.
    post_init_func : callable, optional
        Generic post-solve callback for publishing any derived parameters
        back to the component after equilibrium is found. The callback
        signature is ``post_init_func(x0, free_params, targets, param_values)``
        and it must return a ``dict`` of extra parameter updates.
    """
    target_states: Dict[str, int]
    input_bindings: Dict[str, Union[str, float, int, None]]
    free_param_map: Dict[str, str]
    state_defaults: Dict[int, float] = field(default_factory=dict)
    target_transforms: Dict[str, Callable] = field(default_factory=dict)
    target_bounds: Dict[str, Tuple[str, str]] = field(default_factory=dict)
    solver_func: Optional[Callable] = None
    post_init_func: Optional[Callable] = None


class SymbolicPHS:
    """Symbolic Port-Hamiltonian System representation.

    Parameters
    ----------
    name : str
        Human-readable component name.
    states : list of Symbol
        Ordered state variables.
    inputs : list of Symbol
        Ordered input variables.
    params : dict of str → Symbol
        Named parameter symbols used in the matrices / Hamiltonian.
    J : Matrix
        Skew-symmetric interconnection matrix (n × n).
    R : Matrix
        Positive semi-definite dissipation matrix (n × n).
    g : Matrix
        Input coupling matrix (n × m).
    H : Expr
        Hamiltonian energy function H(x, params).
    Q : Matrix, optional
        Energy-variable transformation (n × n). Defaults to identity.
    description : str, optional
        Free-text description of the component.
    """

    def __init__(
        self,
        name: str,
        states: List[Symbol],
        inputs: List[Symbol],
        params: Dict[str, Symbol],
        J: Matrix,
        R: Matrix,
        g: Matrix,
        H: sp.Expr,
        Q: Optional[Matrix] = None,
        description: str = "",
        dynamics_expr: Optional[Matrix] = None,
    ):
        self.name = name
        self.states = list(states)
        self.inputs = list(inputs)
        self.params = dict(params)
        self.J = J
        self.R = R
        self.g = g
        self.H = H
        self.Q = Q if Q is not None else sp.eye(len(states))
        self.description = description
        self._dynamics_expr = dynamics_expr
        self.limiters: List[Limiter] = []
        self.saturations: List[Saturation] = []
        self.lead_lags: List[LeadLag] = []
        self.output_map: Optional[Dict[int, sp.Expr]] = None
        self.init_spec: Optional[InitSpec] = None

        n = len(states)
        m = len(inputs)
        assert J.shape == (n, n), f"J must be {n}×{n}, got {J.shape}"
        assert R.shape == (n, n), f"R must be {n}×{n}, got {R.shape}"
        assert g.shape == (n, m), f"g must be {n}×{m}, got {g.shape}"
        assert self.Q.shape == (n, n), f"Q must be {n}×{n}, got {self.Q.shape}"

    @property
    def n_states(self) -> int:
        return len(self.states)

    @property
    def n_inputs(self) -> int:
        return len(self.inputs)

    @property
    def grad_H(self) -> Matrix:
        """Symbolic gradient ∂H/∂x as a column vector."""
        return Matrix([diff(self.H, xi) for xi in self.states])

    @property
    def dynamics(self) -> Matrix:
        """Symbolic ẋ as a column vector.

        Returns the explicit dynamics override when provided, otherwise
        computes (J − R) Q ∂H/∂x + g · u from the PHS matrices.
        """
        if self._dynamics_expr is not None:
            return self._dynamics_expr
        u = Matrix(self.inputs)
        return (self.J - self.R) * self.Q * self.grad_H + self.g * u

    @property
    def phs_dynamics(self) -> Matrix:
        """Always returns (J − R) Q ∂H/∂x + g · u (ignores override).

        Use this for structural/energy analysis of the PHS matrices.
        """
        u = Matrix(self.inputs)
        return (self.J - self.R) * self.Q * self.grad_H + self.g * u

    @property
    def dissipation_rate(self) -> sp.Expr:
        """Symbolic −∇Hᵀ R ∇H (always ≤ 0 for valid PHS)."""
        dH = self.grad_H
        return -(dH.T * self.R * dH)[0, 0]

    @property
    def supply_rate(self) -> sp.Expr:
        """Symbolic port supply rate ∇Hᵀ g u = yᵀ u."""
        u = Matrix(self.inputs)
        dH = self.grad_H
        return (dH.T * self.g * u)[0, 0]

    @property
    def power_balance(self) -> sp.Expr:
        """Symbolic dH/dt = −∇Hᵀ R ∇H + ∇Hᵀ g u (passivity inequality)."""
        return simplify(self.dissipation_rate + self.supply_rate)

    # ------------------------------------------------------------------ #
    # Annotations (limiters, output maps)                                  #
    # ------------------------------------------------------------------ #

    def add_limiter(self, state_idx: int, upper_bound, lower_bound) -> None:
        """Declare an anti-windup limiter on ``dxdt[state_idx]``.

        Parameters
        ----------
        state_idx : int
            Index into the state vector.
        upper_bound, lower_bound
            Either a string (parameter name available in C++) or a SymPy
            expression that will be rendered as C++ code.
        """
        self.limiters.append(Limiter(state_idx, upper_bound, lower_bound))

    def add_saturation(self, state_idx: int, sat_A: str, sat_B: str,
                        base_coeff: str, time_const: str) -> None:
        """Declare quadratic saturation on ``x[state_idx]``.

        SE(|x|) = B*(|x|-A)^2/|x| when |x|>A, else 0.
        Effective dissipation becomes ``(base_coeff + SE) * x / time_const``.
        """
        self.saturations.append(Saturation(
            state_idx, sat_A, sat_B, base_coeff, time_const))

    def add_lead_lag(self, state_idx: int, input_expr: sp.Expr,
                     output_var: str, tc_param: str, tb_param: str) -> None:
        """Declare a lead-lag compensator block on ``x[state_idx]``.

        Transfer function: ``(1 + s*TC) / (1 + s*TB)``.
        When TB~0 the block is bypassed (output = input, dxdt = 0).
        """
        self.lead_lags.append(LeadLag(
            state_idx, input_expr, output_var, tc_param, tb_param))

    def set_output_map(self, mapping: Dict[int, sp.Expr]) -> None:
        """Declare output equations ``outputs[k] = expr(x, inputs, params)``.

        Parameters
        ----------
        mapping : dict  {output_index: sympy_expression}
            Each expression is written in terms of the component's state,
            input, and parameter symbols.
        """
        self.output_map = mapping

    def set_init_spec(
        self,
        target_states: Dict[str, int],
        input_bindings: Dict[str, Union[str, float, int, None]],
        free_param_map: Dict[str, str],
        state_defaults: Optional[Dict[int, float]] = None,
        target_transforms: Optional[Dict[str, Callable]] = None,
        target_bounds: Optional[Dict[str, Tuple[str, str]]] = None,
        solver_func: Optional[Callable] = None,
        post_init_func: Optional[Callable] = None,
    ) -> None:
        """Declare equilibrium specification for auto-deriving init.

        See :class:`InitSpec` for parameter descriptions.
        """
        self.init_spec = InitSpec(
            target_states=target_states,
            input_bindings=input_bindings,
            free_param_map=free_param_map,
            state_defaults=state_defaults or {},
            target_transforms=target_transforms or {},
            target_bounds=target_bounds or {},
            solver_func=solver_func,
            post_init_func=post_init_func,
        )

    def substitute_params(self, values: Dict[str, float]) -> "SymbolicPHS":
        """Return a new SymbolicPHS with parameter symbols replaced by floats."""
        subs = {}
        for k, v in values.items():
            if k in self.params:
                subs[self.params[k]] = v
        return SymbolicPHS(
            name=self.name,
            states=self.states,
            inputs=self.inputs,
            params={},
            J=self.J.subs(subs),
            R=self.R.subs(subs),
            g=self.g.subs(subs),
            H=self.H.subs(subs),
            Q=self.Q.subs(subs),
            description=self.description,
        )

    def __repr__(self) -> str:
        return (
            f"SymbolicPHS('{self.name}', "
            f"states={[str(s) for s in self.states]}, "
            f"inputs={[str(s) for s in self.inputs]}, "
            f"n_params={len(self.params)})"
        )
