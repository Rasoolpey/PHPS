"""
Symbolic-to-code generation for Port-Hamiltonian systems.

Bridges the symbolic PHS layer to the simulation pipeline, generating
C++ step code and Python callables from SymbolicPHS definitions.  This
ensures the simulation kernel and analysis tools are *provably consistent*
with the symbolic PHS structure definitions — the same symbolic matrices
are the single source of truth for all downstream artefacts.

Three categories of output:

  C++ dynamics code
      generate_phs_cpp_dynamics()   →  C++ body for get_cpp_step_code()
      generate_hamiltonian_cpp_expr()  →  C++ expression for H(x)

  Python callables (NumPy)
      make_hamiltonian_func()       →  H(x) → float
      make_grad_hamiltonian_func()  →  ∇H(x) → ndarray

  Numerical matrix evaluation
      evaluate_phs_matrices()       →  {J, R, g, Q} as ndarrays
"""

import math
import re
import numpy as np
import sympy as sp
from sympy.printing.c import C99CodePrinter
from typing import Dict, List, Optional, Callable

from src.symbolic.core import SymbolicPHS


# -------------------------------------------------------------------- #
# Custom C code printer                                                  #
# -------------------------------------------------------------------- #

class _PHSCCodePrinter(C99CodePrinter):
    """C99 code printer tuned for PHS dynamics."""

    def __init__(self):
        super().__init__(settings={'precision': 17})

    def _print_Pow(self, expr):
        # x**2 → (x)*(x), x**(-1) → (1.0/(x))
        if expr.exp == 2:
            base = self._print(expr.base)
            return f"(({base})*({base}))"
        if expr.exp == -1:
            base = self._print(expr.base)
            return f"(1.0/({base}))"
        if expr.exp == sp.Rational(1, 2):
            base = self._print(expr.base)
            return f"sqrt({base})"
        return super()._print_Pow(expr)


_printer = _PHSCCodePrinter()


# -------------------------------------------------------------------- #
# C++ code generation                                                    #
# -------------------------------------------------------------------- #

def generate_phs_cpp_dynamics(
    sphs: SymbolicPHS,
    state_c_names: List[str],
    input_c_names: List[str],
    param_c_names: Optional[Dict[str, str]] = None,
) -> str:
    """Generate C++ code for ẋ = (J − R) Q ∇H + g · u.

    Parameters
    ----------
    sphs : SymbolicPHS
        Symbolic PHS definition.
    state_c_names : list of str
        C++ variable names for states, ordered (e.g. ``['Vm', 'Vr', 'Efd', 'Xf']``).
    input_c_names : list of str
        C++ variable names for inputs, ordered (e.g. ``['Vterm', 'Vref']``).
    param_c_names : dict, optional
        Mapping ``{param_key: c_name}``.  Defaults to using param keys as-is.

    Returns
    -------
    str
        C++ code block that declares local state/input variables, then
        assigns ``dxdt[i]`` for each state from the PHS dynamics.
    """
    if param_c_names is None:
        param_c_names = {k: k for k in sphs.params}

    # Build substitution: SymPy symbol → C-safe SymPy symbol
    subs = {}
    for sym, c_name in zip(sphs.states, state_c_names):
        subs[sym] = sp.Symbol(c_name)
    for sym, c_name in zip(sphs.inputs, input_c_names):
        subs[sym] = sp.Symbol(c_name)
    for key, sym in sphs.params.items():
        if key in param_c_names:
            subs[sym] = sp.Symbol(param_c_names[key])

    # Compute symbolic dynamics
    dynamics = sphs.dynamics  # (J - R) Q ∇H + g · u

    lines = []
    lines.append(f"// PHS dynamics: dx/dt = (J - R) Q grad_H + g * u")
    lines.append(f"// Auto-generated from SymbolicPHS '{sphs.name}'")
    lines.append("")

    # Extract states from x[]
    for i, name in enumerate(state_c_names):
        lines.append(f"double {name} = x[{i}];")
    lines.append("")

    # Extract inputs from inputs[]
    for i, name in enumerate(input_c_names):
        lines.append(f"double {name} = inputs[{i}];")
    lines.append("")

    # Build set of limited state indices for easy lookup
    limited_indices = {lim.state_idx: lim for lim in sphs.limiters}
    saturated_indices = {sat.state_idx: sat for sat in sphs.saturations}
    lead_lag_indices = {ll.state_idx: ll for ll in sphs.lead_lags}

    # --- Scan for parameter symbols in denominators (1/T terms) ---
    # IEEE convention: T=0 means instantaneous (bypass), implemented as
    # T_eff = max(T, 1e-4).  The saturation and lead-lag codegen emit their
    # own guards; this handles standard dynamics and lead-lag input_expr.
    all_c_param_syms = set()
    for key, sym in sphs.params.items():
        if key in param_c_names:
            all_c_param_syms.add(sp.Symbol(param_c_names[key]))

    def _find_denom_params(expr):
        found = set()
        for atom in sp.preorder_traversal(expr):
            if isinstance(atom, sp.Pow) and atom.exp < 0:
                if atom.base in all_c_param_syms:
                    found.add(atom.base)
        return found

    denom_params = set()
    for i in range(sphs.n_states):
        if i in lead_lag_indices or i in saturated_indices:
            continue
        denom_params |= _find_denom_params(dynamics[i].subs(subs))
    for ll in sphs.lead_lags:
        denom_params |= _find_denom_params(ll.input_expr.subs(subs))

    eff_subs = {}
    for dp in sorted(denom_params, key=str):
        dp_name = str(dp)
        eff_name = f"{dp_name}_eff"
        lines.append(f"double {eff_name} = ({dp_name} > 1e-4) ? {dp_name} : 1e-4;")
        eff_subs[dp] = sp.Symbol(eff_name)
    if eff_subs:
        lines.append("")

    # Pre-emit saturation helper variables
    for sat in sphs.saturations:
        sname = state_c_names[sat.state_idx]
        lines.append(f"// Saturation on {sname}")
        lines.append(f"double {sname}_abs = ({sname} > 0.0) ? {sname} : -{sname};")
        lines.append(f"double SE_{sname} = 0.0;")
        lines.append(f"if ({sat.sat_B} > 0.0 && {sname}_abs > {sat.sat_A}) {{")
        lines.append(f"    SE_{sname} = {sat.sat_B} * ({sname}_abs - {sat.sat_A}) * ({sname}_abs - {sat.sat_A}) / {sname}_abs;")
        lines.append(f"}}")
        lines.append("")

    # Pre-emit lead-lag blocks (must compute output variables before dynamics)
    for ll in sphs.lead_lags:
        ll_input_cpp = _printer.doprint(ll.input_expr.subs(subs))
        sname = state_c_names[ll.state_idx]
        lines.append(f"// Lead-lag block on {sname}")
        lines.append(f"double {ll.output_var};")
        lines.append("")

    # Generate dxdt assignments
    for i in range(sphs.n_states):
        sname = state_c_names[i]
        lines.append(f"// d{sname}/dt")

        # --- Lead-lag: overrides the entire dxdt generation for this state ---
        if i in lead_lag_indices:
            ll = lead_lag_indices[i]
            ll_input_cpp = _printer.doprint(ll.input_expr.subs(subs).subs(eff_subs))
            tb = ll.tb_param
            tc = ll.tc_param
            lines.append(f"if ({tb} > 1e-4) {{")
            lines.append(f"    dxdt[{i}] = ({ll_input_cpp} - {sname}) / {tb};")
            lines.append(f"    {ll.output_var} = {sname} + ({tc} / {tb}) * ({ll_input_cpp} - {sname});")
            lines.append(f"}} else {{")
            lines.append(f"    dxdt[{i}] = 0.0;")
            lines.append(f"    {ll.output_var} = {ll_input_cpp};")
            lines.append(f"}}")
            continue

        # --- Saturation: replace the PHS-derived dynamics with the correct
        #     formula that includes the SE(x) term ---
        if i in saturated_indices:
            sat = saturated_indices[i]
            # The dynamics for the saturated state should be the regulator
            # output minus the effective dissipation:
            #   dxdt[i] = (Vr - (KE + SE) * x_i) / TE
            # We need to find the regulator input for this state.
            # The PHS dynamics = (Vr - KE*Efd) / TE, so we extract Vr
            # by adding back the KE*Efd/TE term to the base dynamics.
            KE_sym = subs.get(sphs.params.get(sat.base_coeff), sp.Symbol(sat.base_coeff))
            TE_sym = subs.get(sphs.params.get(sat.time_const), sp.Symbol(sat.time_const))
            base_dyn = dynamics[i].subs(subs)
            # base_dyn = (Vr - KE*x_i) / TE  → Vr_term = base_dyn * TE + KE*x_i
            x_i_sym = subs[sphs.states[i]]
            Vr_term = sp.expand(base_dyn * TE_sym + KE_sym * x_i_sym)
            Vr_cpp = _printer.doprint(Vr_term)
            te_cpp = sat.time_const
            te_eff = f"{te_cpp}_eff"
            lines.append(f"double {te_eff} = ({te_cpp} > 1e-4) ? {te_cpp} : 1e-4;")
            cpp_expr = f"({Vr_cpp} - ({sat.base_coeff} + SE_{sname}) * {sname}) / {te_eff}"

            if i in limited_indices:
                lim = limited_indices[i]
                ub = _render_bound(lim.upper_bound, subs)
                lb = _render_bound(lim.lower_bound, subs)
                var = f"_dx{i}_raw"
                lines.append(f"double {var} = {cpp_expr};")
                lines.append(f"if ({sname} >= {ub} && {var} > 0.0) {var} = 0.0;")
                lines.append(f"if ({sname} <= {lb} && {var} < 0.0) {var} = 0.0;")
                lines.append(f"dxdt[{i}] = {var};")
            else:
                lines.append(f"dxdt[{i}] = {cpp_expr};")
            continue

        # --- Standard: PHS dynamics with optional limiter ---
        expr = dynamics[i].subs(subs).subs(eff_subs)
        expr = sp.expand(expr)
        cpp_expr = _printer.doprint(expr)

        if i in limited_indices:
            lim = limited_indices[i]
            ub = _render_bound(lim.upper_bound, subs)
            lb = _render_bound(lim.lower_bound, subs)
            var = f"_dx{i}_raw"
            lines.append(f"double {var} = {cpp_expr};")
            lines.append(f"if ({sname} >= {ub} && {var} > 0.0) {var} = 0.0;")
            lines.append(f"if ({sname} <= {lb} && {var} < 0.0) {var} = 0.0;")
            lines.append(f"dxdt[{i}] = {var};")
        else:
            lines.append(f"dxdt[{i}] = {cpp_expr};")

    return "\n".join(lines)


def _render_bound(bound, subs: dict) -> str:
    """Render a limiter bound as a C++ expression string.

    A string is passed through as-is (parameter name).
    A SymPy expression is substituted and printed as C++.
    """
    if isinstance(bound, str):
        return bound
    # SymPy expression — apply the same symbol substitution
    return _printer.doprint(sp.expand(bound.subs(subs)))


def generate_phs_cpp_outputs(
    sphs: SymbolicPHS,
    state_c_names: List[str],
    input_c_names: List[str],
    param_c_names: Optional[Dict[str, str]] = None,
) -> str:
    """Generate C++ code for ``outputs[k] = expr`` from ``sphs.output_map``.

    Returns
    -------
    str
        C++ code block, or empty string if no output_map is defined.
    """
    if sphs.output_map is None:
        return ""

    if param_c_names is None:
        param_c_names = {k: k for k in sphs.params}

    # Build substitution: SymPy symbol → C-safe SymPy symbol
    subs = {}
    for sym, c_name in zip(sphs.states, state_c_names):
        subs[sym] = sp.Symbol(c_name)
    for sym, c_name in zip(sphs.inputs, input_c_names):
        subs[sym] = sp.Symbol(c_name)
    for key, sym in sphs.params.items():
        if key in param_c_names:
            subs[sym] = sp.Symbol(param_c_names[key])

    lines = []
    lines.append(f"// Output map — auto-generated from SymbolicPHS '{sphs.name}'")

    # Extract states from x[]
    for i, name in enumerate(state_c_names):
        lines.append(f"double {name} = x[{i}];")

    # Extract inputs only if any output expression uses them
    all_output_syms = set()
    for expr in sphs.output_map.values():
        all_output_syms |= expr.free_symbols
    needs_inputs = any(s in all_output_syms for s in sphs.inputs)
    if needs_inputs:
        for i, name in enumerate(input_c_names):
            lines.append(f"double {name} = inputs[{i}];")

    for k in sorted(sphs.output_map):
        expr = sphs.output_map[k].subs(subs)
        cpp_expr = _printer.doprint(sp.expand(expr))
        lines.append(f"outputs[{k}] = {cpp_expr};")

    return "\n".join(lines)


def generate_hamiltonian_cpp_expr(
    sphs: SymbolicPHS,
    param_values: Dict[str, float],
    x_var: str = "x",
) -> str:
    """Generate a C++ expression string for H(x) with parameters substituted.

    Returns a parenthesised expression suitable for embedding in C++ code
    such as an ``observables`` ``cpp_expr`` field.
    """
    # Substitute parameter values
    param_subs = {}
    for key, sym in sphs.params.items():
        if key in param_values:
            val = param_values[key]
            if isinstance(val, (int, float)):
                param_subs[sym] = float(val)

    H_concrete = sphs.H.subs(param_subs)

    # Replace state symbols with indexed placeholders
    placeholders = [sp.Symbol(f'__s{i}__') for i in range(sphs.n_states)]
    H_sub = H_concrete.subs(list(zip(sphs.states, placeholders)))

    cpp = _printer.doprint(sp.expand(H_sub))

    # Replace placeholders with x[i]
    for i in range(sphs.n_states):
        cpp = cpp.replace(f'__s{i}__', f'{x_var}[{i}]')

    return f"({cpp})"


def _symbolic_to_cpp_expr(
    sphs: SymbolicPHS,
    expr: sp.Expr,
    param_values: Dict[str, float],
    x_var: str = "x",
    input_var: str = "inputs",
) -> str:
    """Convert a symbolic expression (in states, inputs, params) to C++.

    Parameters are numerically substituted; state and input symbols are
    replaced with indexed array references.  Positive-declared parameters
    are clamped to 1e-4 (matching the dynamics code guard for T=0 bypass).
    """
    # Use a floor of 1e-4 for positive params (matching dynamics guards)
    param_subs = {}
    for key, sym in sphs.params.items():
        if key in param_values:
            val = param_values[key]
            if isinstance(val, (int, float)):
                if getattr(sym, 'is_positive', False) and val <= 0.0:
                    val = 1e-4
                param_subs[sym] = float(val)

    concrete = expr.subs(param_subs)

    # Replace state and input symbols with indexed placeholders
    subs = {}
    for i, sym in enumerate(sphs.states):
        subs[sym] = sp.Symbol(f'__s{i}__')
    for i, sym in enumerate(sphs.inputs):
        subs[sym] = sp.Symbol(f'__u{i}__')

    concrete = concrete.subs(subs)
    cpp = _printer.doprint(sp.expand(concrete))

    for i in range(sphs.n_states):
        cpp = cpp.replace(f'__s{i}__', f'{x_var}[{i}]')
    for i in range(sphs.n_inputs):
        cpp = cpp.replace(f'__u{i}__', f'{input_var}[{i}]')

    return f"({cpp})"


def generate_dissipation_cpp_expr(
    sphs: SymbolicPHS,
    param_values: Dict[str, float],
    x_var: str = "x",
) -> str:
    """Generate C++ expression for −∇Hᵀ R ∇H (dissipation rate, ≤ 0).

    Decomposes per-state since R is diagonal.  For states where R[i,i]
    evaluates to infinity (because a time-constant parameter is 0, meaning
    IEEE "bypass"), the term is emitted as 0 — matching the dynamics code
    which sets dx/dt = 0 for bypassed states.
    """
    n = sphs.n_states
    dH = sphs.grad_H  # n×1 column vector

    # Build param substitution (actual values, including zeros)
    raw_param_subs = {}
    for key, sym in sphs.params.items():
        if key in param_values:
            val = param_values[key]
            if isinstance(val, (int, float)):
                raw_param_subs[sym] = float(val)

    terms = []
    for i in range(n):
        Rii = sphs.R[i, i]
        if Rii == 0:
            continue
        # Evaluate R[i,i] numerically to detect bypass (infinity)
        Rii_sub = Rii.subs(raw_param_subs)
        if Rii_sub.free_symbols:
            # R[i,i] still has unresolved symbols (e.g. omega_s string param)
            # — cannot be a bypass, include normally
            pass
        else:
            Rii_eval = Rii_sub.evalf()
            try:
                Rii_val = complex(Rii_eval).real
            except (TypeError, ValueError):
                Rii_val = 0.0
            if not math.isfinite(Rii_val):
                # Bypass state: T=0 → R[i,i]=∞ → actual dx/dt=0 → dissipation=0
                continue
        # Normal term: -(∂H/∂x_i)² * R[i,i]
        term = -(dH[i] ** 2) * Rii
        terms.append(term)

    if not terms:
        return "(0.0)"

    total = sp.Add(*terms)
    return _symbolic_to_cpp_expr(sphs, total, param_values, x_var=x_var)


def generate_supply_rate_cpp_expr(
    sphs: SymbolicPHS,
    param_values: Dict[str, float],
    x_var: str = "x",
    input_var: str = "inputs",
) -> str:
    """Generate C++ expression for ∇Hᵀ g u (port supply rate).

    This is the power flowing into the component through its ports.
    """
    return _symbolic_to_cpp_expr(sphs, sphs.supply_rate, param_values,
                                 x_var=x_var, input_var=input_var)


# -------------------------------------------------------------------- #
# Python callable generation (NumPy)                                     #
# -------------------------------------------------------------------- #

def make_hamiltonian_func(
    sphs: SymbolicPHS,
    param_values: Dict[str, float],
) -> Callable[[np.ndarray], float]:
    """Create a fast Python callable ``H(x) → float`` from symbolic H.

    Parameters are baked in; only the state vector varies at call time.
    """
    param_subs = _param_subs(sphs, param_values)
    H_concrete = sphs.H.subs(param_subs)
    f = sp.lambdify(sphs.states, H_concrete, modules='numpy')

    def hamiltonian(x: np.ndarray) -> float:
        return float(f(*x))

    return hamiltonian


def make_grad_hamiltonian_func(
    sphs: SymbolicPHS,
    param_values: Dict[str, float],
) -> Callable[[np.ndarray], np.ndarray]:
    """Create a fast Python callable ``∇H(x) → ndarray`` from symbolic H."""
    param_subs = _param_subs(sphs, param_values)
    grad = sphs.grad_H  # Column vector of partial derivatives
    grad_concrete = grad.subs(param_subs)

    funcs = []
    for i in range(sphs.n_states):
        f = sp.lambdify(sphs.states, grad_concrete[i], modules='numpy')
        funcs.append(f)

    def grad_hamiltonian(x: np.ndarray) -> np.ndarray:
        return np.array([f(*x) for f in funcs])

    return grad_hamiltonian


# -------------------------------------------------------------------- #
# Numerical matrix evaluation                                            #
# -------------------------------------------------------------------- #

def evaluate_phs_matrices(
    sphs: SymbolicPHS,
    param_values: Dict[str, float],
    x: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """Evaluate PHS matrices {J, R, g, Q} numerically.

    Parameters are substituted from *param_values*.  If any matrix
    contains state symbols and *x* is provided, state values are
    substituted as well.
    """
    subs = _param_subs(sphs, param_values)

    # Substitute state values if matrices depend on x
    if x is not None:
        for i, sym in enumerate(sphs.states):
            subs[sym] = float(x[i])

    n, m = sphs.n_states, sphs.n_inputs
    J = np.array(sphs.J.subs(subs).evalf().tolist(), dtype=float).reshape(n, n)
    R = np.array(sphs.R.subs(subs).evalf().tolist(), dtype=float).reshape(n, n)
    g = np.array(sphs.g.subs(subs).evalf().tolist(), dtype=float).reshape(n, m)
    Q = np.array(sphs.Q.subs(subs).evalf().tolist(), dtype=float).reshape(n, n)
    return {'J': J, 'R': R, 'g': g, 'Q': Q}


# -------------------------------------------------------------------- #
# Internal helpers                                                       #
# -------------------------------------------------------------------- #

def _param_subs(sphs: SymbolicPHS, param_values: Dict[str, float]) -> dict:
    """Build a SymPy substitution dict from param_values.

    Parameters declared as ``positive=True`` are clamped to a small
    epsilon (1e-10) if the supplied value is zero or negative, avoiding
    division-by-zero when the symbol appears in a denominator (e.g.
    ``1/TB`` with ``TB=0``).
    """
    subs = {}
    for key, sym in sphs.params.items():
        if key in param_values:
            val = param_values[key]
            if isinstance(val, (int, float)):
                # Clamp positive parameters away from zero
                if getattr(sym, 'is_positive', False) and val <= 0.0:
                    val = 1e-10
                subs[sym] = float(val)
    return subs


# -------------------------------------------------------------------- #
# Generic equilibrium solver (A6)                                        #
# -------------------------------------------------------------------- #

def _resolve_param(val):
    """Convert a param value to float, evaluating string expressions."""
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        return float(eval(val, {"M_PI": math.pi, "math": math, "__builtins__": {}}))
    return float(val)


def _resolve_bound(bound, param_values: Dict[str, any], default: float) -> float:
    """Resolve a limiter bound from either a param name or literal string."""
    if isinstance(bound, str) and bound in param_values:
        return _resolve_param(param_values[bound])
    if bound is None:
        return default
    return _resolve_param(bound)


def solve_equilibrium(
    sphs: SymbolicPHS,
    param_values: Dict[str, any],
    targets: Dict[str, float],
) -> tuple:
    """Solve *dynamics_expr = 0* for unknown states and free reference params.

    Uses the ``InitSpec`` declared on *sphs* to determine which inputs
    and states are known from *targets*, which states are constrained by
    target values, and which input symbols are free (to be solved for as
    reference parameters like Vref or Pref).

    Parameters
    ----------
    sphs : SymbolicPHS
        Must have ``dynamics`` (``dynamics_expr`` or PHS matrices) and
        ``init_spec`` set.
    param_values : dict
        Component parameter values (``self.params``).
    targets : dict
        Initialization targets from the generator (Efd, Tm, Vt, omega, …).

    Returns
    -------
    x0 : np.ndarray
        Equilibrium state vector.
    free_params : dict
        Solved reference parameters, e.g. ``{'Vref': 1.05}``.
    """
    spec = sphs.init_spec

    if spec.solver_func is not None:
        return spec.solver_func(targets, param_values)

    n = sphs.n_states
    dyn = sphs.dynamics.copy()

    # ---- Step 1: At equilibrium, lead-lag output = input_expr ----------
    for ll in sphs.lead_lags:
        out_sym = sp.Symbol(ll.output_var)
        dyn = dyn.subs(out_sym, ll.input_expr)

    # ---- Step 2: Compute target state values (with transforms) ---------
    target_state_values = {}
    for tkey, state_idx in spec.target_states.items():
        val = float(targets.get(tkey, 0.0))
        # Apply target bounds (pre-solve clamping)
        if tkey in spec.target_bounds:
            lo_param, hi_param = spec.target_bounds[tkey]
            lo = _resolve_param(param_values.get(lo_param, -1e6))
            hi = _resolve_param(param_values.get(hi_param, 1e6))
            val = max(min(val, hi), lo)
        # Apply target transform
        if tkey in spec.target_transforms:
            val = spec.target_transforms[tkey](val, targets)
        target_state_values[state_idx] = val

    # ---- Step 3: Handle saturation — compute SE, modify base_coeff -----
    sat_adjustments = {}  # param Symbol → effective value (base + SE)
    for sat in sphs.saturations:
        if sat.state_idx not in target_state_values:
            continue
        x_val = target_state_values[sat.state_idx]
        sat_A_val = _resolve_param(param_values.get(sat.sat_A, 0.0))
        sat_B_val = _resolve_param(param_values.get(sat.sat_B, 0.0))
        SE = 0.0
        if sat_B_val > 0.0 and abs(x_val) > sat_A_val:
            SE = sat_B_val * (abs(x_val) - sat_A_val) ** 2 / abs(x_val)
        base_sym = sphs.params.get(sat.base_coeff)
        if base_sym is not None:
            base_val = _resolve_param(param_values.get(sat.base_coeff, 0.0))
            sat_adjustments[base_sym] = base_val + SE

    # ---- Step 4: Build substitution dict --------------------------------
    subs = {}

    # Parameters (with saturation adjustments)
    for pname, psym in sphs.params.items():
        if psym in sat_adjustments:
            subs[psym] = sat_adjustments[psym]
        elif pname in param_values:
            subs[psym] = _resolve_param(param_values[pname])

    # Known inputs from targets or literals
    for input_sym in sphs.inputs:
        sym_name = str(input_sym)
        binding = spec.input_bindings.get(sym_name)
        if binding is None:
            continue  # free — don't substitute
        elif isinstance(binding, str):
            subs[input_sym] = float(targets.get(binding, 0.0))
        else:
            subs[input_sym] = float(binding)

    # Target state constraints
    for state_idx, val in target_state_values.items():
        subs[sphs.states[state_idx]] = val

    # ---- Step 5: Substitute and simplify --------------------------------
    dyn_sub = dyn.subs(subs)

    # ---- Step 6: Collect unknowns ---------------------------------------
    unknowns = []
    for i, sym in enumerate(sphs.states):
        if sym not in subs:
            unknowns.append(sym)
    free_input_syms = {}  # param_name → input Symbol
    for input_sym in sphs.inputs:
        sym_name = str(input_sym)
        if spec.input_bindings.get(sym_name) is None:  # free
            unknowns.append(input_sym)
            for isym_name, pname in spec.free_param_map.items():
                if isym_name == sym_name:
                    free_input_syms[pname] = input_sym

    # ---- Step 7: Build equation list (skip trivial 0 = 0) --------------
    eqs = []
    for i in range(n):
        expr = sp.nsimplify(dyn_sub[i], rational=False)
        if expr != 0:
            eqs.append(expr)

    # ---- Step 8: Solve --------------------------------------------------
    solution = {}
    if eqs and unknowns:
        sol = sp.solve(eqs, unknowns, dict=True)
        if sol:
            solution = sol[0]

    # ---- Step 9: Build state array --------------------------------------
    x0 = np.zeros(n)
    for i, sym in enumerate(sphs.states):
        if sym in subs:
            x0[i] = float(subs[sym])
        elif sym in solution:
            x0[i] = float(solution[sym])
        elif i in spec.state_defaults:
            x0[i] = spec.state_defaults[i]

    # ---- Step 10: Clamp limited states and re-solve if needed -----------
    clamped = False
    for lim in sphs.limiters:
        ub = _resolve_bound(lim.upper_bound, param_values, 1e6)
        lb = _resolve_bound(lim.lower_bound, param_values, -1e6)
        if x0[lim.state_idx] > ub:
            x0[lim.state_idx] = ub
            clamped = True
        elif x0[lim.state_idx] < lb:
            x0[lim.state_idx] = lb
            clamped = True

    if clamped:
        # Re-solve for free params with clamped state values pinned
        extra_subs = {}
        for lim in sphs.limiters:
            sym = sphs.states[lim.state_idx]
            if sym not in subs:
                extra_subs[sym] = x0[lim.state_idx]
        if extra_subs:
            dyn_re = dyn_sub.subs(extra_subs)
            # Also substitute the newly-known non-limited states
            known_state_subs = {}
            for i, sym in enumerate(sphs.states):
                if sym not in subs and sym not in extra_subs and sym in solution:
                    known_state_subs[sym] = x0[i]
            dyn_re = dyn_re.subs(known_state_subs)
            re_unknowns = [s for s in unknowns
                           if s not in extra_subs and s not in known_state_subs]
            re_eqs = [sp.nsimplify(dyn_re[i], rational=False)
                      for i in range(n) if sp.nsimplify(dyn_re[i], rational=False) != 0]
            if re_eqs and re_unknowns:
                re_sol = sp.solve(re_eqs, re_unknowns, dict=True)
                if re_sol:
                    solution.update(re_sol[0])

    # ---- Step 11: Extract free param values -----------------------------
    free_params = {}
    for pname, input_sym in free_input_syms.items():
        if input_sym in solution:
            free_params[pname] = float(solution[input_sym])

    return x0, free_params
