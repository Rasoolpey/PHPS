"""
PHS structural validation — machine-verifiable checks.

Validates that a SymbolicPHS satisfies the fundamental PHS axioms:

  1. Skew-symmetry:   J + Jᵀ = 0
  2. Positive semi-definiteness:  R = Rᵀ, eigenvalues(R) ≥ 0
  3. Passivity:  dH/dt = −∇Hᵀ R ∇H + ∇Hᵀ g u  (structural)
  4. Energy symmetry: Q = Qᵀ (if non-identity)

Returns a structured report with pass/fail and diagnostic messages.
"""

import numpy as np
import sympy as sp
from sympy import Matrix, simplify, zeros, eye
from typing import List, Optional
from src.symbolic.core import SymbolicPHS


class PHSValidationReport:
    """Structured report from PHS structural validation."""

    def __init__(self, component_name: str):
        self.component_name = component_name
        self.checks: List[dict] = []

    def add_check(self, name: str, passed: bool, detail: str = ""):
        self.checks.append({"name": name, "passed": passed, "detail": detail})

    @property
    def all_passed(self) -> bool:
        return all(c["passed"] for c in self.checks)

    @property
    def n_passed(self) -> int:
        return sum(1 for c in self.checks if c["passed"])

    @property
    def n_failed(self) -> int:
        return sum(1 for c in self.checks if not c["passed"])

    def summary(self) -> str:
        lines = [f"PHS Validation Report: {self.component_name}"]
        lines.append("=" * 60)
        for c in self.checks:
            status = "PASS" if c["passed"] else "FAIL"
            lines.append(f"  [{status}] {c['name']}")
            if c["detail"]:
                for dl in c["detail"].split("\n"):
                    lines.append(f"         {dl}")
        lines.append("-" * 60)
        lines.append(
            f"Result: {self.n_passed}/{len(self.checks)} checks passed"
            + (" -- ALL OK" if self.all_passed else " -- FAILURES DETECTED")
        )
        return "\n".join(lines)


def validate_phs_structure(
    sphs: SymbolicPHS,
    check_eigenvalues: bool = True,
    simplify_level: int = 1,
) -> PHSValidationReport:
    """Validate the structural properties of a SymbolicPHS.

    Parameters
    ----------
    sphs : SymbolicPHS
        The symbolic PHS to validate.
    check_eigenvalues : bool
        If True, attempt to compute eigenvalues of R symbolically.
        Can be slow for large matrices with many parameters.
    simplify_level : int
        0: no simplification, 1: simplify (default), 2: trigsimp + simplify.

    Returns
    -------
    PHSValidationReport
        Structured report with pass/fail for each check.
    """
    report = PHSValidationReport(sphs.name)
    n = sphs.n_states

    def _simplify(expr):
        if simplify_level == 0:
            return expr
        elif simplify_level >= 2:
            return sp.trigsimp(simplify(expr))
        return simplify(expr)

    # --- Check 1: J dimensions ---
    report.add_check(
        "J dimensions",
        sphs.J.shape == (n, n),
        f"Expected ({n}, {n}), got {sphs.J.shape}",
    )

    # --- Check 2: Skew-symmetry  J + Jᵀ = 0 ---
    skew_test = _simplify(sphs.J + sphs.J.T)
    is_skew = skew_test.equals(zeros(n, n))
    detail = ""
    if not is_skew:
        nonzero = []
        for i in range(n):
            for j in range(n):
                entry = _simplify(skew_test[i, j])
                if entry != 0:
                    nonzero.append(f"  (J + Jᵀ)[{i},{j}] = {entry}")
        detail = "\n".join(nonzero[:10])  # limit output
    report.add_check("J skew-symmetry (J + Jᵀ = 0)", is_skew, detail)

    # --- Check 3: R dimensions ---
    report.add_check(
        "R dimensions",
        sphs.R.shape == (n, n),
        f"Expected ({n}, {n}), got {sphs.R.shape}",
    )

    # --- Check 4: R symmetry  R = Rᵀ ---
    sym_test = _simplify(sphs.R - sphs.R.T)
    is_sym = sym_test.equals(zeros(n, n))
    detail = ""
    if not is_sym:
        nonzero = []
        for i in range(n):
            for j in range(n):
                entry = _simplify(sym_test[i, j])
                if entry != 0:
                    nonzero.append(f"  (R − Rᵀ)[{i},{j}] = {entry}")
        detail = "\n".join(nonzero[:10])
    report.add_check("R symmetry (R = Rᵀ)", is_sym, detail)

    # --- Check 5: R positive semi-definiteness (eigenvalues ≥ 0) ---
    if check_eigenvalues:
        try:
            eigs = sphs.R.eigenvals()
            all_nonneg = True
            detail_lines = []
            for eig, mult in eigs.items():
                eig_s = _simplify(eig)
                detail_lines.append(f"  λ = {eig_s} (multiplicity {mult})")
                # For symbolic eigenvalues, check if obviously negative
                if eig_s.is_negative:
                    all_nonneg = False
            detail = "\n".join(detail_lines)
            report.add_check("R positive semi-definite (eigenvalues ≥ 0)",
                             all_nonneg, detail)
        except Exception as e:
            report.add_check(
                "R positive semi-definite (eigenvalues ≥ 0)",
                False,
                f"Could not compute eigenvalues: {e}\n"
                "  (Diagonal entries checked instead)",
            )
            # Fallback: check diagonal entries
            diag_ok = True
            for i in range(n):
                if _simplify(sphs.R[i, i]).is_negative:
                    diag_ok = False
            report.add_check("R diagonal entries ≥ 0", diag_ok, "")
    else:
        # Quick check: diagonal entries only
        diag_ok = True
        detail_lines = []
        for i in range(n):
            entry = _simplify(sphs.R[i, i])
            detail_lines.append(f"  R[{i},{i}] = {entry}")
            if entry.is_negative:
                diag_ok = False
        report.add_check("R diagonal entries ≥ 0 (eigenvalue check skipped)",
                         diag_ok, "\n".join(detail_lines))

    # --- Check 6: g dimensions ---
    m = sphs.n_inputs
    report.add_check(
        "g dimensions",
        sphs.g.shape == (n, m),
        f"Expected ({n}, {m}), got {sphs.g.shape}",
    )

    # --- Check 7: Q symmetry (if non-identity) ---
    if not sphs.Q.equals(eye(n)):
        q_sym = _simplify(sphs.Q - sphs.Q.T)
        is_q_sym = q_sym.equals(zeros(n, n))
        report.add_check("Q symmetry (Q = Qᵀ)", is_q_sym, "")
    else:
        report.add_check("Q = Identity", True, "")

    # --- Check 8: Hamiltonian gradient consistency ---
    grad_H = sphs.grad_H
    detail_lines = []
    for i, xi in enumerate(sphs.states):
        detail_lines.append(f"  ∂H/∂{xi} = {grad_H[i]}")
    report.add_check(
        "Hamiltonian gradient computed",
        True,
        "\n".join(detail_lines),
    )

    # --- Check 9: Power balance structure ---
    # dH/dt = ∇Hᵀ ẋ = ∇Hᵀ (J−R) Q ∇H + ∇Hᵀ g u
    # Since Jᵀ = −J:  ∇Hᵀ J ∇H = 0  →  dH/dt = −∇Hᵀ R ∇H + ∇Hᵀ g u
    # Verify ∇Hᵀ J Q ∇H = 0 symbolically
    dH = sphs.grad_H
    skew_power = _simplify((dH.T * sphs.J * sphs.Q * dH)[0, 0])
    report.add_check(
        "Power conservation (∇Hᵀ J Q ∇H = 0)",
        skew_power == 0,
        f"∇Hᵀ J Q ∇H = {skew_power}" if skew_power != 0 else "",
    )

    return report


def validate_phs_passivity(
    sphs: SymbolicPHS,
    param_values: dict,
    x0: "np.ndarray | None" = None,
    u0: "np.ndarray | None" = None,
) -> PHSValidationReport:
    """Design-time numerical passivity check at an operating point.

    Evaluates the PHS matrices with concrete parameter values and (optionally)
    a state vector *x0* / input vector *u0*, then checks:

      1. R ≥ 0  (eigenvalues of the numerical R matrix)
      2. Dissipation ≤ 0  at the operating point  (−∇H^T R ∇H)
      3. H(x0) > 0  (positive energy)
      4. Per-state dissipation breakdown  (diagnostic, always passes)

    This is intended for component design validation, **not** runtime logging.

    Parameters
    ----------
    sphs : SymbolicPHS
    param_values : dict
        ``{param_name: float}`` — must cover all symbolic parameters.
    x0 : ndarray, optional
        State vector at operating point.  Defaults to zeros.
    u0 : ndarray, optional
        Input vector at operating point.  Defaults to zeros.
    """
    import math
    import numpy as np
    from src.symbolic.codegen import (
        evaluate_phs_matrices,
        make_hamiltonian_func,
        make_grad_hamiltonian_func,
    )

    report = PHSValidationReport(sphs.name)
    n = sphs.n_states
    m = sphs.n_inputs

    if x0 is None:
        x0 = np.zeros(n)
    if u0 is None:
        u0 = np.zeros(m)

    # Resolve string-valued params (e.g. "2.0 * M_PI * 60.0") to floats
    resolved = {}
    for k, v in param_values.items():
        if isinstance(v, str):
            try:
                resolved[k] = float(eval(v, {"M_PI": math.pi, "pi": math.pi, "__builtins__": {}}))
            except Exception:
                pass  # skip un-evaluable strings
        elif isinstance(v, (int, float)):
            resolved[k] = float(v)
        else:
            resolved[k] = v

    # --- Evaluate matrices numerically ---
    mats = evaluate_phs_matrices(sphs, resolved, x=x0)
    R_num = mats['R']
    g_num = mats['g']

    # --- Check 1: R eigenvalues ≥ 0 ---
    eigs = np.linalg.eigvalsh(R_num)
    min_eig = float(eigs.min())
    all_nonneg = min_eig >= -1e-12
    eig_str = ", ".join(f"{e:.6g}" for e in sorted(eigs))
    report.add_check(
        "R numerical eigenvalues >= 0",
        all_nonneg,
        f"eigenvalues: [{eig_str}]\nmin = {min_eig:.6e}",
    )

    # --- Check 2: H(x0) > 0 ---
    H_func = make_hamiltonian_func(sphs, resolved)
    H_val = H_func(x0)
    report.add_check(
        "H(x0) >= 0",
        H_val >= -1e-12,
        f"H(x0) = {H_val:.8g}",
    )

    # --- Check 3: Dissipation ≤ 0 at operating point ---
    grad_H_func = make_grad_hamiltonian_func(sphs, resolved)
    grad_H = grad_H_func(x0)
    dissipation = float(-grad_H @ R_num @ grad_H)
    report.add_check(
        "Dissipation <= 0 at x0",
        dissipation <= 1e-12,
        f"-grad_H^T R grad_H = {dissipation:.8g}",
    )

    # --- Check 4: Supply rate at operating point (informational) ---
    supply = float(grad_H @ g_num @ u0)
    power_balance = dissipation + supply
    report.add_check(
        "Power balance at x0 (informational)",
        True,
        f"dissipation = {dissipation:.8g}\n"
        f"supply      = {supply:.8g}\n"
        f"dH/dt (PHS) = {power_balance:.8g}",
    )

    # --- Check 5: Per-state dissipation breakdown (diagnostic) ---
    lines = []
    bypass_warnings = []
    for i in range(n):
        Rii = R_num[i, i]
        dHi = grad_H[i]
        di = -dHi**2 * Rii
        state_name = str(sphs.states[i])
        tag = ""
        if Rii > 1e6:
            tag = "  [BYPASS: T=0 -> R~inf, dx/dt=0 in dynamics]"
            bypass_warnings.append(state_name)
        lines.append(f"  {state_name:>20s}:  R[{i},{i}]={Rii:>12.4g}  "
                      f"dH/dx={dHi:>12.6g}  diss_i={di:>12.6g}{tag}")
    if bypass_warnings:
        lines.append(f"\n  NOTE: States {bypass_warnings} have T=0 (bypass).")
        lines.append("  Actual dissipation for these states is 0 at runtime.")
    report.add_check(
        "Per-state dissipation breakdown",
        True,
        "\n".join(lines),
    )

    return report
