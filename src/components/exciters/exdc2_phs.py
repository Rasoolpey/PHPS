"""
EXDC2 — Port-Hamiltonian Formulation.

IEEE Type DC2A Exciter with explicit (J, R, H) structure.

States: x = [Vm, Vr, Efd, Xf]

Storage function (generalised Hamiltonian):
    H = ½·TR·Vm² + ½·TA·Vr²/KA + ½·TE·Efd² + ½·TF1·Xf²

This is constructed so that each state's natural dissipation term
appears correctly: the storage function weights each state by its
dominant time constant, making ∂H/∂xi ∝ xi (diagonal quadratic).

Gradient:
    ∂H/∂Vm  = TR · Vm
    ∂H/∂Vr  = TA · Vr / KA
    ∂H/∂Efd = TE · Efd
    ∂H/∂Xf  = TF1 · Xf

PHS Dynamics:
    ẋ = [J(x) − R(x)] ∇H + g · u

where u = [Vterm, Vref] are the port inputs.

The EXDC2 is inherently dissipative — each first-order lag dissipates
energy proportional to its "throughput". The J matrix captures the
feedforward signal path (Vm → Verr → Vr → Efd, with Xf feedback).

References
----------
- IEEE Std 421.5-2016, "IEEE Recommended Practice for Excitation
  System Models for Power System Stability Studies"
"""

import math
import numpy as np
import sympy as sp
from typing import Dict, List, Tuple, Any, Optional
from src.core import PowerComponent


def compute_saturation_coeffs(E1, SE1, E2, SE2):
    """Pre-compute quadratic saturation coefficients A, B from IEEE data."""
    if E1 == 0 or E2 == 0 or abs(E2 - E1) < 1e-10:
        return 0.0, 0.0
    sqrt_SE1E1 = math.sqrt(SE1 * E1) if SE1 * E1 > 0 else 0.0
    sqrt_SE2E2 = math.sqrt(SE2 * E2) if SE2 * E2 > 0 else 0.0
    if abs(sqrt_SE2E2 - sqrt_SE1E1) < 1e-10:
        return 0.0, 0.0
    A = (E1 * sqrt_SE2E2 - E2 * sqrt_SE1E1) / (sqrt_SE2E2 - sqrt_SE1E1)
    B_denom = E1 - A
    if abs(B_denom) < 1e-10:
        return 0.0, 0.0
    B = (sqrt_SE1E1 / B_denom) ** 2
    return A, B


class Exdc2PHS(PowerComponent):
    """
    IEEE EXDC2 exciter in Port-Hamiltonian form.

    ẋ = (J − R) ∇H + g · u

    The storage function is a weighted quadratic ensuring each first-order
    lag's dissipation appears naturally through the R matrix.
    """

    @property
    def port_schema(self) -> Dict[str, List[Tuple[str, str, str]]]:
        return {
            'in':  [('Vterm', 'signal', 'pu'), ('Vref', 'signal', 'pu')],
            'out': [('Efd', 'effort', 'pu')]
        }

    @property
    def state_schema(self) -> List[str]:
        return ['Vm', 'Vr', 'Efd', 'Xf']

    @property
    def param_schema(self) -> Dict[str, str]:
        return {
            'TR':    'Voltage transducer time constant [s]',
            'KA':    'Regulator gain',
            'TA':    'Regulator time constant [s]',
            'KE':    'Exciter self-excitation constant',
            'TE':    'Exciter time constant [s]',
            'KF':    'Rate feedback gain',
            'TF1':   'Rate feedback time constant [s]',
            'VRMAX': 'Max regulator output [pu]',
            'VRMIN': 'Min regulator output [pu]',
            'SAT_A': 'Saturation coefficient A (pre-computed)',
            'SAT_B': 'Saturation coefficient B (pre-computed)',
        }

    @property
    def observables(self) -> Dict[str, Dict[str, str]]:
        return {
            'Efd': {'description': 'Field Voltage', 'unit': 'pu',
                    'cpp_expr': 'x[2]'},
            'H_exc': {'description': 'Exciter storage function', 'unit': 'pu',
                      'cpp_expr': self._hamiltonian_cpp_expr()},
        }

    def get_cpp_compute_outputs_code(self) -> str:
        """Auto-generated from SymbolicPHS output_map: Efd = x[2]."""
        return super().get_cpp_compute_outputs_code()

    def get_cpp_step_code(self) -> str:
        """Auto-generated from SymbolicPHS with saturation + limiter annotations."""
        return super().get_cpp_step_code()

    # ------------------------------------------------------------------ #
    # Python-side PHS interface                                            #
    # ------------------------------------------------------------------ #

    def get_symbolic_phs(self):
        """Return a SymbolicPHS with SymPy matrices for (J, R, g, Q, H)."""
        from src.symbolic.core import SymbolicPHS

        # State symbols
        Vm, Vr, Efd_s, Xf = sp.symbols('V_m V_r E_{fd} X_f')
        states = [Vm, Vr, Efd_s, Xf]

        # Input symbols
        Vterm, Vref = sp.symbols('V_{term} V_{ref}')
        inputs = [Vterm, Vref]

        # Parameter symbols
        TR_s = sp.Symbol('T_R', positive=True)
        KA_s = sp.Symbol('K_A', positive=True)
        TA_s = sp.Symbol('T_A', positive=True)
        KE_s = sp.Symbol('K_E')
        TE_s = sp.Symbol('T_E', positive=True)
        KF_s = sp.Symbol('K_F', nonnegative=True)
        TF1_s = sp.Symbol('T_{F1}', positive=True)

        params = {
            'TR': TR_s, 'KA': KA_s, 'TA': TA_s, 'KE': KE_s,
            'TE': TE_s, 'KF': KF_s, 'TF1': TF1_s,
        }

        # Hamiltonian: H = ½||x||² (identity metric)
        H_expr = sp.Rational(1, 2) * (Vm**2 + Vr**2 + Efd_s**2 + Xf**2)

        # Dissipation matrix R
        R = sp.diag(1/TR_s, KA_s/TA_s, KE_s/TE_s, 1/TF1_s)

        # Interconnection matrix J (skew-symmetric)
        J = sp.zeros(4, 4)
        J[2, 1] = KA_s / (TA_s * TE_s)
        J[1, 2] = -J[2, 1]
        J[3, 2] = 1 / (TE_s * TF1_s)
        J[2, 3] = -J[3, 2]

        # Input coupling
        g = sp.zeros(4, 2)
        g[0, 0] = 1 / TR_s   # Vterm → dVm/dt

        sphs = SymbolicPHS(
            name='EXDC2_PHS',
            states=states,
            inputs=inputs,
            params=params,
            J=J, R=R, g=g, H=H_expr,
            description=(
                'IEEE Type DC2A Exciter in Port-Hamiltonian form. '
                'Identity-weighted storage function H = ½||x||².'
            ),
            dynamics_expr=sp.Matrix([
                (Vterm - Vm) / TR_s,
                (KA_s * (Vref - Vm - KF_s * (Efd_s - Xf) / TF1_s) - Vr) / TA_s,
                (Vr - KE_s * Efd_s) / TE_s,
                (Efd_s - Xf) / TF1_s,
            ]),
        )

        # Anti-windup limiter on regulator output Vr
        sphs.add_limiter(state_idx=1, upper_bound='VRMAX', lower_bound='VRMIN')

        # Saturation on Efd (state 2): SE adds to KE in dissipation
        sphs.add_saturation(state_idx=2, sat_A='SAT_A', sat_B='SAT_B',
                            base_coeff='KE', time_const='TE')

        # Output: Efd = x[2]
        sphs.set_output_map({0: Efd_s})

        # Equilibrium specification for auto-derived init_from_targets():
        #   Efd (state 2) = targets['Efd'], Vref is free, Vterm from targets
        #   Saturation on Efd handled automatically via add_saturation above
        #   Vr limiter handled by post-solve clamping + re-solve
        sphs.set_init_spec(
            target_states={'Efd': 2},      # state 2 (Efd) = targets['Efd']
            input_bindings={
                'V_{term}': 'Vt',          # from targets dict
                'V_{ref}': None,           # free — solve for this
            },
            free_param_map={'V_{ref}': 'Vref'},
            post_init_func=lambda x0, free_params, targets, param_values: {
                'Efd_eff': float(self.compute_efd_output(x0))
            },
        )

        return sphs

    # ------------------------------------------------------------------ #
    # Initialization                                                       #
    # ------------------------------------------------------------------ #

    @property
    def component_role(self) -> str:
        return 'exciter'

    def compute_efd_output(self, x_slice: np.ndarray) -> float:
        return float(x_slice[self.state_schema.index('Efd')])

    def efd_output_expr(self, state_offset: int) -> str:
        efd_i = self.state_schema.index('Efd')
        return f"x[{state_offset + efd_i}]"
