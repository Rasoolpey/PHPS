"""
IEEEX1 — Port-Hamiltonian Formulation.

IEEE Type DC1A Exciter with explicit (J, R, H) structure.

States: x = [Vm, LLx, Vr, Vp, Vf]
  Vm  – Voltage transducer (1st-order lag, TR)
  LLx – Lead-lag compensator state (TC/TB)
  Vr  – Regulator output (KA/TA, V-dependent anti-windup)
  Vp  – Exciter output (KE + saturation SE, TE)
  Vf  – Rate feedback washout state (KF1/TF1)

Signal path:
  Vterm → [1/(1+sTR)] → Vm
  Vf_out = KF1*(Vp − Vf)/TF1        (washout feedback)
  Verr = Vref − Vm − Vf_out
  Verr → [(1+sTC)/(1+sTB)] → LLx_out  (lead-lag)
  LLx_out → [KA/(1+sTA)] → Vr       (V-dependent limits)
  Vr → [1/(KE+SE(Vp)+sTE)] → Vp     (exciter with saturation)
  Efd = Vp * omega                   (speed compensation)
  Vp → [1/(1+sTF1)] → Vf            (feedback state)

Storage function (identity metric):
    H = ½||x||²

References
----------
- IEEE Std 421.5-2005, Type DC1A (IEEEX1)
"""

import math
import numpy as np
import sympy as sp
from typing import Dict, List, Tuple, Any, Optional
from src.core import PowerComponent
from src.components.exciters.exdc2 import compute_saturation_coeffs


class Ieeex1PHS(PowerComponent):
    """
    IEEEX1 exciter in Port-Hamiltonian form.

    ẋ = (J − R) ∇H + g · u

    Storage: H = ½||x||² (identity metric)
    """

    @property
    def port_schema(self) -> Dict[str, List[Tuple[str, str, str]]]:
        return {
            'in':  [('Vterm', 'signal', 'pu'),
                    ('Vref', 'signal', 'pu'),
                    ('omega', 'flow', 'pu')],
            'out': [('Efd', 'effort', 'pu')]
        }

    @property
    def state_schema(self) -> List[str]:
        return ['Vm', 'LLx', 'Vr', 'Vp', 'Vf']

    @property
    def param_schema(self) -> Dict[str, str]:
        return {
            'TR':    'Voltage transducer time constant [s]',
            'TC':    'Lead-lag numerator time constant [s]',
            'TB':    'Lead-lag denominator time constant [s]',
            'KA':    'Regulator gain',
            'TA':    'Regulator time constant [s]',
            'VRMAX': 'Max regulator output – absolute limit from pilot exciter [pu]',
            'VRMIN': 'Min regulator output – absolute limit from pilot exciter [pu]',
            'TE':    'Exciter time constant [s]',
            'KE':    'Exciter self-excitation constant',
            'KF1':   'Rate feedback gain',
            'TF1':   'Rate feedback time constant [s]',
            'SAT_A': 'Saturation coefficient A (pre-computed)',
            'SAT_B': 'Saturation coefficient B (pre-computed)',
        }

    _DEFAULTS = {'TC': 0.0, 'TB': 0.0}

    def __init__(self, name: str, params: Dict[str, Any]):
        for k, v in self._DEFAULTS.items():
            params.setdefault(k, v)
        if 'SAT_A' not in params or 'SAT_B' not in params:
            E1  = params.pop('E1', 0.0)
            SE1 = params.pop('SE1', 0.0)
            E2  = params.pop('E2', 0.0)
            SE2 = params.pop('SE2', 0.0)
            A, B = compute_saturation_coeffs(E1, SE1, E2, SE2)
            params['SAT_A'] = A
            params['SAT_B'] = B
        super().__init__(name, params)

    @property
    def observables(self) -> Dict[str, Dict[str, str]]:
        return {
            'Efd': {'description': 'Field Voltage', 'unit': 'pu',
                    'cpp_expr': 'x[3] * inputs[2]'},
            'H_exc': {'description': 'Exciter Hamiltonian', 'unit': 'pu',
                      'cpp_expr': '0.5*(x[0]*x[0]+x[1]*x[1]+x[2]*x[2]'
                                  '+x[3]*x[3]+x[4]*x[4])'},
        }

    def get_cpp_compute_outputs_code(self) -> str:
        return """
            double Vp    = x[3];
            double omega = inputs[2];
            outputs[0] = Vp * omega;
        """

    def get_cpp_step_code(self) -> str:
        """Auto-generated from SymbolicPHS with lead-lag, saturation, and limiter."""
        return super().get_cpp_step_code()

    # ------------------------------------------------------------------ #
    # Python-side PHS interface                                            #
    # ------------------------------------------------------------------ #

    def get_symbolic_phs(self):
        """Return a SymbolicPHS with SymPy matrices for (J, R, g, Q, H)."""
        from src.symbolic.core import SymbolicPHS

        Vm, LLx, Vr, Vp, Vf = sp.symbols('V_m LLx V_r V_p V_f')
        states = [Vm, LLx, Vr, Vp, Vf]

        Vterm, Vref = sp.symbols('V_{term} V_{ref}')
        inputs = [Vterm, Vref]

        TR_s = sp.Symbol('T_R', positive=True)
        TB_s = sp.Symbol('T_B', positive=True)
        TC_s = sp.Symbol('T_C', nonnegative=True)
        KA_s = sp.Symbol('K_A', positive=True)
        TA_s = sp.Symbol('T_A', positive=True)
        KE_s = sp.Symbol('K_E')
        TE_s = sp.Symbol('T_E', positive=True)
        KF1_s = sp.Symbol('K_{F1}', nonnegative=True)
        TF1_s = sp.Symbol('T_{F1}', positive=True)

        params = {
            'TR': TR_s, 'TB': TB_s, 'TC': TC_s, 'KA': KA_s, 'TA': TA_s,
            'KE': KE_s, 'TE': TE_s, 'KF1': KF1_s, 'TF1': TF1_s,
        }

        H_expr = sp.Rational(1, 2) * (Vm**2 + LLx**2 + Vr**2 + Vp**2 + Vf**2)

        R_mat = sp.diag(1/TR_s, 1/TB_s, KA_s/TA_s, KE_s/TE_s, 1/TF1_s)

        J = sp.zeros(5, 5)

        g = sp.zeros(5, 2)
        g[0, 0] = 1 / TR_s

        # Lead-lag output placeholder for dynamics_expr
        LLx_out_sym = sp.Symbol('LLx_out')

        # Washout feedback
        Vf_out_expr = KF1_s * (Vp - Vf) / TF1_s
        Verr_expr = Vref - Vm - Vf_out_expr

        sphs = SymbolicPHS(
            name='IEEEX1_PHS',
            states=states,
            inputs=inputs,
            params=params,
            J=J, R=R_mat, g=g, H=H_expr,
            description=(
                'IEEE DC1A Exciter (IEEEX1) in Port-Hamiltonian form. '
                'Purely dissipative structure (J = 0). '
                'Identity-weighted storage H = ½||x||².'
            ),
            dynamics_expr=sp.Matrix([
                (Vterm - Vm) / TR_s,
                (Verr_expr - LLx) / TB_s,          # overridden by lead-lag
                (KA_s * LLx_out_sym - Vr) / TA_s,
                (Vr - KE_s * Vp) / TE_s,           # saturation adds SE
                (Vp - Vf) / TF1_s,
            ]),
        )

        # Lead-lag on state 1
        sphs.add_lead_lag(state_idx=1, input_expr=Verr_expr,
                          output_var='LLx_out', tc_param='TC', tb_param='TB')

        # Anti-windup limiter on regulator output Vr
        sphs.add_limiter(state_idx=2, upper_bound='VRMAX', lower_bound='VRMIN')

        # Saturation on Vp (state 3): SE adds to KE in dissipation
        sphs.add_saturation(state_idx=3, sat_A='SAT_A', sat_B='SAT_B',
                            base_coeff='KE', time_const='TE')

        # Equilibrium specification for auto-derived init_from_targets():
        #   Vp (state 3) = Efd / omega, Vref is free, Vterm from targets
        #   Lead-lag output → input_expr at SS (handled automatically)
        #   Saturation on Vp handled automatically via add_saturation above
        #   Vr limiter handled by post-solve clamping + re-solve
        sphs.set_init_spec(
            target_states={'Efd': 3},      # state 3 (Vp) = Efd / omega
            input_bindings={
                'V_{term}': 'Vt',          # from targets dict
                'V_{ref}': None,           # free — solve for this
            },
            free_param_map={'V_{ref}': 'Vref'},
            target_transforms={
                'Efd': lambda efd, t: efd / t.get('omega', 1.0)
                       if t.get('omega', 1.0) > 0.1 else efd,
            },
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
        """Efd = Vp * omega.  At initialization omega=1.0, so Efd = Vp."""
        return float(x_slice[self.state_schema.index('Vp')])

    def efd_output_expr(self, state_offset: int) -> str:
        vp_i = self.state_schema.index('Vp')
        return f"x[{state_offset + vp_i}]"
