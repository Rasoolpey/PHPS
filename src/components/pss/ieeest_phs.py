"""
IEEEST — Port-Hamiltonian Formulation.

IEEE Standard Type ST Power System Stabilizer with explicit (J, R, H) structure.

States: x = [xf1, xf2, xll1, xll2, xll3, xll4, xwo]
  xf1  – First lag filter state (A1)
  xf2  – Second lag filter state (A2)
  xll1 – Lead-lag 1 state (A3/A4)
  xll2 – Lead-lag 2 state (A5/A6)
  xll3 – Lead-lag 3 state (T1/T2)
  xll4 – Lead-lag 4 state (T3/T4)
  xwo  – Washout filter state (T5/T6)

Signal path:
    Input signal (MODE selection)
    -> [lag filter A1] -> [lag filter A2]
    -> [lead-lag A3/A4] -> [lead-lag A5/A6]
    -> [lead-lag T1/T2] -> [lead-lag T3/T4]
    -> [gain KS]
    -> [washout T5/T6]
    -> clip(LSMIN, LSMAX)
    -> Vss

Storage function (identity metric):
    H = ½||x||²

Each lag/lead-lag block is a first-order dissipative element.
The washout block is a high-pass filter (passivity-preserving).

References
----------
- IEEE Std 421.5-2005, PSS Type IEEEST
"""

import numpy as np
import sympy as sp
from typing import Dict, List, Tuple, Any, Optional
from src.core import PowerComponent
from src.signal_flow import SignalFlowGraph


class IeeestPHS(PowerComponent):
    """
    IEEEST PSS in Port-Hamiltonian form.

    ẋ = (J − R) ∇H + g · u

    Storage: H = ½||x||² (identity metric)
    """

    @property
    def port_schema(self) -> Dict[str, List[Tuple[str, str, str]]]:
        return {
            'in': [
                ('omega', 'flow', 'pu'),   # Generator speed [pu]
                ('Pe',    'flow', 'pu'),   # Electrical power
                ('Tm',    'flow', 'pu'),   # Mechanical torque
            ],
            'out': [
                ('Vss', 'effort', 'pu'),   # Stabilizing signal to exciter Vref
            ]
        }

    @property
    def state_schema(self) -> List[str]:
        return ['xf1', 'xf2', 'xll1', 'xll2', 'xll3', 'xll4', 'xwo']

    @property
    def param_schema(self) -> Dict[str, str]:
        return {
            'MODE':  'Input signal mode (1=speed, 3=Pe, 4=Pacc)',
            'A1':    'Lag filter 1 time constant [s]',
            'A2':    'Lag filter 2 time constant [s]',
            'A3':    'Lead-lag 1 numerator time constant [s]',
            'A4':    'Lead-lag 1 denominator time constant [s]',
            'A5':    'Lead-lag 2 numerator time constant [s]',
            'A6':    'Lead-lag 2 denominator time constant [s]',
            'T1':    'Lead-lag 3 numerator time constant [s]',
            'T2':    'Lead-lag 3 denominator time constant [s]',
            'T3':    'Lead-lag 4 numerator time constant [s]',
            'T4':    'Lead-lag 4 denominator time constant [s]',
            'KS':    'Gain',
            'T5':    'Washout numerator time constant [s]',
            'T6':    'Washout denominator time constant [s]',
            'LSMAX': 'Maximum output [pu]',
            'LSMIN': 'Minimum output [pu]',
        }

    @property
    def observables(self) -> Dict[str, Dict[str, str]]:
        return {
            'Vss': {'description': 'PSS stabilizing signal', 'unit': 'pu',
                    'cpp_expr': 'outputs[0]'},
            'H_pss': {'description': 'PSS Hamiltonian', 'unit': 'pu',
                      'cpp_expr': '0.5*(x[0]*x[0]+x[1]*x[1]+x[2]*x[2]'
                                  '+x[3]*x[3]+x[4]*x[4]+x[5]*x[5]+x[6]*x[6])'},
        }

    def get_signal_flow_graph(self):
        g = SignalFlowGraph('IEEEST_PHS')

        g.add_mode_select(
            name='sig_in',
            mode_param='MODE',
            cases={
                1: g.ref('omega') - 1.0,
                3: g.ref('Pe'),
                4: g.ref('Tm') - g.ref('Pe'),
            },
            default=0.0,
        )

        g.add_lag('xf1', 'sig_in', 'A1', output_name='f1_out')
        g.add_lag('xf2', 'f1_out', 'A2', output_name='f2_out')

        g.add_lead_lag('xll1', 'f2_out', 'A3', 'A4', output_name='ll1_out')
        g.add_lead_lag('xll2', 'll1_out', 'A5', 'A6', output_name='ll2_out')
        g.add_lead_lag('xll3', 'll2_out', 'T1', 'T2', output_name='ll3_out')
        g.add_lead_lag('xll4', 'll3_out', 'T3', 'T4', output_name='ll4_out')

        g.add_gain('vks', 'll4_out', 'KS')
        g.add_washout('xwo', 'vks', 'T5', 'T6', output_name='wo_out')

        g.add_clamp('Vss', 'wo_out', 'LSMIN', 'LSMAX')
        g.add_output(0, 'Vss')
        return g

    # ------------------------------------------------------------------ #
    # Python-side PHS interface                                            #
    # ------------------------------------------------------------------ #

    def get_symbolic_phs(self):
        """Return a SymbolicPHS with SymPy matrices for (J, R, g, Q, H)."""
        from src.symbolic.core import SymbolicPHS

        # State symbols
        xf1, xf2, xll1, xll2, xll3, xll4, xwo = sp.symbols(
            'x_{f1} x_{f2} x_{ll1} x_{ll2} x_{ll3} x_{ll4} x_{wo}'
        )
        states = [xf1, xf2, xll1, xll2, xll3, xll4, xwo]

        # Input symbols
        omega_s, Pe_s, Tm_s = sp.symbols('omega P_e T_m')
        inputs = [omega_s, Pe_s, Tm_s]

        # Parameter symbols
        A1_s = sp.Symbol('A_1', positive=True)
        A2_s = sp.Symbol('A_2', positive=True)
        A4_s = sp.Symbol('A_4', positive=True)
        A6_s = sp.Symbol('A_6', positive=True)
        T2_s = sp.Symbol('T_2', positive=True)
        T4_s = sp.Symbol('T_4', positive=True)
        T6_s = sp.Symbol('T_6', positive=True)

        params = {
            'A1': A1_s, 'A2': A2_s, 'A4': A4_s, 'A6': A6_s,
            'T2': T2_s, 'T4': T4_s, 'T6': T6_s,
        }

        # Hamiltonian: H = ½||x||² (identity metric)
        H_expr = sp.Rational(1, 2) * (
            xf1**2 + xf2**2 + xll1**2 + xll2**2 + xll3**2 + xll4**2 + xwo**2
        )

        # Dissipation matrix R (diagonal)
        R = sp.diag(1/A1_s, 1/A2_s, 1/A4_s, 1/A6_s, 1/T2_s, 1/T4_s, 1/T6_s)

        # Interconnection matrix J = 0 (no skew-symmetric coupling)
        J = sp.zeros(7, 7)

        # Input coupling g (from numerical get_phs_matrices: g = 0)
        g = sp.zeros(7, 3)

        sphs = SymbolicPHS(
            name='IEEEST_PHS',
            states=states,
            inputs=inputs,
            params=params,
            J=J, R=R, g=g, H=H_expr,
            description=(
                'IEEE Standard Type ST PSS in Port-Hamiltonian form. '
                'Identity-weighted storage function H = ½||x||². '
                'Seven dissipative lag/lead-lag/washout blocks.'
            ),
        )

        def equilibrium_solver(targets: dict, param_values: dict):
            mode = int(param_values.get('MODE', 1))
            omega = float(targets.get('omega', 1.0))
            vd = float(targets.get('vd', 0.0))
            vq = float(targets.get('vq', 0.0))
            id_val = float(targets.get('id', 0.0))
            iq_val = float(targets.get('iq', 0.0))
            Pe = float(targets.get('Pe', vd * id_val + vq * iq_val))
            Tm = float(targets.get('Tm', Pe))

            if mode == 1:
                sig_in = omega - 1.0
            elif mode == 3:
                sig_in = Pe
            elif mode == 4:
                sig_in = Tm - Pe
            else:
                sig_in = 0.0

            xf1 = sig_in
            xf2 = sig_in
            xll1 = xf2
            xll2 = xll1
            xll3 = xll2
            xll4 = xll3
            vks = float(param_values.get('KS', 0.0)) * xll4
            xwo = vks
            return np.array([xf1, xf2, xll1, xll2, xll3, xll4, xwo]), {}

        sphs.set_init_spec(
            target_states={},
            input_bindings={},
            free_param_map={},
            solver_func=equilibrium_solver,
        )

        return sphs

    # ------------------------------------------------------------------ #
    # Initialization                                                      #
    # ------------------------------------------------------------------ #

    @property
    def component_role(self) -> str:
        return 'pss'
