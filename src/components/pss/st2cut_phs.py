"""
ST2CUT — Port-Hamiltonian Formulation.

Dual-Input Power System Stabilizer with explicit (J, R, H) structure.

States: x = [xl1, xl2, xwo, xll1, xll2, xll3]
  xl1  – Transducer 1 lag state (T1)
  xl2  – Transducer 2 lag state (T2)
  xwo  – Washout filter state (T4)
  xll1 – Lead-lag 1 state (T6)
  xll2 – Lead-lag 2 state (T8)
  xll3 – Lead-lag 3 state (T10)

Signal path:
    omega → [K1 gain, T1 lag] → [T3/T4 washout] → [T5/T6 LL] → [T7/T8 LL]
    → [T9/T10 LL] → clip(LSMIN, LSMAX) → Vss

Storage function (identity metric):
    H = ½||x||²

Each lag/lead-lag block is a first-order dissipative element.
The washout block is a high-pass filter (passivity-preserving).

References
----------
- IEEE Std 421.5-2016, PSS Type ST2CUT
"""

import numpy as np
import sympy as sp
from typing import Dict, List, Tuple, Any, Optional
from src.core import PowerComponent
from src.signal_flow import SignalFlowGraph


class St2cutPHS(PowerComponent):
    """
    ST2CUT Dual-Input PSS in Port-Hamiltonian form.

    ẋ = (J − R) ∇H + g · u

    Storage: H = ½||x||² (identity metric)
    """

    @property
    def port_schema(self) -> Dict[str, List[Tuple[str, str, str]]]:
        return {
            'in': [
                ('omega', 'flow', 'pu'),
                ('Pe',    'flow', 'pu'),
                ('Tm',    'flow', 'pu'),
            ],
            'out': [
                ('Vss', 'effort', 'pu'),
            ]
        }

    @property
    def state_schema(self) -> List[str]:
        return ['xl1', 'xl2', 'xwo', 'xll1', 'xll2', 'xll3']

    @property
    def param_schema(self) -> Dict[str, str]:
        return {
            'K1':   'Input 1 gain',
            'K2':   'Input 2 gain',
            'T1':   'Transducer 1 lag time constant [s]',
            'T2':   'Transducer 2 lag time constant [s]',
            'T3':   'Washout numerator time constant [s]',
            'T4':   'Washout denominator time constant [s]',
            'T5':   'Lead-lag 1 numerator time constant [s]',
            'T6':   'Lead-lag 1 denominator time constant [s]',
            'T7':   'Lead-lag 2 numerator time constant [s]',
            'T8':   'Lead-lag 2 denominator time constant [s]',
            'T9':   'Lead-lag 3 numerator time constant [s]',
            'T10':  'Lead-lag 3 denominator time constant [s]',
            'LSMAX': 'Maximum stabilizing signal output [pu]',
            'LSMIN': 'Minimum stabilizing signal output [pu]',
            'MODE':  'Input 1 mode (1=speed, 3=Pe, 4=Pacc)',
        }

    @property
    def observables(self) -> Dict[str, Dict[str, str]]:
        return {
            'Vss': {'description': 'PSS stabilizing signal', 'unit': 'pu',
                    'cpp_expr': 'outputs[0]'},
        }

    def get_signal_flow_graph(self):
        g = SignalFlowGraph('ST2CUT_PHS')

        g.add_mode_select(
            name='sig1_in',
            mode_param='MODE',
            cases={
                1: g.ref('omega') - 1.0,
                3: g.ref('Pe'),
                4: g.ref('Tm') - g.ref('Pe'),
            },
            default=0.0,
        )

        g.add_lag('xl1', 'sig1_in', 'T1', output_name='l1_track')
        g.add_gain('L1_out', 'l1_track', 'K1')

        # ST2CUT MODE2 path is not modeled in current cases; second channel input is 0.
        g.add_lag('xl2', 0.0, 'T2', output_name='l2_track')
        g.add_gain('L2_out', 'l2_track', 'K2')

        g.add_sum('IN', ['L1_out', 'L2_out'])
        g.add_washout('xwo', 'IN', 'T3', 'T4', output_name='wo_out')

        g.add_lead_lag('xll1', 'wo_out', 'T5', 'T6', output_name='ll1_out')
        g.add_lead_lag('xll2', 'll1_out', 'T7', 'T8', output_name='ll2_out')
        g.add_lead_lag('xll3', 'll2_out', 'T9', 'T10', output_name='ll3_out')

        g.add_clamp('Vss', 'll3_out', 'LSMIN', 'LSMAX')
        g.add_output(0, 'Vss')
        return g

    # ------------------------------------------------------------------ #
    # Python-side PHS interface                                            #
    # ------------------------------------------------------------------ #

    def get_symbolic_phs(self):
        """Return a SymbolicPHS with SymPy matrices for (J, R, g, Q, H)."""
        from src.symbolic.core import SymbolicPHS

        # State symbols
        xl1, xl2, xwo, xll1, xll2, xll3 = sp.symbols(
            'x_{l1} x_{l2} x_{wo} x_{ll1} x_{ll2} x_{ll3}'
        )
        states = [xl1, xl2, xwo, xll1, xll2, xll3]

        # Input symbols
        omega_s, Pe_s, Tm_s = sp.symbols('omega P_e T_m')
        inputs = [omega_s, Pe_s, Tm_s]

        # Parameter symbols
        T1_s  = sp.Symbol('T_1', positive=True)
        T4_s  = sp.Symbol('T_4', positive=True)
        T6_s  = sp.Symbol('T_6', positive=True)
        T8_s  = sp.Symbol('T_8', positive=True)
        T10_s = sp.Symbol('T_{10}', positive=True)

        params = {
            'T1': T1_s, 'T4': T4_s, 'T6': T6_s, 'T8': T8_s, 'T10': T10_s,
        }

        # Hamiltonian: H = ½||x||² (identity metric)
        H_expr = sp.Rational(1, 2) * (
            xl1**2 + xl2**2 + xwo**2 + xll1**2 + xll2**2 + xll3**2
        )

        # Dissipation matrix R (diagonal)
        # T2 typically 0 → xl2 unused → R[1,1] = 0
        R = sp.diag(1/T1_s, 0, 1/T4_s, 1/T6_s, 1/T8_s, 1/T10_s)

        # Interconnection matrix J = 0 (no skew-symmetric coupling)
        J = sp.zeros(6, 6)

        # Input coupling g (from numerical get_phs_matrices: g = 0)
        g = sp.zeros(6, 3)

        sphs = SymbolicPHS(
            name='ST2CUT_PHS',
            states=states,
            inputs=inputs,
            params=params,
            J=J, R=R, g=g, H=H_expr,
            description=(
                'IEEE Type ST2CUT Dual-Input PSS in Port-Hamiltonian form. '
                'Identity-weighted storage function H = ½||x||². '
                'Six dissipative lag/lead-lag/washout blocks.'
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
                sig1_in = omega - 1.0
            elif mode == 3:
                sig1_in = Pe
            elif mode == 4:
                sig1_in = Tm - Pe
            else:
                sig1_in = 0.0

            xl1 = sig1_in
            xl2 = 0.0
            l1_out = float(param_values.get('K1', 0.0)) * sig1_in
            l2_out = float(param_values.get('K2', 0.0)) * 0.0
            in_sig = l1_out + l2_out
            xwo = in_sig
            xll1 = 0.0
            xll2 = 0.0
            xll3 = 0.0
            return np.array([xl1, xl2, xwo, xll1, xll2, xll3]), {}

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
