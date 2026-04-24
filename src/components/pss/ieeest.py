import numpy as np
from typing import Dict, List, Tuple, Any
from src.core import PowerComponent
from src.signal_flow import SignalFlowGraph


class Ieeest(PowerComponent):
    """
    IEEEST Power System Stabilizer for C++ code generation.

    IEEE Standard Type ST PSS (IEEE Std 421.5-2005).

    Signal path:
        Input signal (MODE selection)
        -> [lag filter A1] -> [lag filter A2]
        -> [lead-lag A3/A4] -> [lead-lag A5/A6]
        -> [lead-lag T1/T2] -> [lead-lag T3/T4]
        -> [gain KS]
        -> [washout T5/T6]
        -> clip(LSMIN, LSMAX)
        -> Vss

    States: [xf1, xf2, xll1, xll2, xll3, xll4, xwo]
        xf1:  First lag filter state
        xf2:  Second lag filter state
        xll1: Lead-lag 1 state (A3/A4)
        xll2: Lead-lag 2 state (A5/A6)
        xll3: Lead-lag 3 state (T1/T2)
        xll4: Lead-lag 4 state (T3/T4)
        xwo:  Washout filter state (T5/T6)

    Input modes:
        1: Rotor speed deviation (omega - 1)
        3: Electrical power / torque (Pe)
        4: Accelerating power (Tm - Pe)

    Output: outputs[0] = Vss (stabilizing signal to exciter Vref)
    """

    @property
    def port_schema(self) -> Dict[str, List[Tuple[str, str, str]]]:
        return {
            'in': [
                ('omega', 'flow', 'pu'),  # Generator speed [pu]
                ('Pe',    'flow', 'pu'),  # Electrical power
                ('Tm',    'flow', 'pu'),  # Mechanical torque
            ],
            'out': [
                ('Vss', 'effort', 'pu'),  # Stabilizing signal to exciter Vref
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
        }

    def get_signal_flow_graph(self):
        g = SignalFlowGraph('IEEEST')

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

    # --- Initialization Contract ---

    @property
    def component_role(self) -> str:
        return 'pss'

    def init_from_targets(self, targets: dict) -> np.ndarray:
        p = self.params
        mode = int(p.get('MODE', 1))

        # Equilibrium inputs (defaults keep steady-state ω=1.0, Pe=Tm)
        omega = float(targets.get('omega', 1.0))
        vd     = float(targets.get('vd', 0.0))
        vq     = float(targets.get('vq', 0.0))
        id_val = float(targets.get('id', 0.0))
        iq_val = float(targets.get('iq', 0.0))
        Pe     = float(targets.get('Pe', vd * id_val + vq * iq_val))
        Tm     = float(targets.get('Tm', Pe))

        if mode == 1:
            sig_in = omega - 1.0
        elif mode == 3:
            sig_in = Pe
        elif mode == 4:
            sig_in = Tm - Pe
        else:
            sig_in = 0.0

        # Steady-state through cascaded filters (all states track their inputs)
        xf1  = sig_in
        xf2  = sig_in
        xll1 = xf2
        xll2 = xll1
        xll3 = xll2
        xll4 = xll3

        # Washout state tracks vks so that wo_out = 0 at equilibrium
        vks = float(p.get('KS', 0.0)) * xll4
        xwo = vks

        return np.array([xf1, xf2, xll1, xll2, xll3, xll4, xwo])
