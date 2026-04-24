import numpy as np
from typing import Dict, List, Tuple, Any
from src.core import PowerComponent
from src.signal_flow import SignalFlowGraph


class St2cut(PowerComponent):
    """
    ST2CUT Dual-Input Power System Stabilizer for C++ code generation.

    States: [xl1, xl2, xwo, xll1, xll2, xll3]
        xl1:  Transducer 1 lag state (T1)
        xl2:  Transducer 2 lag state (T2)
        xwo:  Washout filter state (T4)
        xll1: Lead-lag 1 state (T6)
        xll2: Lead-lag 2 state (T8)
        xll3: Lead-lag 3 state (T10)

    Signal path:
        omega (or other mode signal) -> [K1 gain, T1 lag] -> L1_out
        -> [T3/T4 washout] -> [T5/T6 lead-lag] -> [T7/T8 lead-lag]
        -> [T9/T10 lead-lag] -> clip(LSMIN, LSMAX) -> Vss

    Output: outputs[0] = Vss (stabilizing signal added to exciter Vref)

    Only MODE=1 (speed deviation) is implemented, which covers ST2CUT_3.
    VCU=VCL=0 in PSS/E means no voltage gating -> PSS always active.
    """

    @property
    def port_schema(self) -> Dict[str, List[Tuple[str, str, str]]]:
        return {
            'in': [
                ('omega', 'flow', 'pu'),  # Generator speed [pu]
                ('Pe',    'flow', 'pu'),  # Electrical power (for MODE=3/4)
                ('Tm',    'flow', 'pu'),  # Mechanical torque (for MODE=4)
            ],
            'out': [
                ('Vss', 'effort', 'pu'),  # Stabilizing signal to exciter Vref
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
        g = SignalFlowGraph('ST2CUT')

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

        # MODE2 is not modeled in current ST2CUT cases; second channel input is zero.
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

    # --- Initialization Contract ---

    @property
    def component_role(self) -> str:
        return 'pss'

    def init_from_targets(self, targets: dict) -> np.ndarray:
        p = self.params
        mode = int(p.get('MODE', 1))

        omega = float(targets.get('omega', 1.0))
        vd     = float(targets.get('vd', 0.0))
        vq     = float(targets.get('vq', 0.0))
        id_val = float(targets.get('id', 0.0))
        iq_val = float(targets.get('iq', 0.0))
        Pe     = float(targets.get('Pe', vd * id_val + vq * iq_val))
        Tm     = float(targets.get('Tm', Pe))

        if mode == 1:
            sig1_in = omega - 1.0
        elif mode == 3:
            sig1_in = Pe
        elif mode == 4:
            sig1_in = Tm - Pe
        else:
            sig1_in = 0.0

        # Transducer states track their inputs at equilibrium
        xl1 = sig1_in
        xl2 = 0.0  # second input unused in current configurations

        L1_out = float(p.get('K1', 0.0)) * sig1_in
        L2_out = float(p.get('K2', 0.0)) * 0.0
        IN = L1_out + L2_out

        # Washout state tracks IN so wo_out = 0
        xwo = IN
        wo_out = 0.0

        # Lead-lag states track their inputs to keep dx/dt = 0
        xll1 = wo_out
        xll2 = xll1
        xll3 = xll2

        return np.array([xl1, xl2, xwo, xll1, xll2, xll3])
