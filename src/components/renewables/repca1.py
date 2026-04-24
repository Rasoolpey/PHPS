import math
import numpy as np
from typing import Dict, List, Tuple, Any
from src.core import PowerComponent

class Repca1(PowerComponent):
    """
    REPCA1 Renewable Energy Plant Controller.
    """
    @property
    def port_schema(self) -> Dict[str, List[Tuple[str, str, str]]]:
        return {
            'in': [
                ('Vterm', 'effort', 'pu'),
                ('f', 'effort', 'pu'),
                ('Pline', 'flow', 'pu'),
                ('Qline', 'flow', 'pu')
            ],
            'out': [
                ('Pext', 'effort', 'pu'),
                ('Qext', 'effort', 'pu')
            ]
        }

    @property
    def state_schema(self) -> List[str]:
        return ['s0_y', 's1_y', 's2_xi', 's4_y', 's5_xi']

    @property
    def param_schema(self) -> Dict[str, str]:
        return {
            'Tfltr': 'Voltage filter time constant',
            'Kp': 'Reactive power PI proportional gain',
            'Ki': 'Reactive power PI integral gain',
            'Tft': 'Lead time constant',
            'Tfv': 'Lag time constant',
            'Vfrz': 'Voltage freeze threshold',
            'Rc': 'Line drop compensation resistance',
            'Xc': 'Line drop compensation reactance',
            'Kc': 'Reactive power droop gain',
            'emax': 'Maximum error',
            'emin': 'Minimum error',
            'dbd1': 'Deadband lower limit',
            'dbd2': 'Deadband upper limit',
            'Qmax': 'Maximum reactive power',
            'Qmin': 'Minimum reactive power',
            'Kpg': 'Active power PI proportional gain',
            'Kig': 'Active power PI integral gain',
            'Tp': 'Active power filter time constant',
            'fdbd1': 'Frequency deadband lower limit',
            'fdbd2': 'Frequency deadband upper limit',
            'femax': 'Maximum frequency error',
            'femin': 'Minimum frequency error',
            'Pmax': 'Maximum active power',
            'Pmin': 'Minimum active power',
            'Tg': 'Active power lag time constant',
            'Ddn': 'Droop down',
            'Dup': 'Droop up',
            'VCFlag': 'Voltage control flag',
            'RefFlag': 'Reference flag',
            'Fflag': 'Frequency flag',
            'PLflag': 'Plant level flag'
        }

    @property
    def component_role(self) -> str:
        return 'renewable_controller'

    @property
    def observables(self) -> Dict[str, Dict[str, str]]:
        return {
            'Pext':   {'description': 'Active power reference output to REECA1',   'unit': 'pu',  'cpp_expr': 'outputs[0]'},
            'Qext':   {'description': 'Reactive power reference output to REECA1', 'unit': 'pu',  'cpp_expr': 'outputs[1]'},
            'Vterm':  {'description': 'Terminal voltage (plant level)',            'unit': 'pu',  'cpp_expr': 'inputs[0]'},
            'f':      {'description': 'Frequency input',                          'unit': 'pu',  'cpp_expr': 'inputs[1]'},
            'Pline':  {'description': 'Line active power (measured)',             'unit': 'pu',  'cpp_expr': 'inputs[2]'},
            'Qline':  {'description': 'Line reactive power (measured)',           'unit': 'pu',  'cpp_expr': 'inputs[3]'},
        }

    def get_associated_generator(self, comp_map: dict):
        """Return the REGCA1 name by walking rep → ree → reg."""
        ree = comp_map.get(self.params.get('ree'))
        return ree.params.get('reg') if ree else None

    def init_from_targets(self, targets: Dict[str, float]) -> Dict[str, float]:
        V0 = targets.get('Vterm', 1.0)
        P0 = targets.get('Pline', 0.0)
        Q0 = targets.get('Qline', 0.0)

        return self._init_states({
            's0_y': V0,
            's1_y': Q0,
            's2_xi': 0.0,
            's4_y': P0,
            's5_xi': 0.0,
        })

    def get_cpp_step_code(self) -> str:
        return """
        // Simplified REPCA1 for now
        dxdt[0] = (inputs[0] - x[0]) / std::max(Tfltr, 0.001);
        dxdt[1] = (inputs[3] - x[1]) / std::max(Tfltr, 0.001);
        dxdt[2] = 0.0;
        dxdt[3] = (inputs[2] - x[3]) / std::max(Tp, 0.001);
        dxdt[4] = 0.0;
        """

    def get_cpp_compute_outputs_code(self) -> str:
        return """
        outputs[0] = 0.0;
        outputs[1] = 0.0;
        """
