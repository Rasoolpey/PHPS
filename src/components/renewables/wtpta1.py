import math
import numpy as np
from typing import Dict, List, Tuple, Any
from src.core import PowerComponent

class Wtpta1(PowerComponent):
    """
    WTPTA1 Wind Turbine Pitch Control Model.
    """
    @property
    def port_schema(self) -> Dict[str, List[Tuple[str, str, str]]]:
        return {
            'in': [
                ('wt', 'effort', 'pu'),
                ('Pord', 'effort', 'pu'),
                ('Pref', 'effort', 'pu')
            ],
            'out': [
                ('theta', 'flow', 'pu')
            ]
        }

    @property
    def state_schema(self) -> List[str]:
        return ['piw_xi', 'pic_xi', 'lg_y']

    @property
    def param_schema(self) -> Dict[str, str]:
        return {
            'Kiw': 'Speed PI integral gain',
            'Kpw': 'Speed PI proportional gain',
            'Kic': 'Power PI integral gain',
            'Kpc': 'Power PI proportional gain',
            'Kcc': 'Power compensation gain',
            'Tp': 'Pitch lag time constant',
            'thmax': 'Maximum pitch angle',
            'thmin': 'Minimum pitch angle',
            'dthmax': 'Maximum pitch rate',
            'dthmin': 'Minimum pitch rate'
        }

    @property
    def component_role(self) -> str:
        return 'renewable_controller'

    @property
    def observables(self) -> Dict[str, Dict[str, str]]:
        return {
            'theta':   {'description': 'Pitch angle command output',       'unit': 'deg', 'cpp_expr': 'outputs[0]'},
            'piw_xi':  {'description': 'Speed PI integrator state',        'unit': 'pu',  'cpp_expr': 'x[0]'},
            'pic_xi':  {'description': 'Power compensation PI integrator', 'unit': 'pu',  'cpp_expr': 'x[1]'},
            'wt':      {'description': 'Turbine speed input',              'unit': 'pu',  'cpp_expr': 'inputs[0]'},
            'Pord':    {'description': 'Active power order input',         'unit': 'pu',  'cpp_expr': 'inputs[1]'},
        }

    def get_associated_generator(self, comp_map: dict):
        """Return the REGCA1 name by walking wtpta → rea (Wtara1) → rego."""
        wtar = comp_map.get(self.params.get('rea'))
        return wtar.params.get('rego') if wtar else None

    def init_from_targets(self, targets: Dict[str, float]) -> Dict[str, float]:
        return self._init_states({
            'piw_xi': 0.0,
            'pic_xi': 0.0,
            'lg_y': 0.0,
            'theta': 0.0,
        })

    def get_cpp_step_code(self) -> str:
        return """
        // Simplified WTPTA1 for now
        dxdt[0] = 0.0;
        dxdt[1] = 0.0;
        dxdt[2] = (0.0 - x[2]) / std::max(Tp, 0.001);
        """

    def get_cpp_compute_outputs_code(self) -> str:
        return """
        outputs[0] = x[2];
        """
