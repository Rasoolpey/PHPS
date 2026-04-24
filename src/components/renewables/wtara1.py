import math
import numpy as np
from typing import Dict, List, Tuple, Any
from src.core import PowerComponent

class Wtara1(PowerComponent):
    """
    WTARA1 Wind Turbine Aerodynamics Model.
    """
    @property
    def port_schema(self) -> Dict[str, List[Tuple[str, str, str]]]:
        return {
            'in': [
                ('theta', 'effort', 'pu')
            ],
            'out': [
                ('Pm', 'flow', 'pu')
            ]
        }

    @property
    def state_schema(self) -> List[str]:
        return []

    @property
    def param_schema(self) -> Dict[str, str]:
        return {
            'Ka': 'Aerodynamic gain',
            'theta0': 'Initial pitch angle'
        }

    @property
    def component_role(self) -> str:
        return 'renewable_controller'

    @property
    def observables(self) -> Dict[str, Dict[str, str]]:
        return {
            'Pm':    {'description': 'Mechanical power output to drive train', 'unit': 'pu',  'cpp_expr': 'outputs[0]'},
            'theta': {'description': 'Pitch angle input (from WTPTA1)',       'unit': 'deg', 'cpp_expr': 'inputs[0]'},
        }

    def get_associated_generator(self, comp_map: dict):
        """Return the REGCA1 name referenced by param 'rego'."""
        return self.params.get('rego')

    def init_from_targets(self, targets: Dict[str, float]) -> Dict[str, float]:
        P0 = targets.get('Pe', 0.0)
        self.params['Pe0'] = P0
        
        return self._init_states({'Pm': P0})

    def get_cpp_step_code(self) -> str:
        return """
        // No states
        """

    def get_cpp_compute_outputs_code(self) -> str:
        Pe0 = self.params.get('Pe0', 0.0)
        prefix = f"""
        double Pe0 = {Pe0};
        """
        return prefix + """
        double theta0r = theta0 * M_PI / 180.0;
        double theta_rad = inputs[0] * M_PI / 180.0;
        outputs[0] = Pe0 - Ka * theta_rad * (theta_rad - theta0r);
        """
