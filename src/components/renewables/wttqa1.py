import math
import numpy as np
from typing import Dict, List, Tuple, Any
from src.core import PowerComponent

class Wttqa1(PowerComponent):
    """
    WTTQA1 Wind Turbine Torque/Power Reference Control.
    """
    @property
    def port_schema(self) -> Dict[str, List[Tuple[str, str, str]]]:
        return {
            'in': [
                ('Pe', 'effort', 'pu'),
                ('wg', 'effort', 'pu')
            ],
            'out': [
                ('Pref', 'flow', 'pu')
            ]
        }

    @property
    def state_schema(self) -> List[str]:
        return ['s1_y', 's2_y', 'pi_xi']

    @property
    def param_schema(self) -> Dict[str, str]:
        return {
            'Kip': 'Power PI integral gain',
            'Kpp': 'Power PI proportional gain',
            'Tp': 'Power filter time constant',
            'Twref': 'Speed reference filter time constant',
            'Temax': 'Maximum torque/power',
            'Temin': 'Minimum torque/power',
            'Tflag': 'Torque flag',
            'p1': 'Power point 1',
            'sp1': 'Speed point 1',
            'p2': 'Power point 2',
            'sp2': 'Speed point 2',
            'p3': 'Power point 3',
            'sp3': 'Speed point 3',
            'p4': 'Power point 4',
            'sp4': 'Speed point 4'
        }

    @property
    def component_role(self) -> str:
        return 'renewable_controller'

    @property
    def observables(self) -> Dict[str, Dict[str, str]]:
        return {
            'Pref':  {'description': 'Active power reference command to REECA1', 'unit': 'pu',  'cpp_expr': 'outputs[0]'},
            'Pe':    {'description': 'Active power input (from REGCA1)',         'unit': 'pu',  'cpp_expr': 'inputs[0]'},
            'wg':    {'description': 'Generator speed input (from WTDTA1)',     'unit': 'pu',  'cpp_expr': 'inputs[1]'},
        }

    def get_associated_generator(self, comp_map: dict):
        """Return the REGCA1 name by walking wttqa → rep (Repca1) → ree (Reeca1) → reg."""
        rep  = comp_map.get(self.params.get('rep'))
        if not rep:
            return None
        ree = comp_map.get(rep.params.get('ree'))
        return ree.params.get('reg') if ree else None

    def init_from_targets(self, targets: Dict[str, float]) -> Dict[str, float]:
        P0 = targets.get('Pe', 0.0)
        
        return self._init_states({
            's1_y': P0,
            's2_y': 1.0,
            'pi_xi': P0,
        })

    def get_cpp_step_code(self) -> str:
        return """
        // Simplified WTTQA1 for now
        dxdt[0] = (inputs[0] - x[0]) / std::max(Tp, 0.001);
        dxdt[1] = (1.0 - x[1]) / std::max(Twref, 0.001);
        dxdt[2] = 0.0;
        """

    def get_cpp_compute_outputs_code(self) -> str:
        return """
        outputs[0] = x[2];
        """
