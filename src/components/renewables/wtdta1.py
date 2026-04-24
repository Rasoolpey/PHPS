import math
import numpy as np
from typing import Dict, List, Tuple, Any
from src.core import PowerComponent

class Wtdta1(PowerComponent):
    """
    WTDTA1 Wind Turbine Drive-Train Model.
    """
    @property
    def port_schema(self) -> Dict[str, List[Tuple[str, str, str]]]:
        return {
            'in': [
                ('Pm', 'effort', 'pu'),
                ('Pe', 'effort', 'pu')
            ],
            'out': [
                ('wt', 'flow', 'pu'),
                ('wg', 'flow', 'pu')
            ]
        }

    @property
    def state_schema(self) -> List[str]:
        return ['theta_tw', 'p_t', 'p_g']

    @property
    def param_schema(self) -> Dict[str, str]:
        return {
            'H': 'Total inertia constant',
            'DAMP': 'Damping coefficient',
            'Htfrac': 'Turbine inertia fraction',
            'Freq1': 'Shaft spring frequency',
            'Dshaft': 'Shaft damping',
            'w0': 'Nominal speed'
        }

    @property
    def component_role(self) -> str:
        return 'renewable_controller'

    @property
    def observables(self) -> Dict[str, Dict[str, str]]:
        return {
            'wt':          {'description': 'Turbine shaft speed',              'unit': 'pu',  'cpp_expr': 'outputs[0]'},
            'wg':          {'description': 'Generator shaft speed',            'unit': 'pu',  'cpp_expr': 'outputs[1]'},
            'theta_tw':    {'description': 'Shaft twist angle',               'unit': 'rad', 'cpp_expr': 'x[0]'},
            'theta_tw_deg':{'description': 'Shaft twist angle',               'unit': 'deg', 'cpp_expr': 'x[0] * 180.0 / 3.14159265359'},
            'p_t':         {'description': 'Turbine generalized momentum p_t', 'unit': 'pu',  'cpp_expr': 'x[1]'},
            'p_g':         {'description': 'Generator generalized momentum p_g','unit': 'pu', 'cpp_expr': 'x[2]'},
            'Pm':          {'description': 'Mechanical power input',           'unit': 'pu',  'cpp_expr': 'inputs[0]'},
            'Pe':          {'description': 'Electrical power input',           'unit': 'pu',  'cpp_expr': 'inputs[1]'},
        }

    def get_associated_generator(self, comp_map: dict):
        """Return the REGCA1 name by walking wdta → ree → reg."""
        ree = comp_map.get(self.params.get('ree'))
        return ree.params.get('reg') if ree else None

    def init_from_targets(self, targets: Dict[str, float]) -> Dict[str, float]:
        P0 = targets.get('Pe', 0.0)
        w0 = self.params.get('w0', 1.0)
        
        Jt = 2 * self.params.get('Htfrac', 0.5) * self.params.get('H', 6.0)
        Jg = 2 * (1 - self.params.get('Htfrac', 0.5)) * self.params.get('H', 6.0)
        
        Freq1 = self.params.get('Freq1', 1.0)
        Kshaft = Jt * Jg / (Jt + Jg) * (2 * math.pi * Freq1)**2
        
        self.params['Jt'] = Jt
        self.params['Jg'] = Jg
        self.params['Kshaft'] = Kshaft
        
        theta_tw0 = P0 / Kshaft if Kshaft > 0 else 0.0
        
        return self._init_states({
            'theta_tw': theta_tw0,
            'p_t': Jt * w0,
            'p_g': Jg * w0,
            'wt': w0,
            'wg': w0,
        })

    def get_cpp_step_code(self) -> str:
        Jt = self.params.get('Jt', 1.0)
        Jg = self.params.get('Jg', 1.0)
        Kshaft = self.params.get('Kshaft', 1.0)
        prefix = f"""
        double Jt = {Jt};
        double Jg = {Jg};
        double Kshaft = {Kshaft};
        """
        return prefix + """
        double wt = x[1] / std::max(Jt, 0.001);
        double wg = x[2] / std::max(Jg, 0.001);
        
        double T_shaft = Kshaft * x[0] + Dshaft * (wt - wg);
        
        dxdt[0] = wt - wg;
        dxdt[1] = inputs[0] / std::max(wt, 0.01) - T_shaft;
        dxdt[2] = T_shaft - inputs[1] / std::max(wg, 0.01) - DAMP * (wg - w0);
        """

    def get_cpp_compute_outputs_code(self) -> str:
        Jt = self.params.get('Jt', 1.0)
        Jg = self.params.get('Jg', 1.0)
        prefix = f"""
        double Jt = {Jt};
        double Jg = {Jg};
        """
        return prefix + """
        double wt = x[1] / std::max(Jt, 0.001);
        double wg = x[2] / std::max(Jg, 0.001);
        
        outputs[0] = wt;
        outputs[1] = wg;
        """
