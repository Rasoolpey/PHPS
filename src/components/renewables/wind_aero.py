import math
from typing import Dict, List, Tuple, Any
from src.core import PowerComponent

class WindAero(PowerComponent):
    """
    Detailed Wind Turbine Aerodynamics Model.
    Computes Mechanical Torque (Tm) given Wind Speed (vw_pu), Rotor Speed (omega_pu), and Pitch (theta_deg).
    Models standard Cp(lambda, theta) curves.
    """
    @property
    def port_schema(self) -> Dict[str, List[Tuple[str, str, str]]]:
        return {
            'in': [
                ('vw', 'effort', 'pu'),      # Wind speed [pu, base usually nominal wind speed, e.g., 12m/s]
                ('omega', 'effort', 'pu'),   # Rotor mechanical speed [pu]
                ('theta', 'effort', 'pu')    # Pitch angle [deg]
            ],
            'out': [
                ('Tm', 'flow', 'pu'),        # Mechanical torque applied to the rotor [pu]
                ('Pm', 'flow', 'pu')         # Mechanical active power [pu]
            ]
        }

    @property
    def state_schema(self) -> List[str]: return []

    @property
    def param_schema(self) -> Dict[str, str]:
        return {}

    @property
    def component_role(self) -> str: return 'renewable_controller'

    @property
    def observables(self) -> Dict[str, Dict[str, str]]:
        return {
            'Tm':      {'description': 'Aerodynamic mechanical torque', 'unit': 'pu', 'cpp_expr': 'outputs[0]'},
            'Pm':      {'description': 'Aerodynamic power generated',  'unit': 'pu', 'cpp_expr': 'outputs[1]'},
            'vw':      {'description': 'Wind speed input',             'unit': 'pu', 'cpp_expr': 'inputs[0]'},
            'Cp':      {'description': 'Power coefficient',            'unit': 'pu', 'cpp_expr': 'Cp_factor'},
        }

    def init_from_targets(self, targets: Dict[str, float]) -> Dict[str, float]:
        Tm0 = float(targets.get('Tm', 0.5))
        self.params['Tm0'] = Tm0
        return self._init_states({})

    def get_cpp_step_code(self) -> str: return ""

    def get_cpp_compute_outputs_code(self) -> str:
        Tm0 = self.params.get('Tm0', 0.5)
        prefix = f"double Tm0 = {Tm0};\n"
        return prefix + """
        double vw_pu = inputs[0];
        double omega_pu = inputs[1];
        double theta = inputs[2];
        
        vw_pu = std::max(vw_pu, 0.01);
        omega_pu = std::max(omega_pu, 0.01);
        
        // Analytical Cp curve
        double lambda_i = 1.0 / (1.0 / (vw_pu / omega_pu + 0.08 * theta) - 0.035 / (theta * theta + 1.0));
        double Cp = 0.5176 * (116.0 * lambda_i - 0.4 * theta - 5.0) * exp(-21.0 * lambda_i) + 0.0068 * (vw_pu / omega_pu);
        
        // Normalize against base Cp when vw=1, w=1, theta=0
        double lambda_i_0 = 1.0 / (1.0 - 0.035);
        double Cp_0 = 0.5176 * (116.0 * lambda_i_0 - 5.0) * exp(-21.0 * lambda_i_0) + 0.0068;
        
        double Cp_factor = std::max(0.0, Cp / Cp_0);
        
        // P = P_nom * Vw_pu^3 * Cp/Cp0
        // Torque = P / omega
        double Pm = Tm0 * vw_pu * vw_pu * vw_pu * Cp_factor;
        double Tm = Pm / omega_pu;
        
        outputs[0] = Tm;
        outputs[1] = Pm;
        """
