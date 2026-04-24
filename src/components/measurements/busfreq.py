import math
import numpy as np
from typing import Dict, List, Tuple, Any
from src.core import PowerComponent

class BusFreq(PowerComponent):
    """
    Bus Frequency Measurement Device.
    Uses a low-pass filter on the unwrapped bus voltage angle to compute frequency deviation.
    """
    @property
    def port_schema(self) -> Dict[str, List[Tuple[str, str, str]]]:
        return {
            'in': [
                ('Vd', 'effort', 'pu'),
                ('Vq', 'effort', 'pu')
            ],
            'out': [
                ('f_pu', 'signal', 'pu'),
                ('omega_dev', 'signal', 'pu')
            ]
        }

    @property
    def state_schema(self) -> List[str]:
        return ['x_theta']

    @property
    def param_schema(self) -> Dict[str, str]:
        return {
            'Tf': 'Filter time constant (s)',
            'fn': 'Nominal frequency (Hz)'
        }

    @property
    def component_role(self) -> str:
        return 'passive'

    @property
    def observables(self) -> Dict[str, Dict[str, str]]:
        return {
            'f_pu':      {'description': 'Measured frequency deviation (pu)',     'unit': 'pu',  'cpp_expr': 'outputs[0]'},
            'omega_dev': {'description': 'Angular frequency deviation (rad/s)',    'unit': 'pu',  'cpp_expr': 'outputs[1]'},
            'theta':     {'description': 'Filtered bus voltage angle (rad)',      'unit': 'rad', 'cpp_expr': 'x[0]'},
            'Vd':        {'description': 'Bus voltage real component (input)',    'unit': 'pu',  'cpp_expr': 'inputs[0]'},
            'Vq':        {'description': 'Bus voltage imaginary component (input)','unit': 'pu', 'cpp_expr': 'inputs[1]'},
        }

    def init_from_phasor(self, V_phasor: complex, I_phasor: complex) -> tuple:
        Vd = V_phasor.real
        Vq = V_phasor.imag
        theta0 = math.atan2(Vq, Vd)
        
        x_init = np.array([theta0])
        
        targets = {
            'f_pu': 1.0,
            'omega_dev': 0.0,
            'Vd': Vd,
            'Vq': Vq,
            'Efd': 0.0,
            'Tm': 0.0,
            'vd': 0.0,
            'vq': 0.0,
            'id': 0.0,
            'iq': 0.0,
            'vd_ri': Vd,
            'vq_ri': Vq
        }
        return x_init, targets

    def compute_norton_current(self, x_slice: np.ndarray) -> complex:
        return 0j

    def refine_at_kron_voltage(self, x_slice: np.ndarray, vd: float, vq: float) -> np.ndarray:
        # vd and vq here are the network voltages Vd and Vq, because BusFreq is not a generator
        # and doesn't have a rotor angle. Wait, initialization.py calls _park_transform
        # which uses x0[off] as delta. But BusFreq's x0[0] is x_theta, not delta!
        # So vd and vq passed here are NOT the network voltages!
        return x_slice

    def compute_stator_currents(self, x_slice: np.ndarray, vd: float, vq: float) -> tuple:
        return 0.0, 0.0

    def init_from_targets(self, targets: Dict[str, float]) -> Dict[str, float]:
        Vd = targets.get('Vd', 1.0)
        Vq = targets.get('Vq', 0.0)
        theta0 = math.atan2(Vq, Vd)
        return self._init_states({
            'x_theta': theta0,
            'f_pu': 1.0,
            'omega_dev': 0.0,
        })

    def get_cpp_step_code(self) -> str:
        return """
        double Tf_val = std::max(Tf, 0.001);
        double fn_val = fn > 0 ? fn : 60.0;
        double omega_b = 2.0 * M_PI * fn_val;
        
        double Vd = inputs[0];
        double Vq = inputs[1];
        double theta = atan2(Vq, Vd);
        
        double dtheta = theta - x[0];
        while (dtheta > M_PI) dtheta -= 2.0 * M_PI;
        while (dtheta < -M_PI) dtheta += 2.0 * M_PI;
        
        dxdt[0] = dtheta / Tf_val;
        """

    def get_cpp_compute_outputs_code(self) -> str:
        return """
        double Tf_val = std::max(Tf, 0.001);
        double fn_val = fn > 0 ? fn : 60.0;
        double omega_b = 2.0 * M_PI * fn_val;
        
        double Vd = inputs[0];
        double Vq = inputs[1];
        double theta = atan2(Vq, Vd);
        
        double dtheta = theta - x[0];
        while (dtheta > M_PI) dtheta -= 2.0 * M_PI;
        while (dtheta < -M_PI) dtheta += 2.0 * M_PI;
        
        double omega_dev = dtheta / Tf_val;
        
        outputs[0] = 1.0 + omega_dev / omega_b; // f_pu
        outputs[1] = omega_dev;                 // omega_dev
        """
