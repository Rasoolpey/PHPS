import numpy as np
from typing import Dict, List, Tuple, Any
from src.core import PowerComponent

class AvrType1(PowerComponent):
    """
    IEEE Type 1 Exciter (AVR) implementation for C++ code generation.
    """
    
    @property
    def port_schema(self) -> Dict[str, List[Tuple[str, str, str]]]:
        return {
            'in': [
                ('Vterm', 'signal', 'pu'), 
                ('Vref', 'signal', 'pu')
            ],
            'out': [
                ('Efd', 'effort', 'pu')
            ]
        }

    @property
    def state_schema(self) -> List[str]:
        return ['Vm', 'Vr', 'Efd', 'Vf']

    @property
    def param_schema(self) -> Dict[str, str]:
        return {
            'Tr': 'Voltage transducer time constant',
            'Ka': 'Regulator gain',
            'Ta': 'Regulator time constant',
            'Ke': 'Exciter constant',
            'Te': 'Exciter time constant',
            'Kf': 'Rate feedback gain',
            'Tf': 'Rate feedback time constant',
            'Vr_min': 'Min regulator output',
            'Vr_max': 'Max regulator output',
            'Efd_min': 'Min field voltage',
            'Efd_max': 'Max field voltage'
        }
        
    @property
    def observables(self) -> Dict[str, Dict[str, str]]:
        return {
            'Efd': {'description': 'Field Voltage', 'unit': 'pu', 'cpp_expr': 'x[2]'}
        }

    def get_cpp_compute_outputs_code(self) -> str:
        return """
            outputs[0] = x[2]; // Efd
        """

    def get_cpp_step_code(self) -> str:
        return """
            double Vterm = inputs[0];
            double Vref = inputs[1];
            
            double Vm = x[0];
            double Vr = x[1];
            double Efd = x[2];
            double Vf = x[3];
            
            // 1. Measurement Filter
            dxdt[0] = (Vterm - Vm) / Tr;
            
            // 2. Regulator
            double error = Vref - Vm - Vf;
            double dVr = (Ka * error - Vr) / Ta;
            
            if (Vr >= Vr_max && dVr > 0) dVr = 0;
            if (Vr <= Vr_min && dVr < 0) dVr = 0;
            
            dxdt[1] = dVr;
            
            // 3. Exciter
            dxdt[2] = (Vr - Ke * Efd) / Te;
            
            // 4. Rate Feedback
            double dEfd = (Vr - Ke * Efd) / Te;
            dxdt[3] = -Vf/Tf + (Kf/Tf) * dEfd;
        """

    # --- Initialization Contract ---

    @property
    def component_role(self) -> str:
        return 'exciter'

    def compute_efd_output(self, x_slice: np.ndarray) -> float:
        """Efd state (state[2])."""
        return float(x_slice[self.state_schema.index('Efd')])

    def init_from_targets(self, targets: dict) -> np.ndarray:
        """Back-solve 4 states: [Vm, Vr, Efd, Vf=0] from Efd target."""
        Efd_req = float(targets.get('Efd', 1.0))
        Vt      = float(targets.get('Vt',  1.0))
        p  = self.params
        Ke = float(p.get('Ke', 1.0))
        Ka = float(p.get('Ka', 10.0))
        Vr = Ke * Efd_req
        Vm = Vt
        self.params['Efd_eff'] = float(Efd_req)
        self.params['Vref']    = Vm + Vr / Ka
        return np.array([Vm, Vr, Efd_req, 0.0])

    def compute_efd_output(self, x_slice: np.ndarray) -> float:
        """Efd is state[2]."""
        return float(x_slice[self.state_schema.index('Efd')])

    def efd_output_expr(self, state_offset: int) -> str:
        efd_i = self.state_schema.index('Efd')
        return f"x[{state_offset + efd_i}]"
