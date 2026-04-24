import numpy as np
from typing import Dict, List, Tuple, Any
from src.core import PowerComponent

class Ieeeg1(PowerComponent):
    """
    IEEE Type G1 Steam Turbine Governor (IEEEG1) implementation for C++ code generation.
    """
    
    @property
    def port_schema(self) -> Dict[str, List[Tuple[str, str, str]]]:
        return {
            'in': [
                ('omega', 'flow',   'pu'),
                ('Pref',  'signal', 'pu'),
                ('u_agc', 'signal', 'pu'),   # AGC correction (+ve = increase setpoint)
            ],
            'out': [
                ('Tm', 'effort', 'pu')
            ]
        }

    @property
    def required_ports(self):
        """u_agc is optional; defaults to 0.0 when no AGC is wired."""
        return ['omega', 'Pref']

    @property
    def state_schema(self) -> List[str]:
        return ['x1', 'x2']

    @property
    def param_schema(self) -> Dict[str, str]:
        return {
            'K': 'Gain',
            'T1': 'Time Constant 1',
            'T2': 'Time Constant 2',
            'T3': 'Time Constant 3',
            'UO': 'Max opening rate',
            'UC': 'Max closing rate',
            'PMAX': 'Max Power',
            'PMIN': 'Min Power',
            'T4': '', 'K1': '', 'K2': '', 'T5': '', 'K3': '', 'K4': '', 
            'T6': '', 'K5': '', 'K6': '', 'T7': '', 'K7': '', 'K8': ''
        }
        
    @property
    def observables(self) -> Dict[str, Dict[str, str]]:
        return {
            'Tm': {'description': 'Mechanical Torque', 'unit': 'pu', 'cpp_expr': 'x[1]'}
        }

    def get_cpp_compute_outputs_code(self) -> str:
        return "outputs[0] = x[1];"

    def get_cpp_step_code(self) -> str:
        return """
            double omega  = inputs[0];
            double Pref   = inputs[1];
            double u_agc  = inputs[2];   // AGC Pref correction (0.0 when AGC not wired)

            double x1 = x[0];
            double x2 = x[1];

            double err = Pref + u_agc - omega;
            dxdt[0] = (K * err - x1) / T1;
            dxdt[1] = (x1 - x2) / T3;
        """

    # --- Initialization Contract ---

    @property
    def component_role(self) -> str:
        return 'governor'

    def init_from_targets(self, targets: dict) -> np.ndarray:
        """Set x1=x2=Tm, Pref = 1 + Tm/K."""
        Tm = float(targets.get('Tm', 0.0))
        K  = float(self.params.get('K', 20.0))
        PMIN = float(self.params.get('PMIN', -999.0))
        PMAX = float(self.params.get('PMAX', 999.0))
        if Tm < PMIN:
            print(f"  [WARN] {self.name}: Required Tm={Tm:.6f} < PMIN={PMIN}. "
                  f"Clamping governor output to PMIN.")
            Tm = PMIN
        elif Tm > PMAX:
            print(f"  [WARN] {self.name}: Required Tm={Tm:.6f} > PMAX={PMAX}. "
                  f"Clamping governor output to PMAX.")
            Tm = PMAX
        self.params['Pref'] = 1.0 + Tm / K
        return np.array([Tm, Tm])

    def update_from_te(self, x_slice: np.ndarray, Te: float) -> tuple:
        """Update governor states and Pref so Tm = Te at equilibrium."""
        x = x_slice.copy()
        K  = float(self.params.get('K', 20.0))
        PMIN = float(self.params.get('PMIN', -999.0))
        PMAX = float(self.params.get('PMAX', 999.0))
        Tm = Te
        if Tm < PMIN:
            print(f"  [WARN] {self.name}: Required Tm={Tm:.6f} < PMIN={PMIN}. "
                  f"Clamping governor output to PMIN.")
            Tm = PMIN
        elif Tm > PMAX:
            print(f"  [WARN] {self.name}: Required Tm={Tm:.6f} > PMAX={PMAX}. "
                  f"Clamping governor output to PMAX.")
            Tm = PMAX
        old_Pref = float(self.params.get('Pref', 0.0))
        Pref_new = 1.0 + Tm / K
        self.params['Pref'] = Pref_new
        x[0] = Tm
        x[1] = Tm
        print(f"  [Init] {self.name}: Pref {old_Pref:.3f}\u2192{Pref_new:.3f}, Tm\u2192Te={Te:.4f}")
        return x, Pref_new
