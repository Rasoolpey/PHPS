import math
import numpy as np
from typing import Dict, List, Tuple, Any
from src.core import PowerComponent

class Dgprct1(PowerComponent):
    """
    DGPRCT1 IEEE 1547.2018 DG Protection Model.
    """
    @property
    def port_schema(self) -> Dict[str, List[Tuple[str, str, str]]]:
        return {
            'in': [
                ('Vterm', 'effort', 'pu'),
                ('f_pu', 'signal', 'pu')
            ],
            'out': [
                ('ue', 'signal', 'pu')
            ]
        }

    @property
    def state_schema(self) -> List[str]:
        return ['IAWfl1', 'IAWfl2', 'IAWfu1', 'IAWfu2',
                'IAWVl1', 'IAWVl2', 'IAWVl3', 'IAWVu1', 'IAWVu2']

    @property
    def param_schema(self) -> Dict[str, str]:
        return {
            'fn': 'Nominal frequency',
            'fen': 'Frequency protection enable',
            'Ven': 'Voltage protection enable',
            'fl3': 'Frequency threshold l3',
            'fl2': 'Frequency threshold l2',
            'fl1': 'Frequency threshold l1',
            'fu1': 'Frequency threshold u1',
            'fu2': 'Frequency threshold u2',
            'fu3': 'Frequency threshold u3',
            'Tfl1': 'Time limit fl1',
            'Tfl2': 'Time limit fl2',
            'Tfu1': 'Time limit fu1',
            'Tfu2': 'Time limit fu2',
            'vl4': 'Voltage threshold l4',
            'vl3': 'Voltage threshold l3',
            'vl2': 'Voltage threshold l2',
            'vl1': 'Voltage threshold l1',
            'vu1': 'Voltage threshold u1',
            'vu2': 'Voltage threshold u2',
            'vu3': 'Voltage threshold u3',
            'Tvl1': 'Time limit vl1',
            'Tvl2': 'Time limit vl2',
            'Tvl3': 'Time limit vl3',
            'Tvu1': 'Time limit vu1',
            'Tvu2': 'Time limit vu2',
            'Tres': 'Reset time constant'
        }

    @property
    def component_role(self) -> str:
        return 'passive'

    @property
    def observables(self) -> Dict[str, Dict[str, str]]:
        return {
            'ue':     {'description': 'Protection trip signal (1=connected, 0=tripped)', 'unit': '',   'cpp_expr': 'outputs[0]'},
            'Vterm':  {'description': 'Terminal voltage magnitude',                     'unit': 'pu', 'cpp_expr': 'inputs[0]'},
            'f_pu':   {'description': 'Frequency input (from BusFreq)',                 'unit': 'pu', 'cpp_expr': 'inputs[1]'},
        }

    def init_from_phasor(self, V_phasor: complex, I_phasor: complex) -> tuple:
        x_init = np.zeros(9)
        targets = {
            'Vterm': abs(V_phasor),
            'f_pu': 1.0,
            'ue': 0.0,
            'Efd': 0.0,
            'Tm': 0.0,
            'vd': 0.0,
            'vq': 0.0,
            'id': 0.0,
            'iq': 0.0,
            'vd_ri': 0.0,
            'vq_ri': 0.0
        }
        return x_init, targets

    def compute_norton_current(self, x_slice: np.ndarray) -> complex:
        return 0j

    def refine_at_kron_voltage(self, x_slice: np.ndarray, vd: float, vq: float) -> np.ndarray:
        return x_slice

    def compute_stator_currents(self, x_slice: np.ndarray, vd: float, vq: float) -> tuple:
        return 0.0, 0.0

    def init_from_targets(self, targets: Dict[str, float]) -> Dict[str, float]:
        return self._init_states({
            'IAWfl1': 0.0, 'IAWfl2': 0.0, 'IAWfu1': 0.0, 'IAWfu2': 0.0,
            'IAWVl1': 0.0, 'IAWVl2': 0.0, 'IAWVl3': 0.0, 'IAWVu1': 0.0, 'IAWVu2': 0.0,
            'ue': 0.0,
        })

    def get_cpp_step_code(self) -> str:
        return """
        double V = inputs[0];
        double f_pu = inputs[1];
        
        double fn_val = fn > 0 ? fn : 60.0;
        double fHz = fn_val * f_pu;
        
        double Tres_val = std::max(Tres, 0.001);
        
        // Frequency protection
        if (fen > 0.5 && fHz > fl2 && fHz < fl1) {
            dxdt[0] = 1.0;
        } else {
            dxdt[0] = (x[0] > 1e-8) ? -x[0] / Tres_val : 0.0;
        }
        
        if (fen > 0.5 && fHz > fl3 && fHz <= fl2) {
            dxdt[1] = 1.0;
        } else {
            dxdt[1] = (x[1] > 1e-8) ? -x[1] / Tres_val : 0.0;
        }
        
        if (fen > 0.5 && fHz > fu1 && fHz < fu2) {
            dxdt[2] = 1.0;
        } else {
            dxdt[2] = (x[2] > 1e-8) ? -x[2] / Tres_val : 0.0;
        }
        
        if (fen > 0.5 && fHz >= fu2 && fHz < fu3) {
            dxdt[3] = 1.0;
        } else {
            dxdt[3] = (x[3] > 1e-8) ? -x[3] / Tres_val : 0.0;
        }
        
        // Voltage protection
        if (Ven > 0.5 && V > vl2 && V < vl1) {
            dxdt[4] = 1.0;
        } else {
            dxdt[4] = (x[4] > 1e-8) ? -x[4] / Tres_val : 0.0;
        }
        
        if (Ven > 0.5 && V > vl3 && V <= vl2) {
            dxdt[5] = 1.0;
        } else {
            dxdt[5] = (x[5] > 1e-8) ? -x[5] / Tres_val : 0.0;
        }
        
        if (Ven > 0.5 && V > vl4 && V <= vl3) {
            dxdt[6] = 1.0;
        } else {
            dxdt[6] = (x[6] > 1e-8) ? -x[6] / Tres_val : 0.0;
        }
        
        if (Ven > 0.5 && V > vu1 && V < vu2) {
            dxdt[7] = 1.0;
        } else {
            dxdt[7] = (x[7] > 1e-8) ? -x[7] / Tres_val : 0.0;
        }
        
        if (Ven > 0.5 && V >= vu2 && V < vu3) {
            dxdt[8] = 1.0;
        } else {
            dxdt[8] = (x[8] > 1e-8) ? -x[8] / Tres_val : 0.0;
        }
        
        // Anti-windup
        double limits[9] = {Tfl1, Tfl2, Tfu1, Tfu2, Tvl1, Tvl2, Tvl3, Tvu1, Tvu2};
        for (int i = 0; i < 9; i++) {
            if (x[i] >= limits[i] && dxdt[i] > 0) {
                dxdt[i] = 0.0;
            }
            if (x[i] <= 0 && dxdt[i] < 0) {
                dxdt[i] = 0.0;
            }
        }
        """

    def get_cpp_compute_outputs_code(self) -> str:
        return """
        double threshold = 0.8;
        double dsum = 0.0;
        
        if (fen > 0.5) {
            if (x[0] >= threshold * Tfl1) dsum += 1.0;
            if (x[1] >= threshold * Tfl2) dsum += 1.0;
            if (x[2] >= threshold * Tfu1) dsum += 1.0;
            if (x[3] >= threshold * Tfu2) dsum += 1.0;
        }
        
        if (Ven > 0.5) {
            if (x[4] >= threshold * Tvl1) dsum += 1.0;
            if (x[5] >= threshold * Tvl2) dsum += 1.0;
            if (x[6] >= threshold * Tvl3) dsum += 1.0;
            if (x[7] >= threshold * Tvu1) dsum += 1.0;
            if (x[8] >= threshold * Tvu2) dsum += 1.0;
        }
        
        outputs[0] = (dsum > 0.5) ? 1.0 : 0.0;
        """
