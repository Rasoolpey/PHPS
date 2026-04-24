import math
import numpy as np
from typing import Dict, List, Tuple, Any
from src.core import PowerComponent

class Regca1(PowerComponent):
    """
    REGCA1 Renewable Energy Generator/Converter Model.
    """
    @property
    def port_schema(self) -> Dict[str, List[Tuple[str, str, str]]]:
        return {
            'in': [
                ('Ipcmd', 'effort', 'pu'),
                ('Iqcmd', 'effort', 'pu'),
                ('Vterm', 'effort', 'pu'),
                ('Vd', 'effort', 'pu'),
                ('Vq', 'effort', 'pu')
            ],
            'out': [
                ('Ipout', 'flow', 'pu'),
                ('Iqout', 'flow', 'pu'),
                ('Pe', 'flow', 'pu'),
                ('Qe', 'flow', 'pu'),
                ('Id', 'flow', 'pu'),
                ('Iq', 'flow', 'pu')
            ]
        }

    @property
    def state_schema(self) -> List[str]:
        return ['s0_y', 's1_y', 's2_y']

    @property
    def param_schema(self) -> Dict[str, str]:
        return {
            'Tg': 'Converter time constant',
            'Rrpwr': 'Active power rate limit',
            'Brkpt': 'LVPL breakpoint',
            'Zerox': 'LVPL zero crossing',
            'Lvplsw': 'LVPL switch',
            'Lvpl1': 'LVPL gain',
            'Volim': 'Voltage limit for high voltage reactive current management',
            'Lvpnt1': 'High voltage point for LVG',
            'Lvpnt0': 'Low voltage point for LVG',
            'Iolim': 'Lower limit for reactive current',
            'Tfltr': 'Voltage filter time constant',
            'Khv': 'High voltage reactive current gain',
            'Iqrmax': 'Maximum reactive current rate',
            'Iqrmin': 'Minimum reactive current rate'
        }

    @property
    def component_role(self) -> str:
        return 'generator'

    @property
    def observables(self) -> Dict[str, Dict[str, str]]:
        return {
            'Pe':              {'description': 'Active power output',           'unit': 'pu',  'cpp_expr': 'outputs[2]'},
            'Qe':              {'description': 'Reactive power output',         'unit': 'pu',  'cpp_expr': 'outputs[3]'},
            'Ipout':           {'description': 'Active current output',         'unit': 'pu',  'cpp_expr': 'outputs[0]'},
            'Iqout':           {'description': 'Reactive current output',       'unit': 'pu',  'cpp_expr': 'outputs[1]'},
            'Vterm':           {'description': 'Terminal voltage magnitude',    'unit': 'pu',  'cpp_expr': 'inputs[2]'},
            'apparent_power':  {'description': 'Apparent power |S|',           'unit': 'pu',  'cpp_expr': 'sqrt(outputs[2]*outputs[2] + outputs[3]*outputs[3])'},
        }

    @property
    def contributes_norton_admittance(self) -> bool:
        return False  # REGCA1 is a current-source injector, not a Norton equivalent

    def refine_current_source_init(self, x_slice: np.ndarray,
                                   targets: dict,
                                   V_bus: complex) -> np.ndarray:
        """Recompute Ipcmd/Iqcmd/Vterm at the Kron-reduced network voltage.
        Accounts for the Low-Voltage Guard (LVG) and High-Voltage Guard (HVG)
        so that the Python initial condition matches the C++ step function."""
        Vterm = abs(V_bus)

        Pe_pf = targets.get('Pe', 0.0)
        Qe_pf = targets.get('Qe', 0.0)
        # Allow explicit p0 override for slack-bus renewables
        if 'p0' in self.params:
            Pe_pf = float(self.params['p0'])
            targets['Pe'] = Pe_pf

        Lvpnt1 = self.params.get('Lvpnt1', 1.0)
        Lvpnt0 = self.params.get('Lvpnt0', 0.4)
        LVG_y = 1.0
        if Vterm <= Lvpnt0:
            LVG_y = 0.0
        elif Vterm <= Lvpnt1:
            LVG_y = (Vterm - Lvpnt0) / (Lvpnt1 - Lvpnt0)

        Khv   = self.params.get('Khv', 0.0)
        Volim = self.params.get('Volim', 1.2)
        HVG_y = max(0.0, Khv * (Vterm - Volim))

        if Vterm > 0.01 and LVG_y > 0.01:
            Ipcmd_new = Pe_pf / (Vterm * LVG_y)
        else:
            Ipcmd_new = 0.0

        print(f"  [Init] {self.name}: Vterm={Vterm:.5f}, "
              f"LVG_y={LVG_y:.5f}, Ipcmd_new={Ipcmd_new:.5f}")

        if Vterm > 0.01:
            Iqcmd_new = Qe_pf / Vterm + HVG_y
        else:
            Iqcmd_new = 0.0

        targets['Vterm']  = Vterm
        targets['Ipcmd']  = Ipcmd_new
        targets['Iqcmd']  = Iqcmd_new

        x_new = x_slice.copy()
        x_new[0] = Ipcmd_new   # s0_y
        x_new[1] = Iqcmd_new   # s1_y
        x_new[2] = Vterm        # s2_y
        return x_new

    def init_from_phasor(self, V_phasor: complex, I_phasor: complex) -> tuple:
        V0 = abs(V_phasor)
        S0 = V_phasor * np.conj(I_phasor)
        P0 = S0.real
        Q0 = S0.imag

        Ipcmd0 = P0 / V0 if V0 > 0.01 else 0.0
        Iqcmd0 = Q0 / V0 if V0 > 0.01 else 0.0

        x_init = np.array([Ipcmd0, Iqcmd0, V0])
        
        self.params['vd_ri'] = V_phasor.real
        self.params['vq_ri'] = V_phasor.imag
        
        targets = {
            'Vterm': V0,
            'Pe': P0,
            'Qe': Q0,
            'Ipcmd': Ipcmd0,
            'Iqcmd': Iqcmd0,
            'Efd': 0.0,
            'Tm': P0,
            'vd': 0.0,
            'vq': V0,
            'id': 0.0,
            'iq': 0.0,
            'vd_ri': V_phasor.real,
            'vq_ri': V_phasor.imag
        }
        return x_init, targets

    def compute_norton_current(self, x_slice: np.ndarray) -> complex:
        """Return the RI-frame Norton current injection for the Kron Z-bus solve.

        Applies the same LVG and HVG corrections as the C++ step_REGCA1_out,
        so the Python initialization and the C++ simulation see identical
        current injections at the same operating point.  Without this the
        LVG scaling applied in C++ (especially for buses with Vterm < Lvpnt1)
        would create a Te mismatch for the slack generator.
        """
        Vd = self.params.get('vd_ri', 1.0)
        Vq = self.params.get('vq_ri', 0.0)
        Vterm = math.hypot(Vd, Vq)
        if Vterm < 0.01:
            return 0j

        Ip_raw = x_slice[0]  # s0_y = Ipcmd (before LVG)
        Iq_raw = x_slice[1]  # s1_y = Iqcmd (before HVG)

        # --- Low Voltage Gate (LVG) — matches C++ logic ---
        Lvpnt0 = float(self.params.get('Lvpnt0', 0.4))
        Lvpnt1 = float(self.params.get('Lvpnt1', 1.0))
        if Vterm <= Lvpnt0:
            LVG_y = 0.0
        elif Vterm < Lvpnt1:
            LVG_y = (Vterm - Lvpnt0) / (Lvpnt1 - Lvpnt0)
        else:
            LVG_y = 1.0
        Ip = Ip_raw * LVG_y

        # --- High Voltage Gate (HVG) — matches C++ logic ---
        Khv   = float(self.params.get('Khv',   0.7))
        Volim = float(self.params.get('Volim',  1.2))
        Iolim = float(self.params.get('Iolim', -999.0))
        HVG_y = max(0.0, Khv * (Vterm - Volim))
        Iq    = max(Iq_raw - HVG_y, Iolim)

        # Network-frame (Re/Im) current
        It_Re = (Ip * Vd + Iq * Vq) / Vterm
        It_Im = (Ip * Vq - Iq * Vd) / Vterm
        return complex(It_Re, It_Im)

    def refine_at_kron_voltage(self, x_slice: np.ndarray, vd: float, vq: float) -> np.ndarray:
        # For Regca1, vd and vq passed here are from _park_transform.
        # But we need the RI-frame voltages. We can just use the fact that
        # _park_transform was called with delta = x_slice[0] (which is Ipcmd).
        # Actually, it's better to just update vd_ri and vq_ri directly from the network voltage.
        # But we don't have V_bus here.
        # Wait, _park_transform(V_bus, delta) returns:
        # vd = V_Re * sin(delta) - V_Im * cos(delta)
        # vq = V_Re * cos(delta) + V_Im * sin(delta)
        # We can invert this to get V_Re and V_Im, but it's easier to just store V_bus in params.
        # Let's just return x_slice. We will update vd_ri and vq_ri in _compute_network_voltages.
        return x_slice

    def compute_stator_currents(self, x_slice: np.ndarray, vd: float, vq: float) -> tuple:
        return 0.0, 0.0

    def init_from_targets(self, targets: Dict[str, float]) -> Dict[str, float]:
        V0 = targets.get('Vterm', 1.0)
        P0 = targets.get('Pe', 0.0)
        Q0 = targets.get('Qe', 0.0)

        Ipcmd0 = P0 / V0 if V0 > 0.01 else 0.0
        Iqcmd0 = Q0 / V0 if V0 > 0.01 else 0.0

        return self._init_states({
            's0_y': Ipcmd0,
            's1_y': Iqcmd0,
            's2_y': V0,
        })

    def get_cpp_step_code(self) -> str:
        return """
        double Tg_val = std::max(Tg, 0.001);
        double Tfltr_val = std::max(Tfltr, 0.001);
        
        // LVPL
        double LVPL_y = 9999.0;
        if (Lvplsw > 0.5) {
            if (x[2] <= Zerox) {
                LVPL_y = 0.0;
            } else if (x[2] <= Brkpt) {
                double kLVPL = Lvpl1 / (Brkpt - Zerox);
                LVPL_y = (x[2] - Zerox) * kLVPL;
            }
        }
        
        // LVG
        double LVG_y = 1.0;
        if (inputs[2] <= Lvpnt0) {
            LVG_y = 0.0;
        } else if (inputs[2] <= Lvpnt1) {
            double kLVG = 1.0 / (Lvpnt1 - Lvpnt0);
            LVG_y = (inputs[2] - Lvpnt0) * kLVG;
        }
        
        // S0: Lag for Ip
        double s0_upper = std::min(LVPL_y, 9999.0);
        double s0_error = inputs[0] - x[0];
        double s0_rate = s0_error / Tg_val;
        s0_rate = std::min(s0_rate, Rrpwr);
        if (x[0] >= s0_upper && s0_rate > 0) {
            s0_rate = 0.0;
        }
        dxdt[0] = s0_rate;
        
        // S1: Lag for Iq
        double s1_error = inputs[1] - x[1];
        double s1_rate = s1_error / Tg_val;
        s1_rate = std::max(std::min(s1_rate, Iqrmax / Tg_val), Iqrmin / Tg_val);
        dxdt[1] = s1_rate;
        
        // S2: Voltage filter
        dxdt[2] = (inputs[2] - x[2]) / Tfltr_val;
        
        // Outputs
        """

    def get_cpp_compute_outputs_code(self) -> str:
        return """
        // LVG
        double LVG_y = 1.0;
        if (inputs[2] <= Lvpnt0) {
            LVG_y = 0.0;
        } else if (inputs[2] <= Lvpnt1) {
            double kLVG = 1.0 / (Lvpnt1 - Lvpnt0);
            LVG_y = (inputs[2] - Lvpnt0) * kLVG;
        }
        
        outputs[0] = x[0] * LVG_y;
        double HVG_y = std::max(0.0, Khv * (inputs[2] - Volim));
        outputs[1] = std::max(x[1] - HVG_y, Iolim);
        
        outputs[2] = outputs[0] * inputs[2];
        outputs[3] = outputs[1] * inputs[2];
        
        if (inputs[2] > 0.01) {
            outputs[4] = (outputs[0] * inputs[3] + outputs[1] * inputs[4]) / inputs[2];
            outputs[5] = (outputs[0] * inputs[4] - outputs[1] * inputs[3]) / inputs[2];
        } else {
            outputs[4] = 0.0;
            outputs[5] = 0.0;
        }
        """
