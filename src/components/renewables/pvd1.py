import math
import numpy as np
from typing import Dict, List, Tuple, Any
from src.core import PowerComponent

class Pvd1(PowerComponent):
    """
    PVD1 WECC Distributed PV Model.
    """

    @property
    def contributes_norton_admittance(self) -> bool:
        return False  # PVD1 is a current-source injector, not a Norton equivalent
    @property
    def port_schema(self) -> Dict[str, List[Tuple[str, str, str]]]:
        return {
            'in': [
                ('Vterm', 'effort', 'pu'),
                ('f_pu', 'signal', 'pu'),
                ('Vd', 'effort', 'pu'),
                ('Vq', 'effort', 'pu'),
                ('ue', 'signal', 'pu')
            ],
            'out': [
                ('Ipout', 'flow', 'pu'),
                ('Iqout', 'flow', 'pu'),
                ('Pe', 'flow', 'pu'),
                ('Qe', 'flow', 'pu'),
                ('It_Re', 'flow', 'pu'),
                ('It_Im', 'flow', 'pu')
            ]
        }

    @property
    def state_schema(self) -> List[str]:
        return ['Ipout', 'Iqout']

    @property
    def param_schema(self) -> Dict[str, str]:
        return {
            'tip': 'Active current lag time constant',
            'tiq': 'Reactive current lag time constant',
            'fn': 'Nominal frequency',
            'pqflag': 'P/Q priority flag',
            'qmx': 'Max reactive power',
            'qmn': 'Min reactive power',
            'pmx': 'Max active power',
            'v0': 'Voltage droop point 0',
            'v1': 'Voltage droop point 1',
            'dqdv': 'Voltage droop gain',
            'fdbd': 'Frequency deadband',
            'ddn': 'Frequency droop gain',
            'vt0': 'Voltage trip point 0',
            'vt1': 'Voltage trip point 1',
            'vt2': 'Voltage trip point 2',
            'vt3': 'Voltage trip point 3',
            'ft0': 'Frequency trip point 0',
            'ft1': 'Frequency trip point 1',
            'ft2': 'Frequency trip point 2',
            'ft3': 'Frequency trip point 3',
            'ialim': 'Current limit',
            'recflag': 'Recovery flag',
            'gammap': 'Active power fraction',
            'gammaq': 'Reactive power fraction',
            'p0': 'Initial active power',
            'q0': 'Initial reactive power',
            'Pref': 'Active power reference',
            'Qref': 'Reactive power reference'
        }

    @property
    def component_role(self) -> str:
        return 'generator'

    @property
    def observables(self) -> Dict[str, Dict[str, str]]:
        return {
            'Pe':             {'description': 'Active power output',            'unit': 'pu',  'cpp_expr': 'outputs[2]'},
            'Qe':             {'description': 'Reactive power output',          'unit': 'pu',  'cpp_expr': 'outputs[3]'},
            'Ipout':          {'description': 'Active current (state, lagged)',  'unit': 'pu',  'cpp_expr': 'x[0]'},
            'Iqout':          {'description': 'Reactive current (state, lagged)','unit': 'pu',  'cpp_expr': 'x[1]'},
            'Vterm':          {'description': 'Terminal voltage magnitude',     'unit': 'pu',  'cpp_expr': 'inputs[0]'},
            'f_pu':           {'description': 'Frequency input (from BusFreq)', 'unit': 'pu',  'cpp_expr': 'inputs[1]'},
            'apparent_power': {'description': 'Apparent power |S|',            'unit': 'pu',  'cpp_expr': 'sqrt(outputs[2]*outputs[2] + outputs[3]*outputs[3])'},
        }

    def init_from_phasor(self, V_phasor: complex, I_phasor: complex) -> tuple:
        V0 = abs(V_phasor)
        S0 = V_phasor * np.conj(I_phasor)
        P0 = S0.real
        Q0 = S0.imag

        # Use p0 and q0 from params if available, otherwise use P0, Q0
        p0_param = self.params.get('p0', P0)
        q0_param = self.params.get('q0', Q0)
        
        gammap = self.params.get('gammap', 1.0)
        gammaq = self.params.get('gammaq', 1.0)
        
        Pref = p0_param
        Qref = q0_param
        
        self.params['Pref'] = Pref
        self.params['Qref'] = Qref

        P0_scaled = gammap * Pref
        Q0_scaled = gammaq * Qref

        Ip0 = P0_scaled / V0 if V0 > 0.01 else 0.0
        Iq0 = Q0_scaled / V0 if V0 > 0.01 else 0.0

        x_init = np.array([Ip0, Iq0])
        
        self.params['vd_ri'] = V_phasor.real
        self.params['vq_ri'] = V_phasor.imag
        
        targets = {
            'Vterm': V0,
            'f_pu': 1.0,
            'Pe': P0_scaled,
            'Qe': Q0_scaled,
            'Ipout': Ip0,
            'Iqout': Iq0,
            'Vd': V_phasor.real,
            'Vq': V_phasor.imag,
            'Efd': 0.0,
            'Tm': 0.0,
            'vd': V_phasor.real,
            'vq': V_phasor.imag,
            'id': 0.0,
            'iq': 0.0,
            'vd_ri': V_phasor.real,
            'vq_ri': V_phasor.imag
        }
        return x_init, targets

    def compute_norton_current(self, x_slice: np.ndarray) -> complex:
        Vd = self.params.get('vd_ri', 1.0)
        Vq = self.params.get('vq_ri', 0.0)
        Vterm = math.hypot(Vd, Vq)
        if Vterm < 0.01:
            return 0j

        Ip = x_slice[0]
        Iq = x_slice[1]

        It_Re = (Ip * Vd + Iq * Vq) / Vterm
        It_Im = (Ip * Vq - Iq * Vd) / Vterm
        return complex(It_Re, It_Im)

    def refine_at_kron_voltage(self, x_slice: np.ndarray, vd: float, vq: float) -> np.ndarray:
        Vterm = math.hypot(vd, vq)
        if Vterm < 0.01:
            return x_slice
            
        gammap = self.params.get('gammap', 1.0)
        gammaq = self.params.get('gammaq', 1.0)
        Pref = self.params.get('Pref', 0.0)
        Qref = self.params.get('Qref', 0.0)
        
        P0_scaled = gammap * Pref
        Q0_scaled = gammaq * Qref
        
        x_slice[0] = P0_scaled / Vterm
        x_slice[1] = Q0_scaled / Vterm
        return x_slice

    def compute_stator_currents(self, x_slice: np.ndarray, vd: float, vq: float) -> tuple:
        return 0.0, 0.0

    def init_from_targets(self, targets: Dict[str, float]) -> Dict[str, float]:
        V0 = targets.get('Vterm', 1.0)
        P0 = targets.get('Pe', 0.0)
        Q0 = targets.get('Qe', 0.0)

        Ip0 = P0 / V0 if V0 > 0.01 else 0.0
        Iq0 = Q0 / V0 if V0 > 0.01 else 0.0

        return self._init_states({
            'Ipout': Ip0,
            'Iqout': Iq0,
        })

    def get_cpp_step_code(self) -> str:
        return """
        double V = inputs[0];
        double f_pu = inputs[1];
        double vp = std::max(V, 0.01);
        
        double fn_val = fn > 0 ? fn : 60.0;
        double fHz = fn_val * f_pu;
        double Fdev = fn_val - fHz;
        
        double abs_fdbd = std::abs(fdbd);
        double DB_y = 0.0;
        if (Fdev > abs_fdbd) {
            DB_y = ddn * (Fdev - abs_fdbd);
        } else if (Fdev < -abs_fdbd) {
            DB_y = ddn * (Fdev + abs_fdbd);
        }
        
        double qref0 = gammaq * Qref;
        double Qdrp = 0.0;
        if (std::abs(dqdv) > 1e-8) {
            double Vql = v0 + (qmx - qref0) / dqdv;
            double Vqu = v1 - (qref0 - qmn) / dqdv;
            Vql = std::min(Vql, v0);
            Vqu = std::max(Vqu, v1);
            
            if (V < Vql) {
                Qdrp = qmx;
            } else if (V < v0) {
                Qdrp = (Vql < v0) ? qmx + dqdv * (Vqu - V) : qmx;
            } else if (V <= v1) {
                Qdrp = 0.0;
            } else if (V <= Vqu) {
                Qdrp = dqdv * (v1 - V);
            } else {
                Qdrp = qmn;
            }
        }
        
        double Fvl = 1.0;
        if (V <= vt0) {
            Fvl = 0.0;
        } else if (V <= vt1) {
            Fvl = (vt1 - vt0) > 1e-6 ? (V - vt0) / (vt1 - vt0) : 1.0;
        }
        
        double Fvh = 1.0;
        if (V >= vt3) {
            Fvh = 0.0;
        } else if (V >= vt2) {
            Fvh = (vt3 - vt2) > 1e-6 ? (vt3 - V) / (vt3 - vt2) : 1.0;
        }
        
        double Ffl = 1.0;
        if (fHz <= ft0) {
            Ffl = 0.0;
        } else if (fHz <= ft1) {
            Ffl = (ft1 - ft0) > 1e-6 ? (fHz - ft0) / (ft1 - ft0) : 1.0;
        }
        
        double Ffh = 1.0;
        if (fHz >= ft3) {
            Ffh = 0.0;
        } else if (fHz >= ft2) {
            Ffh = (ft3 - ft2) > 1e-6 ? (ft3 - fHz) / (ft3 - ft2) : 1.0;
        }
        
        double R_gain = (recflag > 0.5) ? Fvl * Fvh * Ffl * Ffh : 1.0;
        
        double pref0 = gammap * Pref;
        double Psum = pref0 + DB_y;
        Psum = std::max(0.0, std::min(Psum, pmx));
        double Ipul = Psum / vp;
        
        double Qsum = qref0 + Qdrp;
        Qsum = std::max(qmn, std::min(Qsum, qmx));
        double Iqul = Qsum / vp;
        
        double Ipcmd = 0.0;
        double Iqcmd = 0.0;
        
        if (pqflag > 0.5) {
            double Ipmax = ialim;
            Ipcmd = std::max(0.0, std::min(Ipul * R_gain, Ipmax));
            double Iq_remaining = std::max(0.0, ialim * ialim - Ipcmd * Ipcmd);
            double Iqmax = std::sqrt(Iq_remaining);
            Iqcmd = std::max(-Iqmax, std::min(Iqul * R_gain, Iqmax));
        } else {
            double Iqmax = ialim;
            Iqcmd = std::max(-Iqmax, std::min(Iqul * R_gain, Iqmax));
            double Ip_remaining = std::max(0.0, ialim * ialim - Iqcmd * Iqcmd);
            double Ipmax = std::sqrt(Ip_remaining);
            Ipcmd = std::max(0.0, std::min(Ipul * R_gain, Ipmax));
        }
        
        double tip_val = std::max(tip, 0.001);
        double tiq_val = std::max(tiq, 0.001);
        
        dxdt[0] = (Ipcmd - x[0]) / tip_val;
        dxdt[1] = (Iqcmd - x[1]) / tiq_val;
        """

    def get_cpp_compute_outputs_code(self) -> str:
        return """
        double V = inputs[0];
        double ue = inputs[4];
        
        double Ipout_eff = (ue > 0.5) ? 0.0 : x[0];
        double Iqout_eff = (ue > 0.5) ? 0.0 : x[1];
        
        outputs[0] = Ipout_eff;
        outputs[1] = Iqout_eff;
        outputs[2] = Ipout_eff * V;
        outputs[3] = Iqout_eff * V;
        
        if (V > 0.01) {
            outputs[4] = (Ipout_eff * inputs[2] + Iqout_eff * inputs[3]) / V;
            outputs[5] = (Ipout_eff * inputs[3] - Iqout_eff * inputs[2]) / V;
        } else {
            outputs[4] = 0.0;
            outputs[5] = 0.0;
        }
        """
