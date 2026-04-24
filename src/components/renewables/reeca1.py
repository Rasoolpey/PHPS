import math
import numpy as np
from typing import Dict, List, Tuple, Any
from src.core import PowerComponent

class Reeca1(PowerComponent):
    """
    REECA1 Renewable Energy Electrical Control Model.
    """
    @property
    def port_schema(self) -> Dict[str, List[Tuple[str, str, str]]]:
        return {
            'in': [
                ('Vterm', 'effort', 'pu'),
                ('Pe', 'flow', 'pu'),
                ('Qe', 'flow', 'pu'),
                ('Pext', 'effort', 'pu'),
                ('Qext', 'effort', 'pu'),
                ('wg', 'effort', 'pu')
            ],
            'out': [
                ('Ipcmd', 'effort', 'pu'),
                ('Iqcmd', 'flow', 'pu'),
                ('Pord', 'effort', 'pu')
            ]
        }

    @property
    def state_schema(self) -> List[str]:
        return ['s0_y', 's1_y', 'piq_xi', 's5_y']

    @property
    def param_schema(self) -> Dict[str, str]:
        return {
            'Trv': 'Voltage filter time constant',
            'Tp': 'Active power filter time constant',
            'Tpord': 'Active power order time constant',
            'Kqp': 'Reactive power PI proportional gain',
            'Kqi': 'Reactive power PI integral gain',
            'Kvp': 'Voltage PI proportional gain',
            'Kvi': 'Voltage PI integral gain',
            'Vref0': 'Voltage reference 0',
            'Vref1': 'Voltage reference 1',
            'dbd1': 'Deadband lower limit',
            'dbd2': 'Deadband upper limit',
            'Kqv': 'Voltage control gain',
            'Vdip': 'Voltage dip threshold',
            'Vup': 'Voltage up threshold',
            'QMax': 'Maximum reactive power',
            'QMin': 'Minimum reactive power',
            'VMAX': 'Maximum voltage PI output',
            'VMIN': 'Minimum voltage PI output',
            'PMAX': 'Maximum active power',
            'PMIN': 'Minimum active power',
            'Imax': 'Maximum current',
            'dPmax': 'Maximum active power rate',
            'dPmin': 'Minimum active power rate',
            'PFFLAG': 'Power factor control flag',
            'VFLAG': 'Voltage control flag',
            'QFLAG': 'Reactive power control flag',
            'PFLAG': 'Active power control flag',
            'PQFLAG': 'P/Q priority flag'
        }

    @property
    def component_role(self) -> str:
        return 'renewable_controller'

    @property
    def observables(self) -> Dict[str, Dict[str, str]]:
        return {
            'Ipcmd':  {'description': 'Active current command to REGCA1',    'unit': 'pu',  'cpp_expr': 'outputs[0]'},
            'Iqcmd':  {'description': 'Reactive current command to REGCA1',  'unit': 'pu',  'cpp_expr': 'outputs[1]'},
            'Pord':   {'description': 'Active power order',                  'unit': 'pu',  'cpp_expr': 'outputs[2]'},
            'Vterm':  {'description': 'Terminal voltage (measured)',         'unit': 'pu',  'cpp_expr': 'inputs[0]'},
            'wg':     {'description': 'Generator speed input',              'unit': 'pu',  'cpp_expr': 'inputs[5]'},
        }

    def get_associated_generator(self, comp_map: dict):
        """Return the name of the REGCA1 this REECA1 is connected to via param 'reg'."""
        return self.params.get('reg')

    def init_from_targets(self, targets: Dict[str, float]) -> Dict[str, float]:
        V0 = targets.get('Vterm', 1.0)
        P0 = targets.get('Pe', 0.0)
        Q0 = targets.get('Qe', 0.0)

        Ipcmd0 = targets.get('Ipcmd', P0 / V0 if V0 > 0.01 else 0.0)
        Iqcmd0 = targets.get('Iqcmd', Q0 / V0 if V0 > 0.01 else 0.0)
        
        print(f"  [Init] {self.name}: targets['Ipcmd']={targets.get('Ipcmd')}, Ipcmd0={Ipcmd0}")

        # Store initial values in params for use in step
        self.params['Ipcmd0'] = Ipcmd0
        self.params['Iqcmd0'] = Iqcmd0
        self.params['qref0'] = Q0
        self.params['p0'] = Ipcmd0 * V0
        self.params['q0'] = Q0
        
        if abs(P0) > 1e-6:
            self.params['pfaref0'] = math.atan(Q0 / P0)
        else:
            self.params['pfaref0'] = 0.0

        return self._init_states({
            's0_y': V0,
            's1_y': P0,
            'piq_xi': Iqcmd0 if self.params.get('QFLAG', 0) > 0.5 else 0.0,
            's5_y': Ipcmd0 * V0,
        })

    def get_cpp_step_code(self) -> str:
        Ipcmd0 = self.params.get('Ipcmd0', 0.0)
        Iqcmd0 = self.params.get('Iqcmd0', 0.0)
        qref0 = self.params.get('qref0', 0.0)
        p0 = self.params.get('p0', 0.0)
        q0 = self.params.get('q0', 0.0)
        pfaref0 = self.params.get('pfaref0', 0.0)
        
        prefix = f"""
        double Ipcmd0 = {Ipcmd0};
        double Iqcmd0 = {Iqcmd0};
        double qref0 = {qref0};
        double p0 = {p0};
        double q0 = {q0};
        double pfaref0 = {pfaref0};
        """
        return prefix + """
        double vp = std::max(inputs[0], 0.01);
        double Volt_dip = (inputs[0] < Vdip || inputs[0] > Vup) ? 1.0 : 0.0;
        
        // s0: Voltage filter
        dxdt[0] = (inputs[0] - x[0]) / std::max(Trv, 0.001);
        
        // s1: Pe filter
        dxdt[1] = (inputs[1] - x[1]) / std::max(Tp, 0.001);
        
        // Reactive power control path
        double Qref = qref0 + inputs[4];
        double PFsel = Qref;
        if (PFFLAG > 0.5) {
            PFsel = (abs(p0) > 1e-6) ? x[1] * tan(pfaref0) : q0;
        }
        PFsel = std::max(std::min(PFsel, QMax), QMin);
        double Qerr = PFsel - inputs[2];
        
        double piq_input = (VFLAG > 0.5) ? Qerr : 0.0;
        double piq_p = Kqp * piq_input;
        double piq_y = piq_p + x[2];
        double piq_y_clamped = std::max(std::min(piq_y, VMAX), VMIN);
        
        if ((piq_y >= VMAX && Kqi * piq_input > 0) ||
            (piq_y <= VMIN && Kqi * piq_input < 0)) {
            dxdt[2] = 0.0;
        } else {
            dxdt[2] = Kqi * piq_input;
        }
        
        // Active power control path
        double Pref = p0 + inputs[3];
        double Pord = Pref;
        if (PFLAG > 0.5) {
            // Speed control
            double werr = inputs[5] - 1.0;
            // Simplified speed control for now
        }
        Pord = std::max(std::min(Pord, PMAX), PMIN);
        
        double s5_error = Pord - x[3];
        double s5_rate = s5_error / std::max(Tpord, 0.001);
        s5_rate = std::max(std::min(s5_rate, dPmax), dPmin);
        dxdt[3] = s5_rate;
        """

    def get_cpp_compute_outputs_code(self) -> str:
        Ipcmd0 = self.params.get('Ipcmd0', 0.0)
        Iqcmd0 = self.params.get('Iqcmd0', 0.0)
        
        prefix = f"""
        double Ipcmd0 = {Ipcmd0};
        double Iqcmd0 = {Iqcmd0};
        """
        return prefix + """
        double vp = std::max(inputs[0], 0.01);
        double Volt_dip = (inputs[0] < Vdip || inputs[0] > Vup) ? 1.0 : 0.0;
        
        // Iqinj
        double Verr = Vref0 - x[0];
        double dbV_y = 0.0;
        if (Verr < dbd1) {
            dbV_y = Verr - dbd1;
        } else if (Verr > dbd2) {
            dbV_y = Verr - dbd2;
        }
        double Iqinj = dbV_y * Kqv * Volt_dip;
        
        // Qsel
        double Qsel = Iqcmd0;
        if (QFLAG > 0.5) {
            Qsel = Kvp * (Vref1 - x[0]) + x[2];
        }
        
        // Current limits
        double Ipmax, Iqmax;
        if (PQFLAG > 0.5) {
            Ipmax = Imax;
            double temp = Imax * Imax - (x[3] / vp) * (x[3] / vp);
            Iqmax = sqrt(std::max(0.0, temp));
        } else {
            Iqmax = Imax;
            double temp = Imax * Imax - (Qsel + Iqinj) * (Qsel + Iqinj);
            Ipmax = sqrt(std::max(0.0, temp));
        }
        
        outputs[1] = std::max(std::min(Qsel + Iqinj, Iqmax), -Iqmax);
        
        // LVG logic for Ipcmd
        double LVG_y = 1.0;
        // Assuming LVG is handled in REGCA1, but REECA1 also scales Ipcmd
        // We'll just output s5_y / vp for now, limited by Ipmax
        double denom = std::max(vp, 0.1);
        outputs[0] = std::max(std::min(x[3] / denom, Ipmax), 0.0);
        outputs[2] = x[3];
        """
