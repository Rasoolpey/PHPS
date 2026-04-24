"""
DFIG_GSC — Grid-Side Converter Controller (Port-Hamiltonian)
============================================================

The GSC maintains the DC-link voltage and provides grid-side reactive
power control.  It is modelled as a **current-source generator** on the
same bus as the DFIG stator.  It injects (Id_gsc, Iq_gsc) into the
network without an internal Norton admittance (pure current injection,
like REGCA1).

Port-Hamiltonian interpretation
-------------------------------
The GSC consists of two PI controllers (Vdc loop + Q loop), each a
first-order integrator.  By the canonical construction (Ch. 3):

    H_gsc = ½ (x_Vdc² + x_Q²),   R_c = ½ I₂.

States (2):
    [0] x_Vdc  — DC-voltage-loop PI integrator
    [1] x_Q    — Q-loop PI integrator

Inputs (5):
    [0] Vdc      — measured DC-link voltage (from DCLink) [pu]
    [1] Vdc_ref  — DC-link voltage reference [pu]
    [2] Qref     — grid-side Q reference [pu]
    [3] Vd       — grid d-voltage at PCC [pu]
    [4] Vq       — grid q-voltage at PCC [pu]

Outputs (5):
    [0] Id_gsc  — grid-side d-current injection [pu]  (= Id Norton)
    [1] Iq_gsc  — grid-side q-current injection [pu]  (= Iq Norton)
    [2] P_gsc   — active power drawn from grid by GSC [pu]
    [3] Q_gsc   — reactive power from GSC [pu]
    [4] Pe      — alias of P_gsc for WTDTA1 compatibility [pu]

Parameters:
    Kp_dc, Ki_dc  — Vdc-loop PI gains
    Kp_Qg, Ki_Qg  — Q-loop PI gains
    I_max          — current magnitude saturation [pu]
"""

import math
import numpy as np
from typing import Dict, List, Tuple, Any
from src.core import PowerComponent


class DfigGsc(PowerComponent):
    """
    Grid-Side Converter — current-source injector on the DFIG bus.

    Maintains DC-link voltage via active power exchange with the grid
    and controls grid-side reactive power.  Port-Hamiltonian canonical
    form with H = ½ (x_Vdc² + x_Q²).
    """

    @property
    def port_schema(self) -> Dict[str, List[Tuple[str, str, str]]]:
        return {
            'in': [
                ('Vdc',     'signal', 'pu'),   # [0]
                ('Vdc_ref', 'signal', 'pu'),   # [1]
                ('Qref',    'signal', 'pu'),   # [2]
                ('Vd',      'effort', 'pu'),   # [3]
                ('Vq',      'effort', 'pu'),   # [4]
            ],
            'out': [
                ('Id',      'flow', 'pu'),     # [0] Norton injection Re
                ('Iq',      'flow', 'pu'),     # [1] Norton injection Im
                ('P_gsc',   'flow', 'pu'),     # [2] GSC active power
                ('Q_gsc',   'flow', 'pu'),     # [3] GSC reactive power
                ('Pe',      'flow', 'pu'),     # [4] alias for WTDTA1 compat
            ]
        }

    @property
    def state_schema(self) -> List[str]:
        return ['x_Vdc', 'x_Q']

    @property
    def param_schema(self) -> Dict[str, str]:
        return {
            'Kp_dc':  'Proportional gain, Vdc loop',
            'Ki_dc':  'Integral gain, Vdc loop',
            'Kp_Qg':  'Proportional gain, Q loop',
            'Ki_Qg':  'Integral gain, Q loop',
            'I_max':  'Current magnitude saturation [pu]',
        }

    @property
    def component_role(self) -> str:
        # Renewable controller: initialized from DFIG targets so that
        # the DC-link power balance P_rsc = P_gsc holds at t=0.
        return 'renewable_controller'

    def get_associated_generator(self, comp_map: dict):
        """Return the DFIG generator that this GSC supports."""
        return self.params.get('dfig', None)

    @property
    def contributes_norton_admittance(self) -> bool:
        return False    # no bus-level injection

    @property
    def observables(self) -> Dict[str, Dict[str, str]]:
        return {
            'Id_gsc':   {'description': 'GSC d-current injection', 'unit': 'pu',
                         'cpp_expr': 'outputs[0]'},
            'Iq_gsc':   {'description': 'GSC q-current injection', 'unit': 'pu',
                         'cpp_expr': 'outputs[1]'},
            'P_gsc':    {'description': 'GSC active power',        'unit': 'pu',
                         'cpp_expr': 'outputs[2]'},
            'Q_gsc':    {'description': 'GSC reactive power',      'unit': 'pu',
                         'cpp_expr': 'outputs[3]'},
        }

    def get_cpp_compute_outputs_code(self) -> str:
        return """
            // DC-voltage loop  →  active-current reference
            double e_Vdc = inputs[0] - inputs[1];           // Vdc - Vdc_ref
            double Id_cmd = Kp_dc * e_Vdc + Ki_dc * x[0];

            // Q loop  →  reactive-current reference
            double e_Q  = inputs[2];                        // Qref (direct)
            double Iq_cmd = Kp_Qg * e_Q + Ki_Qg * x[1];

            // Current magnitude saturation
            double I_mag = sqrt(Id_cmd * Id_cmd + Iq_cmd * Iq_cmd);
            double scale = (I_mag > I_max && I_mag > 1e-6) ? I_max / I_mag : 1.0;
            double Id_dq = Id_cmd * scale;
            double Iq_dq = Iq_cmd * scale;

            // Rotate dq command (voltage-aligned frame) to network RI frame
            double Vd = inputs[3];
            double Vq = inputs[4];
            double Vmag = sqrt(Vd * Vd + Vq * Vq);
            double c = (Vmag > 1e-9) ? (Vd / Vmag) : 1.0;
            double s = (Vmag > 1e-9) ? (Vq / Vmag) : 0.0;
            double Id_gsc = Id_dq * c - Iq_dq * s;
            double Iq_gsc = Id_dq * s + Iq_dq * c;

            // Power computation at grid terminal
            double P_gsc = Vd * Id_gsc + Vq * Iq_gsc;
            double Q_gsc = Vq * Id_gsc - Vd * Iq_gsc;

            outputs[0] = Id_gsc;
            outputs[1] = Iq_gsc;
            outputs[2] = P_gsc;
            outputs[3] = Q_gsc;
            outputs[4] = P_gsc;      // Pe alias
        """

    def get_cpp_step_code(self) -> str:
        return r"""
            // DC-voltage loop  →  active-current reference
            double e_Vdc = inputs[0] - inputs[1];           // Vdc - Vdc_ref
            double Id_cmd = Kp_dc * e_Vdc + Ki_dc * x[0];

            // Q loop  →  reactive-current reference
            double e_Q  = inputs[2];                        // Qref (direct)
            double Iq_cmd = Kp_Qg * e_Q + Ki_Qg * x[1];

            // Current magnitude saturation
            double I_mag = sqrt(Id_cmd * Id_cmd + Iq_cmd * Iq_cmd);
            double scale = (I_mag > I_max && I_mag > 1e-6) ? I_max / I_mag : 1.0;
            double Id_dq = Id_cmd * scale;
            double Iq_dq = Iq_cmd * scale;

            // Rotate dq command (voltage-aligned frame) to network RI frame
            double Vd = inputs[3];
            double Vq = inputs[4];
            double Vmag = sqrt(Vd * Vd + Vq * Vq);
            double c = (Vmag > 1e-9) ? (Vd / Vmag) : 1.0;
            double s = (Vmag > 1e-9) ? (Vq / Vmag) : 0.0;
            double Id_gsc = Id_dq * c - Iq_dq * s;
            double Iq_gsc = Id_dq * s + Iq_dq * c;

            // Integrator dynamics with simple anti-windup under saturation
            bool sat_hi = (I_mag > I_max + 1e-12);
            if (!sat_hi) {
                dxdt[0] = e_Vdc;
                dxdt[1] = e_Q;
            } else {
                // Freeze integrator when error pushes farther into saturation;
                // allow unwind when error drives command back toward feasible region.
                dxdt[0] = 0.0;
                dxdt[1] = 0.0;
                if ((Id_cmd > 0.0 && e_Vdc < 0.0) || (Id_cmd < 0.0 && e_Vdc > 0.0)) dxdt[0] = e_Vdc;
                if ((Iq_cmd > 0.0 && e_Q < 0.0) || (Iq_cmd < 0.0 && e_Q > 0.0)) dxdt[1] = e_Q;
            }

            // Power computation at grid terminal
            double P_gsc = Vd * Id_gsc + Vq * Iq_gsc;
            double Q_gsc = Vq * Id_gsc - Vd * Iq_gsc;

            outputs[0] = Id_gsc;
            outputs[1] = Iq_gsc;
            outputs[2] = P_gsc;
            outputs[3] = Q_gsc;
            outputs[4] = P_gsc;
        """

    def compute_norton_current(self, x_slice: np.ndarray,
                               V_bus_complex: complex = None) -> complex:
        """Return GSC current injection in RI-frame for Python Z-bus init.

        The GSC is a pure current-source injector (no Norton admittance). During
        initialization, we assume near-equilibrium references so the PI outputs
        reduce to their integral terms.
        """
        Kp_dc = float(self.params.get('Kp_dc', 0.0))
        Ki_dc = float(self.params.get('Ki_dc', 0.0))
        Kp_Qg = float(self.params.get('Kp_Qg', 0.0))
        Ki_Qg = float(self.params.get('Ki_Qg', 0.0))
        I_max = float(self.params.get('I_max', 1e9))

        Vdc_ref = float(self.params.get('Vdc_ref0', 1.0))
        Vdc_meas = float(self.params.get('Vdc0', Vdc_ref))
        Qref = float(self.params.get('Qref0', 0.0))
        if V_bus_complex is None:
            V_bus_complex = complex(float(self.params.get('Vd0', 1.0)),
                                    float(self.params.get('Vq0', 0.0)))

        e_vdc = Vdc_meas - Vdc_ref
        e_q = Qref

        id_cmd = Kp_dc * e_vdc + Ki_dc * float(x_slice[0])
        iq_cmd = Kp_Qg * e_q + Ki_Qg * float(x_slice[1])
        i_mag = math.hypot(id_cmd, iq_cmd)
        if i_mag > I_max and i_mag > 1e-9:
            scale = I_max / i_mag
            id_cmd *= scale
            iq_cmd *= scale

        v_re = float(V_bus_complex.real)
        v_im = float(V_bus_complex.imag)
        v_mag = math.hypot(v_re, v_im)
        c = (v_re / v_mag) if v_mag > 1e-9 else 1.0
        s = (v_im / v_mag) if v_mag > 1e-9 else 0.0
        i_re = id_cmd * c - iq_cmd * s
        i_im = id_cmd * s + iq_cmd * c
        return complex(i_re, i_im)

    def init_from_targets(self, targets: dict) -> np.ndarray:
        """Initialise GSC integrators for DC-link power balance.

        At equilibrium the Vdc loop error is zero (Vdc = Vdc_ref), so
        Id_gsc = Ki_dc * x_Vdc.  We need P_gsc = P_rotor to keep the
        DC link balanced:  P_gsc = Vd * Id_gsc → x_Vdc = P_rotor / (Ki_dc * Vd).

        If P_rotor is not available in targets (pre-Kron pass), fall back
        to zero.
        """
        p = self.params
        Ki_dc = p['Ki_dc']
        Ki_Qg = p['Ki_Qg']

        P_rotor = targets.get('P_rotor', 0.0)
        Vd = targets.get('vd_ri', targets.get('vd', 0.0))
        Vq = targets.get('vq_ri', targets.get('vq', 0.0))
        Vdc_ref = targets.get('Vdc_ref', 1.0)
        Qref = targets.get('Qref', 0.0)

        if 'bus' in targets:
            self.params['bus'] = targets['bus']
        self.params['Vdc_ref0'] = float(Vdc_ref)
        self.params['Vdc0'] = float(Vdc_ref)
        self.params['Qref0'] = float(Qref)
        self.params['Vd0'] = float(Vd)
        self.params['Vq0'] = float(Vq)

        # At equilibrium: Id_gsc = Ki_dc * x_Vdc, P_gsc = Vd * Id_gsc
        Vmag = math.hypot(Vd, Vq)
        if abs(Ki_dc) > 1e-10 and Vmag > 1e-6:
            x_Vdc = P_rotor / (Ki_dc * Vmag)
        else:
            x_Vdc = 0.0

        # Q loop: Qref = 0 at init → x_Q = 0
        x_Q = 0.0

        return self._init_states({'x_Vdc': x_Vdc, 'x_Q': x_Q})
