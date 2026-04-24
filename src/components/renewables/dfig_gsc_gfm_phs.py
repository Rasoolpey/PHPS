"""
DFIG_GSC_GFM_PHS — Grid-Forming Grid-Side Converter (Σ₆_GFM)
=============================================================

Identical to the GFL GSC (DfigGscPHS) except:
  - No PLL: the dq reference frame angle θs is received from the RSC APC
  - θs is an input port used by the system interconnection layer for
    RI↔dq frame rotation. The GSC C++ step function itself does NOT
    use θs directly.

States (4):
    [0] phi_fd  — filter d-axis flux linkage  [pu]
    [1] phi_fq  — filter q-axis flux linkage  [pu]
    [2] x_Vdc  — DC-voltage PI integrator     [pu]
    [3] x_Q    — Q-control PI integrator      [pu]

Inputs (6):
    [0] Vdc      — DC-link voltage measurement from Σ₄
    [1] Vdc_ref  — DC-link voltage reference
    [2] Qref     — Reactive power reference
    [3] Vd       — d-axis grid voltage at PCC
    [4] Vq       — q-axis grid voltage at PCC
    [5] theta_s  — Internal voltage angle from RSC APC (used by system layer)

Outputs (5):
    [0] Id       — d-axis current injection
    [1] Iq       — q-axis current injection
    [2] P_gsc    — GSC active power
    [3] Q_gsc    — GSC reactive power
    [4] Pe       — alias for P_gsc
"""

import math
import numpy as np
import sympy as sp
from typing import Dict, List, Tuple, Any
from src.core import PowerComponent


class DfigGscGfmPHS(PowerComponent):
    """
    GFM Grid-Side Converter (Σ₆_GFM) with filter inductance in PH form.

    Structurally identical to GFL GSC. The only difference is that the
    dq frame is oriented by θs from the RSC APC instead of a PLL.
    The θs input port is consumed by the system interconnection layer.
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
                ('theta_s', 'signal', 'rad'),  # [5] from RSC APC
            ],
            'out': [
                ('Id',      'flow', 'pu'),     # [0]
                ('Iq',      'flow', 'pu'),     # [1]
                ('P_gsc',   'flow', 'pu'),     # [2]
                ('Q_gsc',   'flow', 'pu'),     # [3]
                ('Pe',      'flow', 'pu'),     # [4]
            ]
        }

    @property
    def state_schema(self) -> List[str]:
        return ['phi_fd', 'phi_fq', 'x_Vdc', 'x_Q']

    @property
    def param_schema(self) -> Dict[str, str]:
        return {
            'L_f':     'Filter inductance [pu]',
            'R_f':     'Filter resistance [pu]',
            'omega_s': 'Synchronous speed [pu]',
            'omega_b': 'Base angular frequency [rad/s]',
            'Kp_dc':   'Vdc-loop proportional gain',
            'Ki_dc':   'Vdc-loop integral gain',
            'Kp_Qg':   'Q-loop proportional gain',
            'Ki_Qg':   'Q-loop integral gain',
            'I_max':   'Current magnitude saturation [pu]',
        }

    @property
    def component_role(self) -> str:
        return 'renewable_controller'

    def get_associated_generator(self, comp_map: dict):
        return self.params.get('dfig', None)

    @property
    def contributes_norton_admittance(self) -> bool:
        return False

    @property
    def observables(self) -> Dict[str, Dict[str, str]]:
        return {
            'i_fd':   {'description': 'Filter d-current', 'unit': 'pu', 'cpp_expr': 'x[0] / L_f'},
            'i_fq':   {'description': 'Filter q-current', 'unit': 'pu', 'cpp_expr': 'x[1] / L_f'},
            'P_gsc':  {'description': 'GSC active power',  'unit': 'pu', 'cpp_expr': 'outputs[2]'},
            'Q_gsc':  {'description': 'GSC reactive power', 'unit': 'pu', 'cpp_expr': 'outputs[3]'},
            'H_filter': {'description': 'Filter energy', 'unit': 'pu',
                         'cpp_expr': '(x[0]*x[0] + x[1]*x[1]) / (2.0 * L_f)'},
        }

    # ------------------------------------------------------------------
    # C++ Code Generation
    # ------------------------------------------------------------------

    def get_cpp_compute_outputs_code(self) -> str:
        return """
            // Filter currents: ∇H₆ = φ_f / L_f
            double i_fd = x[0] / L_f;
            double i_fq = x[1] / L_f;

            // Grid voltage
            double Vd = inputs[3];
            double Vq = inputs[4];

            // Power at grid terminal
            double P_gsc = Vd * i_fd + Vq * i_fq;
            double Q_gsc = Vq * i_fd - Vd * i_fq;

            outputs[0] = i_fd;      // Id injection
            outputs[1] = i_fq;      // Iq injection
            outputs[2] = P_gsc;
            outputs[3] = Q_gsc;
            outputs[4] = P_gsc;     // Pe alias
        """

    def get_cpp_step_code(self) -> str:
        return r"""
            // ---- Filter currents ----
            double i_fd = x[0] / L_f;
            double i_fq = x[1] / L_f;

            // ---- Grid voltage ----
            double Vd = inputs[3];
            double Vq = inputs[4];
            // Note: inputs[5] = theta_s from RSC APC, used by system layer only

            // ---- PI controller: Vdc loop → active power command ----
            double e_Vdc = inputs[0] - inputs[1];  // Vdc - Vdc_ref
            double P_cmd = Kp_dc * e_Vdc + Ki_dc * x[2];

            // ---- PI controller: Q loop → reactive power command ----
            double Q_meas = Vq * i_fd - Vd * i_fq;
            double e_Q = inputs[2] - Q_meas;  // Qref - Q_meas
            double Q_neg_cmd = Kp_Qg * e_Q + Ki_Qg * x[3];

            // ---- Power-projection: decouple P/Q control from voltage angle ----
            // From S = V·conj(I): Id=(Vd·P+Vq·Q)/V², Iq=(Vq·P−Vd·Q)/V²
            // Q_neg_cmd tracks (Qref−Q_meas) > 0 when Q below ref → more injection
            // Injection requires Iq < 0, so Iq = −Q_neg_cmd at Vd≈1,Vq≈0: sign is −Vd·Q
            double V_sq = fmax(Vd * Vd + Vq * Vq, 0.01);
            double Id_ref = (Vd * P_cmd + Vq * Q_neg_cmd) / V_sq;
            double Iq_ref = (Vq * P_cmd - Vd * Q_neg_cmd) / V_sq;

            // ---- Current magnitude saturation ----
            double I_ref_mag = sqrt(Id_ref * Id_ref + Iq_ref * Iq_ref);
            double scale = (I_ref_mag > I_max && I_ref_mag > 1e-6) ? I_max / I_ref_mag : 1.0;
            Id_ref *= scale;
            Iq_ref *= scale;
            bool sat = (I_ref_mag > I_max + 1e-12);

            // ---- Converter voltage commands (PI on current + decoupling) ----
            double Kp_i_gsc = 5.0;  // inner current loop gain
            double V_fd = Kp_i_gsc * (Id_ref - i_fd) + Vd - omega_s * x[1];
            double V_fq = Kp_i_gsc * (Iq_ref - i_fq) + Vq + omega_s * x[0];

            // ---- Filter PHS dynamics ----
            dxdt[0] = omega_b * ((V_fd - Vd) - R_f * i_fd + omega_s * L_f * i_fq);
            dxdt[1] = omega_b * ((V_fq - Vq) - R_f * i_fq - omega_s * L_f * i_fd);

            // ---- PI integrator dynamics with anti-windup ----
            if (!sat) {
                dxdt[2] = e_Vdc;
                dxdt[3] = e_Q;
            } else {
                dxdt[2] = 0.0;
                dxdt[3] = 0.0;
                if ((P_cmd > 0.0 && e_Vdc < 0.0) || (P_cmd < 0.0 && e_Vdc > 0.0))
                    dxdt[2] = e_Vdc;
                if ((Q_neg_cmd > 0.0 && e_Q < 0.0) || (Q_neg_cmd < 0.0 && e_Q > 0.0))
                    dxdt[3] = e_Q;
            }

            // ---- Integrator state clamping ----
            double x_Vdc_lim = I_max / Ki_dc;
            if (x[2] > x_Vdc_lim && dxdt[2] > 0.0) dxdt[2] = 0.0;
            if (x[2] < -x_Vdc_lim && dxdt[2] < 0.0) dxdt[2] = 0.0;
            double x_Q_lim = I_max / Ki_Qg;
            if (x[3] > x_Q_lim && dxdt[3] > 0.0) dxdt[3] = 0.0;
            if (x[3] < -x_Q_lim && dxdt[3] < 0.0) dxdt[3] = 0.0;

            // ---- Outputs ----
            double P_gsc = Vd * i_fd + Vq * i_fq;
            double Q_gsc = Vq * i_fd - Vd * i_fq;
            outputs[0] = i_fd;
            outputs[1] = i_fq;
            outputs[2] = P_gsc;
            outputs[3] = Q_gsc;
            outputs[4] = P_gsc;
        """

    # ------------------------------------------------------------------
    # Symbolic PHS
    # ------------------------------------------------------------------

    def get_symbolic_phs(self):
        from src.symbolic.core import SymbolicPHS

        phi_fd, phi_fq = sp.symbols('phi_fd phi_fq')
        x_Vdc, x_Q = sp.symbols('x_Vdc x_Q')
        states = [phi_fd, phi_fq, x_Vdc, x_Q]

        dV_fd = sp.Symbol('Delta_V_fd')
        dV_fq = sp.Symbol('Delta_V_fq')
        e_Vdc_s = sp.Symbol('e_Vdc')
        e_Q_s = sp.Symbol('e_Q')
        inputs = [dV_fd, dV_fq, e_Vdc_s, e_Q_s]

        Lf = sp.Symbol('L_f', positive=True)
        Rf = sp.Symbol('R_f', nonnegative=True)
        ws = sp.Symbol('omega_s', positive=True)
        wb = sp.Symbol('omega_b', positive=True)

        params = {'L_f': Lf, 'R_f': Rf, 'omega_s': ws, 'omega_b': wb}

        H_expr = (phi_fd**2 + phi_fq**2) / (2 * Lf) + sp.Rational(1, 2) * (x_Vdc**2 + x_Q**2)

        J = sp.zeros(4, 4)
        J[0, 1] = ws * Lf;  J[1, 0] = -ws * Lf

        R = sp.zeros(4, 4)
        R[0, 0] = Rf
        R[1, 1] = Rf

        g = sp.zeros(4, 4)
        g[0, 0] = 1 / Lf
        g[1, 1] = 1 / Lf
        g[2, 2] = 1
        g[3, 3] = 1

        return SymbolicPHS(
            name='DFIG_GSC_GFM_PHS',
            states=states, inputs=inputs, params=params,
            J=J, R=R, g=g, H=H_expr,
            description='GFM GSC filter inductance (Σ₆_GFM) + PI controllers in PH form. '
                        'Identical PHS structure to GFL; θs from RSC replaces PLL.',
        )

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def compute_norton_current(self, x_slice: np.ndarray,
                               V_bus_complex: complex = None) -> complex:
        L_f = float(self.params['L_f'])
        i_fd = float(x_slice[0]) / L_f
        i_fq = float(x_slice[1]) / L_f
        return complex(i_fd, i_fq)

    def init_from_targets(self, targets: dict) -> np.ndarray:
        p = self.params
        Ki_dc = p['Ki_dc']
        L_f = p['L_f']

        P_rotor = targets.get('P_rotor', 0.0)
        Vd = targets.get('vd_ri', targets.get('vd', 0.0))
        Vq = targets.get('vq_ri', targets.get('vq', 0.0))
        Vdc_ref = targets.get('Vdc_ref', 1.0)

        if 'bus' in targets:
            p['bus'] = targets['bus']
        p['Vdc_ref0'] = float(Vdc_ref)
        p['Vdc0'] = float(Vdc_ref)

        Vmag = math.hypot(Vd, Vq)

        if Vmag > 1e-6:
            i_fd_ss = P_rotor * Vd / (Vmag**2)
            i_fq_ss = P_rotor * Vq / (Vmag**2)
        else:
            i_fd_ss = 0.0
            i_fq_ss = 0.0

        phi_fd_ss = L_f * i_fd_ss
        phi_fq_ss = L_f * i_fq_ss

        if abs(Ki_dc) > 1e-10:
            x_Vdc = i_fd_ss / Ki_dc
        else:
            x_Vdc = 0.0
        x_Q = 0.0

        return self._init_states({
            'phi_fd': phi_fd_ss,
            'phi_fq': phi_fq_ss,
            'x_Vdc': x_Vdc,
            'x_Q': x_Q,
        })
