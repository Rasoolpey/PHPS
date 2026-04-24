"""
DFIG_RSC_PHS — Rotor-Side Converter Controller (Σ₅ Port-Hamiltonian)
=====================================================================

Cascaded vector-control for the DFIG rotor circuit in PHS form.

Each PI integrator is a natural PH component:
    H_k = ½ x_k²,  J_k = 0,  R_k = 0,  ẋ_k = e_k

The four integrators stack into a 4×4 diagonal PHS:

    J₅ = 0,  R₅ = 0,  g₅ = I₄,  H₅ = ½(x_P² + x_Q² + x_ird² + x_irq²)

States (5):
    [0] x_P    — P-loop PI integrator
    [1] x_Q    — Q-loop PI integrator
    [2] x_ird  — inner i_rd PI integrator
    [3] x_irq  — inner i_rq PI integrator
    [4] x_Pramp — post-fault active power ramp state

Inputs (10):
    [0] Pref, [1] Qref, [2] Pe, [3] Qe,
    [4] i_rd, [5] i_rq, [6] omega_r,
    [7] phi_sd, [8] phi_sq, [9] Vterm

Outputs (4):
    [0] Vrd, [1] Vrq, [2] P_rotor, [3] ird_ref
"""

import numpy as np
import sympy as sp
from typing import Dict, List, Tuple, Any
from src.core import PowerComponent


class DfigRscPHS(PowerComponent):
    """
    RSC controller (Σ₅) in Port-Hamiltonian form.

    Cascaded PI with SFO, cross-coupling decoupling, anti-windup.
    Includes LVRT transient control: QInject, MaxLim, RampLim.
    """

    _LVRT_DEFAULTS = {
        'V_lvrt': 0.9,
        'K_lvrt': 2.0,
        'I_max': 1.0,
        'P_ramp': 0.2,
        'Rvir': 0.0,
        'Xvir': 0.0,
    }

    def __init__(self, name: str, params: dict):
        for k, v in self._LVRT_DEFAULTS.items():
            params.setdefault(k, v)
        super().__init__(name, params)

    @property
    def port_schema(self) -> Dict[str, List[Tuple[str, str, str]]]:
        return {
            'in': [
                ('Pref',    'signal', 'pu'),
                ('Qref',    'signal', 'pu'),
                ('Pe',      'signal', 'pu'),
                ('Qe',      'signal', 'pu'),
                ('i_rd',    'signal', 'pu'),
                ('i_rq',    'signal', 'pu'),
                ('omega_r', 'signal', 'pu'),
                ('phi_sd',  'signal', 'pu'),
                ('phi_sq',  'signal', 'pu'),
                ('Vterm',   'signal', 'pu'),
            ],
            'out': [
                ('Vrd',     'signal', 'pu'),
                ('Vrq',     'signal', 'pu'),
                ('P_rotor', 'signal', 'pu'),
                ('ird_ref', 'signal', 'pu'),
            ]
        }

    @property
    def state_schema(self) -> List[str]:
        return ['x_P', 'x_Q', 'x_ird', 'x_irq', 'x_Pramp']

    @property
    def param_schema(self) -> Dict[str, str]:
        return {
            'Kp_P': 'P-loop proportional gain',
            'Ki_P': 'P-loop integral gain',
            'Kp_Q': 'Q-loop proportional gain',
            'Ki_Q': 'Q-loop integral gain',
            'Kp_i': 'Inner current-loop proportional gain',
            'Ki_i': 'Inner current-loop integral gain',
            'Lm':   'DFIG mutual inductance [pu]',
            'Ls':   'DFIG stator inductance [pu]',
            'Lr':   'DFIG rotor inductance [pu]',
            'omega_s': 'Synchronous speed [pu]',
            'Vrd_max': 'Rotor voltage saturation [pu]',
            'Kv':      'Voltage-dependent Q droop gain',
            'Vref':    'Terminal voltage reference [pu]',
            'V_lvrt':  'LVRT activation threshold [pu] (default 0.9)',
            'K_lvrt':  'LVRT reactive current gain [pu/pu] (default 2.0)',
            'I_max':   'Converter current magnitude limit [pu] (default 1.0)',
            'P_ramp':  'Post-fault active power ramp rate [pu/s] (default 0.2)',
            'Rvir':    'Virtual resistance [pu] (default 0.0)',
            'Xvir':    'Virtual reactance [pu] (default 0.0)',
        }

    @property
    def component_role(self) -> str:
        return 'renewable_controller'

    def get_associated_generator(self, comp_map: dict):
        return self.params.get('dfig', None)

    @property
    def observables(self) -> Dict[str, Dict[str, str]]:
        return {
            'Vrd_cmd': {'description': 'Rotor d-voltage command', 'unit': 'pu', 'cpp_expr': 'outputs[0]'},
            'Vrq_cmd': {'description': 'Rotor q-voltage command', 'unit': 'pu', 'cpp_expr': 'outputs[1]'},
            'P_rotor': {'description': 'Rotor electrical power',  'unit': 'pu', 'cpp_expr': 'outputs[2]'},
            'Pramp':   {'description': 'Active power ramp state', 'unit': 'pu', 'cpp_expr': 'x[4]'},
        }

    # ------------------------------------------------------------------
    # C++ Code Generation
    # ------------------------------------------------------------------

    def get_cpp_compute_outputs_code(self) -> str:
        return """
            double Vterm_meas = inputs[9];
            double Qref_eff = inputs[1] - Kv * (Vterm_meas - Vref);

            double e_P = inputs[0] - inputs[2];
            double e_Q = Qref_eff - inputs[3];

            // Stator Flux Orientation angle
            double phi_sd = inputs[7];
            double phi_sq = inputs[8];
            double phi_s_mag = sqrt(phi_sd*phi_sd + phi_sq*phi_sq);
            if (phi_s_mag < 1e-6) phi_s_mag = 1e-6;
            double cos_th = phi_sd / phi_s_mag;
            double sin_th = phi_sq / phi_s_mag;

            // LVRT parameters
            double V_lvrt_thr = V_lvrt;
            double K_lvrt_gain = K_lvrt;
            double I_max_val = I_max;

            bool lvrt = (Vterm_meas < V_lvrt_thr);

            // Outer PI → SFO current references
            double i_q_sfo_ref = Kp_P * e_P + Ki_P * x[0];
            double i_d_sfo_ref = Kp_Q * e_Q + Ki_Q * x[1];

            // SFO → RI frame
            double i_rd_ref =  i_d_sfo_ref * cos_th - i_q_sfo_ref * sin_th;
            double i_rq_ref =  i_d_sfo_ref * sin_th + i_q_sfo_ref * cos_th;

            if (lvrt) {
                // QInject + MaxLim
                double i_q_inject = K_lvrt_gain * (V_lvrt_thr - Vterm_meas) * I_max_val;
                if (i_q_inject > I_max_val) i_q_inject = I_max_val;
                if (i_q_inject < 0.0) i_q_inject = 0.0;
                double i_d_remain = sqrt(fmax(I_max_val * I_max_val - i_q_inject * i_q_inject, 0.0));

                // SFO frame: d-axis = reactive (Q), q-axis = active (P)
                i_d_sfo_ref = i_q_inject;   // reactive injection → d-axis
                i_q_sfo_ref = i_d_remain;   // reduced active current → q-axis

                // SFO → RI frame
                i_rd_ref =  i_d_sfo_ref * cos_th - i_q_sfo_ref * sin_th;
                i_rq_ref =  i_d_sfo_ref * sin_th + i_q_sfo_ref * cos_th;
            } else {
                // Post-fault: use ramped P reference
                e_P = x[4] - inputs[2];
                i_q_sfo_ref = Kp_P * e_P + Ki_P * x[0];
                i_d_sfo_ref = Kp_Q * e_Q + Ki_Q * x[1];
                i_rd_ref =  i_d_sfo_ref * cos_th - i_q_sfo_ref * sin_th;
                i_rq_ref =  i_d_sfo_ref * sin_th + i_q_sfo_ref * cos_th;
            }

            // Circular current limiter (like GFM)
            double I_ref_mag = sqrt(i_rd_ref * i_rd_ref + i_rq_ref * i_rq_ref);
            if (I_ref_mag > I_max_val && I_ref_mag > 1e-6) {
                double sc = I_max_val / I_ref_mag;
                i_rd_ref *= sc;
                i_rq_ref *= sc;
            }

            // HVRT i_rd clamp
            if (Vterm_meas > 1.1) {
                double i_rd_eff_max = I_max_val - 8.0 * (Vterm_meas - 1.1);
                if (i_rd_eff_max < -0.5) i_rd_eff_max = -0.5;
                if (i_rd_ref > i_rd_eff_max) i_rd_ref = i_rd_eff_max;
            }

            // Inner loop errors
            double e_ird = i_rd_ref - inputs[4];
            double e_irq = i_rq_ref - inputs[5];

            // Cross-coupling decoupling
            double sigma_Lr = (Ls * Lr - Lm * Lm) / Ls;
            double slip_omega = omega_s - inputs[6];
            slip_omega = fmax(-0.2, fmin(0.2, slip_omega));
            double Vrd_dec = -slip_omega * (sigma_Lr * inputs[5] + Lm / Ls * inputs[8]);
            double Vrq_dec =  slip_omega * (Lm / Ls * inputs[7] + sigma_Lr * inputs[4]);

            // Virtual impedance voltage drop (opposes fault current)
            double Vrd_vir = Rvir * inputs[4] - Xvir * inputs[5];
            double Vrq_vir = Rvir * inputs[5] + Xvir * inputs[4];

            // Voltage commands with virtual impedance and saturation
            double Vrd_raw = Kp_i * e_ird + Ki_i * x[2] + Vrd_dec - Vrd_vir;
            double Vrq_raw = Kp_i * e_irq + Ki_i * x[3] + Vrq_dec - Vrq_vir;
            double Vrd_cmd = fmax(-Vrd_max, fmin(Vrd_max, Vrd_raw));
            double Vrq_cmd = fmax(-Vrd_max, fmin(Vrd_max, Vrq_raw));

            outputs[0] = Vrd_cmd;
            outputs[1] = Vrq_cmd;
            outputs[2] = Vrd_cmd * inputs[4] + Vrq_cmd * inputs[5];
            outputs[3] = i_rd_ref;
        """

    def get_cpp_step_code(self) -> str:
        return r"""
            double Vterm_meas = inputs[9];
            double Qref_eff = inputs[1] - Kv * (Vterm_meas - Vref);

            double e_P = inputs[0] - inputs[2];
            double e_Q = Qref_eff - inputs[3];

            // Stator Flux Orientation angle
            double phi_sd = inputs[7];
            double phi_sq = inputs[8];
            double phi_s_mag = sqrt(phi_sd*phi_sd + phi_sq*phi_sq);
            if (phi_s_mag < 1e-6) phi_s_mag = 1e-6;
            double cos_th = phi_sd / phi_s_mag;
            double sin_th = phi_sq / phi_s_mag;

            // ---- LVRT parameters (with defaults) ----
            double V_lvrt_thr = V_lvrt;   // activation threshold (param, default 0.9)
            double K_lvrt_gain = K_lvrt;  // reactive current gain (param, default 2.0)
            double I_max_val = I_max;     // converter current limit (param, default 1.0)
            double P_ramp_rate = P_ramp;  // post-fault ramp rate (param, default 0.2)

            bool lvrt = (Vterm_meas < V_lvrt_thr);

            // ---- Normal-mode: outer PI → SFO current references ----
            double i_q_sfo_ref = Kp_P * e_P + Ki_P * x[0];
            double i_d_sfo_ref = Kp_Q * e_Q + Ki_Q * x[1];

            // SFO → RI frame
            double i_rd_ref =  i_d_sfo_ref * cos_th - i_q_sfo_ref * sin_th;
            double i_rq_ref =  i_d_sfo_ref * sin_th + i_q_sfo_ref * cos_th;

            if (lvrt) {
                // ========================================
                // LVRT MODE: QInject + MaxLim
                // ========================================

                // QInject: reactive current proportional to voltage deviation
                // i*rq_ref = K_lvrt * (V_lvrt_thr - Vterm) * I_max
                // In SFO d-axis (reactive), then transform to RI
                double i_q_inject = K_lvrt_gain * (V_lvrt_thr - Vterm_meas) * I_max_val;
                if (i_q_inject > I_max_val) i_q_inject = I_max_val;
                if (i_q_inject < 0.0) i_q_inject = 0.0;

                // MaxLim: active current uses remaining capacity (circle limiter)
                double i_d_remain = sqrt(fmax(I_max_val * I_max_val - i_q_inject * i_q_inject, 0.0));

                // SFO frame: d-axis = reactive (Q), q-axis = active (P)
                i_d_sfo_ref = i_q_inject;   // reactive injection → d-axis
                i_q_sfo_ref = i_d_remain;   // reduced active current → q-axis

                // SFO → RI frame
                i_rd_ref =  i_d_sfo_ref * cos_th - i_q_sfo_ref * sin_th;
                i_rq_ref =  i_d_sfo_ref * sin_th + i_q_sfo_ref * cos_th;

                // Latch the ramp state to measured active power (not SFO d-axis)
                double P_fault = inputs[2];  // measured Pe
                dxdt[4] = (P_fault - x[4]) * 100.0;  // fast tracking during fault

            } else {
                // ========================================
                // NORMAL / POST-FAULT RECOVERY MODE
                // ========================================

                // RampLim: x[4] ramps from fault-exit value toward Pref
                double Pref_target = inputs[0];  // desired active power reference
                double P_err = Pref_target - x[4];

                // Ramp rate limited
                if (P_err > P_ramp_rate) {
                    dxdt[4] = P_ramp_rate;  // ramp up at P_ramp pu/s
                } else if (P_err < -P_ramp_rate) {
                    dxdt[4] = -P_ramp_rate;
                } else {
                    // Close to target: fast convergence
                    dxdt[4] = P_err * 10.0;
                }

                // Use the ramped P reference instead of raw Pref for outer loop
                e_P = x[4] - inputs[2];  // Pramp - Pe

                // Recompute outer PI with ramped P ref
                i_q_sfo_ref = Kp_P * e_P + Ki_P * x[0];
                i_d_sfo_ref = Kp_Q * e_Q + Ki_Q * x[1];

                // SFO → RI frame
                i_rd_ref =  i_d_sfo_ref * cos_th - i_q_sfo_ref * sin_th;
                i_rq_ref =  i_d_sfo_ref * sin_th + i_q_sfo_ref * cos_th;
            }

            // ---- Circular current limiter (like GFM) ----
            double I_ref_mag = sqrt(i_rd_ref * i_rd_ref + i_rq_ref * i_rq_ref);
            bool cur_sat = (I_ref_mag > I_max_val);
            if (cur_sat && I_ref_mag > 1e-6) {
                double sc = I_max_val / I_ref_mag;
                i_rd_ref *= sc;
                i_rq_ref *= sc;
            }

            // HVRT i_rd clamp
            if (Vterm_meas > 1.1) {
                double i_rd_eff_max = I_max_val - 8.0 * (Vterm_meas - 1.1);
                if (i_rd_eff_max < -0.5) i_rd_eff_max = -0.5;
                if (i_rd_ref > i_rd_eff_max) i_rd_ref = i_rd_eff_max;
            }

            // Inner loop errors
            double e_ird = i_rd_ref - inputs[4];
            double e_irq = i_rq_ref - inputs[5];

            // Cross-coupling decoupling
            double sigma_Lr = (Ls * Lr - Lm * Lm) / Ls;
            double slip_omega = omega_s - inputs[6];
            slip_omega = fmax(-0.2, fmin(0.2, slip_omega));
            double Vrd_dec = -slip_omega * (sigma_Lr * inputs[5] + Lm / Ls * inputs[8]);
            double Vrq_dec =  slip_omega * (Lm / Ls * inputs[7] + sigma_Lr * inputs[4]);

            // Virtual impedance voltage drop (opposes fault current)
            double Vrd_vir = Rvir * inputs[4] - Xvir * inputs[5];
            double Vrq_vir = Rvir * inputs[5] + Xvir * inputs[4];

            // Voltage commands with virtual impedance and saturation
            double Vrd_raw = Kp_i * e_ird + Ki_i * x[2] + Vrd_dec - Vrd_vir;
            double Vrq_raw = Kp_i * e_irq + Ki_i * x[3] + Vrq_dec - Vrq_vir;
            double Vrd_cmd = fmax(-Vrd_max, fmin(Vrd_max, Vrd_raw));
            double Vrq_cmd = fmax(-Vrd_max, fmin(Vrd_max, Vrq_raw));

            // --- Anti-windup ---
            bool inner_sat = (fabs(Vrd_raw) > Vrd_max) || (fabs(Vrq_raw) > Vrd_max);
            bool outer_sat_P = cur_sat;
            bool outer_sat_Q = cur_sat;

            // Outer P integrator
            if (inner_sat || outer_sat_P || lvrt) {
                if (!lvrt && ((x[0] > 0.0 && e_P < 0.0) || (x[0] < 0.0 && e_P > 0.0)))
                    dxdt[0] = e_P;
                else
                    dxdt[0] = 0.0;
            } else {
                dxdt[0] = e_P;
            }

            // Outer Q integrator
            if (inner_sat || outer_sat_Q || lvrt) {
                if (!lvrt && ((x[1] > 0.0 && e_Q < 0.0) || (x[1] < 0.0 && e_Q > 0.0)))
                    dxdt[1] = e_Q;
                else
                    dxdt[1] = 0.0;
            } else {
                dxdt[1] = e_Q;
            }

            // Inner d-axis integrator
            if (Vrd_raw > Vrd_max && e_ird > 0.0) dxdt[2] = 0.0;
            else if (Vrd_raw < -Vrd_max && e_ird < 0.0) dxdt[2] = 0.0;
            else dxdt[2] = e_ird;

            // Inner q-axis integrator
            if (Vrq_raw > Vrd_max && e_irq > 0.0) dxdt[3] = 0.0;
            else if (Vrq_raw < -Vrd_max && e_irq < 0.0) dxdt[3] = 0.0;
            else dxdt[3] = e_irq;

            outputs[0] = Vrd_cmd;
            outputs[1] = Vrq_cmd;
            outputs[2] = Vrd_cmd * inputs[4] + Vrq_cmd * inputs[5];
            outputs[3] = i_rd_ref;
        """

    # ------------------------------------------------------------------
    # Symbolic PHS
    # ------------------------------------------------------------------

    def get_symbolic_phs(self):
        from src.symbolic.core import SymbolicPHS

        x_P, x_Q, x_ird, x_irq = sp.symbols('x_P x_Q x_{ird} x_{irq}')
        states = [x_P, x_Q, x_ird, x_irq]

        e_P, e_Q, e_ird, e_irq = sp.symbols('e_P e_Q e_{ird} e_{irq}')
        inputs = [e_P, e_Q, e_ird, e_irq]

        # Hamiltonian: H₅ = ½ ||x₅||² (only PI integrators; x_Pramp excluded)
        H_expr = sp.Rational(1, 2) * (x_P**2 + x_Q**2 + x_ird**2 + x_irq**2)

        # PHS: J=0, R=0 (lossless integrators), g=I₄
        J = sp.zeros(4, 4)
        R = sp.zeros(4, 4)
        g = sp.eye(4)

        return SymbolicPHS(
            name='DFIG_RSC_PHS',
            states=states, inputs=inputs, params={},
            J=J, R=R, g=g, H=H_expr,
            description=(
                'RSC cascaded PI (Σ₅): 4 lossless integrators in PH form. '
                'Note: x_Pramp (state [4]) is an implementation-only ramp '
                'state outside the PHS Hamiltonian.'
            ),
        )

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def init_from_targets(self, targets: dict) -> np.ndarray:
        p = self.params
        Ls = p['Ls']; Lr = p['Lr']; Lm = p['Lm']
        omega_s = p['omega_s']
        Kp_P = p['Kp_P']; Ki_P = p['Ki_P']
        Kp_Q = p['Kp_Q']; Ki_Q = p['Ki_Q']
        Kp_i = p['Kp_i']; Ki_i = p['Ki_i']
        Kv = p.get('Kv', 0.0)

        i_rd_ss   = targets.get('i_rd', 0.0)
        i_rq_ss   = targets.get('i_rq', 0.0)
        Pe_ss     = targets.get('Pe', 0.0)
        Qe_ss     = targets.get('Qe', 0.0)
        omega_ss  = targets.get('omega', omega_s)
        phi_sd_ss = targets.get('phi_sd', 0.0)
        phi_sq_ss = targets.get('phi_sq', 0.0)
        Vrd_ss    = targets.get('Vrd', 0.0)
        Vrq_ss    = targets.get('Vrq', 0.0)
        Vterm_ss  = targets.get('Vterm', targets.get('Vt', 1.0))

        self.params['Vref'] = float(Vterm_ss)
        Vref = self.params['Vref']

        Pref = targets.get('Pref', Pe_ss)
        Qref = targets.get('Qref', Qe_ss)
        self.params['Pref0'] = float(Pref)
        self.params['Qref0'] = float(Qref)

        Qref_eff = Qref - Kv * (Vterm_ss - Vref)
        e_P_ss = Pref - Pe_ss
        e_Q_ss = Qref_eff - Qe_ss

        # SFO angle
        phi_s_mag_ss = (phi_sd_ss**2 + phi_sq_ss**2)**0.5
        if phi_s_mag_ss < 1e-6:
            phi_s_mag_ss = 1e-6
        cos_th = phi_sd_ss / phi_s_mag_ss
        sin_th = phi_sq_ss / phi_s_mag_ss

        # Reverse SFO rotation
        i_d_sfo_ref = i_rd_ss * cos_th + i_rq_ss * sin_th
        i_q_sfo_ref = -i_rd_ss * sin_th + i_rq_ss * cos_th

        x_P = (i_q_sfo_ref - Kp_P * e_P_ss) / Ki_P if abs(Ki_P) > 1e-10 else 0.0
        x_Q = (i_d_sfo_ref - Kp_Q * e_Q_ss) / Ki_Q if abs(Ki_Q) > 1e-10 else 0.0

        # Inner-loop current references
        i_q_tmp = Kp_P * e_P_ss + Ki_P * x_P
        i_d_tmp = Kp_Q * e_Q_ss + Ki_Q * x_Q
        i_rd_ref_ss = i_d_tmp * cos_th - i_q_tmp * sin_th
        i_rq_ref_ss = i_d_tmp * sin_th + i_q_tmp * cos_th

        # Decoupling
        sigma_Lr = (Ls * Lr - Lm**2) / Ls
        slip_omega = max(-0.2, min(0.2, omega_s - omega_ss))
        Vrd_dec = -slip_omega * (sigma_Lr * i_rq_ss + Lm / Ls * phi_sq_ss)
        Vrq_dec =  slip_omega * (Lm / Ls * phi_sd_ss + sigma_Lr * i_rd_ss)

        e_ird_ss = i_rd_ref_ss - i_rd_ss
        e_irq_ss = i_rq_ref_ss - i_rq_ss

        # Virtual impedance steady-state compensation
        Rvir = p.get('Rvir', 0.0)
        Xvir = p.get('Xvir', 0.0)
        Vrd_vir_ss = Rvir * i_rd_ss - Xvir * i_rq_ss
        Vrq_vir_ss = Rvir * i_rq_ss + Xvir * i_rd_ss

        x_ird = (Vrd_ss - Kp_i * e_ird_ss - Vrd_dec + Vrd_vir_ss) / Ki_i if abs(Ki_i) > 1e-10 else 0.0
        x_irq = (Vrq_ss - Kp_i * e_irq_ss - Vrq_dec + Vrq_vir_ss) / Ki_i if abs(Ki_i) > 1e-10 else 0.0

        return self._init_states({
            'x_P': x_P, 'x_Q': x_Q,
            'x_ird': x_ird, 'x_irq': x_irq,
            'x_Pramp': float(Pref),
        })
