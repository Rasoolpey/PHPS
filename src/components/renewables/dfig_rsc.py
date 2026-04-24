"""
DFIG_RSC — Rotor-Side Converter Controller (Port-Hamiltonian)
=============================================================

Cascaded vector-control for the DFIG rotor circuit.

    Outer loop:  P, Q errors  →  rotor current references  (i_rd_ref, i_rq_ref)
    Inner loop:  current errors  →  rotor voltage commands  (Vrd, Vrq)

Each PI controller is a first-order integrator — a stable linear system
whose canonical PH form is given by the Lyapunov construction (Ch. 3):

    H_k = ½ x_k²,   R_k = ½,   dH_k/dt ≤ u_k y_k.

The overall RSC is the cascade of four such primitives.

States (4):
    [0] x_P    — P-loop PI integrator
    [1] x_Q    — Q-loop PI integrator
    [2] x_ird  — i_rd inner-loop PI integrator
    [3] x_irq  — i_rq inner-loop PI integrator

Inputs (9):
    [0] Pref      — active power reference [pu]
    [1] Qref      — reactive power reference [pu]
    [2] Pe        — measured active power (from DFIG) [pu]
    [3] Qe        — measured reactive power (from DFIG) [pu]
    [4] i_rd      — measured rotor d-current (from DFIG) [pu]
    [5] i_rq      — measured rotor q-current (from DFIG) [pu]
    [6] omega_r   — rotor mechanical speed (from DFIG) [pu]
    [7] phi_sd    — stator d-flux (from DFIG) [pu]
    [8] phi_sq    — stator q-flux (from DFIG) [pu]
    [9] Vterm     — terminal voltage magnitude (from DFIG) [pu]

Outputs (4):
    [0] Vrd       — rotor d-voltage command [pu]
    [1] Vrq       — rotor q-voltage command [pu]
    [2] P_rotor   — rotor-side electrical power [pu]
    [3] ird_ref   — i_rd reference (monitoring) [pu]

Parameters:
    Kp_P, Ki_P    — outer P-loop PI gains
    Kp_Q, Ki_Q    — outer Q-loop PI gains
    Kp_i, Ki_i    — inner current-loop PI gains
    Lm, Ls, Lr    — machine inductances (for cross-coupling decoupling)
    omega_s       — synchronous electrical speed [pu]
    Vrd_max       — voltage command saturation [pu]
"""

import numpy as np
from typing import Dict, List, Tuple, Any
from src.core import PowerComponent


class DfigRsc(PowerComponent):
    """
    Rotor-Side Converter controller for the DFIG.

    Implements cascaded PI control with cross-coupling decoupling.
    The controller is a passive PH component (4 integrator states)
    whose canonical Hamiltonian is H = ½ (x_P² + x_Q² + x_ird² + x_irq²).
    """

    @property
    def port_schema(self) -> Dict[str, List[Tuple[str, str, str]]]:
        return {
            'in': [
                ('Pref',    'signal', 'pu'),   # [0]
                ('Qref',    'signal', 'pu'),   # [1]
                ('Pe',      'signal', 'pu'),   # [2]
                ('Qe',      'signal', 'pu'),   # [3]
                ('i_rd',    'signal', 'pu'),   # [4]
                ('i_rq',    'signal', 'pu'),   # [5]
                ('omega_r', 'signal', 'pu'),   # [6]
                ('phi_sd',  'signal', 'pu'),   # [7]
                ('phi_sq',  'signal', 'pu'),   # [8]
                ('Vterm',   'signal', 'pu'),   # [9]
            ],
            'out': [
                ('Vrd',     'signal', 'pu'),   # [0]
                ('Vrq',     'signal', 'pu'),   # [1]
                ('P_rotor', 'signal', 'pu'),   # [2]
                ('ird_ref', 'signal', 'pu'),   # [3]
            ]
        }

    @property
    def state_schema(self) -> List[str]:
        return ['x_P', 'x_Q', 'x_ird', 'x_irq']

    @property
    def param_schema(self) -> Dict[str, str]:
        return {
            'Kp_P':    'Proportional gain, outer P-loop',
            'Ki_P':    'Integral gain, outer P-loop',
            'Kp_Q':    'Proportional gain, outer Q-loop',
            'Ki_Q':    'Integral gain, outer Q-loop',
            'Kp_i':    'Proportional gain, inner current loop',
            'Ki_i':    'Integral gain, inner current loop',
            'Lm':      'DFIG mutual inductance [pu] (for decoupling)',
            'Ls':      'DFIG stator inductance [pu] (for decoupling)',
            'Lr':      'DFIG rotor inductance [pu] (for decoupling)',
            'omega_s': 'Synchronous electrical speed [pu]',
            'Vrd_max': 'Rotor voltage saturation limit [pu]',
            'Kv':      'Voltage-dependent Q droop gain [pu/pu] (Qref_eff = Qref - Kv*(Vterm-Vref))',
            'Vref':    'Terminal voltage reference for Q droop [pu]',
        }

    @property
    def component_role(self) -> str:
        return 'renewable_controller'

    def get_associated_generator(self, comp_map: dict):
        """Return the DFIG generator that this RSC controls."""
        return self.params.get('dfig', None)

    @property
    def observables(self) -> Dict[str, Dict[str, str]]:
        return {
            'Vrd_cmd':  {'description': 'Rotor d-voltage command', 'unit': 'pu',
                         'cpp_expr': 'outputs[0]'},
            'Vrq_cmd':  {'description': 'Rotor q-voltage command', 'unit': 'pu',
                         'cpp_expr': 'outputs[1]'},
            'P_rotor':  {'description': 'Rotor electrical power',  'unit': 'pu',
                         'cpp_expr': 'outputs[2]'},
            'ird_ref':  {'description': 'i_rd reference',          'unit': 'pu',
                         'cpp_expr': 'outputs[3]'},
        }

    def get_cpp_compute_outputs_code(self) -> str:
        return """
            // Voltage-dependent Q droop: Qref_eff = Qref - Kv*(Vterm - Vref)
            double Vterm_meas = inputs[9];
            double Qref_eff = inputs[1] - Kv * (Vterm_meas - Vref);

            // Outer loop errors
            double e_P = inputs[0] - inputs[2];   // Pref - Pe
            double e_Q = Qref_eff - inputs[3];    // Qref_eff - Qe

            // Stator Flux Angle for SFO Reference Frame
            double phi_sd = inputs[7];
            double phi_sq = inputs[8];
            double phi_s_mag = sqrt(phi_sd*phi_sd + phi_sq*phi_sq);
            if (phi_s_mag < 1e-6) phi_s_mag = 1e-6; // prevent div by zero
            double cos_th = phi_sd / phi_s_mag;
            double sin_th = phi_sq / phi_s_mag;

            // PI output generates SFO currents (q-axis for P, d-axis for Q)
            double i_q_sfo_ref = Kp_P * e_P + Ki_P * x[0];
            double i_d_sfo_ref = Kp_Q * e_Q + Ki_Q * x[1];

            // Transform back to fixed RI frame (which DFIG stator natively uses)
            double i_rd_ref =  i_d_sfo_ref * cos_th - i_q_sfo_ref * sin_th;
            double i_rq_ref =  i_d_sfo_ref * sin_th + i_q_sfo_ref * cos_th;

            // Clamp current references to prevent runaway
            double i_ref_max = 1.0;  // pu
            if (i_rq_ref > i_ref_max) i_rq_ref = i_ref_max;
            if (i_rq_ref < -i_ref_max) i_rq_ref = -i_ref_max;
            if (i_rd_ref > i_ref_max) i_rd_ref = i_ref_max;
            if (i_rd_ref < -i_ref_max) i_rd_ref = -i_ref_max;

            // HVRT: voltage-dependent i_rd clamp.  When Vterm > 1.1 pu,
            // progressively reduce the max positive i_rd to limit Q
            // injection and suppress overvoltage.  Acts directly on the
            // current output (not on the PI reference), so the Q
            // integrator is frozen by anti-windup when this clamp is active.
            double i_rd_eff_max = i_ref_max;
            if (Vterm_meas > 1.1) {
                i_rd_eff_max = i_ref_max - 8.0 * (Vterm_meas - 1.1);
                if (i_rd_eff_max < -0.5) i_rd_eff_max = -0.5;
                if (i_rd_ref > i_rd_eff_max) i_rd_ref = i_rd_eff_max;
            }

            // Inner loop errors
            double e_ird = i_rd_ref - inputs[4];
            double e_irq = i_rq_ref - inputs[5];

            // Cross-coupling decoupling terms
            double sigma_Lr = (Ls * Lr - Lm * Lm) / Ls;
            double slip_omega = omega_s - inputs[6];
            // Limit decoupling action away from nominal slip to avoid
            // excessive rotor-voltage feedforward during transients.
            if (slip_omega > 0.2) slip_omega = 0.2;
            if (slip_omega < -0.2) slip_omega = -0.2;
            // Full RI-frame decoupling: both phi_sd (inputs[7]) and phi_sq (inputs[8])
            double Vrd_dec = -slip_omega * (sigma_Lr * inputs[5] + Lm / Ls * inputs[8]);
            double Vrq_dec =  slip_omega * (Lm / Ls * inputs[7] + sigma_Lr * inputs[4]);

            // Voltage commands with saturation
            double Vrd_raw = Kp_i * e_ird + Ki_i * x[2] + Vrd_dec;
            double Vrq_raw = Kp_i * e_irq + Ki_i * x[3] + Vrq_dec;
            double Vrd_cmd = fmax(-Vrd_max, fmin(Vrd_max, Vrd_raw));
            double Vrq_cmd = fmax(-Vrd_max, fmin(Vrd_max, Vrq_raw));

            outputs[0] = Vrd_cmd;
            outputs[1] = Vrq_cmd;
            outputs[2] = Vrd_cmd * inputs[4] + Vrq_cmd * inputs[5];  // P_rotor
            outputs[3] = i_rd_ref;
        """

    def get_cpp_step_code(self) -> str:
        return r"""
            // Voltage-dependent Q droop: Qref_eff = Qref - Kv*(Vterm - Vref)
            double Vterm_meas = inputs[9];
            double Qref_eff = inputs[1] - Kv * (Vterm_meas - Vref);

            // Outer loop errors
            double e_P = inputs[0] - inputs[2];   // Pref - Pe
            double e_Q = Qref_eff - inputs[3];    // Qref_eff - Qe

            // Stator Flux Angle for SFO Reference Frame
            double phi_sd = inputs[7];
            double phi_sq = inputs[8];
            double phi_s_mag = sqrt(phi_sd*phi_sd + phi_sq*phi_sq);
            if (phi_s_mag < 1e-6) phi_s_mag = 1e-6; // prevent div by zero
            double cos_th = phi_sd / phi_s_mag;
            double sin_th = phi_sq / phi_s_mag;

            // PI output generates SFO currents (q-axis for P, d-axis for Q)
            double i_q_sfo_ref = Kp_P * e_P + Ki_P * x[0];
            double i_d_sfo_ref = Kp_Q * e_Q + Ki_Q * x[1];

            // Transform back to fixed RI frame
            double i_rd_ref =  i_d_sfo_ref * cos_th - i_q_sfo_ref * sin_th;
            double i_rq_ref =  i_d_sfo_ref * sin_th + i_q_sfo_ref * cos_th;
            // Clamp current references to prevent runaway
            double i_ref_max = 1.0;  // pu
            if (i_rq_ref > i_ref_max) i_rq_ref = i_ref_max;
            if (i_rq_ref < -i_ref_max) i_rq_ref = -i_ref_max;
            if (i_rd_ref > i_ref_max) i_rd_ref = i_ref_max;
            if (i_rd_ref < -i_ref_max) i_rd_ref = -i_ref_max;

            // HVRT: voltage-dependent i_rd clamp.  When Vterm > 1.1 pu,
            // progressively reduce max positive i_rd to limit Q injection.
            double i_rd_eff_max = i_ref_max;
            if (Vterm_meas > 1.1) {
                i_rd_eff_max = i_ref_max - 8.0 * (Vterm_meas - 1.1);
                if (i_rd_eff_max < -0.5) i_rd_eff_max = -0.5;
                if (i_rd_ref > i_rd_eff_max) i_rd_ref = i_rd_eff_max;
            }

            // Inner loop errors
            double e_ird = i_rd_ref - inputs[4];
            double e_irq = i_rq_ref - inputs[5];

            // Cross-coupling decoupling terms
            double sigma_Lr = (Ls * Lr - Lm * Lm) / Ls;
            double slip_omega = omega_s - inputs[6];
            // Limit decoupling action away from nominal slip to avoid
            // excessive rotor-voltage feedforward during transients.
            if (slip_omega > 0.2) slip_omega = 0.2;
            if (slip_omega < -0.2) slip_omega = -0.2;
            // Full RI-frame decoupling: both phi_sd (inputs[7]) and phi_sq (inputs[8])
            double Vrd_dec = -slip_omega * (sigma_Lr * inputs[5] + Lm / Ls * inputs[8]);
            double Vrq_dec =  slip_omega * (Lm / Ls * inputs[7] + sigma_Lr * inputs[4]);

            // Voltage commands with saturation
            double Vrd_raw = Kp_i * e_ird + Ki_i * x[2] + Vrd_dec;
            double Vrq_raw = Kp_i * e_irq + Ki_i * x[3] + Vrq_dec;
            double Vrd_cmd = fmax(-Vrd_max, fmin(Vrd_max, Vrd_raw));
            double Vrq_cmd = fmax(-Vrd_max, fmin(Vrd_max, Vrq_raw));

            // --- Outer-loop anti-windup ---
            // When the inner loop is voltage-saturated, the outer-loop
            // integrators must freeze to prevent wind-up.
            bool inner_sat = (fabs(Vrd_raw) > Vrd_max) || (fabs(Vrq_raw) > Vrd_max);

            // Check if outer-loop current references hit the clamp
            bool outer_sat_P = (fabs(i_rq_ref) >= i_ref_max - 1e-9);
            bool outer_sat_Q = (i_rd_ref >= i_rd_eff_max - 1e-9) || (i_rd_ref <= -i_ref_max + 1e-9);

            // LVRT: freeze outer integrators during severe voltage sag
            // (fault ride-through — prevents integrator wind-up during fault)
            bool lvrt = (Vterm_meas < 0.85);

            // Outer P integrator
            if (inner_sat || outer_sat_P || lvrt) {
                // Only allow unwind (error drives integrator toward zero)
                // During LVRT: complete freeze (no unwind either)
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

            // --- Inner-loop anti-windup ---
            if (Vrd_raw > Vrd_max && e_ird > 0.0) {
                dxdt[2] = 0.0;
            } else if (Vrd_raw < -Vrd_max && e_ird < 0.0) {
                dxdt[2] = 0.0;
            } else {
                dxdt[2] = e_ird;
            }

            if (Vrq_raw > Vrd_max && e_irq > 0.0) {
                dxdt[3] = 0.0;
            } else if (Vrq_raw < -Vrd_max && e_irq < 0.0) {
                dxdt[3] = 0.0;
            } else {
                dxdt[3] = e_irq;
            }

            // Update outputs
            double P_rotor = Vrd_cmd * inputs[4] + Vrq_cmd * inputs[5];
            outputs[0] = Vrd_cmd;
            outputs[1] = Vrq_cmd;
            outputs[2] = P_rotor;
            outputs[3] = i_rd_ref;
        """

    def init_from_targets(self, targets: dict) -> np.ndarray:
        """Initialise RSC integrator states from DFIG equilibrium targets.

        The integrators are set so that, at t=0, the RSC outputs exactly the
        steady-state Vrd_ss / Vrq_ss, accounting for any non-zero proportional
        errors (e.g. Qref=0 but Qe_ss != 0 at the Kron operating point).

        For an outer PI loop: output = Kp*e + Ki*x_int
        So at steady state: x_int = (output_ss - Kp * e_ss) / Ki
        """
        p = self.params
        Ls = p['Ls'];  Lr = p['Lr'];  Lm = p['Lm']
        omega_s = p['omega_s']
        Kp_P = p['Kp_P'];  Ki_P = p['Ki_P']
        Kp_Q = p['Kp_Q'];  Ki_Q = p['Ki_Q']
        Kp_i = p['Kp_i'];  Ki_i = p['Ki_i']
        Kv   = p.get('Kv', 0.0)
        Vref = p.get('Vref', 1.0)

        # Extract DFIG equilibrium values
        i_rd_ss   = targets.get('i_rd', 0.0)
        i_rq_ss   = targets.get('i_rq', 0.0)
        Pe_ss     = targets.get('Pe', 0.0)
        Qe_ss     = targets.get('Qe', 0.0)
        omega_ss  = targets.get('omega', omega_s)
        phi_sd_ss = targets.get('phi_sd', 0.0)
        Vrd_ss    = targets.get('Vrd', 0.0)
        Vrq_ss    = targets.get('Vrq', 0.0)
        Vterm_ss  = targets.get('Vterm', 1.0)

        # Override Vref to match the actual post-Kron terminal voltage.
        # This prevents the Q droop from creating a mismatch at t=0 that
        # drives a 20s transient to a new equilibrium point.
        self.params['Vref'] = float(Vterm_ss)
        Vref = self.params['Vref']

        # Steady-state reference signals (from wiring constants if available)
        Pref = targets.get('Pref', Pe_ss)   # default: Pref=Pe_ss → e_P=0
        Qref = targets.get('Qref', Qe_ss)   # default: Qref=Qe_ss → e_Q=0
        
        self.params['Pref0'] = float(Pref)
        self.params['Qref0'] = float(Qref)

        # Voltage-dependent Q droop: Qref_eff = Qref - Kv*(Vterm - Vref)
        Qref_eff = Qref - Kv * (Vterm_ss - Vref)

        # Steady-state outer-loop errors (use Qref_eff, not Qref)
        e_P_ss = Pref - Pe_ss
        e_Q_ss = Qref_eff - Qe_ss

        # Outer integrators: x_int = (cmd_ss - Kp * e_ss) / Ki

        # Calculate SFO angle from stator flux at equilibrium
        phi_sd_ss = targets.get('phi_sd', 0.0)
        phi_sq_ss = targets.get('phi_sq', 0.0)
        phi_s_mag_ss = (phi_sd_ss**2 + phi_sq_ss**2)**0.5
        if phi_s_mag_ss < 1e-6: phi_s_mag_ss = 1e-6
        cos_th_ss = phi_sd_ss / phi_s_mag_ss
        sin_th_ss = phi_sq_ss / phi_s_mag_ss

        # Reverse SFO rotation to find steady-state SFO references
        i_d_sfo_ref = i_rd_ss * cos_th_ss + i_rq_ss * sin_th_ss
        i_q_sfo_ref = -i_rd_ss * sin_th_ss + i_rq_ss * cos_th_ss

        # Compute PI outer integrator states using transformed references
        x_P = (i_q_sfo_ref - Kp_P * e_P_ss) / Ki_P if abs(Ki_P) > 1e-10 else 0.0
        x_Q = (i_d_sfo_ref - Kp_Q * e_Q_ss) / Ki_Q if abs(Ki_Q) > 1e-10 else 0.0

        # Inner-loop commands at equilibrium (outer loop drives these)
        i_q_temp = Kp_P * e_P_ss + Ki_P * x_P
        i_d_temp = Kp_Q * e_Q_ss + Ki_Q * x_Q
        i_rd_ref_ss = i_d_temp * cos_th_ss - i_q_temp * sin_th_ss
        i_rq_ref_ss = i_d_temp * sin_th_ss + i_q_temp * cos_th_ss

        # Decoupling feed-forward terms at equilibrium (full RI-frame)
        sigma_Lr = (Ls * Lr - Lm ** 2) / Ls
        slip_omega = omega_s - omega_ss
        slip_omega = max(-0.2, min(0.2, slip_omega))
        phi_sq_ss = targets.get('phi_sq', 0.0)
        Vrd_dec = -slip_omega * (sigma_Lr * i_rq_ss + Lm / Ls * phi_sq_ss)
        Vrq_dec = slip_omega * (Lm / Ls * phi_sd_ss + sigma_Lr * i_rd_ss)

        # Inner current errors at equilibrium (should be zero if outer loop is correct)
        e_ird_ss = i_rd_ref_ss - i_rd_ss   # = 0 by construction above
        e_irq_ss = i_rq_ref_ss - i_rq_ss   # = 0 by construction above

        # Inner integrators: Vcmd = Kp_i*e + Ki_i*x_int + dec → x_int = (Vss - Kp_i*e - dec) / Ki_i
        x_ird = (Vrd_ss - Kp_i * e_ird_ss - Vrd_dec) / Ki_i if abs(Ki_i) > 1e-10 else 0.0
        x_irq = (Vrq_ss - Kp_i * e_irq_ss - Vrq_dec) / Ki_i if abs(Ki_i) > 1e-10 else 0.0

        return self._init_states({
            'x_P': x_P, 'x_Q': x_Q,
            'x_ird': x_ird, 'x_irq': x_irq,
        })
