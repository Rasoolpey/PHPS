"""
DFIG_RSC_GFM_PHS — Grid-Forming Rotor-Side Converter Controller (Σ₅_GFM)
==========================================================================

VSG-based Active Power Controller (APC) + Reactive Power Controller (RPC)
+ voltage outer loop + inner current loop with decoupling.

Based on Ali et al. (SmallSignarithms, 2025), Wang et al. (TimeSequential,
2025), and Zhan et al. (CircularLimiter/TransientModeling, 2025).

States (7):
    [0] omega_s  — Virtual angular frequency (APC)
    [1] theta_s  — Virtual angle (APC, integrated from omega_s)
    [2] phi_Qs   — RPC integral state
    [3] phi_vqs  — q-axis voltage PI integral
    [4] phi_vds  — d-axis voltage PI integral
    [5] phi_ird  — d-axis inner current PI
    [6] phi_iqr  — q-axis inner current PI

Inputs (11):
    [0] P_star   — Active power reference
    [1] Q_star   — Reactive power reference
    [2] Ps       — Measured stator active power
    [3] Qs       — Measured stator reactive power
    [4] vds      — d-axis stator voltage
    [5] vqs      — q-axis stator voltage
    [6] ird      — d-axis rotor current
    [7] iqr      — q-axis rotor current
    [8] phi_sd   — d-axis stator flux linkage (for decoupling)
    [9] phi_sq   — q-axis stator flux linkage (for decoupling)
    [10] omega_r — Rotor electrical angular speed

Outputs (4):
    [0] Vrd      — d-axis rotor voltage command
    [1] Vrq      — q-axis rotor voltage command
    [2] P_rotor  — Rotor slip power to DC link
    [3] theta_s  — Internal voltage angle → to GSC
"""

import numpy as np
import sympy as sp
from typing import Dict, List, Tuple, Any
from src.core import PowerComponent


class DfigRscGfmPHS(PowerComponent):
    """
    GFM RSC controller (Σ₅_GFM) in Port-Hamiltonian form.

    VSG swing equation + RPC droop + voltage PI outer loop +
    inner current PI + decoupling. 7 differential states.
    """

    def __init__(self, name: str, params: dict):
        super().__init__(name, params)

    @property
    def port_schema(self) -> Dict[str, List[Tuple[str, str, str]]]:
        return {
            'in': [
                ('P_star',  'signal', 'pu'),   # [0]
                ('Q_star',  'signal', 'pu'),   # [1]
                ('Ps',      'signal', 'pu'),   # [2]
                ('Qs',      'signal', 'pu'),   # [3]
                ('vds',     'signal', 'pu'),   # [4]
                ('vqs',     'signal', 'pu'),   # [5]
                ('ird',     'signal', 'pu'),   # [6]
                ('iqr',     'signal', 'pu'),   # [7]
                ('phi_sd',  'signal', 'pu'),   # [8]
                ('phi_sq',  'signal', 'pu'),   # [9]
                ('omega_r', 'signal', 'pu'),   # [10]
            ],
            'out': [
                ('Vrd',     'signal', 'pu'),   # [0]
                ('Vrq',     'signal', 'pu'),   # [1]
                ('P_rotor', 'signal', 'pu'),   # [2]
                ('theta_s', 'signal', 'rad'),  # [3]
            ]
        }

    @property
    def state_schema(self) -> List[str]:
        return ['omega_vs', 'theta_s', 'phi_Qs',
                'phi_vqs', 'phi_vds', 'phi_ird', 'phi_iqr']

    @property
    def param_schema(self) -> Dict[str, str]:
        return {
            'J':        'Virtual inertia [pu·s²/rad]',
            'Dp':       'APC damping / droop gain [pu]',
            'omega_N':  'Nominal angular frequency [pu]',
            'DQ':       'RPC droop coefficient [pu]',
            'kQs':      'RPC integral gain',
            'Rvir':     'Virtual resistance [pu]',
            'Xvir':     'Virtual reactance [pu]',
            'KpPs':     'q-axis voltage loop P gain',
            'KiPs':     'q-axis voltage loop I gain',
            'KpQs':     'd-axis voltage loop P gain',
            'KiQs':     'd-axis voltage loop I gain',
            'Kp_i':     'Inner current loop P gain',
            'Ki_i':     'Inner current loop I gain',
            'omega_b':  'Base angular frequency [rad/s]',
            'Lm':       'DFIG mutual inductance [pu]',
            'Ls':       'DFIG stator inductance [pu]',
            'Lr':       'DFIG rotor inductance [pu]',
            'omega_s':  'Synchronous speed [pu]',
            'Vrd_max':  'Rotor voltage saturation [pu]',
            'I_max':    'Rotor current saturation [pu]',
            'V_nom':    'Nominal terminal voltage from power flow [pu]',
        }

    @property
    def component_role(self) -> str:
        return 'renewable_controller'

    def get_associated_generator(self, comp_map: dict):
        return self.params.get('dfig', None)

    @property
    def observables(self) -> Dict[str, Dict[str, str]]:
        return {
            'Vrd_cmd':  {'description': 'Rotor d-voltage command', 'unit': 'pu', 'cpp_expr': 'outputs[0]'},
            'Vrq_cmd':  {'description': 'Rotor q-voltage command', 'unit': 'pu', 'cpp_expr': 'outputs[1]'},
            'P_rotor':  {'description': 'Rotor electrical power',  'unit': 'pu', 'cpp_expr': 'outputs[2]'},
            'theta_s':  {'description': 'Virtual angle (APC)',     'unit': 'rad', 'cpp_expr': 'x[1]'},
            'omega_vs': {'description': 'Virtual frequency (APC)', 'unit': 'pu', 'cpp_expr': 'x[0]'},
        }

    # ------------------------------------------------------------------
    # C++ Code Generation
    # ------------------------------------------------------------------

    def get_cpp_compute_outputs_code(self) -> str:
        return """
            // --- Measurements (RI frame) ---
            double Ps_meas = inputs[2];
            double Qs_meas = inputs[3];
            double vds_ri  = inputs[4];
            double vqs_ri  = inputs[5];
            double ird_ri  = inputs[6];
            double iqr_ri  = inputs[7];
            double phi_sd_ri = inputs[8];
            double phi_sq_ri = inputs[9];
            double omega_r_m = inputs[10];

            // --- Frame rotation: RI → virtual dq (theta_s) ---
            double theta_s = x[1];
            double cos_ts = cos(theta_s);
            double sin_ts = sin(theta_s);
            double vds_m   =  vds_ri * cos_ts + vqs_ri * sin_ts;
            double vqs_m   = -vds_ri * sin_ts + vqs_ri * cos_ts;
            double ird_m   =  ird_ri * cos_ts + iqr_ri * sin_ts;
            double iqr_m   = -ird_ri * sin_ts + iqr_ri * cos_ts;
            double phi_sd  =  phi_sd_ri * cos_ts + phi_sq_ri * sin_ts;
            double phi_sq  = -phi_sd_ri * sin_ts + phi_sq_ri * cos_ts;

            // --- STEP 2: RPC → Vref ---
            double Q_err  = inputs[1] - Qs_meas;
            double Vref   = V_nom + DQ * Q_err + x[2];

            // --- STEP 3: Virtual impedance (theta_s frame) ---
            double vaux_ds = Rvir * ird_m - Xvir * iqr_m;
            double vaux_qs = Rvir * iqr_m + Xvir * ird_m;

            // --- STEP 4: Voltage loop errors ---
            // Virtual EMF = V_term + Zvir·Ir → real part of drop → d-axis, imag → q-axis
            // e_vqs = -(vqs_m + vaux_qs)   drive q-axis voltage to 0
            // e_vds =  (vds_m - Vref + vaux_ds)  drive d-axis to Vref
            double e_vqs = -vqs_m - vaux_qs;
            double e_vds =  vds_m - Vref + vaux_ds;

            // --- STEP 5: Current references from voltage loop ---
            // vqs = ωs·(Ls·isd + Lm·ird)  →  e_vqs drives i_rd_ref
            // vds = -ωs·(Ls·isq + Lm·iqr) →  e_vds drives i_rq_ref
            double i_rd_ref = KpPs * e_vqs + KiPs * x[3];
            double i_rq_ref = KpQs * e_vds + KiQs * x[4];

            // --- STEP 6: Current saturation (circular limiter) ---
            double I_ref_mag = sqrt(i_rd_ref * i_rd_ref + i_rq_ref * i_rq_ref);
            if (I_ref_mag > I_max && I_ref_mag > 1e-6) {
                double sc = I_max / I_ref_mag;
                i_rd_ref *= sc;
                i_rq_ref *= sc;
            }

            // --- STEP 7: Inner current loop errors (theta_s frame) ---
            double e_ird = i_rd_ref - ird_m;
            double e_iqr = i_rq_ref - iqr_m;

            // --- STEP 8: Decoupling (theta_s frame) ---
            // GFM: slip relative to virtual frame (omega_vs), not fixed omega_N
            double sigma_Lr = (Ls * Lr - Lm * Lm) / Ls;
            double omega_vs_co = x[0];
            double omega_slip = omega_vs_co - omega_r_m;
            omega_slip = fmax(-0.2, fmin(0.2, omega_slip));
            double Vrd_dec = -omega_slip * (sigma_Lr * iqr_m + Lm / Ls * phi_sq);
            double Vrq_dec =  omega_slip * (sigma_Lr * ird_m + Lm / Ls * phi_sd);

            // --- STEP 9: Rotor voltage commands (theta_s frame) ---
            double Vrd_raw = Kp_i * e_ird + Ki_i * x[5] + Vrd_dec;
            double Vrq_raw = Kp_i * e_iqr + Ki_i * x[6] + Vrq_dec;
            double Vrd_cmd = fmax(-Vrd_max, fmin(Vrd_max, Vrd_raw));
            double Vrq_cmd = fmax(-Vrd_max, fmin(Vrd_max, Vrq_raw));

            // --- Rotate back to RI frame ---
            outputs[0] = Vrd_cmd * cos_ts - Vrq_cmd * sin_ts;
            outputs[1] = Vrd_cmd * sin_ts + Vrq_cmd * cos_ts;
            outputs[2] = outputs[0] * ird_ri + outputs[1] * iqr_ri;
            outputs[3] = x[1];  // theta_s
        """

    def get_cpp_step_code(self) -> str:
        return r"""
            // --- Measurements (RI frame) ---
            double P_star  = inputs[0];
            double Q_star  = inputs[1];
            double Ps_meas = inputs[2];
            double Qs_meas = inputs[3];
            double vds_ri  = inputs[4];
            double vqs_ri  = inputs[5];
            double ird_ri  = inputs[6];
            double iqr_ri  = inputs[7];
            double phi_sd_ri = inputs[8];
            double phi_sq_ri = inputs[9];
            double omega_r_m = inputs[10];

            // ==========================================================
            // STEP 1: APC — Virtual Swing Equation (VSG)
            //   J·ω̇s = P*s − Ps − Dp·(ωs − ωN)
            //   θ̇s   = ωs
            // ==========================================================
            double omega_vs = x[0];
            dxdt[0] = (P_star - Ps_meas - Dp * (omega_vs - omega_N)) / J;
            dxdt[1] = omega_vs - omega_N;  // virtual angle deviation [rad]

            // ==========================================================
            // Frame rotation: RI → virtual dq (theta_s frame)
            //   The voltage controller must see voltages/currents in the
            //   theta_s frame so that vqs → 0 and vds → |V|.
            // ==========================================================
            double theta_s = x[1];
            double cos_ts = cos(theta_s);
            double sin_ts = sin(theta_s);

            double vds_m   =  vds_ri * cos_ts + vqs_ri * sin_ts;
            double vqs_m   = -vds_ri * sin_ts + vqs_ri * cos_ts;
            double ird_m   =  ird_ri * cos_ts + iqr_ri * sin_ts;
            double iqr_m   = -ird_ri * sin_ts + iqr_ri * cos_ts;
            double phi_sd  =  phi_sd_ri * cos_ts + phi_sq_ri * sin_ts;
            double phi_sq  = -phi_sd_ri * sin_ts + phi_sq_ri * cos_ts;

            // ==========================================================
            // STEP 2: RPC — Reactive Power Controller
            //   Vref = V_nom + DQ·(Q*−Qs) + ϕQs (spec §4)
            // ==========================================================
            double Q_err  = Q_star - Qs_meas;
            dxdt[2] = kQs * Q_err;
            // Clamp phi_Qs: RPC is a trim around V_nom, ±0.1 pu is sufficient
            double phi_Qs_max = 0.1;
            if (x[2] > phi_Qs_max && dxdt[2] > 0.0) dxdt[2] = 0.0;
            if (x[2] < -phi_Qs_max && dxdt[2] < 0.0) dxdt[2] = 0.0;
            double Vref = V_nom + DQ * Q_err + x[2];

            // ==========================================================
            // STEP 3: Virtual impedance (theta_s frame)
            // ==========================================================
            double vaux_ds = Rvir * ird_m - Xvir * iqr_m;
            double vaux_qs = Rvir * iqr_m + Xvir * ird_m;

            // ==========================================================
            // STEP 4: Outer voltage loop
            //   Virtual EMF = V_term + Zvir·Ir → imag part of drop → q-axis, real → d-axis
            //   e_vqs = −(vqs_m + vaux_qs)    drive q-axis to 0
            //   e_vds =  (vds_m − Vref + vaux_ds)  drive d-axis to Vref
            // ==========================================================
            double e_vqs = -vqs_m - vaux_qs;
            double e_vds =  vds_m - Vref + vaux_ds;
            dxdt[3] = e_vqs;
            dxdt[4] = e_vds;

            // ==========================================================
            // STEP 5: Current references from voltage loop
            // vqs = ωs·(Ls·isd + Lm·ird)  →  e_vqs drives i_rd_ref
            // vds = -ωs·(Ls·isq + Lm·iqr) →  e_vds drives i_rq_ref
            // ==========================================================
            double i_rd_ref = KpPs * e_vqs + KiPs * x[3];
            double i_rq_ref = KpQs * e_vds + KiQs * x[4];

            // ==========================================================
            // STEP 6: Current saturation (circular limiter)
            // ==========================================================
            double I_ref_mag = sqrt(i_rd_ref * i_rd_ref + i_rq_ref * i_rq_ref);
            bool cur_sat = (I_ref_mag > I_max);
            if (cur_sat && I_ref_mag > 1e-6) {
                double sc = I_max / I_ref_mag;
                i_rd_ref *= sc;
                i_rq_ref *= sc;
            }

            // ==========================================================
            // STEP 7: Inner current loop errors (theta_s frame)
            // ==========================================================
            double e_ird = i_rd_ref - ird_m;
            double e_iqr = i_rq_ref - iqr_m;

            // ==========================================================
            // STEP 8: Decoupling (theta_s frame)
            // GFM: slip relative to virtual frame (omega_vs), not fixed omega_N
            // ==========================================================
            double sigma_Lr = (Ls * Lr - Lm * Lm) / Ls;
            double omega_slip = omega_vs - omega_r_m;
            omega_slip = fmax(-0.2, fmin(0.2, omega_slip));
            double Vrd_dec = -omega_slip * (sigma_Lr * iqr_m + Lm / Ls * phi_sq);
            double Vrq_dec =  omega_slip * (sigma_Lr * ird_m + Lm / Ls * phi_sd);

            // ==========================================================
            // STEP 9: Rotor voltage commands (theta_s frame) + saturation
            // ==========================================================
            double Vrd_raw = Kp_i * e_ird + Ki_i * x[5] + Vrd_dec;
            double Vrq_raw = Kp_i * e_iqr + Ki_i * x[6] + Vrq_dec;
            double Vrd_cmd = fmax(-Vrd_max, fmin(Vrd_max, Vrd_raw));
            double Vrq_cmd = fmax(-Vrd_max, fmin(Vrd_max, Vrq_raw));

            // ==========================================================
            // STEP 10: Inner PI integrators with anti-windup
            // ==========================================================
            bool Vrd_sat = (fabs(Vrd_raw) > Vrd_max);
            bool Vrq_sat = (fabs(Vrq_raw) > Vrd_max);

            if (Vrd_sat && ((Vrd_raw > 0.0 && e_ird > 0.0) || (Vrd_raw < 0.0 && e_ird < 0.0)))
                dxdt[5] = 0.0;
            else
                dxdt[5] = e_ird;

            if (Vrq_sat && ((Vrq_raw > 0.0 && e_iqr > 0.0) || (Vrq_raw < 0.0 && e_iqr < 0.0)))
                dxdt[6] = 0.0;
            else
                dxdt[6] = e_iqr;

            if (cur_sat) {
                if ((x[3] > 0.0 && e_vqs > 0.0) || (x[3] < 0.0 && e_vqs < 0.0))
                    dxdt[3] = 0.0;
                if ((x[4] > 0.0 && e_vds > 0.0) || (x[4] < 0.0 && e_vds < 0.0))
                    dxdt[4] = 0.0;
            }

            // ==========================================================
            // STEP 11: Rotate Vrd/Vrq back to RI frame and output
            // ==========================================================
            double Vrd_out = Vrd_cmd * cos_ts - Vrq_cmd * sin_ts;
            double Vrq_out = Vrd_cmd * sin_ts + Vrq_cmd * cos_ts;
            outputs[0] = Vrd_out;
            outputs[1] = Vrq_out;
            outputs[2] = Vrd_out * ird_ri + Vrq_out * iqr_ri;
            outputs[3] = x[1];  // theta_s → to GSC
        """

    # ------------------------------------------------------------------
    # Symbolic PHS
    # ------------------------------------------------------------------

    def get_symbolic_phs(self):
        from src.symbolic.core import SymbolicPHS

        omega_vs_, theta_s_ = sp.symbols('omega_vs theta_s')
        phi_Qs_ = sp.Symbol('phi_Qs')
        phi_vqs_, phi_vds_ = sp.symbols('phi_vqs phi_vds')
        phi_ird_, phi_iqr_ = sp.symbols('phi_ird phi_iqr')
        states = [omega_vs_, theta_s_, phi_Qs_, phi_vqs_, phi_vds_, phi_ird_, phi_iqr_]

        J_sym = sp.Symbol('J', positive=True)
        Dp_sym = sp.Symbol('Dp', nonnegative=True)

        # Hamiltonian: H = ½J·ωs² + ½(θs² + ϕQs² + ϕvqs² + ϕvds² + ϕird² + ϕiqr²)
        H_expr = (sp.Rational(1, 2) * J_sym * omega_vs_**2
                  + sp.Rational(1, 2) * (theta_s_**2 + phi_Qs_**2
                  + phi_vqs_**2 + phi_vds_**2 + phi_ird_**2 + phi_iqr_**2))

        # J matrix: conservative coupling ωs ↔ θs
        J_mat = sp.zeros(7, 7)
        J_mat[0, 1] = -1
        J_mat[1, 0] = 1

        # R matrix: damping on ωs only
        R_mat = sp.zeros(7, 7)
        R_mat[0, 0] = Dp_sym / J_sym

        # g matrix: identity (each state has its own input)
        g_mat = sp.eye(7)

        u_Ps = sp.Symbol('u_Ps')
        u_theta = sp.Symbol('u_theta')
        u_Qs = sp.Symbol('u_Qs')
        u_vqs = sp.Symbol('u_vqs')
        u_vds = sp.Symbol('u_vds')
        u_ird = sp.Symbol('u_ird')
        u_iqr = sp.Symbol('u_iqr')
        inputs = [u_Ps, u_theta, u_Qs, u_vqs, u_vds, u_ird, u_iqr]

        return SymbolicPHS(
            name='DFIG_RSC_GFM_PHS',
            states=states, inputs=inputs,
            params={'J': J_sym, 'Dp': Dp_sym},
            J=J_mat, R=R_mat, g=g_mat, H=H_expr,
            description=(
                'GFM RSC (Σ₅_GFM): VSG + RPC + voltage PI + current PI. '
                '7 PHS states. J₅[0,1]=-1, J₅[1,0]=+1 couples ωs↔θs. '
                'Damping Dp/J on ωs dissipates virtual kinetic energy.'
            ),
        )

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def init_from_targets(self, targets: dict) -> np.ndarray:
        p = self.params
        Ls = p['Ls']; Lr = p['Lr']; Lm = p['Lm']
        omega_s_nom = p['omega_s']
        Kp_i = p['Kp_i']; Ki_i = p['Ki_i']
        KpPs = p['KpPs']; KiPs = p['KiPs']
        KpQs = p['KpQs']; KiQs = p['KiQs']
        Rvir = p['Rvir']; Xvir = p['Xvir']
        DQ = p['DQ']; kQs = p['kQs']
        omega_N = p['omega_N']

        # Steady-state values from power flow (RI frame)
        i_rd_ri   = targets.get('i_rd', 0.0)
        i_rq_ri   = targets.get('i_rq', 0.0)
        Pe_ss     = targets.get('Pe', 0.0)
        Qe_ss     = targets.get('Qe', 0.0)
        omega_ss  = targets.get('omega', omega_s_nom)
        phi_sd_ri = targets.get('phi_sd', 0.0)
        phi_sq_ri = targets.get('phi_sq', 0.0)
        Vrd_ri    = targets.get('Vrd', 0.0)
        Vrq_ri    = targets.get('Vrq', 0.0)
        vds_ri    = targets.get('vd_ri', targets.get('vd', 1.0))
        vqs_ri    = targets.get('vq_ri', targets.get('vq', 0.0))

        Pref = targets.get('Pref', Pe_ss)
        Qref = targets.get('Qref', Qe_ss)
        self.params['Pref0'] = float(Pref)
        self.params['Qref0'] = float(Qref)

        # APC equilibrium: omega_s = omega_N, P*s = Ps
        omega_s_ss = omega_N
        # theta_s from load flow angle (bus voltage angle)
        theta_s_ss = targets.get('theta_s', targets.get('a0', 0.0))

        # Rotate all RI-frame quantities to theta_s frame
        cos_ts = np.cos(theta_s_ss)
        sin_ts = np.sin(theta_s_ss)

        vds_ss   =  vds_ri * cos_ts + vqs_ri * sin_ts
        vqs_ss   = -vds_ri * sin_ts + vqs_ri * cos_ts
        i_rd_ss  =  i_rd_ri * cos_ts + i_rq_ri * sin_ts
        i_rq_ss  = -i_rd_ri * sin_ts + i_rq_ri * cos_ts
        phi_sd_ss =  phi_sd_ri * cos_ts + phi_sq_ri * sin_ts
        phi_sq_ss = -phi_sd_ri * sin_ts + phi_sq_ri * cos_ts
        Vrd_ss   =  Vrd_ri * cos_ts + Vrq_ri * sin_ts
        Vrq_ss   = -Vrd_ri * sin_ts + Vrq_ri * cos_ts

        # Virtual impedance (theta_s frame)
        vaux_ds_ss = Rvir * i_rd_ss - Xvir * i_rq_ss
        vaux_qs_ss = Rvir * i_rq_ss + Xvir * i_rd_ss

        # RPC equilibrium: Vref = V_nom + DQ*(Q*-Qs) + phi_Qs
        # V_nom must equal the virtual EMF E_d = vds + vaux_ds at the
        # operating point so that e_vds_ss = 0 exactly.  With Vmag (the
        # terminal voltage) the integrator phi_vds never reaches
        # equilibrium because the virtual impedance offset persists.
        E_d_ss = vds_ss + vaux_ds_ss          # virtual EMF magnitude
        self.params['V_nom'] = float(E_d_ss)  # store as compiled constant
        phi_Qs_ss = 0.0

        # Voltage loop errors at equilibrium:
        # e_vqs = -(vqs + vaux_qs),  e_vds = (vds - Vref + vaux_ds)
        # With V_nom = E_d_ss and Q*=Qs, phi_Qs=0: Vref = E_d_ss
        e_vqs_ss = -vqs_ss - vaux_qs_ss
        e_vds_ss = vds_ss - (E_d_ss + DQ * (Qref - Qe_ss) + phi_Qs_ss) + vaux_ds_ss

        # Back-solve voltage PI states: e_vqs→i_rd_ref, e_vds→i_rq_ref
        phi_vqs_ss = (i_rd_ss - KpPs * e_vqs_ss) / KiPs if abs(KiPs) > 1e-10 else 0.0
        phi_vds_ss = (i_rq_ss - KpQs * e_vds_ss) / KiQs if abs(KiQs) > 1e-10 else 0.0

        # Inner loop back-solve (theta_s frame quantities)
        sigma_Lr = (Ls * Lr - Lm**2) / Ls
        slip_omega = max(-0.2, min(0.2, omega_s_nom - omega_ss))
        Vrd_dec = -slip_omega * (sigma_Lr * i_rq_ss + Lm / Ls * phi_sq_ss)
        Vrq_dec =  slip_omega * (Lm / Ls * phi_sd_ss + sigma_Lr * i_rd_ss)

        e_ird_ss = 0.0
        e_iqr_ss = 0.0

        phi_ird_ss = (Vrd_ss - Kp_i * e_ird_ss - Vrd_dec) / Ki_i if abs(Ki_i) > 1e-10 else 0.0
        phi_iqr_ss = (Vrq_ss - Kp_i * e_iqr_ss - Vrq_dec) / Ki_i if abs(Ki_i) > 1e-10 else 0.0

        return self._init_states({
            'omega_vs': omega_s_ss,
            'theta_s':  theta_s_ss,
            'phi_Qs':   phi_Qs_ss,
            'phi_vqs':  phi_vqs_ss,
            'phi_vds':  phi_vds_ss,
            'phi_ird':  phi_ird_ss,
            'phi_iqr':  phi_iqr_ss,
        })
