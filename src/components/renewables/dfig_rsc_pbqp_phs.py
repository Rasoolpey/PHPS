"""
DFIG_RSC_PBQP_PHS — Passivity-Based QP Rotor-Side Converter Controller (Σ₅_PBQP)
======================================================================

Replaces the cascaded PI inner loops of the GFM RSC with a one-step
analytical Passivity-Based Quadratic Program (PB-QP) that minimises the
shifted Hamiltonian (Lyapunov function) of the DFIG electromagnetic subsystem.

**Retained from GFM:**
  - VSG swing equation (Active Power Controller)
  - Reactive Power Controller (droop + integral)

**Replaced by PB-QP:**
  - Voltage PI outer loop
  - Current PI inner loop
  - Feed-forward decoupling
  - Anti-windup logic

The PB-QP solves a one-step QP analytically at each time step:

    min_{V_rd, V_rq}  α_V · V₃(z_{k+1}) + α_u · ||u_k − u*||²
                     + α_ω · (ω_s − ω_N)² + α_v · (V_term − V_ref)²
                     + α_tw · (ΔT_e + (ω_t − ω_g))²
                     + α_ff · (T_e + ΔT_e − T_aero)²

    s.t.  z_{k+1} = z_k + dt · f(z_k, u_k)
          |V_rd|, |V_rq| ≤ V_max
          i_rd² + i_rq² ≤ I_max²

The QP reduces to a 2×2 linear system (Cramer's rule) + constraint projection.

States (4):
    [0] omega_vs  — Virtual angular frequency (APC)
    [1] theta_s   — Virtual angle (APC, integrated from omega_s)
    [2] phi_Qs    — RPC integral state
    [3] xi_P      — APC frequency-error integral (secondary control)

Inputs (11):  Same as DfigRscGfmPHS.
Outputs (4):  Same as DfigRscGfmPHS.
"""

import numpy as np
import sympy as sp
from typing import Dict, List, Tuple, Any
from src.core import PowerComponent


class DfigRscPbqpPHS(PowerComponent):
    """
    Passivity-Based QP RSC controller (Σ₅_PBQP) in Port-Hamiltonian form.

    VSG swing equation + RPC droop + analytical one-step passivity-based QP.
    4 differential states (down from 7 in GFM).
    """

    def __init__(self, name: str, params: dict):
        # Defaults for torsional damping / feedforward (backward compatible)
        params.setdefault('alpha_tw', 0.0)
        params.setdefault('gamma_tw', 1.0)
        params.setdefault('alpha_ff', 0.0)
        params.setdefault('Jratio', 0.25)  # J_g/J_t inertia ratio for model-based feedforward
        params.setdefault('D_t', 0.0)     # Turbine friction [pu] for feedforward correction
        params.setdefault('tau_ff', 2.0)  # Washout filter time constant [s]
        params.setdefault('K_td', 0.0)    # Frequency-domain torsional damping gain [pu]
        params.setdefault('K_ff', 0.0)    # Frequency-domain feedforward gain [pu]
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
                ('omega_t', 'signal', 'pu'),   # [11] turbine speed (from drivetrain)
                ('theta_tw','signal', 'rad'),  # [12] shaft twist angle (from drivetrain)
                ('T_aero',  'signal', 'pu'),   # [13] aerodynamic torque (feedforward)
                ('T_shaft', 'signal', 'pu'),   # [14] shaft torque (feedforward)
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
        return ['omega_vs', 'theta_s', 'phi_Qs', 'xi_P', 'T_aero_filt']

    @property
    def param_schema(self) -> Dict[str, str]:
        return {
            'J':        'Virtual inertia [pu·s²/rad]',
            'Dp':       'APC damping / droop gain [pu]',
            'omega_N':  'Nominal angular frequency [pu]',
            'DQ':       'RPC droop coefficient [pu]',
            'kQs':      'RPC integral gain',
            'ki_P':     'APC frequency-error integral gain',
            'omega_b':  'Base angular frequency [rad/s]',
            'Lm':       'DFIG mutual inductance [pu]',
            'Ls':       'DFIG stator inductance [pu]',
            'Lr':       'DFIG rotor inductance [pu]',
            'Rs':       'DFIG stator resistance [pu]',
            'Rr':       'DFIG rotor resistance [pu]',
            'np_poles': 'DFIG pole pairs',
            'j_inertia':'DFIG generator inertia (2H) [pu·s]',
            'f_damp':   'DFIG generator friction [pu]',
            'omega_s':  'Synchronous speed [pu]',
            'Vrd_max':  'Rotor voltage saturation [pu]',
            'I_max':    'Rotor current saturation [pu]',
            'V_nom':    'Nominal terminal voltage from power flow [pu]',
            # MPC-specific parameters
            'alpha_V':  'Lyapunov (energy) cost weight',
            'alpha_u':  'Control effort cost weight',
            'alpha_omega': 'Frequency deviation cost weight',
            'alpha_volt':  'Voltage tracking cost weight',
            'alpha_tw':  'Torsional damping cost weight (active drivetrain damping)',
            'alpha_ff':  'Feedforward torque-matching cost weight (wind gust compensation)',
            'Jratio':    'Generator-to-turbine inertia ratio J_g/J_t for model-based feedforward',
            'D_t':       'Turbine friction coefficient [pu] for feedforward friction correction',
            'tau_ff':    'Washout filter time constant for feedforward [s] (default 2.0)',
            'K_td':      'Frequency-domain torsional damping gain [pu] (APC supplementary)',
            'K_ff':      'Frequency-domain feedforward gain [pu] (APC supplementary)',
            'gamma_tw':  'Max ratio of torsional-to-base Hessian (adaptive clamp, default 1.0)',
            'dt_mpc':   'MPC prediction step [s]',
            # Equilibrium state for shifted Hamiltonian (set during init)
            'phi_sd_ss': 'Stator d-flux at equilibrium [pu]',
            'phi_sq_ss': 'Stator q-flux at equilibrium [pu]',
            'phi_rd_ss': 'Rotor d-flux at equilibrium [pu]',
            'phi_rq_ss': 'Rotor q-flux at equilibrium [pu]',
            'p_g_ss':    'Generator momentum at equilibrium [pu·s]',
            'Vrd_ss':    'Rotor d-voltage at equilibrium [pu]',
            'Vrq_ss':    'Rotor q-voltage at equilibrium [pu]',
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
            'xi_P':     {'description': 'APC freq-error integral',  'unit': 'pu', 'cpp_expr': 'x[3]'},
        }

    # ------------------------------------------------------------------
    # C++ Code Generation — MPC Core
    # ------------------------------------------------------------------

    def _mpc_reconstruct_and_solve(self) -> str:
        """C++ code block: reconstruct DFIG state, run one-step MPC, return
        rotor voltage commands in the theta_s frame."""
        return r"""
            // ============================================================
            // MPC: Reconstruct DFIG electromagnetic state from inputs
            // ============================================================
            double sigma_LS = Ls * Lr - Lm * Lm;

            // Stator fluxes (already in theta_s frame from rotation above)
            double phi_sd_loc = phi_sd;
            double phi_sq_loc = phi_sq;

            // Rotor fluxes reconstructed from rotor currents + stator fluxes
            // i_rd = (-Lm·φ_sd + Ls·φ_rd) / σ  →  φ_rd = (σ·i_rd + Lm·φ_sd) / Ls
            double phi_rd_loc = (sigma_LS * ird_m + Lm * phi_sd_loc) / Ls;
            double phi_rq_loc = (sigma_LS * iqr_m + Lm * phi_sq_loc) / Ls;

            // Generator momentum
            double omega_g = omega_r_m;
            double p_g_loc = j_inertia * omega_g;

            // Current co-energies (∇H₃ = M⁻¹ x₃)
            double i_sd_m =  (Lr * phi_sd_loc - Lm * phi_rd_loc) / sigma_LS;
            double i_sq_m =  (Lr * phi_sq_loc - Lm * phi_rq_loc) / sigma_LS;
            // ird_m, iqr_m already from rotation

            // ============================================================
            // MPC: Equilibrium state (from steady-state init values)
            // ============================================================
            double phi_sd_eq = phi_sd_ss;
            double phi_sq_eq = phi_sq_ss;
            double phi_rd_eq = phi_rd_ss;
            double phi_rq_eq = phi_rq_ss;
            double p_g_eq    = p_g_ss;

            // Error states (shifted Hamiltonian coordinates)
            double z0 = phi_sd_loc - phi_sd_eq;
            double z1 = phi_sq_loc - phi_sq_eq;
            double z2 = phi_rd_loc - phi_rd_eq;
            double z3 = phi_rq_loc - phi_rq_eq;
            double z4 = p_g_loc    - p_g_eq;

            // ============================================================
            // MPC: Compute Lyapunov function V₃ = ½ z^T M⁻¹ z
            // ============================================================
            double dH0 = (Lr * z0 - Lm * z2) / sigma_LS;
            double dH1 = (Lr * z1 - Lm * z3) / sigma_LS;
            double dH2 = (-Lm * z0 + Ls * z2) / sigma_LS;
            double dH3 = (-Lm * z1 + Ls * z3) / sigma_LS;
            double dH4 = z4 / j_inertia;
            double V_lyap = 0.5 * (z0 * dH0 + z1 * dH1 + z2 * dH2
                                  + z3 * dH3 + z4 * dH4);

            // ============================================================
            // MPC: Autonomous dynamics f(z_k) = (J₃ - R₃)∇H₃ + g₃·u_ext
            // (everything except V_rd, V_rq which we control)
            // ============================================================
            double omega_vs_co = x[0];
            double omega_slip = omega_vs_co - omega_g;
            omega_slip = fmax(-0.2, fmin(0.2, omega_slip));

            // Stator voltages (in theta_s frame) are external inputs
            double Vsd_ext = vds_m;
            double Vsq_ext = vqs_m;

            // J₃·∇H₃ (conservative terms using FULL state, not error)
            double JdH0 =  omega_s * Ls * i_sq_m  + omega_s * Lm * iqr_m;
            double JdH1 = -omega_s * Ls * i_sd_m  - omega_s * Lm * ird_m;
            double JdH2 =  omega_s * Lm * i_sq_m  + omega_s * Lr * iqr_m
                           - np_poles * phi_rq_loc * omega_g;
            double JdH3 = -omega_s * Lm * i_sd_m  - omega_s * Lr * ird_m
                           + np_poles * phi_rd_loc * omega_g;
            double JdH4 =  np_poles * (phi_rq_loc * ird_m - phi_rd_loc * iqr_m);

            // R₃·∇H₃ (dissipation)
            double RdH0 = Rs * i_sd_m;
            double RdH1 = Rs * i_sq_m;
            double RdH2 = Rr * ird_m;
            double RdH3 = Rr * iqr_m;
            double RdH4 = f_damp * omega_g;

            // Shaft torque from drivetrain (external input to p_g)
            // T_shaft comes through the DFIG, not directly here.
            // At steady state p_g is balanced; we predict zero net torque change.

            // Autonomous prediction (without V_rd, V_rq)
            double f0 = omega_b * (JdH0 - RdH0 + Vsd_ext);
            double f1 = omega_b * (JdH1 - RdH1 + Vsq_ext);
            double f2 = omega_b * (JdH2 - RdH2);  // V_rd will be added
            double f3 = omega_b * (JdH3 - RdH3);  // V_rq will be added
            double f4 = JdH4 - RdH4;               // mechanical, no control

            // Predicted state WITHOUT control (p = z + dt * f_auto)
            double p0 = z0 + dt_mpc * f0;
            double p1 = z1 + dt_mpc * f1;
            double p2 = z2 + dt_mpc * f2;
            double p3 = z3 + dt_mpc * f3;
            double p4 = z4 + dt_mpc * f4;

            // ============================================================
            // MPC: Solve 2×2 QP analytically
            //
            // z_{k+1} = p + dt_mpc * omega_b * B * u
            // where B = [0,0; 0,0; 1,0; 0,1; 0,0], u = [V_rd; V_rq]
            //
            // Cost: α_V · ½ z_{k+1}^T Q z_{k+1} + α_u · (u − u_ss)^T (u − u_ss)
            //        + α_tw · (ΔTe − Δω)²
            //
            // ∂cost/∂u = 0 gives:
            // (α_V·dt²·ωb²·B^T Q B + α_u·I + α_tw·A^T A) u
            //   = -α_V·dt·ωb·B^T Q p + α_u·u_ss + α_tw·A^T·Δω
            // ============================================================

            double dtw = dt_mpc * omega_b;  // dt * omega_b

            // ============================================================
            // Drivetrain signals for torsional damping
            // ============================================================
            double omega_t_m  = inputs[11];   // turbine speed [pu]
            double theta_tw_m = inputs[12];   // shaft twist [rad]
            double T_aero_m   = inputs[13];   // aerodynamic torque [pu]
            double T_shaft_m  = inputs[14];   // shaft torque [pu]

            // Torsional speed deviation (twist rate)
            double dtheta_tw = omega_t_m - omega_g;

            // Current electromagnetic torque: Te = np · (phi_rd · iqr - phi_rq · ird)
            double Te_current = np_poles * (phi_rd_loc * iqr_m - phi_rq_loc * ird_m);

            // ============================================================
            // Active drivetrain damping via TORQUE-RATE cost
            //
            // cost_tw = α_tw · (ΔTe + Δω)²
            //
            // where:
            //   ΔTe = Te⁺ − Te  ≈  a_rd·Vrd + a_rq·Vrq   (linearised)
            //   Δω  = ω_t − ω_g                            (twist rate)
            //
            //   a_rd = ∂Te⁺/∂Vrd = −n_p · dtw · i_qr
            //   a_rq = ∂Te⁺/∂Vrq =  n_p · dtw · i_rd
            //
            // Sign convention:  The generator equation is
            //    J_g · dω_g/dt = T_shaft − T_e
            // When ω_t > ω_g (shaft winding up, Δω > 0) we need the
            // generator to ACCELERATE → T_e must DECREASE → ΔTe < 0.
            // The cost (ΔTe + Δω)² drives ΔTe → −Δω, correctly
            // reducing Te when the shaft is winding up.
            //
            // At steady state  ω_t ≈ ω_g → Δω → 0  so the cost merely
            // regularises torque changes.  NO steady-state bias.
            //
            // Hessian: H_tw = α_tw · [a_rd²,     a_rd·a_rq]
            //                        [a_rd·a_rq,  a_rq²    ]
            //
            // RHS:    rhs_tw = α_tw · [a_rd · (−Δω)]
            //                         [a_rq · (−Δω)]
            // ============================================================
            double a_rd = -np_poles * dtw * iqr_m;
            double a_rq =  np_poles * dtw * ird_m;
            double r_tw  = -dtheta_tw;  // NEGATIVE twist rate: active damping opposes twist

            // ============================================================
            // Slow-reference feedforward — total-change detector
            //
            // x[4] = T_aero_ref: a SLOW low-pass filter of T_aero.
            //   dT_aero_ref/dt = (T_aero − T_aero_ref) / tau_ff
            //
            // tau_ff ≈ 5 s: LONGER than the gust ramp duration.
            // During a 10 s gust ramp, the reference barely moves, so
            // (T_aero − T_aero_ref) captures the FULL aerodynamic change.
            //
            // deficit = −Jratio · (T_aero − T_aero_ref)
            //
            // Physics:  The generator equation is
            //   J_g · dω_g/dt = T_shaft − T_e
            // During a gust-up, T_aero rises → T_shaft rises → to keep
            // twist rate zero we need Te to DECREASE (generator decelerates
            // less than turbine accelerates when Jratio < 1).
            //   deficit < 0 → pushes Te DOWN → less twist  ✓
            //
            // Deadband: |deficit| < 0.02 pu → deficit = 0.
            // This eliminates steady-state noise from small T_aero
            // fluctuations.  The QP is completely clean during no-gust.
            // ============================================================
            double Te_deficit_raw = -Jratio * (T_aero_m - x[4]);
            double Te_deficit = Te_deficit_raw;
            if (fabs(Te_deficit) < 0.02) Te_deficit = 0.0;

            // ============================================================
            // Adaptive torsional weight clamping
            //
            // At high alpha_tw the torsional Hessian  α_tw·(a_rd²+a_rq²)
            // can dominate the base Hessian (Lyapunov + regularisation),
            // causing the QP to prioritise torque-rate matching over
            // voltage regulation.  The resulting aggressive Te modulation
            // becomes a positive-feedback excitation source for torsional
            // oscillations — the exact opposite of the desired effect.
            //
            // Fix: cap the effective alpha_tw so that
            //   alpha_tw_eff · |a|²  ≤  gamma_tw · H_base
            // where  H_base = α_V·dtw²·BtQB + α_u.
            //
            // gamma_tw = 1.0 → torsional term ≤ base term (default)
            // gamma_tw = 0.5 → torsional term ≤ 50 % of base term
            // ============================================================
            double a_sq = a_rd * a_rd + a_rq * a_rq;
            double alpha_tw_eff = alpha_tw;
            if (a_sq > 1e-12) {
                double H_base_diag = alpha_V * dtw * dtw * (Ls / (Ls * Lr - Lm * Lm)) + alpha_u;
                double alpha_damp_max = gamma_tw * H_base_diag / a_sq;
                if (alpha_tw > alpha_damp_max) {
                    alpha_tw_eff = alpha_damp_max;
                }
            }

            // ============================================================
            // B^T Q_r p  — block-diagonal Lyapunov gradient (ROTOR ONLY)
            //
            // The full M⁻¹ has off-diagonal blocks (-Lm/σ) that couple
            // stator flux errors to rotor voltage commands.  During a
            // fault the stator flux is driven by the collapsed grid
            // voltage and cannot be corrected through V_rd/V_rq.  The
            // cross-coupling causes the MPC to over-react to stator
            // transients, leading to sustained post-fault oscillations.
            //
            // Fix: use block-diagonal Q_r = diag(·, ·, Ls/σ, Ls/σ, ·)
            // so only rotor flux errors drive rotor voltage commands.
            // ============================================================
            double BtQp_0 = (Ls * p2) / sigma_LS;
            double BtQp_1 = (Ls * p3) / sigma_LS;

            // B^T Q_r B  (2×2 submatrix — unchanged)
            double BtQB_diag = Ls / sigma_LS;

            // ============================================================
            // QP Hessian and RHS  (Lyapunov + regularisation + torsional)
            //
            // Model-inversion feedforward: Te_deficit is projected onto
            // the voltage sensitivity direction A = [a_rd; a_rq] and
            // added to the RHS with its own weight alpha_ff.  The
            // feedforward does NOT enter the Hessian (no A^T·A term),
            // so it cannot dominate the QP.  It acts as an external
            // gradient that biases the solution toward Te*.
            //
            // The bounded shift ΔTe_ff = clamp(Te_deficit, ±ΔTe_max)
            // prevents the feedforward from requesting more than one
            // step can deliver (ΔTe_max = |A|·Vrd_max).
            // ============================================================
            double ff_rhs0 = 0.0;
            double ff_rhs1 = 0.0;
            if (a_sq > 1e-12 && fabs(alpha_ff) > 1e-12) {
                double a_norm = sqrt(a_sq);
                double dTe_max = a_norm * Vrd_max;
                double deficit_clamp = fmax(-dTe_max, fmin(dTe_max, Te_deficit));
                ff_rhs0 = alpha_ff * a_rd * deficit_clamp;
                ff_rhs1 = alpha_ff * a_rq * deficit_clamp;
            }

            double H00 = alpha_V * dtw * dtw * BtQB_diag + alpha_u
                       + alpha_tw_eff * a_rd * a_rd;
            double H01 = alpha_tw_eff * a_rd * a_rq;
            double H11 = alpha_V * dtw * dtw * BtQB_diag + alpha_u
                       + alpha_tw_eff * a_rq * a_rq;

            double rhs0 = -alpha_V * dtw * BtQp_0 + alpha_u * Vrd_ss
                        + alpha_tw_eff * a_rd * r_tw + ff_rhs0;
            double rhs1 = -alpha_V * dtw * BtQp_1 + alpha_u * Vrq_ss
                        + alpha_tw_eff * a_rq * r_tw + ff_rhs1;

            // NOTE: Voltage tracking via ∂v_ds/∂V_rd removed.
            // The electromagnetic sensitivity dv_dVrd ≈ -1226 gives a
            // closed-loop gain of ~4e6, causing violent oscillations.
            // Terminal voltage is an algebraic (network) variable with
            // much smaller true sensitivity.  Voltage regulation is
            // handled by the outer-loop integrators (phi_Qs, xi_P).

            // Solve 2×2 system via Cramer's rule (H is now non-diagonal
            // due to the torsional damping cross-coupling term)
            double det_H = H00 * H11 - H01 * H01;
            if (fabs(det_H) < 1e-12) det_H = 1e-12;
            double Vrd_opt = ( H11 * rhs0 - H01 * rhs1) / det_H;
            double Vrq_opt = (-H01 * rhs0 + H00 * rhs1) / det_H;

            // ============================================================
            // Passivity-based damping on TOTAL rotor flux derivative.
            // The total derivative: df_rd/dt = f2 + omega_b * Vrd.
            // At equilibrium this is zero, so damping vanishes exactly.
            //
            // u_final = u_opt - R_inj * (f2 + omega_b * u_final)
            // Solving: u_final = (u_opt - R_inj * f2) / (1 + R_inj * omega_b)
            // ============================================================
            double R_inj = 0.05;
            double damp_denom = 1.0 + R_inj * omega_b;
            Vrd_opt = (Vrd_opt - R_inj * f2) / damp_denom;
            Vrq_opt = (Vrq_opt - R_inj * f3) / damp_denom;

            // ============================================================
            // Constraint projection
            // ============================================================

            // Rotor voltage saturation
            Vrd_opt = fmax(-Vrd_max, fmin(Vrd_max, Vrd_opt));
            Vrq_opt = fmax(-Vrd_max, fmin(Vrd_max, Vrq_opt));

            // Current limit: predict currents after applying this voltage
            double i_rd_pred = ird_m + dtw * Vrd_opt / (Lr - Lm * Lm / Ls);
            double i_rq_pred = iqr_m + dtw * Vrq_opt / (Lr - Lm * Lm / Ls);
            double I_pred_mag = sqrt(i_rd_pred * i_rd_pred + i_rq_pred * i_rq_pred);
            if (I_pred_mag > I_max && I_pred_mag > 1e-6) {
                double sc = I_max / I_pred_mag;
                Vrd_opt = Vrd_ss + sc * (Vrd_opt - Vrd_ss);
                Vrq_opt = Vrq_ss + sc * (Vrq_opt - Vrq_ss);
            }

            double Vrd_cmd = Vrd_opt;
            double Vrq_cmd = Vrq_opt;
"""

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

            // --- RPC → Vref ---
            double Q_err  = inputs[1] - Qs_meas;
            double Vref   = V_nom + DQ * Q_err + x[2];
""" + self._mpc_reconstruct_and_solve() + """
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
            // STEP 1: APC — Virtual Swing Equation (VSG) + integral
            //
            // Frequency-domain torsional damping (K_td):
            //   Like adjusting the governor, not the exciter.
            //   When ω_t > ω_g (shaft winding up), REDUCE power ref
            //   → VSG slows → Te decreases → generator accelerates
            //   → twist reduces.  Natural, smooth, no voltage noise.
            //
            // Frequency-domain feedforward (K_ff):
            //   When wind increases, pre-emptively reduce power ref
            //   so Te drops before T_shaft rise reaches generator.
            //
            // J·dω_vs/dt = P* - Ps - Dp·(ω_vs-ω_N) + xi_P
            //              - K_td·(ω_t - ω_g)    [torsional damping]
            //              + K_ff·deficit          [feedforward]
            // ==========================================================
            double omega_vs = x[0];
            double xi_P = x[3];

            // Torsional damping: supplementary power signal
            double omega_t_apc = inputs[11];  // turbine speed
            double omega_g_apc = inputs[10];  // generator speed
            double twist_rate  = omega_t_apc - omega_g_apc;
            double P_td = -K_td * twist_rate;

            // Feedforward: supplementary power signal
            double T_aero_apc  = inputs[13];
            double ff_deficit  = -Jratio * (T_aero_apc - x[4]);
            double P_ff = K_ff * ff_deficit;

            dxdt[0] = (P_star - Ps_meas - Dp * (omega_vs - omega_N)
                       + xi_P + P_td + P_ff) / J;
            dxdt[1] = omega_vs - omega_N;

            // Secondary frequency control: integral of frequency error
            dxdt[3] = ki_P * (omega_N - omega_vs);
            // Anti-windup on xi_P
            double xi_P_max = 0.5;
            if (x[3] > xi_P_max && dxdt[3] > 0.0) dxdt[3] = 0.0;
            if (x[3] < -xi_P_max && dxdt[3] < 0.0) dxdt[3] = 0.0;

            // ==========================================================
            // STEP 2: Frame rotation: RI → virtual dq (theta_s)
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
            // STEP 3: RPC — Reactive Power Controller — retained
            // ==========================================================
            double Q_err  = Q_star - Qs_meas;
            dxdt[2] = kQs * Q_err;
            double phi_Qs_max = 0.1;
            if (x[2] > phi_Qs_max && dxdt[2] > 0.0) dxdt[2] = 0.0;
            if (x[2] < -phi_Qs_max && dxdt[2] < 0.0) dxdt[2] = 0.0;
            double Vref = V_nom + DQ * Q_err + x[2];

            // ==========================================================
            // STEP 3b: Slow-reference filter on T_aero for feedforward
            //
            // tau_ff ≈ 5 s: slow reference tracks the "old" T_aero.
            // Fast-snap: when |T_aero − x[4]| > 0.1 (e.g. at startup
            // when x[4]=0 but T_aero≈0.75), use tau=0.01s so the
            // filter converges in ~30 ms.  This prevents false deficit
            // during the initial transient.
            // ==========================================================
            double T_aero_step = inputs[13];
            if (tau_ff > 1e-12) {
                double ff_err = T_aero_step - x[4];
                double tau_eff = (fabs(ff_err) > 0.1) ? 0.01 : tau_ff;
                dxdt[4] = ff_err / tau_eff;
            } else {
                dxdt[4] = 0.0;
            }
""" + self._mpc_reconstruct_and_solve() + r"""
            // ==========================================================
            // STEP 4: Rotate Vrd/Vrq back to RI frame and output
            // ==========================================================
            double Vrd_out = Vrd_cmd * cos_ts - Vrq_cmd * sin_ts;
            double Vrq_out = Vrd_cmd * sin_ts + Vrq_cmd * cos_ts;
            outputs[0] = Vrd_out;
            outputs[1] = Vrq_out;
            outputs[2] = Vrd_out * ird_ri + Vrq_out * iqr_ri;
            outputs[3] = x[1];
        """

    # ------------------------------------------------------------------
    # Symbolic PHS (reduced: only VSG + RPC states)
    # ------------------------------------------------------------------

    def get_symbolic_phs(self):
        from src.symbolic.core import SymbolicPHS

        omega_vs_, theta_s_ = sp.symbols('omega_vs theta_s')
        phi_Qs_ = sp.Symbol('phi_Qs')
        xi_P_ = sp.Symbol('xi_P')
        states = [omega_vs_, theta_s_, phi_Qs_, xi_P_]

        J_sym = sp.Symbol('J', positive=True)
        Dp_sym = sp.Symbol('Dp', nonnegative=True)
        ki_P_sym = sp.Symbol('ki_P', nonnegative=True)

        # Hamiltonian: H = ½J·ωs² + ½(θs² + ϕQs² + ξP²)
        H_expr = (sp.Rational(1, 2) * J_sym * omega_vs_**2
                  + sp.Rational(1, 2) * (theta_s_**2 + phi_Qs_**2 + xi_P_**2))

        # J matrix: conservative coupling ωs ↔ θs, ωs ↔ ξP
        J_mat = sp.zeros(4, 4)
        J_mat[0, 1] = -1
        J_mat[1, 0] = 1
        J_mat[0, 3] = 1
        J_mat[3, 0] = -1

        # R matrix: damping on ωs only
        R_mat = sp.zeros(4, 4)
        R_mat[0, 0] = Dp_sym / J_sym

        # g matrix: identity
        g_mat = sp.eye(4)

        u_Ps = sp.Symbol('u_Ps')
        u_theta = sp.Symbol('u_theta')
        u_Qs = sp.Symbol('u_Qs')
        u_xi = sp.Symbol('u_xi')
        inputs = [u_Ps, u_theta, u_Qs, u_xi]

        return SymbolicPHS(
            name='DFIG_RSC_PBQP_PHS',
            states=states, inputs=inputs,
            params={'J': J_sym, 'Dp': Dp_sym, 'ki_P': ki_P_sym},
            J=J_mat, R=R_mat, g=g_mat, H=H_expr,
            description=(
                'PB-QP RSC (Σ₅_PBQP): VSG + RPC + freq integral, 4 PHS states. '
                'PI cascades replaced by analytical one-step passivity-based QP.'
            ),
        )

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def init_from_targets(self, targets: dict) -> np.ndarray:
        p = self.params

        # Default values for torsional damping params (backward compatible)
        p.setdefault('alpha_tw', 0.0)
        p.setdefault('gamma_tw', 1.0)
        p.setdefault('alpha_ff', 0.0)
        p.setdefault('tau_ff', 2.0)
        p.setdefault('K_td', 0.0)
        p.setdefault('K_ff', 0.0)

        Ls = p['Ls']; Lr = p['Lr']; Lm = p['Lm']
        Rs = p['Rs']; Rr = p['Rr']
        omega_s_nom = p['omega_s']
        DQ = p['DQ']; kQs = p['kQs']
        omega_N = p['omega_N']
        np_poles = p['np_poles']
        j_inertia = p['j_inertia']
        sigma_LS = Ls * Lr - Lm**2

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

        # APC equilibrium: omega_s = omega_N
        omega_s_ss = omega_N
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

        # V_nom = vds at equilibrium (d-axis aligned)
        self.params['V_nom'] = float(vds_ss)
        phi_Qs_ss = 0.0

        # Store DFIG equilibrium for the MPC prediction model
        phi_rd_ss = (sigma_LS * i_rd_ss + Lm * phi_sd_ss) / Ls
        phi_rq_ss = (sigma_LS * i_rq_ss + Lm * phi_sq_ss) / Ls
        p_g_ss = j_inertia * omega_ss

        # These become compiled constants for the MPC
        self.params['phi_sd_ss'] = float(phi_sd_ss)
        self.params['phi_sq_ss'] = float(phi_sq_ss)
        self.params['phi_rd_ss'] = float(phi_rd_ss)
        self.params['phi_rq_ss'] = float(phi_rq_ss)
        self.params['p_g_ss']    = float(p_g_ss)
        self.params['Vrd_ss']    = float(Vrd_ss)
        self.params['Vrq_ss']    = float(Vrq_ss)

        return self._init_states({
            'omega_vs': omega_s_ss,
            'theta_s':  theta_s_ss,
            'phi_Qs':   phi_Qs_ss,
            'xi_P':     0.0,
            'T_aero_filt': 0.0,
        })
